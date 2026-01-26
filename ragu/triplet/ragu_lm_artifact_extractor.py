from __future__ import annotations

import asyncio
import itertools
import re
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Union

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from tqdm.asyncio import tqdm_asyncio

from ragu.chunker.types import Chunk
from ragu.common.batch_generator import BatchGenerator
from ragu.common.cache import TextCache, make_llm_cache_key
from ragu.common.logger import logger
from ragu.common.prompts.messages import ChatMessages, UserMessage, render
from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.graph.types import Entity, Relation
from ragu.triplet.base_artifact_extractor import BaseArtifactExtractor


@dataclass
class ChunkContext:
    """
    Tracks extraction state for a single chunk through all pipeline stages.
    """
    chunk: Chunk
    raw_entities: List[str] = field(default_factory=list)
    normalized_entities: List[str] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)


class RaguLmArtifactExtractor(BaseArtifactExtractor):
    """
    RAGU-LM artifact extractor with stage-by-stage batch processing.
    """

    def __init__(
        self,
        ragu_lm_vllm_url: str="",
        model_name: str = "RaguTeam/RAGU-lm",
        api_token: str="EMPTY",
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 100,
        concurrency: int = 64,
        request_timeout: int = 120,
        cache_flush_every: int = 100,
    ) -> None:
        """
        Artifact extractor powered by RAGU-LM with optimized batch processing.

        :param ragu_lm_vllm_url: Base URL of the deployed vLLM server.
        :param model_name: Model name used for inference.
        :param temperature: Sampling temperature used in generation.
        :param top_p: Probability mass for nucleus sampling.
        :param top_k: Number of tokens considered in top-k sampling.
        :param concurrency: Maximum concurrent requests (default 64 for small models).
        :param request_timeout: Timeout in seconds for each request.
        :param cache_flush_every: Flush cache to disk every N requests (default 100).
        """
        super().__init__(prompts=[
            "ragu_lm_entity_extraction",
            "ragu_lm_entity_normalization",
            "ragu_lm_entity_description",
            "ragu_lm_relation_description",
        ])

        self.base_url = ragu_lm_vllm_url
        self.model_name = model_name

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        self._sem = asyncio.Semaphore(max(1, concurrency))

        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=api_token,
            timeout=request_timeout,
        )

        self._cache = TextCache(flush_every_n_writes=cache_flush_every)

    async def extract(
        self,
        chunks: List[Chunk],
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Run optimized knowledge extraction pipeline via RAGU-LM.

        Uses stage-by-stage batch processing for better vLLM utilization:
        1. Extract entities from chunks
        2. Normalize entities across chunks
        3. Generate descriptions for entities
        4. Extract relations for inner product of entities from every chunk

        :param chunks: Text chunks to process.
        :return: Tuple of (entities, relations) extracted from all chunks.
        """
        if not chunks:
            return [], []

        await self._check_connection()
        start_time = time.time()

        contexts = [ChunkContext(chunk=chunk) for chunk in chunks]

        # Stage 1: Extract raw entities from chunks
        logger.info(f"Stage 1/4: Extracting entities from {len(chunks)} chunks...")
        await self._batch_extract_entities(contexts)
        total_raw = sum(len(ctx.raw_entities) for ctx in contexts)
        logger.info(f"Extracted {total_raw} raw entities from {len(chunks)} chunks")

        # Stage 2: Normalize entities
        logger.info(f"Stage 2/4: Normalizing {total_raw} entities...")
        await self._batch_normalize_entities(contexts)
        total_normalized = sum(len(ctx.normalized_entities) for ctx in contexts)
        logger.info(f"Normalized to {total_normalized} entities")

        # Stage 3: Generate descriptions for entities
        logger.info(f"Stage 3/4: Generating descriptions for {total_normalized} entities...")
        await self._batch_generate_descriptions(contexts)
        total_entities = sum(len(ctx.entities) for ctx in contexts)
        logger.info(f"Created {total_entities} entity objects")

        # Stage 4: Extract relations for entity pairs
        total_pairs = sum(len(ctx.entities) * (len(ctx.entities) - 1) for ctx in contexts)
        logger.info(f"Stage 4/4: Extracting relations for {total_pairs} entity pairs...")
        await self._batch_extract_relations(contexts)
        total_relations = sum(len(ctx.relations) for ctx in contexts)
        logger.info(f"Extracted {total_relations} relations")

        all_entities = [e for ctx in contexts for e in ctx.entities]
        all_relations = [r for ctx in contexts for r in ctx.relations]

        await self._cache.flush_cache()

        elapsed = time.time() - start_time
        logger.info(
            f"Extraction complete: {len(all_entities)} entities, {len(all_relations)} relations "
            f"from {len(chunks)} chunks in {elapsed:.2f}s"
        )

        return all_entities, all_relations

    async def _batch_extract_entities(self, contexts: List[ChunkContext]) -> None:
        """
        Stage 1: Extract raw entities from all chunks in a single batch.
        """
        instruction: RAGUInstruction = self.get_prompt("ragu_lm_entity_extraction")

        conversations: List[ChatMessages] = render(
            instruction.messages,
            text=[ctx.chunk.content for ctx in contexts]
        )

        if not conversations:
            return

        responses = await self._run(conversations, description="Extracting entities.")

        # Parse responses back to contexts
        for ctx, response in zip(contexts, responses):
            if not self._ok(response):
                continue

            lines = self._content(response).splitlines()
            entities = [ln.strip() for ln in lines if ln.strip()]
            unique_entities = list(dict.fromkeys(entities))  # Preserve order, remove duplicates

            if len(unique_entities) != len(entities):
                logger.debug(f"Removed {len(entities) - len(unique_entities)} duplicate entities from chunk {ctx.chunk.id}")

            ctx.raw_entities = unique_entities

    async def _batch_normalize_entities(self, contexts: List[ChunkContext]) -> None:
        """
        Stage 2: Normalize all entities across all chunks in a single batch.
        """
        instruction: RAGUInstruction = self.get_prompt("ragu_lm_entity_normalization")

        conversations = []
        prompt_map: List[Tuple[ChunkContext, int]] = []

        for ctx in contexts:
            if not ctx.raw_entities:
                continue

            chunk_conversations: List[ChatMessages] = render(
                instruction.messages,
                source_text=ctx.chunk.content,
                source_entity=ctx.raw_entities,
            )

            for i, conversation in enumerate(chunk_conversations):
                conversations.append(conversation)
                prompt_map.append((ctx, i))

        if not conversations:
            return

        responses = await self._run(conversations, description="Normalizing entities")

        # Parse responses back to contexts
        for (ctx, entity_idx), response in zip(prompt_map, responses):
            if not self._ok(response):
                continue
            normalized = self._content(response)
            if normalized:
                ctx.normalized_entities.append(normalized)

    async def _batch_generate_descriptions(self, contexts: List[ChunkContext]) -> None:
        """
        Stage 3: Generate descriptions for all entities in a single batch.
        """
        instruction: RAGUInstruction = self.get_prompt("ragu_lm_entity_description")

        conversations = []
        prompt_map: List[Tuple[ChunkContext, str]] = []  # (context, entity_name)

        for ctx in contexts:
            if not ctx.normalized_entities:
                continue

            chunk_conversations: List[ChatMessages] = render(
                instruction.messages,
                normalized_entity=ctx.normalized_entities,
                source_text=ctx.chunk.content,
            )

            for conversation, entity_name in zip(chunk_conversations, ctx.normalized_entities):
                conversations.append(conversation)
                prompt_map.append((ctx, entity_name))

        if not conversations:
            return

        responses = await self._run(conversations, description="Generating descriptions")

        # Parse responses back to contexts
        for (ctx, entity_name), response in zip(prompt_map, responses):
            if not self._ok(response):
                continue
            description = self._content(response)
            if not description:
                continue

            entity = Entity(
                entity_name=entity_name,
                entity_type="UNKNOWN",
                description=description,
                source_chunk_id=[ctx.chunk.id],
                documents_id=[ctx.chunk.doc_id] if getattr(ctx.chunk, "doc_id", None) else [],
                clusters=[],
            )
            ctx.entities.append(entity)

    async def _batch_extract_relations(self, contexts: List[ChunkContext]) -> None:
        """
        Stage 4: Extract relations for all entity pairs in a single batch.
        """
        instruction: RAGUInstruction = self.get_prompt("ragu_lm_relation_description")

        conversations = []
        prompt_map: List[Tuple[ChunkContext, Entity, Entity]] = []  # (context, subject, object)

        for ctx in contexts:
            if len(ctx.entities) < 2:
                continue

            entity_pairs = list(itertools.permutations(ctx.entities, 2))
            first_entities = [pair[0].entity_name for pair in entity_pairs]
            second_entities = [pair[1].entity_name for pair in entity_pairs]

            chunk_conversations: List[ChatMessages] = render(
                instruction.messages,
                first_normalized_entity=first_entities,
                second_normalized_entity=second_entities,
                source_text=ctx.chunk.content,
            )

            for conversation, (subject, obj) in zip(chunk_conversations, entity_pairs):
                conversations.append(conversation)
                prompt_map.append((ctx, subject, obj))

        if not conversations:
            return

        responses = await self._run(conversations, description="Extracting relations")

        # Parse responses and collect candidates per context
        context_candidates: Dict[int, List[Relation]] = {id(ctx): [] for ctx in contexts}

        for (ctx, subject, obj), response in zip(prompt_map, responses):
            if not self._ok(response):
                continue
            description = self._content(response)
            relation = Relation(
                subject_id=subject.id,
                object_id=obj.id,
                subject_name=subject.entity_name,
                object_name=obj.entity_name,
                relation_type="UNKNOWN",
                description=description,
                source_chunk_id=[ctx.chunk.id],
            )
            context_candidates[id(ctx)].append(relation)

        # Filter relations per context
        for ctx in contexts:
            candidates = context_candidates[id(ctx)]
            ctx.relations = self.filter_relations(candidates)

    async def _async_call(self, conversation: ChatMessages) -> ChatCompletion:
        """
        Make async call to the LLM with a ChatMessages conversation.

        :param conversation: ChatMessages object containing the conversation history.
        :return: ChatCompletion response from the model.
        """
        return await self.client.chat.completions.create(
            messages=conversation.to_openai(),
            model=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
        )

    @staticmethod
    def _ok(resp: Any) -> bool:
        return (resp is not None) and (not isinstance(resp, Exception))

    @staticmethod
    def _content(resp: Any) -> str:
        if isinstance(resp, str):
            return resp.strip()
        if isinstance(resp, dict):
            try:
                choices = resp.get("choices", [])
                first = choices[0] if choices else {}
                content = first.get("message", {}).get("content", {})
                return str(content).strip()
            except Exception:
                return ""
        try:
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return ""

    async def _check_connection(self) -> None:
        """
        Check if the server with RAGU-lm is reachable and the model is available.
        """
        try:
            test_conversation = ChatMessages.from_messages([UserMessage(content="Hello, how are you?")])
            _ = await self._async_call(test_conversation)
        except openai.APIConnectionError:
            raise ConnectionError(
                "It looks like the vllm with RAGU-LM is not running. "
                "Run it via 'vllm serve'. See docs for more details."
            )
        except openai.NotFoundError:
            raise ValueError(
                "It looks like the model is not available. "
                "Check the model name that you pass to vllm."
            )

    async def _run(self, conversations: List[ChatMessages], description: str = "") -> List[Any]:
        """
        Run LLM inference on a batch of conversations with caching support.

        :param conversations: List of ChatMessages to process.
        :param description: Description for progress bar.
        :return: List of responses (either strings or Exceptions).
        """
        if not conversations:
            return []

        with tqdm_asyncio(total=len(conversations), desc=description if description else None) as pbar:
            results: List[Any] = [None] * len(conversations)
            pending: List[Tuple[int, ChatMessages, str]] = []  # (index, conversation, cache_key)
            cache_hits = 0

            for idx, conversation in enumerate(conversations):
                conversation_str = conversation.to_str()
                cache_key = make_llm_cache_key(
                    content=conversation_str,
                    model_name=self.model_name,
                )
                cached = await self._cache.get(cache_key)
                if cached is not None:
                    results[idx] = cached
                    cache_hits += 1
                    pbar.update(1)
                else:
                    pending.append((idx, conversation, cache_key))

            if cache_hits:
                logger.info(f"Cache served {cache_hits}/{len(conversations)} responses")

            # Process pending requests in batches
            if pending:
                async def process_request(
                        idx: int,
                        conversation: ChatMessages,
                        cache_key: str
                ) -> Tuple[int, str, str, Any]:
                    input_instruction = conversation.to_str()
                    async with self._sem:
                        try:
                            response = await self._async_call(conversation)
                            content = self._content(response)
                            return idx, cache_key, input_instruction, content
                        except Exception as e:
                            return idx, cache_key, input_instruction, e
                        finally:
                            pbar.update(1)

                # Process in batches to save cache periodically
                for batch in BatchGenerator(pending, self._cache.flush_every_n_writes).get_batches():
                    tasks = [
                        process_request(idx, conversation, cache_key)
                        for idx, conversation, cache_key in batch
                    ]
                    completed = await asyncio.gather(*tasks, return_exceptions=True)

                    for result in completed:
                        if isinstance(result, Exception):
                            continue
                        idx, cache_key, input_instruction, content = result
                        results[idx] = content
                        if not isinstance(content, Exception):
                            await self._cache.set(
                                cache_key,
                                content,
                                input_instruction=input_instruction,
                                model_name=self.model_name,
                            )

                    await self._cache.flush_cache()

            await self._cache.flush_cache()
            return results

    @staticmethod
    def filter_relations(
            relations: List[Relation],
            negative_pattern: Optional[Union[str, re.Pattern[str]]] = None,
    ) -> List[Relation]:
        """
        Filter out empty, irrelevant, or negated relations.
        """
        def _clean_bullet(s: str) -> str:
            return re.sub(r"^[\-\u2022]\s*", "", (s or "").strip())

        NEGATION_PATTERNS = [
            r"^\s*$",
            r"^\s*[\-–—]\s*$",
            r"^(?:[-•]\s*)?(?:отсутств\w*\s+(?:связ\w*|отнош\w*)|нет\s+(?:связ\w*|отнош\w*|информац\w*|данн\w*|сведен\w*))\b",
            r"\bтекст\s+не\s+содерж\w*\b",
            r"\b(?:текст\s+)?не\s+содерж\w*\s+информац\w*\s+о\b",
            r"\bнет\s+(?:информац\w*|сведен\w*|данн\w*)(?:\s+о\b|\b)",
            r"\bне\s+явля\w*\s+\w*отнош\w*",
            r"\bнет\s+\w*отнош\w*",
            r"\bотсутств\w*\s+\w*отнош\w*",
            r"\bне\s+содерж\w*\s+\w*отнош\w*",
            r"\bнет\s+явн\w*\s+\w*отнош\w*",
            r"\bнет\s+\w*связ\w*",
            r"\bотсутств\w*\s+\w*связ\w*",
            r"\bсвяз\w*\s+не\s+(?:установ\w*|прослежива\w*|подтвержд\w*|обнаруж\w*)",
            r"\bотнош\w*\s+не\s+(?:установ\w*|прослежива\w*|подтвержд\w*|обнаруж\w*)",
            r"\bне[^.\n]{0,60}(?:содерж\w*|ука\w*|упомина\w*|найд\w*|обнаруж\w*|подтвержд\w*|установ\w*|прослеж\w*)[^.\n]{0,80}(?:связ\w*|отнош\w*|информац\w*)",
        ]

        if isinstance(negative_pattern, re.Pattern):
            neg = negative_pattern
        else:
            neg = re.compile(
                negative_pattern or r"(?:" + "|".join(NEGATION_PATTERNS) + r")",
                flags=re.IGNORECASE | re.UNICODE
            )

        kept: List[Relation] = []
        for rel in relations:
            cleaned = _clean_bullet(rel.description)
            if not cleaned:
                continue
            if neg.search(cleaned):
                continue
            rel.description = cleaned
            kept.append(rel)

        return kept