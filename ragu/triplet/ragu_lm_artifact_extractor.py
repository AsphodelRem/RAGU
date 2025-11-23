from __future__ import annotations

import asyncio
import itertools
import re
import time
from typing import List, Tuple, Dict, Any, Optional, Union

import openai
from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from tqdm.asyncio import tqdm_asyncio

from ragu.chunker.types import Chunk
from ragu.common.logger import logger
from ragu.graph.types import Entity, Relation
from ragu.common.cache import CachedLLMCall, CacheResult
from ragu.triplet.base_artifact_extractor import BaseArtifactExtractor
from ragu.utils.ragu_utils import AsyncRunner

# TODO: move system prompts to PromptTemplate class
SYSTEM_PROMPT_RU = "Вы - эксперт в области анализа текстов и извлечения семантической информации из них."


class RaguLmArtifactExtractor(BaseArtifactExtractor):
    def __init__(
        self,
        ragu_lm_vllm_url: str,
        model: str = "RaguTeam/RAGU-lm",
        system_prompt: str = SYSTEM_PROMPT_RU,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 100,
        max_requests_per_minute: Optional[int] = 600,
        max_requests_per_second: Optional[int] = 10,
        concurrency: int = 10,
        request_timeout: int = 60,
    ) -> None:
        """
        Artifact extractor powered by RAGU-LM. Supports only Russian language.

        The full pipeline:
        1. Extract unnormalized entities from raw text.
            For example: "Мама дала кошке воды." -> ["Мама", "кошке"]
        2. Entity normalization (lemmatization).
            For example: кошке -> кошка
        3. Entity description generation.
        4. Relation extraction between entities.

        :param ragu_lm_vllm_url: Base URL of the deployed vLLM server.
        :param model: Model name used for inference.
        :param system_prompt: System instruction for the language model.
        :param temperature: Sampling temperature used in generation.
        :param top_p: Probability mass for nucleus sampling.
        :param top_k: Number of tokens considered in top-k sampling.
        :param max_requests_per_minute: Optional rate limit per minute.
        :param max_requests_per_second: Optional rate limit per second.
        :param concurrency: Maximum number of concurrent asynchronous requests.
        :param request_timeout: Timeout in seconds for each request.
        """
        super().__init__(prompts=[
            "ragu_lm_entity_extraction",
            "ragu_lm_entity_normalization",
            "ragu_lm_entity_description",
            "ragu_lm_relation_description",
        ])

        self.base_url = ragu_lm_vllm_url
        self.model = model

        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        self._sem = asyncio.Semaphore(max(1, concurrency))
        self._rpm = AsyncLimiter(max_requests_per_minute, time_period=60) if max_requests_per_minute else None
        self._rps = AsyncLimiter(max_requests_per_second, time_period=1) if max_requests_per_second else None

        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key="EMPTY",
            timeout=request_timeout,
        )

        self._cached_llm_call = CachedLLMCall(
            fn=lambda text: self._async_call(system_prompt=self.system_prompt, prompt=text),
        )

    async def extract(
        self,
        chunks: List[Chunk],
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Run the full knowledge extraction pipeline via RAGU-LM. Aims to Russian language.

        Perform the following steps:
        - Extract unnormalized entities from raw text.
        - Normalize entities.
        - Generate descriptions for normalized entities.
        - Extract relations between extracted entities within each chunk.

        :param chunks: Text chunks to process.

        :return: Tuple of lists of `Entity` and `Relation` objects - extracted entities and relations.
        """

        await self._check_connection()

        durations = []
        result_entities, result_relations = [], []
        for i, chunk in enumerate(chunks):
            logger.info(f"[{i}/{len(chunks)}] Processing chunk: {chunk.id}")
            start_time = time.time()

            # Extract unnormalized entities from raw text.
            raw = await self.extract_artifacts(chunk)

            # Lemmatize entities.
            normalized_payloads = await self.normalize_entities(raw, chunk)

            # Extract descriptions for entities.
            entities = await self.extract_entity_descriptions(normalized_payloads, chunk)

            # Extract relations between entities.
            relations = await self.extract_relations(entities, chunk)

            end_time = time.time()
            durations.append(end_time - start_time)

            logger.info(f"Chunk processing time: {end_time - start_time:.2f} seconds")
            logger.info(f"Extracted {len(entities)} entities and {len(relations)} relations from chunk {chunk.id}\n")

            result_entities.extend(entities)
            result_relations.extend(relations)

        logger.info(
            f"Extracted {len(result_entities)} entities and {len(result_relations)} relations in total from {len(chunks)} chunks"
        )
        logger.info(f"Total processing time: {sum(durations):.2f} seconds, "
                    f"mean time per chunk: {sum(durations) / len(durations):.2f} seconds"
        )

        return result_entities, result_relations

    async def _async_call(self, system_prompt: str, prompt: str) -> ChatCompletion:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        return await self.client.chat.completions.create(  # type: ignore
            messages=messages,  # type: ignore
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
        )

    @staticmethod
    def _ok(resp: Any) -> bool:
        if isinstance(resp, CacheResult):
            resp = resp.response
        return (resp is not None) and (not isinstance(resp, Exception))

    @staticmethod
    def _content(resp: Any) -> str:
        if isinstance(resp, CacheResult):
            resp = resp.response
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
        try:
            _ = await self._async_call("", "")
        except openai.APIConnectionError:
            raise ConnectionError("It looks like the vllm with RAGU-LM is not running. Run it via 'vllm serve'. See docs for more details.")
        except openai.NotFoundError:
            raise ValueError("It looks like the model is not available. Check the model name that you pass to vllm.")

    async def _run(self, prompts: List[str], description: str = "") -> List[Any]:
        if not prompts:
            return []

        with tqdm_asyncio(total=len(prompts), desc=description if description else None) as pbar:
            runner = AsyncRunner(self._sem, self._rps, self._rpm, pbar)
            results: List[Any] = [None] * len(prompts)
            tasks: list[tuple[int, asyncio.Task[Any]]] = []
            cache_hits = 0

            for idx, prompt in enumerate(prompts):
                cached = self._cached_llm_call.peek(prompt, system_prompt=self.system_prompt)
                if cached is not None:
                    results[idx] = cached
                    cache_hits += 1
                    if pbar:
                        pbar.update(1)
                    continue
                task = asyncio.create_task(
                    runner.make_request(
                        self._cached_llm_call,
                        prompt=prompt,
                        system_prompt=self.system_prompt,
                    )
                )
                tasks.append((idx, task))

            if tasks:
                gathered = await asyncio.gather(*(task for _, task in tasks), return_exceptions=True)
                for (idx, _), value in zip(tasks, gathered):
                    results[idx] = value

            if cache_hits:
                logger.info(f"Cache served {cache_hits}/{len(prompts)} responses for {description or 'LLM call'}")

            return results

    async def extract_artifacts(self, chunk) -> List[Dict[str, Any]]:
        """
        Extract raw entity candidates from input chunks via RAGU-LM.

        :param chunk: Text chunk to process.
        :returns: List of dictionaries with entity lists and chunks from which they were extracted..
        """

        prompt, _ = self.get_prompt("ragu_lm_entity_extraction").get_instruction(text=chunk.content)
        responses = await self._run(prompt, description="Extracting entities")

        extracted = []
        for response in responses:
            if not self._ok(response):
                continue
            lines = self._content(response).splitlines()

            entities = [ln.strip() for ln in lines if ln.strip()]
            unique_entities = list(set(entities))

            if len(unique_entities) != len(entities):
                removed = [entity for entity in entities if entities.count(entity) > 1]
                logger.info(
                    f"Removed {len(entities) - len(unique_entities)} duplicates from entities. "
                    f"Maybe hallucination? Removed entities: {removed}"
                )
            extracted.extend(unique_entities)

        logger.info(f"Extracted {len(extracted)} entities")
        return extracted

    async def normalize_entities(self, entities: List[Dict], chunk: Chunk) -> List[Dict[str, Any]]:
        """
        Normalize extracted entities for consistency.

        Example:
            "Софье Алексеевной" -> "Софья Алексеева"
            "Петру Первому" -> "Петр Первый"
            "Искусственного интеллекта" -> "Искусственный интеллект"

        :param entities: List of raw extracted entities.
        :param chunk: Chunk from which entities were extracted.
        :returns: List of normalized entity payloads.
        """
        prompts, _ = self.get_prompt("ragu_lm_entity_normalization").get_instruction(
            source_text=chunk.content,
            source_entity=entities,
        )

        responses = await self._run(prompts, description="Normalizing entities")

        normalized_entities = []
        for response in responses:
            if not self._ok(response):
                continue
            normalized = self._content(response)
            if not normalized:
                continue
            normalized_entities.append(normalized)

        return normalized_entities

    async def extract_entity_descriptions(self, entities: List[str], chunk: Chunk) -> List[Entity]:
        """
        Generate descriptive summaries for normalized entities.

        :param entities: List of normalized entities and their associated chunks.
        :param chunk: Chunk from which entities were extracted.
        :returns: List of fully described `Entity` objects.
        """
        prompts, _ = self.get_prompt("ragu_lm_entity_description").get_instruction(
            normalized_entity=entities,
            source_text=chunk.content,
        )

        responses = await self._run(prompts, description="Generating entity descriptions")

        described: List[Entity] = []
        candidates: List[tuple[str, str, Chunk]] = []
        for response, old_entity in zip(responses, entities):
            if not self._ok(response):
                continue
            description = self._content(response)
            if not description:
                continue

            candidates.append((old_entity, description, chunk))

        for (name, description, chunk) in candidates:
            entity = Entity(
                entity_name=name,
                entity_type="UNKNOWN",
                description=description,
                source_chunk_id=[chunk.id],
                documents_id=[chunk.doc_id] if getattr(chunk, "doc_id", None) else [],
                clusters=[],
            )
            described.append(entity)

        return described

    async def extract_relations(self, entities: List[Entity], chunk: Chunk) -> List[Relation]:
        """
        Extract relations between extracted entities within each chunk.

        Generate relation description between inner product of entities.

        :param entities: List of `Entity` objects.
        :param chunk: Chunk from which entities were extracted.
        :returns: List of `Relation` objects describing inter-entity links.
        """
        if not entities:
            return []

        template = self.get_prompt("ragu_lm_relation_description")

        prompts, pairs = [], []
        entity_inner_product = list(itertools.permutations(entities, 2))
        for subject_entity, object_entity in entity_inner_product:
            instructions, _ = template.get_instruction(
                first_normalized_entity=subject_entity.entity_name,
                second_normalized_entity=object_entity.entity_name,
                source_text=chunk.content,
            )
            for instruction in instructions:
                pairs.append((subject_entity, object_entity))
                prompts.append(instruction)

        responses = await self._run(prompts, description="Extract relations")

        candidates: List[Relation] = []
        for resp, (subject_entity, object_entity) in zip(responses, pairs):
            if not self._ok(resp):
                continue
            description = self._content(resp)
            relation = Relation(
                subject_id=subject_entity.id,
                object_id=object_entity.id,
                subject_name=subject_entity.entity_name,
                object_name=object_entity.entity_name,
                description=description,
                source_chunk_id=[chunk.id],
            )
            candidates.append(relation)

        relations = self.filter_relations(candidates)
        logger.info(f"Extracted {len(relations)} relations from {len(entities)} entities")

        return relations


    @staticmethod
    def filter_relations(
            relations: List[Relation],
            negative_pattern: Optional[Union[str, re.Pattern[str]]] = None,
    ) -> List[Relation]:
        """
        Filter out relations extracted by RAGU-LM that are empty, irrelevant, or explicitly negated.

        This function applies a combined regular expression pattern that detects
        negations and absence phrases such as "нет связи", "не содержит информации",
        "отсутствует отношение", etc.

        :param relations: List of extracted `Relation` objects.
        :param negative_pattern: Optional custom regular expression pattern to override default.
        :returns: Filtered list of relations with only meaningful descriptions.
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
            neg = re.compile(negative_pattern or r"(?:" + "|".join(NEGATION_PATTERNS) + r")", flags=re.IGNORECASE | re.UNICODE)

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
