from __future__ import annotations

import asyncio
import itertools
import re
from typing import Iterable, List, Tuple, Dict, Any, Optional, Union

import openai
from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from tqdm.asyncio import tqdm_asyncio

from ragu.chunker.types import Chunk
from ragu.common.logger import logger
from ragu.graph.types import Entity, Relation
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

        self._chunk_cache: dict[str, Chunk] = {}

    async def extract(
        self,
        chunks: Iterable[Chunk],
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

        # Extract unnormalized entities from raw text.
        raw = await self.extract_artifacts(chunks)

        # Lemmatize entities.
        normalized_payloads = await self.normalize_entities(raw)

        # Extract descriptions for entities.
        entities = await self.extract_entity_descriptions(normalized_payloads)

        # Extract relations between entities.
        relations = await self.extract_relations(entities)

        return entities, relations

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
        return (resp is not None) and (not isinstance(resp, Exception))

    @staticmethod
    def _content(resp: ChatCompletion) -> str:
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

    async def _run(self, prompts: List[str]) -> List[Any]:
        if not prompts:
            return []
        with tqdm_asyncio(total=len(prompts)) as pbar:
            runner = AsyncRunner(self._sem, self._rps, self._rpm, pbar)
            tasks = [
                runner.make_request(
                    self._async_call,
                    system_prompt=self.system_prompt,
                    prompt=p,
                )
                for p in prompts
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

    async def extract_artifacts(self, chunks: Iterable[Chunk]) -> List[Dict[str, Any]]:
        """
        Extract raw entity candidates from input chunks via RAGU-LM.

        :param chunks: Iterable of text chunks.
        :returns: List of dictionaries with entity lists and chunks from which they were extracted..
        """
        chunk_list = list(chunks)
        texts = [c.content for c in chunk_list]

        prompt_template = self.get_prompt("ragu_lm_entity_extraction")
        prompts, _ = prompt_template.get_instruction(text=texts)

        responses = await self._run(prompts)

        extracted: List[Dict[str, Any]] = []
        for resp, chunk in zip(responses, chunk_list):
            if not self._ok(resp):
                continue
            lines = self._content(resp).splitlines()

            entities = [ln.strip() for ln in lines if ln.strip()]
            extracted.append({"entity_list": entities, "chunk": chunk})

        return extracted

    async def normalize_entities(self, entities_and_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize extracted entities for consistency.

        Example:
            "Софье Алексеевной" -> "Софья Алексеева"
            "Петру Первому" -> "Петр Первый"
            "Искусственного интеллекта" -> "Искусственный интеллект"

        :param entities_and_chunks: List of payloads with extracted entity names and their source chunks.
        :returns: List of normalized entity payloads.
        """
        all_prompts: List[str] = []
        backrefs: List[Chunk] = []

        norm_template = self.get_prompt("ragu_lm_entity_normalization")
        for payload in entities_and_chunks:
            entity_list: List[str] = payload.get("entity_list", [])
            chunk: Chunk = payload["chunk"]
            if not entity_list:
                continue

            instructions, _ = norm_template.get_instruction(
                source_entity=entity_list,
                source_text=chunk.content,
            )

            for instr in instructions:
                all_prompts.append(instr)
                backrefs.append(chunk)

        responses = await self._run(all_prompts)

        normalized_payloads: List[Dict[str, Any]] = []
        for resp, chunk in zip(responses, backrefs):
            if not self._ok(resp):
                continue
            normalized = self._content(resp)
            if not normalized:
                continue
            normalized_payloads.append({
                "normalized_entity": normalized,
                "chunk": chunk,
            })

        return normalized_payloads

    async def extract_entity_descriptions(self, entities: List[Dict[str, Any]]) -> List[Entity]:
        """
        Generate descriptive summaries for normalized entities.

        :param entities: List of normalized entities and their associated chunks.
        :returns: List of fully described `Entity` objects.
        """
        desc_template = self.get_prompt("ragu_lm_entity_description")

        prompts: List[str] = []
        meta: List[tuple[str, Chunk]] = []

        for item in entities:
            name: Optional[str] = item.get("normalized_entity")
            chunk: Optional[Chunk] = item.get("chunk")
            if not name or chunk is None:
                continue

            self._chunk_cache[chunk.id] = chunk

            instructions, _ = desc_template.get_instruction(
                normalized_entity=name,
                source_text=chunk.content,
            )
            for instr in instructions:
                prompts.append(instr)
                meta.append((name, chunk))

        responses = await self._run(prompts)

        described: List[Entity] = []
        for resp, (name, chunk) in zip(responses, meta):
            if not self._ok(resp):
                continue
            description = self._content(resp)
            ent = Entity(
                entity_name=name,
                entity_type="UNKNOWN",
                description=description,
                source_chunk_id=[chunk.id],
                documents_id=[chunk.doc_id] if getattr(chunk, "doc_id", None) else [],
                clusters=[],
            )
            described.append(ent)

        return described

    async def extract_relations(self, entities: List[Entity]) -> List[Relation]:
        """
        Extract relations between extracted entities within each chunk.

        Generate relation description between inner product of entities.

        :param entities: List of `Entity` objects.
        :returns: List of `Relation` objects describing inter-entity links.
        """
        if not entities:
            return []

        rel_template = self.get_prompt("ragu_lm_relation_description")

        by_chunk: Dict[str, List[Entity]] = {}
        for entity in entities:
            if not entity.source_chunk_id:
                continue
            by_chunk.setdefault(entity.source_chunk_id[0], []).append(entity)

        prompts: List[str] = []
        meta: List[tuple[Entity, Entity, Chunk]] = []

        for chunk_id, entities in by_chunk.items():
            chunk = self._chunk_cache.get(chunk_id)
            if not chunk:
                continue

            valid = [entity for entity in entities if entity.entity_name]

            if len(valid) < 2:
                continue

            for subject_entity, object_entity in itertools.permutations(valid, 2):
                instructions, _ = rel_template.get_instruction(
                    first_normalized_entity=subject_entity.entity_name,
                    second_normalized_entity=object_entity.entity_name,
                    source_text=chunk.content,
                )
                for instruction in instructions:
                    prompts.append(instruction)
                    meta.append((subject_entity, object_entity, chunk))


        responses = await self._run(prompts)

        relations: List[Relation] = []
        for resp, (subject_entity, object_entity, chunk) in zip(responses, meta):
            if not self._ok(resp):
                continue
            description = self._content(resp)
            relations.append(Relation(
                subject_id=subject_entity.id,
                object_id=object_entity.id,
                subject_name=subject_entity.entity_name,
                object_name=object_entity.entity_name,
                description=description,
                source_chunk_id=[chunk.id],
            ))

        logger.info(f"Extracted {len(relations)} relations from {len(entities)} entities")
        relations = self.filter_relations(relations)
        logger.info(f"Number of relations after filtering: {len(relations)}")

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

        COMBINED_NEG_RU = (
            r"(?:"
            r"^\s*$" 
            r"|^\s*[\-–—]\s*$"  
            r"|^(?:[-•]\s*)?(?:отсутств\w*\s+(?:связ\w*|отнош\w*)|"
            r"нет\s+(?:связ\w*|отнош\w*|информац\w*|данн\w*|сведен\w*))\b"
            r"|\\bтекст\\s+не\\s+содерж\\w*\\b"
            r"|\\b(?:текст\\s+)?не\\s+содерж\\w*\\s+информац\\w*\\s+о\\b"
            r"|\\bнет\\s+(?:информац\\w*|сведен\\w*|данн\\w*)(?:\\s+о\\b|\\b)"
            r"|\\bне\\s+явля\\w*\\s+\\w*отнош\\w*"
            r"|\\bнет\\s+\\w*отнош\\w*"
            r"|\\bотсутств\\w*\\s+\\w*отнош\\w*"
            r"|\\bне\\s+содерж\\w*\\s+\\w*отнош\\w*"
            r"|\\bнет\\s+явн\\w*\\s+\\w*отнош\\w*"
            r"|\\bнет\\s+\\w*связ\\w*"
            r"|\\bотсутств\\w*\\s+\\w*связ\\w*"
            r"|\\bсвяз\\w*\\s+не\\s+(?:установ\\w*|прослежива\\w*|подтвержд\\w*|обнаруж\\w*)"
            r"|\\bотнош\\w*\\s+не\\s+(?:установ\\w*|прослежива\\w*|подтвержд\\w*|обнаруж\\w*)"
            r"|\\bне[^.\n]{0,60}(?:содерж\\w*|ука\\w*|упомина\\w*|найд\\w*|обнаруж\\w*|подтвержд\\w*|установ\\w*|прослеж\\w*)"
            r"[^.\n]{0,80}(?:связ\\w*|отнош\\w*|информац\\w*)"
            r")"
            )

        if isinstance(negative_pattern, re.Pattern):
            NEG = negative_pattern
        else:
            NEG = re.compile(negative_pattern or COMBINED_NEG_RU, flags=re.IGNORECASE | re.UNICODE)

        kept: List[Relation] = []
        for rel in relations:
            desc = rel.description
            cleaned = _clean_bullet(desc)
            if NEG.search(cleaned):
                continue
            rel.description = cleaned
            kept.append(rel)

        return kept


