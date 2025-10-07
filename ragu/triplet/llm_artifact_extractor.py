from __future__ import annotations

from typing import List, Tuple, Optional, Iterable

from ragu.chunker.types import Chunk
from ragu.graph.types import Entity, Relation
from ragu.llm.base_llm import BaseLLM
from ragu.triplet.base_artifact_extractor import BaseArtifactExtractor
from ragu.triplet.types import NEREL_ENTITY_TYPES


class ArtifactsExtractorLLM(BaseArtifactExtractor):
    """
    Extracts entities and relations from text chunks using LLM.

    The class implements an LLM-driven pipeline for artifact extraction:
    - Extract entities and relations from raw texts.
    - Optionally performs LLM-based validation to refine the extracted artifacts.
    """

    def __init__(
        self,
        client: BaseLLM,
        do_validation: bool = False,
        language: str = "russian",
        entity_types: Optional[List[str]] = NEREL_ENTITY_TYPES,
        relation_types: Optional[List[str]] = None
    ):
        """
        Initialize a new :class:`ArtifactsExtractorLLM`.

        :param client: Language model client for generation and validation.
        :param do_validation: Whether to perform additional LLM-based validation of artifacts.
        :param language: Input text language (used for prompt conditioning).
        :param entity_types: List of entity types to guide extraction prompts.
        :param relation_types: List of relation types to guide extraction prompts.
        """
        _PROMPTS = ["artifact_extraction", "artifact_validation"]
        super().__init__(prompts=_PROMPTS)

        self.client = client
        self.do_validation = do_validation
        self.language = language
        self.entity_types = ", ".join(entity_types) if entity_types else None
        self.relation_types = ", ".join(relation_types) if relation_types else None

    async def extract(self, chunks: Iterable[Chunk], *args, **kwargs) -> Tuple[List[Entity], List[Relation]]:
        """
        Extract entities and relations from a collection of text chunks.

        The method performs two sequential steps:
        1. **Extraction:** Extract entities and relations from each chunk/
        2. **Validation (optional):** Refine extracted artifacts..

        For each chunk, entities and relations are created as :class:`Entity` and :class:`Relation`
        objects, preserving source metadata (chunk IDs).

        :param chunks: Iterable of :class:`Chunk` objects containing text content.
        :param args: Additional positional arguments (ignored by default).
        :param kwargs: Additional keyword arguments (ignored by default).
        :return: A tuple ``(entities, relations)`` with all extracted artifacts.
        """
        entities_result, relations_result = [], []

        context = [chunk.content for chunk in chunks]
        prompts, schema = self.get_prompt("artifact_extraction").get_instruction(
            context=context,
            language=self.language,
            entity_types=self.entity_types
        )

        result_list = await self.client.generate(
            prompt=prompts,
            schema=schema,
        )

        if self.do_validation:
            prompts, schema = self.get_prompt("artifact_validation").get_instruction(
                artifacts=result_list,
                context=context,
                entity_types=self.entity_types,
                language=self.language
            )

            result_list = await self.client.generate(
                prompt=prompts,
                schema=schema
            )

        result_list = list(filter(lambda x: x is not None, result_list))

        for artifacts, chunk in zip(result_list, chunks):
            current_chunk_entities = []

            # Parse entities
            for result in artifacts.model_dump().get("entities", []):
                if result is not None:
                    entity = Entity(
                        entity_name=result.get("entity_name", ""),
                        entity_type=result.get("entity_type", ""),
                        description=result.get("description", ""),
                        source_chunk_id=[chunk.id],
                        documents_id=[],
                        clusters=[],
                    )
                    current_chunk_entities.append(entity)
            entities_result.extend(current_chunk_entities)

            # Parse relations
            for result in artifacts.model_dump().get("relations", []):
                if result is not None:
                    subject_name = result.get("source_entity", "")
                    object_name = result.get("target_entity", "")

                    if subject_name and object_name:
                        subject_entity = next(
                            (e for e in current_chunk_entities if e.entity_name == subject_name), None
                        )
                        object_entity = next(
                            (e for e in current_chunk_entities if e.entity_name == object_name), None
                        )

                        if subject_entity and object_entity:
                            relation = Relation(
                                subject_name=subject_name,
                                object_name=object_name,
                                subject_id=subject_entity.id,
                                object_id=object_entity.id,
                                description=result.get("description", ""),
                                relation_strength=result.get("relation_strength", 1.0),
                                source_chunk_id=[chunk.id],
                            )
                            relations_result.append(relation)

        return entities_result, relations_result
