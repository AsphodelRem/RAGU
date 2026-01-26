from __future__ import annotations

from typing import List, Tuple, Optional

from ragu.chunker.types import Chunk
from ragu.common.global_parameters import Settings
from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.common.prompts.messages import ChatMessages, render
from ragu.graph.types import Entity, Relation
from ragu.llm.base_llm import BaseLLM
from ragu.triplet.base_artifact_extractor import BaseArtifactExtractor
from ragu.triplet.types import NEREL_ENTITY_TYPES, NEREL_RELATION_TYPES


class ArtifactsExtractorLLM(BaseArtifactExtractor):
    """
    Extracts entities and relations from text chunks using LLM.

    Pipeline:
      1. Render the `artifact_extraction` instruction in batch mode over chunk texts.
      2. Call the LLM to produce structured artifacts for each chunk.
      3. Optionally render and run `artifact_validation` to refine extracted artifacts.
      4. Convert model outputs into Entity/Relation objects, preserving source chunk ids.
    """

    def __init__(
        self,
        client: BaseLLM,
        do_validation: bool = False,
        language: str | None = None,
        entity_types: Optional[List[str]] = NEREL_ENTITY_TYPES,
        relation_types: Optional[List[str]] = NEREL_RELATION_TYPES,
    ):
        """
        Initialize a new :class:`ArtifactsExtractorLLM`.

        :param client: Language model client for generation and validation.
        :param do_validation: Whether to perform additional LLM-based validation of artifacts.
        :param language: Output text language.
        :param entity_types: List of entity types to guide extraction prompts.
        :param relation_types: List of relation types to guide extraction prompts.
        """
        _PROMPTS = ["artifact_extraction", "artifact_validation"]
        super().__init__(prompts=_PROMPTS)

        self.client = client
        self.do_validation = do_validation
        self.language = language if language else Settings.language
        self.entity_types = ", ".join(entity_types) if entity_types else None
        self.relation_types = ", ".join(relation_types) if relation_types else None

    async def extract(self, chunks: List[Chunk], *args, **kwargs) -> Tuple[List[Entity], List[Relation]]:
        """
        Extract entities and relations from a collection of chunks.

        Steps:
          1) Batch-render the extraction prompt with `context=<chunk_texts>`,
          2) Generate structured artifacts per chunk,
          3) Optionally validate artifacts against the original context,
          4) Convert artifacts into Entity/Relation objects.

        :param chunks: Iterable of Chunk objects.
        :return: (entities, relations) extracted from all chunks.
        """

        entities_result: List[Entity] = []
        relations_result: List[Relation] = []

        context: List[str] = [chunk.content for chunk in chunks]

        extraction_instruction: RAGUInstruction = self.get_prompt("artifact_extraction")
        extraction_conversations: List[ChatMessages] = render(
            extraction_instruction.messages,
            context=context,
            language=self.language,
            entity_types=self.entity_types,
            relation_types=self.relation_types,
        )
        print(extraction_conversations)
        result_list = await self.client.generate(
            conversations=extraction_conversations,
            response_model=extraction_instruction.pydantic_model,
            progress_bar_desc="Extracting a knowledge graph from chunks",
        )

        if self.do_validation:
            validation_instruction: RAGUInstruction = self.get_prompt("artifact_validation")

            validation_conversations: List[ChatMessages] = render(
                validation_instruction.messages,
                artifacts=result_list,
                context=context,
                entity_types=self.entity_types,
                relation_types=self.relation_types,
                language=self.language,
            )

            result_list = await self.client.generate(
                conversations=validation_conversations,
                response_model=validation_instruction.pydantic_model,
                progress_bar_desc="Validation of extracted artifacts",
            )

        result_list = [x for x in result_list if x is not None and not isinstance(x, Exception)]

        for artifacts, chunk in zip(result_list, chunks):
            current_chunk_entities: List[Entity] = []

            # Parse entities
            for result in artifacts.model_dump().get("entities", []):
                if result is None:
                    continue
                entity = Entity(
                    entity_name=result.get("entity_name", ""),
                    entity_type=result.get("entity_type", "UNKNOWN"),
                    description=result.get("description", ""),
                    source_chunk_id=[chunk.id],
                    documents_id=[],
                    clusters=[],
                )
                current_chunk_entities.append(entity)

            entities_result.extend(current_chunk_entities)

            # Parse relations
            for result in artifacts.model_dump().get("relations", []):
                if result is None:
                    continue

                subject_name = result.get("source_entity", "")
                object_name = result.get("target_entity", "")

                if not (subject_name and object_name):
                    continue

                subject_entity = next(
                    (e for e in current_chunk_entities if e.entity_name == subject_name),
                    None,
                )
                object_entity = next(
                    (e for e in current_chunk_entities if e.entity_name == object_name),
                    None,
                )

                if subject_entity and object_entity:
                    relation = Relation(
                        subject_name=subject_name,
                        object_name=object_name,
                        subject_id=subject_entity.id,
                        object_id=object_entity.id,
                        relation_type=result.get("relation_type", "UNKNOWN"),
                        description=result.get("description", ""),
                        relation_strength=result.get("relation_strength", 1.0),
                        source_chunk_id=[chunk.id],
                    )
                    relations_result.append(relation)

        return entities_result, relations_result
