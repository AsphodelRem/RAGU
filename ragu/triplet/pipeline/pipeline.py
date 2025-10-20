from typing import Any, Dict, List, Tuple, Iterable
from tqdm import tqdm
from dataclasses import asdict

from ragu.triplet.base_artifact_extractor import BaseArtifactExtractor
from ragu.chunker.types import Chunk
from ragu.graph.types import Entity, Relation
from ragu.triplet.pipeline.base import PipelineStep
from ragu.common.logger import logger
from ragu.triplet.pipeline.models import Triplet


class Pipeline(BaseArtifactExtractor):
    """
    A pipeline for triplet extraction from text.
    """

    def __init__(self, steps: list[PipelineStep]):
        super().__init__(prompts=[])
        self.steps = steps

    async def extract(
        self,
        chunks: Iterable[Chunk],
        *args,
        **kwargs
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Runs the entire pipeline.
        """
        from ragu.triplet.pipeline.steps import NERStep, NENStep, REStep, DescriptionStep

        logger.info("Starting triplet extraction pipeline.")

        ner_step = next((s for s in self.steps if isinstance(s, NERStep)), None)
        nen_step = next((s for s in self.steps if isinstance(s, NENStep)), None)
        re_step = next((s for s in self.steps if isinstance(s, REStep)), None)
        description_step = next((s for s in self.steps if isinstance(s, DescriptionStep)), None)

        if not all([ner_step, nen_step, re_step, description_step]):
            raise ValueError("Missing one or more required pipeline steps (NER, NEN, RE, Description).")

        all_chunks_text: List[str] = []
        all_normalized_entities_for_re: List[List[Dict[str, Any]]] = []
        final_entities: List[Entity] = []
        chunk_map: Dict[int, Chunk] = {}
        chunk_to_entities_map: Dict[str, List[Entity]] = {}

        chunk_list = list(chunks)
        for i, chunk in enumerate(tqdm(chunk_list, desc="Extracting entities from chunks")):
            logger.info(f"Processing chunk {chunk.chunk_order_idx} from doc {chunk.doc_id}.")
            context = {"text": chunk.content}
            chunk_map[i] = chunk

            context = await ner_step.run(context)
            entities_for_nen = context.get("entities", [])

            current_chunk_entities = []
            for entity_dict in entities_for_nen:
                entity = Entity(
                    entity_name=entity_dict["name"],
                    entity_type=entity_dict["type"],
                    description="",
                    source_chunk_id=[chunk.id],
                )
                current_chunk_entities.append(entity)
            
            final_entities.extend(current_chunk_entities)
            chunk_to_entities_map[chunk.id] = current_chunk_entities

            context["entities"] = entities_for_nen
            context = await nen_step.run(context)
            normalized_entities = context.get("normalized_entities", [])
            all_normalized_entities_for_re.append(normalized_entities)
            all_chunks_text.append(chunk.content)

        entities_for_re_service = []
        for chunk_entities in all_normalized_entities_for_re:
            entities_for_re_service.append(
                [[e['start'], e['end'], e['type']] for e in chunk_entities]
            )

        logger.info("Extracting relations for all chunks in a batch.")
        raw_relations_dicts = await re_step.client.extract_relations(all_chunks_text, entities_for_re_service)

        relations_for_description: List[Relation] = []
        relations_by_chunk: Dict[int, List[Dict[str, Any]]] = {}

        for raw_rel_dict in raw_relations_dicts:
            chunk_idx = raw_rel_dict.get("chunk_id")
            if chunk_idx is not None and chunk_idx in chunk_map:
                chunk = chunk_map[chunk_idx]
                subject_name = raw_rel_dict.get("source_entity", "")
                object_name = raw_rel_dict.get("target_entity", "")

                if subject_name and object_name:
                    subject_entity = next(
                        (e for e in chunk_to_entities_map.get(chunk.id, []) if e.entity_name == subject_name), None
                    )
                    object_entity = next(
                        (e for e in chunk_to_entities_map.get(chunk.id, []) if e.entity_name == object_name), None
                    )

                    if subject_entity and object_entity:
                        relation = Relation(
                            subject_name=subject_name,
                            object_name=object_name,
                            relationship_type=raw_rel_dict.get("relationship_type", ""),
                            subject_id=subject_entity.id,
                            object_id=object_entity.id,
                            description="",
                            relation_strength=raw_rel_dict.get("relationship_strength", 1.0),
                            source_chunk_id=[chunk.id],
                        )
                        relations_for_description.append(relation)
                        if chunk_idx not in relations_by_chunk:
                            relations_by_chunk[chunk_idx] = []
                        relations_by_chunk[chunk_idx].append(asdict(relation))

        logger.info(f"Generating descriptions for {len(relations_for_description)} relations.")
        
        all_triplets: List[Triplet] = []
        for chunk_idx, relations_in_chunk in relations_by_chunk.items():
            chunk = chunk_map[chunk_idx]
            relations_for_desc_client = []
            for r_dict in relations_in_chunk:
                relations_for_desc_client.append({
                    "source_entity": r_dict["subject_name"],
                    "target_entity": r_dict["object_name"],
                    "relationship_type": r_dict["relationship_type"],
                })
            context = {"text": chunk.content, "relations": relations_for_desc_client}
            context = await description_step.run(context)
            all_triplets.extend([Triplet(**t) for t in context.get("triplets", [])])

        for triplet in all_triplets:
            for relation in relations_for_description:
                if relation.subject_name == triplet.source_entity and relation.object_name == triplet.target_entity:
                    relation.description = triplet.description
                    break

        logger.info(f"Triplet extraction pipeline complete. Total extracted: {len(final_entities)} entities, {len(relations_for_description)} relations.")
        return final_entities, relations_for_description
