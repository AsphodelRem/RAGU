from typing import Any, Dict, List, Tuple, Iterable
from tqdm import tqdm
from collections import defaultdict

from ragu.triplet.base_artifact_extractor import BaseArtifactExtractor
from ragu.chunker.types import Chunk
from ragu.graph.types import Entity, Relation
from ragu.triplet.pipeline.base import PipelineStep
from ragu.common.logger import logger


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
        Runs the entire pipeline in a structured, multi-pass process.
        """
        from ragu.triplet.pipeline.steps import (
            NERStep, NENStep, REStep, EntityDescriptionStep, RelationDescriptionStep
        )

        logger.info("Starting triplet extraction pipeline.")

        # 1. Get step instances from the pipeline
        ner_step = next((s for s in self.steps if isinstance(s, NERStep)), None)
        nen_step = next((s for s in self.steps if isinstance(s, NENStep)), None)
        entity_desc_step = next((s for s in self.steps if isinstance(s, EntityDescriptionStep)), None)
        re_step = next((s for s in self.steps if isinstance(s, REStep)), None)
        relation_desc_step = next((s for s in self.steps if isinstance(s, RelationDescriptionStep)), None)

        if not all([ner_step, nen_step, entity_desc_step, re_step, relation_desc_step]):
            raise ValueError("Missing one or more required pipeline steps.")

        # 2. Pass 1: Process entities for each chunk
        chunk_list = list(chunks)
        chunk_map = {i: chunk for i, chunk in enumerate(chunk_list)}
        all_entities_by_chunk_id: Dict[str, List[Entity]] = defaultdict(list)
        all_normalized_entities_for_re: List[List[Dict[str, Any]]] = []

        for i, chunk in enumerate(tqdm(chunk_list, desc="Pass 1/3: Processing Entities")):
            context = {"text": chunk.content}

            # NER -> NEN -> Entity Description
            context = await ner_step.run(context)
            context = await nen_step.run(context)
            context = await entity_desc_step.run(context)

            # Create Entity objects and store them
            described_entities = context.get("described_entities", [])
            for entity_data in described_entities:
                entity = Entity(
                    entity_name=entity_data["name"],
                    entity_type=entity_data["type"],
                    description=entity_data.get("description", ""),
                    source_chunk_id=[chunk.id],
                )
                all_entities_by_chunk_id[chunk.id].append(entity)
            
            # Prepare data for the RE step
            all_normalized_entities_for_re.append(
                [[e['start'], e['end'], e['type']] for e in described_entities]
            )

        # Flatten the list of all entities
        final_entities = [entity for entities in all_entities_by_chunk_id.values() for entity in entities]

        # 3. Pass 2: Extract relations from all chunks
        logger.info("Pass 2/3: Extracting Relations...")
        re_context = {
            "chunks": [c.content for c in chunk_list],
            "entities_list": all_normalized_entities_for_re
        }
        re_output_context = await re_step.run(re_context)
        raw_relations = re_output_context.get("relations", [])

        # 4. Pass 3: Process and describe relations
        logger.info("Pass 3/3: Processing and Describing Relations...")
        final_relations: List[Relation] = []
        relations_for_description_by_chunk: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

        for raw_rel in tqdm(raw_relations, desc="Processing Relations"):
            chunk_idx = raw_rel.chunk_id
            if chunk_idx in chunk_map:
                chunk = chunk_map[chunk_idx]
                subject_entity = next((e for e in all_entities_by_chunk_id.get(chunk.id, []) if e.entity_name == raw_rel.source_entity), None)
                object_entity = next((e for e in all_entities_by_chunk_id.get(chunk.id, []) if e.entity_name == raw_rel.target_entity), None)

                if subject_entity and object_entity:
                    relation = Relation(
                        subject_id=subject_entity.id,
                        object_id=object_entity.id,
                        subject_name=subject_entity.entity_name,
                        object_name=object_entity.entity_name,
                        relationship_type=raw_rel.relationship_type,
                        description="",  # Placeholder for now
                        relation_strength=raw_rel.relationship_strength,
                        source_chunk_id=[chunk.id]
                    )
                    final_relations.append(relation)
                    relations_for_description_by_chunk[chunk_idx].append({
                        "source_entity": relation.subject_name,
                        "target_entity": relation.object_name,
                        "relationship_type": relation.relationship_type,
                    })

        # Generate descriptions for relations chunk by chunk
        all_described_triplets = []
        for chunk_idx, relations_in_chunk in tqdm(relations_for_description_by_chunk.items(), desc="Describing Relations"):
            if not relations_in_chunk:
                continue
            desc_context = {
                "text": chunk_map[chunk_idx].content,
                "relations_for_description": relations_in_chunk
            }
            described_context = await relation_desc_step.run(desc_context)
            all_described_triplets.extend(described_context.get("triplets", []))

        # Map descriptions back to the final Relation objects
        for triplet in all_described_triplets:
            for relation in final_relations:
                if (relation.subject_name == triplet["source_entity"] and
                    relation.object_name == triplet["target_entity"] and
                    relation.relationship_type == triplet["relationship_type"]):
                    relation.description = triplet.get("description", "")
                    break

        logger.info(f"Triplet extraction pipeline complete. Total extracted: {len(final_entities)} entities, {len(final_relations)} relations.")
        return final_entities, final_relations