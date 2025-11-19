from typing import Any, Dict, List, Tuple, Iterable, Optional, Union
import re
import asyncio
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

        # 2. Pass 1: Process entities for each chunk in parallel
        chunk_list = list(chunks)
        chunk_map = {i: chunk for i, chunk in enumerate(chunk_list)}
        all_entities_by_chunk_id: Dict[str, List[Entity]] = defaultdict(list)
        all_normalized_entities_for_re: List[List[Dict[str, Any]]] = [[] for _ in chunk_list]

        async def _process_chunk_entities(chunk: Chunk, index: int):
            context = {"text": chunk.content}
            context = await ner_step.run(context)
            context = await nen_step.run(context)
            context = await entity_desc_step.run(context)

            described_entities = context.get("described_entities", [])
            chunk_entities = []
            for entity_data in described_entities:
                entity = Entity(
                    entity_name=entity_data["name"],
                    entity_type=entity_data["type"],
                    description=entity_data.get("description", ""),
                    source_chunk_id=[chunk.id],
                )
                chunk_entities.append(entity)
            
            normalized_for_re = [[e['start'], e['end'], e['type']] for e in described_entities]
            return index, chunk.id, chunk_entities, normalized_for_re

        entity_processing_tasks = [_process_chunk_entities(chunk, i) for i, chunk in enumerate(chunk_list)]
        
        for task in tqdm(asyncio.as_completed(entity_processing_tasks), total=len(entity_processing_tasks), desc="Pass 1/3: Processing Entities"):
            index, chunk_id, chunk_entities, normalized_for_re = await task
            all_entities_by_chunk_id[chunk_id].extend(chunk_entities)
            if index < len(all_normalized_entities_for_re):
                all_normalized_entities_for_re[index] = normalized_for_re

        # Flatten the list of all entities
        final_entities = [entity for entities in all_entities_by_chunk_id.values() for entity in entities]

        # 3. Pass 2: Extract relations from all chunks
        logger.info("Pass 2/3: Extracting Relations...")
        
        def batch_generator(data, batch_size):
            for i in range(0, len(data), batch_size):
                yield data[i:i + batch_size]

        batch_size = 5
        chunk_batches = list(batch_generator(chunk_list, batch_size))
        entity_batches = list(batch_generator(all_normalized_entities_for_re, batch_size))
        
        raw_relations = []
        for i, (chunk_batch, entity_batch) in enumerate(tqdm(zip(chunk_batches, entity_batches), desc="Pass 2/3: Extracting Relations in Batches", total=len(chunk_batches))):
            batch_start_index = i * batch_size
            re_context = {
                "chunks": [c.content for c in chunk_batch],
                "entities_list": entity_batch
            }
            re_output_context = await re_step.run(re_context)
            relations_in_batch = re_output_context.get("relations", [])
            for rel in relations_in_batch:
                rel.chunk_id += batch_start_index
            raw_relations.extend(relations_in_batch)

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

        final_relations = self.filter_relations(final_relations)
        logger.info(f"Number of relations after filtering: {len(final_relations)}")

        return final_entities, final_relations

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
            r"|\bтекст\s+не\s+содерж\w*\b"
            r"|\b(?:текст\s+)?не\s+содерж\w*\s+информац\w*\s+о\b"
            r"|\bнет\s+(?:информац\w*|сведен\w*|данн\w*)(?:\s+о\b|\b)"
            r"|\bне\s+явля\w*\s+\w*отнош\w*"
            r"|\bнет\s+\w*отнош\w*"
            r"|\bотсутств\w*\s+\w*отнош\w*"
            r"|\bне\s+содерж\w*\s+\w*отнош\w*"
            r"|\bнет\s+явн\w*\s+\w*отнош\w*"
            r"|\bнет\s+\w*связ\w*"
            r"|\bотсутств\w*\s+\w*связ\w*"
            r"|\bсвяз\w*\s+не\s+(?:установ\w*|прослежива\w*|подтвержд\w*|обнаруж\w*)"
            r"|\bотнош\w*\s+не\s+(?:установ\w*|прослежива\w*|подтвержд\w*|обнаруж\w*)"
            r"|\bне[^.\n]{0,60}(?:содерж\w*|ука\w*|упомина\w*|найд\w*|обнаруж\w*|подтвержд\w*|установ\w*|прослеж\w*)"
            r"[^.\n]{0,80}(?:связ\w*|отнош\w*|информац\w*)"
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
