from typing import Any, Dict, List

from ragu.triplet.pipeline.base import PipelineStep
from ragu.triplet.pipeline.clients import (DescriptionClient, NERClient, NENClient, REClient)


class NERStep(PipelineStep):
    """
    Named Entity Recognition step.
    """

    def __init__(self, client: NERClient):
        self.client = client

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        text = context["text"]
        raw_entities_dict = await self.client.extract_entities(text)
        
        processed_entities = []
        if raw_entities_dict and isinstance(raw_entities_dict, dict):
            for entity_type, entities in raw_entities_dict.items():
                if isinstance(entities, list):
                    for entity_data in entities:
                        start, end, _ = entity_data
                        processed_entities.append({
                            "name": text[start:end],
                            "type": entity_type,
                            "start": start,
                            "end": end,
                        })
        context["entities"] = processed_entities
        return context


class NENStep(PipelineStep):
    """
    Named Entity Normalization step.
    """

    def __init__(self, client: NENClient):
        self.client = client

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        entities = context["entities"]
        normalized_entities = await self.client.normalize_entities(entities, context["text"])
        context["normalized_entities"] = normalized_entities
        return context


class REStep(PipelineStep):
    """
    Relation Extraction step.
    """

    def __init__(self, client: REClient):
        self.client = client

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        text = context["text"]
        entities = context["normalized_entities"]
        relations = await self.client.extract_relations(text, entities)
        context["relations"] = relations
        return context


class DescriptionStep(PipelineStep):
    """
    Description Generation step.
    """

    def __init__(self, client: DescriptionClient):
        self.client = client

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        relations = context["relations"]
        source_text = context["text"]
        triplets = await self.client.generate_descriptions(relations, source_text)
        context["triplets"] = triplets
        return context




