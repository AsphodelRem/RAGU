from typing import Any, Dict, List

from ragu.triplet.pipeline.base import PipelineStep
from ragu.triplet.pipeline.clients import (
    DescriptionClient, NERClient, NENClient, REClient
)
from ragu.triplet.pipeline.io_models import NER_IN, RE_IN


class NERStep(PipelineStep):
    """
    Named Entity Recognition step.
    """

    def __init__(self, client: NERClient):
        self.client = client

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        text = context["text"]
        ner_input = NER_IN(root=text)
        ner_output = await self.client.extract_entities(ner_input)

        processed_entities = []
        for entity_data in ner_output.ners:
            start, end, entity_type = entity_data.root
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
        chunks = context["chunks"]
        entities_list = context["entities_list"]
        re_input = RE_IN(chunks=chunks, entities_list=entities_list)
        relations = await self.client.extract_relations(re_input)
        context["relations"] = relations.root
        return context

class EntityDescriptionStep(PipelineStep):
    """
    Entity Description Generation step.
    """

    def __init__(self, client: DescriptionClient):
        self.client = client

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        entities = context["normalized_entities"]
        source_text = context["text"]
        described_entities = await self.client.generate_entity_descriptions(entities, source_text)
        context["described_entities"] = described_entities
        return context


class RelationDescriptionStep(PipelineStep):
    """
    Relation Description Generation step.
    """

    def __init__(self, client: DescriptionClient):
        self.client = client

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        relations = context["relations_for_description"]
        source_text = context["text"]
        triplets = await self.client.generate_relation_description(relations, source_text)
        context["triplets"] = triplets
        return context