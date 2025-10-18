from typing import Any, Dict, List

from ragu.triplet.pipeline.base import PipelineStep
from ragu.triplet.pipeline.clients import (DescriptionClient, NERClient, NENClient, REClient)
from ragu.triplet.pipeline.models import (Entity, NormalizedEntity, Relation, Triplet)


class NERStep(PipelineStep):
    """
    Named Entity Recognition step.
    """

    def __init__(self, client: NERClient):
        self.client = client

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        text = context["text"]
        entities = await self.client.extract_entities(text)
        context["entities"] = [Entity(**e) for e in entities]
        return context


class NENStep(PipelineStep):
    """
    Named Entity Normalization step.
    """

    def __init__(self, client: NENClient):
        self.client = client

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        text = context["text"] # Get text from context
        entities = context["entities"]
        normalized_entities = await self.client.normalize_entities([e.dict() for e in entities], source_text=text)
        context["normalized_entities"] = [NormalizedEntity(**e) for e in normalized_entities]
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
        relations = await self.client.extract_relations(text, [e.dict() for e in entities])
        context["relations"] = [Relation(**r) for r in relations]
        return context


class DescriptionStep(PipelineStep):
    """
    Description Generation step.
    """

    def __init__(self, client: DescriptionClient):
        self.client = client

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        text = context["text"] # Get text from context
        relations = context["relations"]
        triplets = await self.client.generate_descriptions([r.dict() for r in relations], source_text=text)
        context["triplets"] = [Triplet(**t) for t in triplets]
        return context