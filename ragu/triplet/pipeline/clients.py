import httpx
from typing import Any, Dict, List

from ragu.triplet.pipeline.io_models import RE_IN, RE_OUT, NER_OUT, NER_IN


class BaseClient:
    """
    Base class for a client that communicates with an external service.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url

    async def post(self, endpoint: str, data: Any) -> Any:
        async with httpx.AsyncClient(timeout=1200) as client:
            response = await client.post(f"{self.base_url}{endpoint}", json=data)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                print(f"HTTP error occurred: {e}")
                print(f"Response body: {e.response.text}")
                raise
            return response.json()


class NERClient(BaseClient):
    """
    Client for the Named Entity Recognition service.
    """

    async def extract_entities(self, text: NER_IN) -> NER_OUT:
        """
        Input: "..." (a string)
        Output: {'ners': [[start, end, 'TYPE'], ...], 'text': '...'}
        """
        response = await self.post("/recognize", data=text.dict())
        return NER_OUT.parse_obj(response)


class NENClient(BaseClient):
    """
    Client for the Named Entity Normalization service.
    """

    async def normalize_entities(self, entities: List[Dict[str, Any]], source_text: str) -> List[Dict[str, Any]]:
        response = await self.post("/nen", data={"entities": entities, "source_text": source_text})
        return response.get("normalized_entities", [])


class REClient(BaseClient):
    """
    Client for the Relation Extraction service.
    """

    async def extract_relations(self, data: RE_IN) -> RE_OUT:
        relations = await self.post("/predict", data=data.dict())
        return RE_OUT.parse_obj(relations)


class DescriptionClient(BaseClient):
    """
    Client for the Description Generation service.
    """

    async def generate_entity_descriptions(self, entities: List[Dict[str, Any]], source_text: str) -> List[Dict[str, Any]]:
        response = await self.post("/describe", data={"entities": entities, "source_text": source_text})
        return response.get("described_entities", [])

    async def generate_relation_description(self, relations: List[Dict[str, Any]], source_text: str) -> List[Dict[str, Any]]:
        response = await self.post("/describe_relation", data={"relations": relations, "source_text": source_text})
        return response.get("triplets", [])
