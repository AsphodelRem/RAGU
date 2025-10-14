import httpx
from typing import Any, Dict, List


class BaseClient:
    """
    Base class for a client that communicates with an external service.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url

    async def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{self.base_url}{endpoint}", json=data)
            response.raise_for_status()
            return response.json()


class NERClient(BaseClient):
    """
    Client for the Named Entity Recognition service.
    """

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        response = await self.post("/ner", data={"text": text})
        return response.get("entities", [])


class NENClient(BaseClient):
    """
    Client for the Named Entity Normalization service.
    """

    async def normalize_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        response = await self.post("/nen", data={"entities": entities})
        return response.get("normalized_entities", [])


class REClient(BaseClient):
    """
    Client for the Relation Extraction service.
    """

    async def extract_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        response = await self.post("/re", data={"text": text, "entities": entities})
        return response.get("relations", [])


class DescriptionClient(BaseClient):
    """
    Client for the Description Generation service.
    """

    async def generate_descriptions(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        response = await self.post("/describe", data={"relations": relations})
        return response.get("triplets", [])
