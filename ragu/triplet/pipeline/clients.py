import httpx
from typing import Any, Dict, List

from ragu.llm.openai_client import OpenAIClient


# Prompts from ragu-lm-examples/build_custom_knowledge_graph.ipynb
SYSTEM_PROMPT = 'Вы - эксперт в области анализа текстов и извлечения семантической информации из них.'
PROMPT_FOR_ENTITY_NORMALIZATION = 'Выполните нормализацию именованной сущности, встретившейся в тексте.\n\nИсходная (ненормализованная) именованная сущность: {source_entity}\n\nТекст: {source_text}\n\nНормализованная именованная сущность: '
PROMPT_FOR_RELATION_DESCRIPTION = 'Напишите, что означает именованная сущность в тексте, то есть раскройте её смысл относительно текста.\n\nИменованная сущность: {normalized_entity}\n\nТекст: {source_text}\n\nСмысл именованной сущности: '


class BaseClient:
    """
    Base class for a client that communicates with an external service.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url

    async def post(self, endpoint: str, data: Any) -> Any:
        async with httpx.AsyncClient() as client:
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

    async def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Input: "..." (a string)
        Output: {'TYPE': [[start, end, 'TYPE'], ...]}
        """
        return await self.post("/recognize", data=text)


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

    async def extract_relations(self, chunks: List[str], entities_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        relations = await self.post("/predict", data={"chunks": chunks, "entities_list": entities_list})
        return relations


class DescriptionClient(BaseClient):
    """
    Client for the Description Generation service.
    """

    async def generate_descriptions(self, relations: List[Dict[str, Any]], source_text: str) -> List[Dict[str, Any]]:
        response = await self.post("/describe", data={"relations": relations, "source_text": source_text})
        return response.get("triplets", [])
