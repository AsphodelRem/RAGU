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


class NENClient:
    """
    Client for the Named Entity Normalization service.
    """

    def __init__(self, client: OpenAIClient):
        self.client = client

    async def normalize_entities(self, entities: List[Dict[str, Any]], source_text: str) -> List[Dict[str, Any]]:
        
        normalized_entities = []
        for entity in entities:
            prompt = PROMPT_FOR_ENTITY_NORMALIZATION.format(source_entity=entity['name'], source_text=source_text)
            responses = await self.client.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
            
            normalized_name = responses[0].strip() if responses and responses[0] else entity['name']
            normalized_entities.append({
                "name": entity["name"],
                "type": entity["type"],
                "start": entity["start"],
                "end": entity["end"],
                "normalized_name": normalized_name,
            })
        return normalized_entities


class REClient(BaseClient):
    """
    Client for the Relation Extraction service.
    """

    async def extract_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        response = await self.post("/re", data={"text": text, "entities": entities})
        return response.get("relations", [])


class DescriptionClient:
    """
    Client for the Description Generation service.
    """

    def __init__(self, client: OpenAIClient):
        self.client = client

    async def generate_descriptions(self, relations: List[Dict[str, Any]], source_text: str) -> List[Dict[str, Any]]:
        triplets = []
        for relation in relations:
            prompt = PROMPT_FOR_RELATION_DESCRIPTION.format(normalized_entity=relation['source'], source_text=source_text)
            responses = await self.client.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)

            description = responses[0].strip() if responses and responses[0] else ""
            triplets.append({
                "source": relation["source"],
                "target": relation["target"],
                "type": relation["type"],
                "description": description,
            })
        return triplets
