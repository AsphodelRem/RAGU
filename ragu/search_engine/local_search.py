# Partially based on https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/

import asyncio

from ragu.common.embedder import BaseEmbedder
from ragu.common.index import Index
from ragu.common.llm import BaseLLM
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.search_engine.base_engine import BaseEngine
from ragu.search_engine.search_functional import (
    _find_most_related_community_from_entities,
    _find_most_related_edges_from_entities,
    _find_most_related_text_unit_from_entities,
)
from ragu.search_engine.types import SearchResult
from ragu.utils.ragu_utils import TokenTruncation

from ragu.common import GlobalPromptStorage


class LocalSearchEngine(BaseEngine):
    """
    Perform local search on a knowledge graph using a given query.

    Reference: https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/_op.py#L919
    """
    def __init__(
        self,
        client: BaseLLM,
        knowledge_graph: KnowledgeGraph,
        embedder: BaseEmbedder,
        index: Index,
        max_context_length: int = 30_000,
        tokenizer_backend: str = "tiktoken",
        tokenizer_model: str = "gpt-4",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.truncation = TokenTruncation(
            tokenizer_model,
            tokenizer_backend,
            max_context_length
        )

        self.community_reports = None
        self.graph = knowledge_graph
        self.embedder = embedder
        self.index = index
        self.client = client

        self.prompt_tool = GlobalPromptStorage.local_search_engine_prompt

    async def build_index(self):
        pass

    async def a_search(self, query: str, top_k: int = 20, *args, **kwargs) -> SearchResult:
        """
        Find the most related entities/chunks/relations to the given query.

        :param query: The query to search for.
        :param top_k: The number of relevant artifacts used to build local context.
        :return: A text report containing the most related entities/chunks/relations.
        """
        entities = await self.index.entity_vector_db.query(
            query,
            top_k=top_k
        )
        nodes_data = [
            {**n, "entity_name": k["entity_name"]}
            for k in entities
            if (n := self.graph.graph.nodes.get(k["entity_name"])) is not None
        ]
        relations = await _find_most_related_edges_from_entities(
            nodes_data,
            self.graph
        )

        relevant_summaries = await _find_most_related_community_from_entities(
            nodes_data,
            self.index.communities_kv_storage
        )

        relevant_chunks = await _find_most_related_text_unit_from_entities(
            nodes_data,
            self.index.chunks_kv_storage,
            self.graph
        )

        relevant_entities = [
            {
                "entity_name": entity["entity_name"],
                "entity_type": entity.get("entity_type", "UNKNOWN"),
                "description": entity.get("description", "UNKNOWN")
            } for entity in nodes_data
        ]

        relevant_relations = [
            {
                "source_entity": relation["source_entity"],
                "target_entity": relation["target_entity"],
                "description": relation.get("description", "UNKNOWN"),
                "rank": relation.get("rank", "UNKNOWN")
            } for relation in relations
        ]

        search_result = SearchResult(
            entities=relevant_entities,
            relations=relevant_relations,
            summaries=relevant_summaries,
            chunks=relevant_chunks
        )

        return search_result

    async def a_query(self, query: str) -> str:
        """
        Perform RAG on knowledge graph using the local context.
        :param query: User query
        :return: RAG response
        """

        context: SearchResult = await self.a_search(query)
        truncated_contest: str = self.truncation(str(context))

        return self.prompt_tool.forward(
            self.client,
            query=query,
            context=truncated_contest
        )

    def search(self, query, *args, **kwargs) -> SearchResult:
        return asyncio.run(self.a_search(query, *args, **kwargs))

    def query(self, query: str, *args, **kwargs) -> str:
        return asyncio.run(self.a_query(query))