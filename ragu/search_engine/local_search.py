# Partially based on https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/

import asyncio

from ragu.embedder.base_embedder import BaseEmbedder
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.llm.base_llm import BaseLLM
from ragu.search_engine.base_engine import BaseEngine
from ragu.search_engine.search_functional import (
    _find_most_related_edges_from_entities,
    _find_most_related_text_unit_from_entities,
)
from ragu.search_engine.types import LocalSearchResult
from ragu.utils.token_truncation import TokenTruncation


class LocalSearchEngine(BaseEngine):
    """
    Performs local retrieval-augmented search (RAG) over a knowledge graph.

    This engine finds entities, relations, and text units most relevant to a given
    query, builds a local context, and passes it to an LLM for response generation.

    Reference
    ---------
    Based on: https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/_op.py#L919
    """

    def __init__(
        self,
        client: BaseLLM,
        knowledge_graph: KnowledgeGraph,
        embedder: BaseEmbedder,
        max_context_length: int = 30_000,
        tokenizer_backend: str = "tiktoken",
        tokenizer_model: str = "gpt-4",
        *args,
        **kwargs
    ):
        """
        Initialize a `LocalSearchEngine`.

        :param client: Language model client for generation.
        :param knowledge_graph: Knowledge graph used for entity and relation retrieval.
        :param embedder: Embedding model for similarity search.
        :param max_context_length: Maximum number of tokens allowed in the truncated context.
        :param tokenizer_backend: Tokenizer backend to use (e.g. ``tiktoken``).
        :param tokenizer_model: Model name used for token counting and truncation.
        """
        _PROMPTS_NAMES = ["local_search"]
        super().__init__(prompts=_PROMPTS_NAMES, *args, **kwargs)

        self.truncation = TokenTruncation(
            tokenizer_model,
            tokenizer_backend,
            max_context_length
        )

        self.knowledge_graph = knowledge_graph
        self.embedder = embedder
        self.client = client
        self.community_reports = None

    async def a_search(self, query: str, top_k: int = 20, *args, **kwargs) -> LocalSearchResult:
        """
        Perform a local search on the knowledge graph.

        :param query: The input text query to search for.
        :param top_k: Number of top entities to include in context (default: 20).
        :return: A :class:`SearchResult` object containing entities, relations, and chunks.
        """

        entities_id = await self.knowledge_graph.index.entity_vector_db.query(query, top_k=top_k)
        entities = await asyncio.gather(*[
            self.knowledge_graph.get_entity(entity["__id__"])
            for entity in entities_id
        ])
        entities = [data for data in entities if data is not None]

        relations = await _find_most_related_edges_from_entities(entities, self.knowledge_graph)
        relations = [relation for relation in relations if relation is not None]

        relevant_chunks = await _find_most_related_text_unit_from_entities(entities, self.knowledge_graph)
        relevant_chunks = [chunk for chunk in relevant_chunks if chunk is not None]

        return LocalSearchResult(
            entities=entities,
            relations=relations,
            chunks=relevant_chunks
        )

    async def a_query(self, query: str, top_k: int = 20) -> str:
        """
        Execute a retrieval-augmented query over the local knowledge graph.

        :param query: User query in natural language.
        :param top_k: Number of entities to search in the local context (default: 20).
        :return: Generated response text from the language model.
        """

        context: LocalSearchResult = await self.a_search(query, top_k)
        truncated_context: str = self.truncation(str(context))

        prompt, schema = self.get_prompt("local_search").get_instruction(
            query=query,
            context=truncated_context
        )

        return await self.client.generate(prompt=prompt, schema=schema)
