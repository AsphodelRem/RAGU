import asyncio
import logging

from ragu.common.index import Index
from ragu.common.llm import BaseLLM
from ragu.search_engine.base_engine import BaseEngine
from ragu.utils.parse_json_output import create_text_from_community
from ragu.utils.ragu_utils import TokenTruncation

from ragu.common import GlobalPromptStorage


class GlobalSearchEngine(BaseEngine):
    def __init__(
            self,
            client: BaseLLM,
            index: Index,
            max_context_length: int = 30_000,
            tokenizer_backend: str = "tiktoken",
            tokenizer_model: str = "gpt-4",
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.index = index
        self.client = client

        self.truncation = TokenTruncation(
            tokenizer_model,
            tokenizer_backend,
            max_context_length
        )

        self.prompt_tool = GlobalPromptStorage.global_search_engine_prompt
        self.prompt_tool_for_context = GlobalPromptStorage.global_search_context_engine_prompt

    async def a_search(self, query, *args, **kwargs):
        responses: list[str] = []
        for community_cluster_id in await self.index.communities_kv_storage.all_keys():
            try:
                community = await self.index.communities_kv_storage.get_by_id(community_cluster_id)
                responses.append(await self.get_meta_responses(query, create_text_from_community(community)))
            except ValueError as e:
                logging.warning(e)

        responses = list(filter(lambda x: int(x.get("rating", 0)) > 0, responses))
        responses: list[dict] = sorted(responses, key=lambda x: int(x.get("rating", 0)), reverse=True)

        return "\n".join([r.get("response", "") for r in responses])

    async def get_meta_responses(self, query: str, context: str) -> str:
        return self.prompt_tool_for_context.forward(
            self.client,
            query=query,
            context=context
        )

    async def a_query(self, query: str):
        context = await self.a_search(query)
        truncated_context: str = self.truncation(str(context))

        return self.prompt_tool.forward(
            self.client,
            query=query,
            context=truncated_context
        )

    def search(self, query, *args, **kwargs):
        return asyncio.run(self.a_search(query))

    def query(self, query: str):
        return asyncio.run(self.a_query(query))
