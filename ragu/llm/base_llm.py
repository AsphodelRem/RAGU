import asyncio
from abc import ABC, abstractmethod
from typing import List, Type, Any

from aiolimiter import AsyncLimiter
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

from ragu.common.batch_generator import BatchGenerator
from ragu.common.cache import PendingRequest, make_llm_cache_key, TextCache
from ragu.common.logger import logger
from ragu.common.prompts import ChatMessages
from ragu.utils.ragu_utils import AsyncRunner


class BaseLLM(ABC):
    """
    Abstract base class for language model (LLM) clients.

    Provides a unified interface for text or structured output generation
    and maintains statistics about token usage and request outcomes.
    """

    def __init__(
            self,
            model_name: str,
            max_requests_per_minute: int = 60,
            max_requests_per_second: int = 1,
            concurrency: int = 10,
            time_period: int | float = 1,
            cache_flush_every: int = 100,
    ):
        """
        Initialize the LLM base client with default usage statistics.
        """
        self.model_name = model_name
        self._sem = asyncio.Semaphore(max(1, concurrency))
        self._rpm = AsyncLimiter(max_requests_per_minute, time_period=60)
        self._rps = AsyncLimiter(max_requests_per_second, time_period=time_period)
        self._cache_flush_every = cache_flush_every

        self.cache = TextCache(flush_every_n_writes=cache_flush_every)

        self._save_stats = True
        self.statistics = {
            "total_tokens": 0,
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "requests": 0,
            "success": 0,
            "fail": 0,
        }

    def get_statistics(self):
        """
        Retrieve a copy of current usage statistics.

        :return: Dictionary containing total tokens, requests, and success/failure counts.
        """
        if not self._save_stats:
            return {}
        return self.statistics.copy()

    def reset_statistics(self):
        """
        Reset all stored usage statistics to zero.
        """
        for k in list(self.statistics.keys()):
            self.statistics[k] = 0

    @abstractmethod
    async def complete(
        self,
        messages: ChatMessages,
        response_model: Type[BaseModel] | None = None,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> str | BaseModel | None:
        ...

    async def generate(
            self,
            conversations: List[ChatMessages],
            response_model: Type[BaseModel] | None = None,
            model_name: str | None = None,
            progress_bar_desc: str = "Processing",
            **kwargs: Any,
    ) -> List[str | BaseModel | None]:

        results: List[str | BaseModel | None] = [None] * len(conversations)
        pending: List[PendingRequest] = []

        for i, conversation in enumerate(conversations):
            key = make_llm_cache_key(
                content=conversation.to_str(),
                model_name=model_name or self.model_name,
                schema=response_model,
                kwargs=kwargs,
            )

            cached = await self.cache.get(key, schema=response_model)
            if cached is not None:
                results[i] = cached
            else:
                pending.append(PendingRequest(i, conversation, key))

        logger.info(
            f"[OpenAIClientService]: Found {len(conversations) - len(pending)}/{len(conversations)} requests in cache.")

        if not pending:
            return results

        with tqdm_asyncio(total=len(pending), desc=progress_bar_desc) as pbar:
            runner = AsyncRunner(self._sem, self._rps, self._rpm, pbar)

            for batch in BatchGenerator(pending, self._cache_flush_every).get_batches():
                tasks = [
                    runner.make_request(
                        self.complete,
                        messages=req.messages,
                        model_name=model_name or self.model_name,
                        response_model=response_model,
                        **kwargs
                    )
                    for req in batch
                ]

                generated = await asyncio.gather(*tasks)

                for req, value in zip(batch, generated):
                    if not isinstance(value, Exception) and value is not None:
                        await self.cache.set(
                            req.cache_key,
                            value,
                            input_instruction=req.messages.to_str(),
                            model_name=model_name or self.model_name,
                        )
                        results[req.index] = value
                    else:
                        results[req.index] = None

                await self.cache.flush_cache()

        return results

