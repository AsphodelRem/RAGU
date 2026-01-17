import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm_asyncio

from ragu.common.batch_generator import BatchGenerator
from ragu.common.cache import EmbeddingCache, make_embedding_cache_key
from ragu.common.logger import logger
from ragu.embedder.base_embedder import BaseEmbedder
from ragu.utils.ragu_utils import AsyncRunner


@dataclass(frozen=True, slots=True)
class PendingEmbeddingRequest:
    """
    Represents an embedding request pending generation (not found in cache).
    """
    index: int
    text: str
    cache_key: str


class OpenAIEmbedder(BaseEmbedder):
    def __init__(
            self,
            model_name: str,
            base_url: str,
            api_token: str,
            dim: int,
            concurrency: int = 8,
            request_timeout: float = 60.0,
            max_requests_per_second: int = 1,
            max_requests_per_minute: int = 60,
            time_period: int | float = 1,
            use_cache: bool = False,
            cache_path: Optional[str | Path] = None,
            cache_flush_every: int=100,
            *args,
            **kwargs
    ):
        super().__init__(dim=dim)

        self.model_name = model_name
        self.client = AsyncOpenAI(
            api_key=api_token,
            base_url=base_url,
            timeout=request_timeout
        )

        self._sem = asyncio.Semaphore(max(1, concurrency))
        self._rpm = AsyncLimiter(max_requests_per_minute, time_period=60) if max_requests_per_minute else None
        self._rps = AsyncLimiter(max_requests_per_second, time_period=time_period) if max_requests_per_second else None
        self._cache_flush_every = cache_flush_every

        self._use_cache = use_cache
        self._cache = EmbeddingCache(cache_path=cache_path, flush_every_n_writes=cache_flush_every)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def _one_call(self, text: str) -> List[float] | None:
        try:
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=text,
            )
            return [item.embedding for item in response.data][0]
        except Exception as e:
            logger.error(f"[OpenAI API Embedder] Exception occurred: {e}")
            raise

    async def embed(
            self,
            texts: Union[str, List[str]],
            progress_bar_desc=None
    ) -> List[List[float] | None]:
        if isinstance(texts, str):
            texts = [texts]

        results: List[Optional[List[float]]] = [None] * len(texts)
        pending: List[PendingEmbeddingRequest] = []

        # Check cache for all texts first
        for i, text in enumerate(texts):
            if self._cache is not None:
                cache_key = make_embedding_cache_key(text, self.model_name)
                cached = await self._cache.get(cache_key)
                if cached is not None:
                    results[i] = cached
                else:
                    pending.append(PendingEmbeddingRequest(i, text, cache_key))
            else:
                pending.append(PendingEmbeddingRequest(i, text, ""))

        logger.info(f"[OpenAIEmbedder]: Found {len(texts) - len(pending)}/{len(texts)} embeddings in cache.")

        if not pending:
            return results

        with tqdm_asyncio(total=len(pending), desc=progress_bar_desc) as pbar:
            runner = AsyncRunner(self._sem, self._rps, self._rpm, pbar)

            for batch in BatchGenerator(pending, self._cache_flush_every).get_batches():
                tasks = [runner.make_request(self._one_call, text=req.text) for req in batch]
                generated = await asyncio.gather(*tasks, return_exceptions=True)

                for req, value in zip(batch, generated):
                    if not isinstance(value, Exception) and value is not None:
                        if self._use_cache:
                            await self._cache.set(req.cache_key, value)
                        results[req.index] = value
                    else:
                        results[req.index] = None

                if self._use_cache:
                    await self._cache.flush_cache()

        return results

    async def aclose(self):
        try:
            if self._cache is not None:
                await self._cache.close()
            await self.client.close()
        except Exception as e:
            pass
