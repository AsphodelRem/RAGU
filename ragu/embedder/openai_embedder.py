import asyncio
from typing import List, Union

from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm_asyncio

from ragu.common.logger import logger
from ragu.embedder.base_embedder import BaseEmbedder
from ragu.utils.ragu_utils import AsyncRunner


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
            *args,
            **kwargs
    ):
        super().__init__(dim=dim)

        self.model = model_name
        self.client = AsyncOpenAI(
            api_key=api_token,
            base_url=base_url,
            timeout=request_timeout
        )

        self._sem = asyncio.Semaphore(max(1, concurrency))
        self._rpm = AsyncLimiter(max_requests_per_minute, time_period=60) if max_requests_per_minute else None
        self._rps = AsyncLimiter(max_requests_per_second, time_period=1) if max_requests_per_second else None


    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def _one_call(self, text: str) -> List[float] | None:
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            return [item.embedding for item in response.data][0]
        except Exception as e:
            logger.error(f"[OpenAI API Embedder] Exception occurred: {e}")
            return None

    async def embed(self, texts: Union[str, List[str]], progress_bar_desc=None) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]

        with tqdm_asyncio(total=len(texts), desc=progress_bar_desc) as pbar:
            runner = AsyncRunner(self._sem, self._rps, self._rpm, pbar)
            tasks = [runner.make_request(self._one_call, text=text) for text in texts]

            return await asyncio.gather(*tasks)

    async def aclose(self):
        try:
            await self.client.close()
        except Exception as e:
            pass
