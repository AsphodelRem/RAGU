import asyncio
from typing import List, Tuple

import httpx
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential

from ragu.common.logger import logger
from ragu.rerank.base_reranker import BaseReranker


class VLLMReranker(BaseReranker):
    """
    Reranker that uses vLLM's /v1/score endpoint.
    Compatible with vLLM serve running cross-encoder models.
    """

    def __init__(
            self,
            model_name: str,
            base_url: str = "http://localhost:8000/v1",
            api_token: str | None = None,
            concurrency: int = 8,
            request_timeout: float = 60.0,
            max_requests_per_second: int = 1,
            max_requests_per_minute: int = 60,
            time_period: int | float = 1,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token

        self.client = httpx.AsyncClient(timeout=request_timeout)

        self._sem = asyncio.Semaphore(max(1, concurrency))
        self._rpm = AsyncLimiter(max_requests_per_minute, time_period=60) if max_requests_per_minute else None
        self._rps = AsyncLimiter(max_requests_per_second, time_period=time_period) if max_requests_per_second else None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def rerank(
            self,
            x: str,
            others: List[str],
            top_k: int | None = None,
    ) -> List[Tuple[int, float]]:
        """
        Reranks documents based on relevance to the query.

        :param x: Query text.
        :param others: List of documents/items to rerank.
        :param top_k: Number of top results to return. If None, returns all.
        :return: List of (index, score) tuples sorted by relevance descending.
        """
        if not others:
            return []

        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        payload = {
            "model": self.model_name,
            "text_1": x,
            "text_2": others,
        }

        try:
            response = await self.client.post(
                f"{self.base_url}/score",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            results = [(item["index"], item["score"]) for item in data["data"]]
            results.sort(key=lambda x: x[1], reverse=True)

            if top_k is not None:
                results = results[:top_k]

            return results
        except Exception as e:
            logger.error(f"[VLLM Reranker] Exception occurred: {e}")
            raise

    async def aclose(self):
        try:
            await self.client.aclose()
        except Exception:
            pass
