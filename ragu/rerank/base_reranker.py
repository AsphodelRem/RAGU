import asyncio
from abc import ABC, abstractmethod
from typing import List, Tuple

from tqdm.asyncio import tqdm_asyncio

from ragu.utils.ragu_utils import AsyncRunner


class BaseReranker(ABC):
    _sem: asyncio.Semaphore | None = None
    _rps = None  # AsyncLimiter
    _rpm = None  # AsyncLimiter

    @abstractmethod
    async def rerank(self, x: str, others: List[str], **kwargs) -> List[Tuple[int, float]]:
        ...

    async def batch_rerank(
            self,
            queries: List[str],
            documents: List[List[str]],
            progress_bar_desc: str | None = None,
            **kwargs
    ) -> List[List[Tuple[int, float]]]:
        """
        Reranks multiple query-documents pairs in batch with rate limiting.

        :param queries: List of query texts.
        :param documents: List of document lists, one per query.
        :param progress_bar_desc: Description for progress bar.
        :param kwargs: Additional arguments passed to rerank.
        :return: List of rerank results for each query.
        """
        if len(queries) != len(documents):
            raise ValueError("Length of queries and documents must match")

        if not queries:
            return []

        with tqdm_asyncio(total=len(queries), desc=progress_bar_desc, disable=progress_bar_desc is None) as pbar:
            if self._sem is not None:
                runner = AsyncRunner(self._sem, self._rps, self._rpm, pbar)
                tasks = [
                    runner.make_request(self.rerank, x=query, others=docs, **kwargs)
                    for query, docs in zip(queries, documents)
                ]
                return await asyncio.gather(*tasks)
            else:
                results = []
                for query, docs in zip(queries, documents):
                    result = await self.rerank(query, docs, **kwargs)
                    results.append(result)
                    pbar.update(1)
                return results

    async def __call__(self, x: str, others: List[str], **kwargs):
        return await self.rerank(x, others, **kwargs)
