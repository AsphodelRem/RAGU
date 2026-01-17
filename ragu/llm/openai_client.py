import asyncio
from typing import (
    Any,
    List,
    Optional,
    Union,
)

import instructor
from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import (
    stop_after_attempt,
    wait_exponential,
    retry,
)
from tqdm.asyncio import tqdm_asyncio

from ragu.common.batch_generator import BatchGenerator
from ragu.common.cache import TextCache, PendingRequest, make_llm_cache_key
from ragu.common.logger import logger
from ragu.common.decorator import no_throw
from ragu.llm.base_llm import BaseLLM
from ragu.utils.ragu_utils import AsyncRunner


class OpenAIClient(BaseLLM):
    """
    Asynchronous client for OpenAI-compatible LLMs with instructor integration.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_token: str,
        concurrency: int = 8,
        request_timeout: float = 60.0,
        instructor_mode: instructor.Mode = instructor.Mode.JSON,
        max_requests_per_minute: int = 60,
        max_requests_per_second: int = 1,
        time_period: int | float = 1,
        cache_flush_every: int = 100,
        **openai_kwargs: Any,
    ):
        """
        Initialize a new OpenAIClient.

        :param model_name: Name of the OpenAI model to use.
        :param base_url: Base API endpoint.
        :param api_token: Authentication token.
        :param concurrency: Maximum number of concurrent requests.
        :param request_timeout: Request timeout in seconds.
        :param instructor_mode: Output parsing mode for `instructor`.
        :param max_requests_per_minute: Limit of requests per minute (RPM).
        :param max_requests_per_second: Limit of requests per second (RPS).
        :param cache_flush_every: Flush cache to disk every N requests (default 100).
        :param openai_kwargs: Additional keyword arguments passed to AsyncOpenAI.
        """
        super().__init__()

        self.model_name = model_name
        self._sem = asyncio.Semaphore(max(1, concurrency))
        self._rpm = AsyncLimiter(max_requests_per_minute, time_period=60)
        self._rps = AsyncLimiter(max_requests_per_second, time_period=time_period)
        self._cache_flush_every = cache_flush_every

        base_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_token,
            timeout=request_timeout,
            **openai_kwargs,
        )

        self._client = instructor.from_openai(client=base_client, mode=instructor_mode)

        self.cache = TextCache(flush_every_n_writes=cache_flush_every)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _one_call(
        self,
        prompt: str,
        schema: Optional[BaseModel] = None,
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[Union[str, BaseModel]]:
        """
        Perform a single generation request to the LLM with retry logic.

        :param prompt: The input text or instruction prompt.
        :param schema: Optional Pydantic model defining the structured response format.
        :param system_prompt: Optional system-level instruction prepended to the prompt.
        :param model_name: Override model name for this call (defaults to client model).
        :param kwargs: Additional API call parameters.
        :return: Parsed model output or raw string, or ``None`` if failed.
        """

        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        try:
            self.statistics["requests"] += 1
            parsed: BaseModel = await self._client.chat.completions.create(
                model=model_name or self.model_name,
                messages=messages,  # type: ignore
                response_model=schema,
                **kwargs,
            )
            self.statistics["success"] += 1
            return parsed

        except Exception as e:
            logger.error(f"[RemoteLLM] request failed after retries: {e}", e, exc_info=True)
            self.statistics["fail"] += 1
            raise

    @no_throw
    async def generate(
            self,
            prompt: str | list[str],
            *,
            system_prompt: Optional[str] = None,
            model_name: Optional[str] = None,
            progress_bar_desc: Optional[str] = "Processing",
            schema: Optional[type[BaseModel]] = None,
            **kwargs: Any,
    ) -> List[Optional[Union[str, BaseModel]]]:

        prompts: List[str] = [prompt] if isinstance(prompt, str) else list(prompt)

        results: list[Optional[Union[str, BaseModel]]] = [None] * len(prompts)
        pending: list[PendingRequest] = []

        for i, p in enumerate(prompts):
            key = make_llm_cache_key(
                prompt=p,
                system_prompt=system_prompt,
                model_name=model_name or self.model_name,
                schema=schema,
                kwargs=kwargs,
            )

            cached = await self.cache.get(key, schema=schema)
            if cached is not None:
                results[i] = cached
            else:
                pending.append(PendingRequest(i, p, key))

        logger.info(f"[OpenAIClientService]: Found {len(prompts) - len(pending)}/{len(prompts)} requests in cache.")

        if not pending:
            return results

        with tqdm_asyncio(total=len(pending), desc=progress_bar_desc) as pbar:
            runner = AsyncRunner(self._sem, self._rps, self._rpm, pbar)

            for batch in BatchGenerator(pending, self._cache_flush_every).get_batches():
                tasks = [
                    runner.make_request(
                        self._one_call,
                        prompt=req.prompt,
                        system_prompt=system_prompt,
                        model_name=model_name,
                        schema=schema,
                        **kwargs
                    )
                    for req in batch
                ]

                generated = await asyncio.gather(*tasks, return_exceptions=True)

                for req, value in zip(batch, generated):
                    if not isinstance(value, Exception) and value is not None:
                        if system_prompt:
                            input_instruction = f"[system]: {system_prompt}\n[user]: {req.prompt}"
                        else:
                            input_instruction = req.prompt

                        await self.cache.set(
                            req.cache_key,
                            value,
                            input_instruction=input_instruction,
                            model_name=model_name or self.model_name,
                        )
                        results[req.index] = value
                    else:
                        results[req.index] = None

                await self.cache.flush_cache()

        return results

    async def async_close(self) -> None:
        """
        Close the underlying asynchronous OpenAI client and flush cache.
        """
        try:
            await self.cache.close()
        except Exception:
            pass

        try:
            await self._client.close()
        except Exception:
            pass
