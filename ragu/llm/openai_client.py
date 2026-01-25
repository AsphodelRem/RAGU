from typing import (
    Any,
    Optional,
    Union, Type,
)

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import (
    stop_after_attempt,
    wait_exponential,
    retry,
)

from ragu.common.logger import logger
from ragu.common.prompts import ChatMessages
from ragu.llm.base_llm import BaseLLM


class OpenAIClient(BaseLLM):
    """
    Asynchronous client for OpenAI-compatible LLMs with instructor integration.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_token: str,
        max_requests_per_minute: int = 60,
        max_requests_per_second: int = 1,
        concurrency: int = 10,
        time_period: int | float = 1,
        cache_flush_every: int = 100,
        request_timeout: float = 60.0,
        instructor_mode: instructor.Mode = instructor.Mode.JSON,
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
        super().__init__(
            model_name=model_name,
            max_requests_per_minute=max_requests_per_minute,
            max_requests_per_second=max_requests_per_second,
            concurrency=concurrency,
            time_period=time_period,
            cache_flush_every=cache_flush_every,
        )

        base_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_token,
            timeout=request_timeout,
            **openai_kwargs,
        )

        self._client = instructor.from_openai(client=base_client, mode=instructor_mode)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def complete(
        self,
        messages: ChatMessages,
        response_model: Type[BaseModel]=None,
        model_name: str = None,
        **kwargs: Any,
    ) -> Optional[Union[str, BaseModel]]:
        """
        Perform a single generation request to the LLM with retry logic.

        :param model_name: Override model name for this call (defaults to client model).
        :param kwargs: Additional API call parameters.
        :return: Parsed model output or raw string, or ``None`` if failed.
        """

        try:
            self.statistics["requests"] += 1
            parsed: BaseModel = await self._client.chat.completions.create(
                model=model_name or self.model_name,
                messages=messages.to_openai(),
                response_model=response_model,
                **kwargs,
            )
            self.statistics["success"] += 1
            return parsed

        except Exception as e:
            logger.error(f"[RemoteLLM] request failed after retries: {e}", e, exc_info=True)
            self.statistics["fail"] += 1
            raise

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