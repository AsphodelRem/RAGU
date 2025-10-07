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
        max_requests_per_minute: Optional[int] = None,
        max_requests_per_second: Optional[int] = None,
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
        :param openai_kwargs: Additional keyword arguments passed to AsyncOpenAI.
        """
        super().__init__()

        self.model_name = model_name
        self._sem = asyncio.Semaphore(max(1, concurrency))
        self._rpm = AsyncLimiter(max_requests_per_minute, time_period=60) if max_requests_per_minute else None
        self._rps = AsyncLimiter(max_requests_per_second, time_period=1) if max_requests_per_second else None

        base_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_token,
            timeout=request_timeout,
            **openai_kwargs,
        )

        self._client = instructor.from_openai(client=base_client, mode=instructor_mode)

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
            return None

    @no_throw
    async def generate(
        self,
        prompt: str | list[str],
        *,
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        progress_bar_desc: Optional[str] = "Processing",
        **kwargs: Any,
    ) -> List[Optional[Union[str, BaseModel]]]:
        """
        Generate one or multiple completions asynchronously.

        This method automatically batches multiple prompts, runs them with
        concurrency and rate limits, and provides a live progress bar.

        :param prompt: Single prompt string or a list of prompts to process.
        :param system_prompt: Optional system prompt applied to all items.
        :param model_name: Optional override for model name.
        :param progress_bar_desc: Label shown in the progress bar (default: ``"Processing"``).
        :param kwargs: Additional keyword arguments passed to the model call.
        :return: List of responses (strings or Pydantic models). Items may be ``None`` if failed.
        """
        prompts: List[str] = [prompt] if isinstance(prompt, str) else list(prompt)
        with tqdm_asyncio(total=len(prompts), desc=progress_bar_desc) as pbar:
            runner = AsyncRunner(self._sem, self._rps, self._rpm, pbar)
            tasks = [
                runner.make_request(
                    self._one_call,
                    prompt=p,
                    system_prompt=system_prompt,
                    model_name=model_name,
                    **kwargs
                ) for p in prompts
            ]

            return await asyncio.gather(*tasks, return_exceptions=False)

    async def async_close(self) -> None:
        """
        Close the underlying asynchronous OpenAI client.
        """
        try:
            await self._client.close()
        except Exception:
            pass
