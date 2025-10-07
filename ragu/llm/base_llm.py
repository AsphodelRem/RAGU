from abc import ABC, abstractmethod
from typing import (
    Optional,
    Union
)

from pydantic import BaseModel


class BaseLLM(ABC):
    """
    Abstract base class for language model (LLM) clients.

    Provides a unified interface for text or structured output generation
    and maintains statistics about token usage and request outcomes.
    """

    def __init__(self):
        """
        Initialize the LLM base client with default usage statistics.
        """
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
    async def generate(
        self,
        prompt: str | list[str],
        system_prompt: str = None,
        pydantic_model: type[BaseModel] = None,
        model_name: str = None,
        **kwargs
    ) -> Optional[Union[BaseModel, str]]:
        """
        Abstract method for text or structured output generation.

        Implementations must perform the actual LLM call and return
        either a plain string response or a parsed Pydantic model.

        :param prompt: User prompt or a list of prompts for batch generation.
        :param system_prompt: Optional system-level context or instruction.
        :param pydantic_model: Optional Pydantic model for structured output parsing.
        :param model_name: Optional override for model selection.
        :param kwargs: Additional generation parameters (e.g., temperature, max_tokens).
        :return: Generated text or structured model instance, or ``None`` on failure.
        """
        pass
