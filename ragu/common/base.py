"""
Base class for RAGU modules that use LLM.

This class provides unified management of Jinja-based prompt templates
(:class:`PromptTemplate`) used across different components of the RAGU
system.

Each generative module declares which prompt templates it uses, either by
referencing their names from :data:`DEFAULT_PROMPT_TEMPLATES`.

Classes
-------
- :class:`RaguGenerativeModule` — Manages prompt templates for modules that
  perform LLM-driven generation or structured response tasks.
"""

from ragu.common.prompts import PromptTemplate, DEFAULT_PROMPT_TEMPLATES


class RaguGenerativeModule:
    """
    Base class for generative components in the RAGU framework.

    Provides consistent handling of prompt templates across modules.
    The class can load default templates by name or accept custom
    :class:`PromptTemplate` instances directly.
    """

    def __init__(self, prompts: list[str] | dict[str, PromptTemplate]):
        """
        Initialize the generative module with one or more prompts.

        :param prompts: Either a list of prompt names (loaded from
                        :data:`DEFAULT_PROMPT_TEMPLATES`) or a dictionary
                        mapping prompt names to :class:`PromptTemplate` objects.
        :raises ValueError: If the input format is neither list nor dict.
        """
        super().__init__()

        if isinstance(prompts, list):
            self.prompts: dict[str, PromptTemplate] = {
                prompt_name: DEFAULT_PROMPT_TEMPLATES.get(prompt_name) for prompt_name in prompts
            }
        elif isinstance(prompts, dict):
            self.prompts = prompts
        else:
            raise ValueError(
                f"Prompts must be a list of prompt names or a dictionary of prompt names and PromptTemplate objects, "
                f"got {type(prompts)}"
            )

    def get_prompts(self) -> dict:
        """
        Retrieve all prompt templates registered in the module.

        :return: Dictionary mapping prompt names to :class:`PromptTemplate` objects.
        """
        return self.prompts

    def get_prompt(self, prompt_name: str) -> PromptTemplate:
        """
        Retrieve a specific prompt template by name.

        :param prompt_name: The name of the prompt to retrieve.
        :return: The corresponding :class:`PromptTemplate` instance.
        :raises ValueError: If the prompt name is not found.
        """
        if prompt_name in self.prompts:
            return self.prompts[prompt_name]
        else:
            raise ValueError(f"Prompt {prompt_name} not found")

    def update_prompt(self, prompt_name: str, prompt: PromptTemplate) -> None:
        """
        Replace or add a prompt template in the module.

        :param prompt_name: The key name under which to store the prompt.
        :param prompt: The :class:`PromptTemplate` object to register.
        """
        self.prompts[prompt_name] = prompt
