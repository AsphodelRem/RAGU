import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from jinja2 import Environment, FileSystemLoader, Template
from pydantic import BaseModel

from ragu.common.llm import BaseLLM
from ragu.common.prompts.structured_data_extractors import (
    dummy_extractor,
    json_extractor,
)


@dataclass
class PromptTool:
    prompt_template: str
    extractor_func: Callable[[str], dict]
    schema: Optional[Type[BaseModel]] = None
    description: Optional[str] = None

    def get_instruction(self, **kwargs) -> str:
        """
        Format the prompt with provided kwargs
        """
        try:
            return Template(self.prompt_template).render(**kwargs)
        except KeyError as e:
            logging.error(f"Missing required parameter for prompt formatting: {e}")
            raise ValueError(f"Missing required parameter: {e}")
        except Exception as e:
            logging.error(f"Error formatting prompt: {e}")
            raise

    def forward(self, llm: BaseLLM, **prompt_inputs) -> dict:
        prompt = self.get_instruction(**prompt_inputs)
        structured_response = llm.generate(prompt)[0]
        return self.extractor_func(structured_response)

    def batch_forward(
            self,
            llm: BaseLLM,
            batched_prompt_inputs: list[dict[str, Any]],
    ) -> list[dict]:

        prompts = []
        for inputs in batched_prompt_inputs:
            prompt = self.get_instruction(**inputs)
            prompts.append(prompt)

        responses = llm.generate(prompts)
        return [self.extract_response(response) for response in responses]

    def extract_response(self, response: str) -> dict:
        return self.extractor_func(response)


class PromptStorage:
    """
    Singleton class for managing prompt templates across the RAGU system.
    Now with Jinja2 template support.
    """
    _language: str = "ru"
    _instance: Optional['PromptStorage'] = None
    _jinja_env: Optional[Environment] = None
    _templates_base_path: str = Path(__file__).resolve().parent

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._init_jinja_environment()
            cls._instance.update_prompt_tools()
        return cls._instance

    # Prompt tools for knowledge graph creation
    artifact_extractor_prompt: Optional[PromptTool] = None
    artifacts_validation_prompt: Optional[PromptTool] = None
    entity_summarizer_prompt: Optional[PromptTool] = None
    relation_summarizer_prompt: Optional[PromptTool] = None
    community_summarizer_prompt: Optional[PromptTool] = None

    # Prompt tools for search engines
    local_search_engine_prompt: Optional[PromptTool] = None
    global_search_engine_prompt: Optional[PromptTool] = None
    global_search_context_engine_prompt: Optional[PromptTool] = None

    def _init_jinja_environment(self):
        """
        Initialize Jinja2 environment with template loading
        """

        self._jinja_env = Environment(
            loader=FileSystemLoader(self._templates_base_path),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False
        )

    def _load_jinja_template(self, template_name: str) -> str:
        """
        Load Jinja2 template for current language.
        """
        if self._jinja_env is None:
            self._init_jinja_environment()

        template_path = f"{self._language}/{template_name}.jinja2"
        source, _, _ = self._jinja_env.loader.get_source(self._jinja_env, template_path)
        return source

    def set_language(self, language: str = "ru") -> None:
        """
        Set the language for prompt templates.
        """
        supported_languages = self.get_supported_languages()
        if language not in supported_languages:
            raise ValueError(f"{language} is an unsupported language. RAGU supports: {supported_languages}")

        if self._language != language:
            self._language = language
            self.update_prompt_tools()
            logging.info(f"Language changed to {language}, prompt tools will be reinitialized")

    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages by checking existing directories.
        """

        languages = []
        for item in Path(self._templates_base_path).iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                languages.append(item.name)

        return languages

    def update_prompt_tools(self) -> None:
        """
        Update all prompt tools by loading Jinja2 templates for current language.
        """
        # Knowledge graph creation prompts
        self.artifact_extractor_prompt = PromptTool(
            prompt_template=self._load_jinja_template("artifacts_extractor_prompt"),
            extractor_func=json_extractor
        )

        self.artifacts_validation_prompt = PromptTool(
            prompt_template=self._load_jinja_template("artifacts_validation_prompt"),
            extractor_func=json_extractor
        )

        self.entity_summarizer_prompt = PromptTool(
            prompt_template=self._load_jinja_template("entity_description_summarization_prompt"),
            extractor_func=json_extractor
        )

        self.relation_summarizer_prompt = PromptTool(
            prompt_template=self._load_jinja_template("relation_description_summarization_prompt"),
            extractor_func=json_extractor
        )

        self.community_summarizer_prompt = PromptTool(
            prompt_template=self._load_jinja_template("community_summary_prompt"),
            extractor_func=json_extractor
        )

        # Search engine prompts
        self.local_search_engine_prompt = PromptTool(
            prompt_template=self._load_jinja_template("local_search_engine_prompt"),
            extractor_func=dummy_extractor
        )

        self.global_search_engine_prompt = PromptTool(
            prompt_template=self._load_jinja_template("global_search_engine_prompt"),
            extractor_func=json_extractor
        )

        self.global_search_context_engine_prompt = PromptTool(
            prompt_template=self._load_jinja_template("global_search_context_prompt"),
            extractor_func=json_extractor
        )

    def add_new_prompt_tool(self, name: str, prompt_tool: PromptTool):
        setattr(self, name, prompt_tool)

    def get_prompt_tool(self, tool_name: str) -> Optional[PromptTool]:
        return getattr(self, tool_name, None)

    def list_available_prompts(self) -> Dict[str, Dict[str, Any]]:
        prompts = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_') and attr_name.endswith('_prompt'):
                prompt_tool = getattr(self, attr_name)
                if isinstance(prompt_tool, PromptTool):
                    prompts[attr_name] = {
                        "prompt": prompt_tool.prompt_template,
                        "schema": prompt_tool.schema,
                        "description": prompt_tool.description,
                    }
        return prompts
