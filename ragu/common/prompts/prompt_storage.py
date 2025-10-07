from __future__ import annotations

from dataclasses import dataclass
from typing import Type, Tuple, List

from jinja2 import Template
from pydantic import BaseModel

from ragu.common.prompts.default_models import (
    ArtifactsModel,
    CommunityReportModel,
    GlobalSearchResponseModel,
    GlobalSearchContextModel,
    DefaultResponseModel,
    EntityDescriptionModel,
    RelationDescriptionModel
)
from ragu.common.prompts.default_templates import (
    DEFAULT_ARTIFACTS_EXTRACTOR_PROMPT,
    DEFAULT_ARTIFACTS_VALIDATOR_PROMPT,
    DEFAULT_COMMUNITY_REPORT_PROMPT,
    DEFAULT_RELATIONSHIP_SUMMARIZER_PROMPT,
    DEFAULT_ENTITY_SUMMARIZER_PROMPT,
    DEFAULT_RESPONSE_ONLY_PROMPT,
    DEFAULT_GLOBAL_SEARCH_CONTEXT_PROMPT,
    DEFAULT_GLOBAL_SEARCH_PROMPT,
)


@dataclass
class PromptTemplate:
    """
    Represents a Jinja2-based prompt template for instruction generation.

    Each template defines:
      - a Jinja2 text pattern (`template`)
      - an optional Pydantic schema for structured output validation (`schema`)
      - a short description of its purpose (`description`)

    The template can be rendered dynamically with keyword arguments,
    supporting both single-instance and batched (list/tuple) generation.
    """

    template: str
    schema: Type[BaseModel] = None
    description: str = ""

    def __post_init__(self):
        """
        Compile the Jinja2 template upon initialization for faster rendering.
        """
        self.compiled_template = Template(self.template)

    def get_instruction(self, **batch_kwargs) -> Tuple[List[str], Type[BaseModel]]:
        """
        Render one or more prompt instructions using the template.

        Supports both single-value rendering and batch processing
        (when lists or tuples are passed as arguments).

        :param batch_kwargs: Key-value pairs passed into the Jinja2 template.
                             Lists and tuples trigger batch rendering.
        :return: A tuple of (list of rendered instructions, associated schema).
        """
        batch_lengths = {
            key: len(value) for key, value in batch_kwargs.items()
            if isinstance(value, (list, tuple))
        }

        # No batched parameters → single instruction
        if not batch_lengths:
            return [self.compiled_template.render(**batch_kwargs)], self.schema

        # Validate that all batched parameters have equal length
        unique_lengths = set(batch_lengths.values())
        if len(unique_lengths) > 1:
            raise ValueError("All batch parameters must have the same length")

        batch_size = next(iter(unique_lengths))
        batch_params = []
        for i in range(batch_size):
            params = {}
            for key, value in batch_kwargs.items():
                if isinstance(value, (list, tuple)):
                    params[key] = value[i]
                else:
                    params[key] = value
            batch_params.append(params)

        instructions = [
            self.compiled_template.render(**params)
            for params in batch_params
        ]

        return instructions, self.schema


DEFAULT_PROMPT_TEMPLATES = {
    "artifact_extraction": PromptTemplate(
        template=DEFAULT_ARTIFACTS_EXTRACTOR_PROMPT,
        schema=ArtifactsModel,
        description="Prompt for extracting artifacts (entities and relations) from a text passage."
    ),
    "artifact_validation": PromptTemplate(
        template=DEFAULT_ARTIFACTS_VALIDATOR_PROMPT,
        schema=ArtifactsModel,
        description="Prompt for validating extracted artifacts against a schema."
    ),
    "community_report": PromptTemplate(
        template=DEFAULT_COMMUNITY_REPORT_PROMPT,
        schema=CommunityReportModel,
        description="Prompt for generating community summaries from contextual data."
    ),
    "entity_summarizer": PromptTemplate(
        template=DEFAULT_ENTITY_SUMMARIZER_PROMPT,
        schema=EntityDescriptionModel,
        description="Prompt for summarizing entity descriptions."
    ),
    "relation_summarizer": PromptTemplate(
        template=DEFAULT_RELATIONSHIP_SUMMARIZER_PROMPT,
        schema=RelationDescriptionModel,
        description="Prompt for summarizing relationship descriptions."
    ),
    "global_search_context": PromptTemplate(
        template=DEFAULT_GLOBAL_SEARCH_CONTEXT_PROMPT,
        schema=GlobalSearchContextModel,
        description="Prompt for generating contextual information for a global search."
    ),
    "global_search": PromptTemplate(
        template=DEFAULT_GLOBAL_SEARCH_PROMPT,
        schema=GlobalSearchResponseModel,
        description="Prompt for generating a synthesized global search response."
    ),
    "local_search": PromptTemplate(
        template=DEFAULT_RESPONSE_ONLY_PROMPT,
        schema=DefaultResponseModel,
        description="Prompt for generating a local context-based search response."
    ),
}
