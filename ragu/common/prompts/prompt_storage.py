from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Type

from pydantic import BaseModel

from ragu.common.prompts.default_models import (
    ArtifactsModel,
    CommunityReportModel,
    GlobalSearchResponseModel,
    GlobalSearchContextModel,
    DefaultResponseModel,
    EntityDescriptionModel,
    RelationDescriptionModel,
    ClusterSummarizationModel,
    QueryPlan,
    RewriteQuery,
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
    DEFAULT_CLUSTER_SUMMARIZER_PROMPT,
    DEFAULT_RAGU_LM_ENTITY_EXTRACTION_PROMPT,
    DEFAULT_RAGU_LM_ENTITY_NORMALIZATION_PROMPT,
    DEFAULT_RAGU_LM_ENTITY_DESCRIPTION_PROMPT,
    DEFAULT_RAGU_LM_RELATION_DESCRIPTION_PROMPT,
    DEFAULT_QUERY_DECOMPOSITION_PROMPT,
    DEFAULT_QUERY_REWRITE_PROMPT,
)
from ragu.common.prompts.messages import (
    ChatMessages,
    UserMessage,
    SystemMessage
)


@dataclass(frozen=True, slots=True)
class RAGUInstruction:
    messages: ChatMessages
    pydantic_model: Optional[Type[BaseModel]] = None
    description: Optional[str] = None


DEFAULT_PROMPT_TEMPLATES: dict[str, RAGUInstruction] = {
    "artifact_extraction": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_ARTIFACTS_EXTRACTOR_PROMPT),
            ]
        ),
        pydantic_model=ArtifactsModel,
        description="Prompt for extracting artifacts (entities and relations) from a text passage.",
    ),

    "artifact_validation": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_ARTIFACTS_VALIDATOR_PROMPT),
            ]
        ),
        pydantic_model=ArtifactsModel,
        description="Prompt for validating extracted artifacts against a schema.",
    ),

    "community_report": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_COMMUNITY_REPORT_PROMPT),
            ]
        ),
        pydantic_model=CommunityReportModel,
        description="Prompt for generating community summaries from contextual data.",
    ),

    "entity_summarizer": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_ENTITY_SUMMARIZER_PROMPT),
            ]
        ),
        pydantic_model=EntityDescriptionModel,
        description="Prompt for summarizing entity descriptions.",
    ),

    "relation_summarizer": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_RELATIONSHIP_SUMMARIZER_PROMPT),
            ]
        ),
        pydantic_model=RelationDescriptionModel,
        description="Prompt for summarizing relationship descriptions.",
    ),

    "global_search_context": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_GLOBAL_SEARCH_CONTEXT_PROMPT),
            ]
        ),
        pydantic_model=GlobalSearchContextModel,
        description="Prompt for generating contextual information for a global search.",
    ),

    "global_search": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_GLOBAL_SEARCH_PROMPT),
            ]
        ),
        pydantic_model=GlobalSearchResponseModel,
        description="Prompt for generating a synthesized global search response.",
    ),

    "local_search": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_RESPONSE_ONLY_PROMPT),
            ]
        ),
        pydantic_model=DefaultResponseModel,
        description="Prompt for generating a local context-based search response.",
    ),

    "naive_search": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_RESPONSE_ONLY_PROMPT),
            ]
        ),
        pydantic_model=DefaultResponseModel,
        description="Prompt for generating a naive vector RAG search response.",
    ),

    "cluster_summarize": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_CLUSTER_SUMMARIZER_PROMPT),
            ]
        ),
        pydantic_model=ClusterSummarizationModel,
        description=None,
    ),

    "ragu_lm_entity_extraction": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_RAGU_LM_ENTITY_EXTRACTION_PROMPT),
            ]
        ),
        pydantic_model=None,
        description="Instruction for RAGU-lm entity extraction stage.",
    ),

    "ragu_lm_entity_normalization": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_RAGU_LM_ENTITY_NORMALIZATION_PROMPT),
            ]
        ),
        pydantic_model=None,
        description="Instruction for RAGU-lm entity normalization stage.",
    ),

    "ragu_lm_entity_description": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_RAGU_LM_ENTITY_DESCRIPTION_PROMPT),
            ]
        ),
        pydantic_model=None,
        description="Instruction for RAGU-lm entity description stage.",
    ),

    "ragu_lm_relation_description": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_RAGU_LM_RELATION_DESCRIPTION_PROMPT),
            ]
        ),
        pydantic_model=None,
        description="Instruction for RAGU-lm relation description stage.",
    ),

    "query_decomposition": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_QUERY_DECOMPOSITION_PROMPT),
            ]
        ),
        pydantic_model=QueryPlan,
        description="Prompt for decomposing a complex query into atomic subqueries with dependencies.",
    ),

    "query_rewrite": RAGUInstruction(
        messages=ChatMessages.from_messages(
            [
                UserMessage(content=DEFAULT_QUERY_REWRITE_PROMPT),
            ]
        ),
        pydantic_model=RewriteQuery,
        description="Prompt for rewriting a subquery using answers from its dependencies.",
    ),
}
