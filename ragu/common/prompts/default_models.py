"""
Pydantic schema definitions for structured outputs of the RAGU prompt system.

These models describe the expected structured responses from the LLM
for various prompt types — such as artifact extraction, community analysis,
entity/relation summarization, and global/local search results.

They serve as type-safe contracts between LLM generations and the downstream
processing pipeline (e.g., graph construction, ranking, reporting).

Modules and models overview
---------------------------

• **EntityModel**, **RelationModel**, **ArtifactsModel**
    Define the structure for extracted entities and relationships
    produced during artifact extraction from text.

• **CommunityFindingModel**, **CommunityReportModel**
    Describe the structure of automatically generated reports summarizing
    clusters or communities discovered in a knowledge graph.

• **GlobalSearchContextModel**, **GlobalSearchResponseModel**
    Define the format of reasoning and response outputs during global
    retrieval-augmented search across multiple graph communities.

• **DefaultResponseModel**
    Provides a minimal schema for generic responses.

• **EntityDescriptionModel**, **RelationDescriptionModel**
    Contain simplified summaries of entities and relations for
    documentation and human-readable output.

Each schema includes field-level descriptions to guide the LLM
in producing well-structured, interpretable results.
"""

import logging
from typing import List

from pydantic import (
    BaseModel,
    Field,
    conint,
    confloat,
    model_validator
)


class EntityModel(BaseModel):
    entity_name: str = Field(..., description="Normalized entity name, capitalized")
    entity_type: str = Field(..., description="Entity type")
    description: str = Field(..., description="Detailed description of the entity from the text")


class RelationModel(BaseModel):
    source_entity: str = Field(..., description="Name of the source entity (matches an Entity.entity_name)")
    target_entity: str = Field(..., description="Name of the target entity (matches an Entity.entity_name)")
    description: str = Field(..., description="Description of the relationship")
    relationship_strength: conint(ge=0, le=5) = Field(
        ..., description="Relationship strength 0–5 (0 = weak, 5 = strong)"
    )


class ArtifactsModel(BaseModel):
    entities: List[EntityModel] = Field(
        default_factory=list,
        description="List of extracted entities"
    )
    relations: List[RelationModel] = Field(
        default_factory=list,
        description="List of extracted relationships"
    )

    @model_validator(mode="after")
    def _validate_relationship_endpoints(self):
        names = {e.entity_name for e in self.entities}
        bad = []
        for r in self.relations:
            if r.source_entity not in names or r.target_entity not in names:
                bad.append((r.source_entity, r.target_entity))
        if bad:
            logging.warning(
                "Relations reference entities not present in 'entities': "
                + "; ".join(f"{s} -> {t}" for s, t in bad)
            )
        return self


class CommunityFindingModel(BaseModel):
    summary: str = Field(..., description="Short description of the finding")
    explanation: str = Field(..., description="Detailed explanation (several paragraphs based on the data)")


class CommunityReportModel(BaseModel):
    title: str = Field(..., description="Report title")
    summary: str = Field(..., description="Short summary of the community")
    rating: confloat(ge=0, le=10) = Field(..., description="Impact rating 0–10")
    rating_explanation: str = Field(..., description="Explanation of the rating")
    findings: List[CommunityFindingModel] = Field(
        default_factory=list,
        description="List of 5–10 key findings"
    )


class GlobalSearchContextModel(BaseModel):
    reasoning: str = Field(..., description="Reasoning about the relevance of the context")
    response: str = Field(..., description="Answer to the query")
    rating: confloat(ge=0, le=10) = Field(..., description="Relevance rating of the context 0–10")


class GlobalSearchResponseModel(BaseModel):
    reasoning: str = Field(..., description="Reasoning about context relevance and the final answer")
    response: str = Field(..., description="Final answer")


class DefaultResponseModel(BaseModel):
    response: str = Field(..., description="Answer based on the provided context; if unknown, explicitly state so")


class EntityDescriptionModel(BaseModel):
    entity_name: str = Field(description="Entity name")
    description: str = Field(description="Summarized description")


class RelationDescriptionModel(BaseModel):
    subject_name: str = Field(description="Subject entity name")
    object_name: str = Field(description="Object entity name")
    description: str = Field(description="Summarized description of the relationship between the entities")
