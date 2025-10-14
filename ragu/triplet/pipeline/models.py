from typing import List, Optional
from pydantic import BaseModel, Field


class Entity(BaseModel):
    name: str = Field(..., description="The name of the entity.")
    type: str = Field(..., description="The type of the entity.")
    start: int = Field(..., description="The starting index of the entity in the text.")
    end: int = Field(..., description="The ending index of the entity in the text.")


class NormalizedEntity(Entity):
    normalized_name: str = Field(..., description="The normalized name of the entity.")
    ontology_id: Optional[str] = Field(None, description="The ID of the entity in the ontology.")


class Relation(BaseModel):
    source: str = Field(..., description="The source entity of the relation.")
    target: str = Field(..., description="The target entity of the relation.")
    type: str = Field(..., description="The type of the relation.")


class Triplet(Relation):
    description: str = Field(..., description="The description of the relation.")
