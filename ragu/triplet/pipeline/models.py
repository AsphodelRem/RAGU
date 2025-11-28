from typing import Optional
from pydantic import BaseModel, Field


class NormalizedEntity(BaseModel):
    name: str = Field(..., description="The name of the entity.")
    type: str = Field(..., description="The type of the entity.")
    start: int = Field(..., description="The starting index of the entity in the text.")
    end: int = Field(..., description="The ending index of the entity in the text.")
    normalized_name: str = Field(..., description="The normalized name of the entity.")
    ontology_id: Optional[str] = Field(None, description="The ID of the entity in the ontology.")