"""
This module contains Pydantic models that define the I/O formats for the 
NER and RE services, as described in the IO_FORMAT.md document.
"""
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, RootModel


class NEREntity(RootModel[Tuple[int, int, str]]):
    """
    Represents a single entity extracted by the NER model.
    It's a list with three elements: [start_char_index, end_char_index, entity_type].
    """
    pass


class NER_IN(RootModel[str]):
    """
    Input for the NER model. A single string.
    """
    pass


class NER_OUT(BaseModel):
    """
    Output of the NER model.
    """
    ners: List[NEREntity] = Field(..., description="A list of extracted entities.")
    text: str = Field(..., description="The original input string.")


class RE_IN(BaseModel):
    """
    Input for the RE model.
    """
    chunks: List[str] = Field(..., description="A list of text strings (e.g., sentences or paragraphs).")
    entities_list: List[List[NEREntity]] = Field(
        ...,
        description="A list where each element is a list of entities found in the corresponding chunk."
    )


class RE_OUT_Item(BaseModel):
    """
    An item in the output of the RE model.
    """
    source_entity: str = Field(..., description="The text of the source entity in the relationship.")
    target_entity: str = Field(..., description="The text of the target entity in the relationship.")
    relationship_type: str = Field(..., description="The type of the relationship.")
    relationship_description: Optional[str] = Field(None, description="A natural language description of the relationship.")
    relationship_strength: float = Field(..., description="A confidence score for the extracted relationship.")
    chunk_id: int = Field(..., description="The index of the chunk from the RE_IN chunks list where this relationship was found.")


class RE_OUT(RootModel[List[RE_OUT_Item]]):
    """
    Output of the RE model. A list of extracted relationships.
    """
    pass
