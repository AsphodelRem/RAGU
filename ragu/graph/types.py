"""
Data structures for representing entities, relations, and communities
within the RAGU knowledge graph system.

These lightweight dataclasses define the core graph components used
throughout the indexing, storage, and retrieval pipelines. Each object
includes an auto-generated hashed ID to ensure consistent deduplication
and reproducibility across sessions.

Modules overview
----------------
• **Entity**, **Relation** — primary units representing semantic nodes
  and edges within the knowledge graph.

• **EntityEmbedding**, **RelationEmbedding** — hold numeric vector
  representations used for embedding-based retrieval and similarity search.

• **Community**, **CommunitySummary** — represent clustered subgraphs
  (communities) and their human-readable summaries produced during
  community detection or summarization.
"""


from dataclasses import dataclass, field
from typing import List

import numpy as np

from ragu.utils.ragu_utils import compute_mdhash_id


@dataclass(slots=True)
class Entity:
    """
    Represents a semantic entity (graph node).

    :param entity_name: Canonical name of the entity.
    :param entity_type: Type or category of the entity (e.g., Person, Organization).
    :param description: Textual description extracted from the source text.
    :param source_chunk_id: Identifiers of text chunks where the entity was found.
    :param documents_id: Identifiers of documents containing this entity.
    :param clusters: List of cluster memberships from community detection.
    :param id: Unique identifier; auto-generated if not provided.
    """
    entity_name: str
    entity_type: str
    description: str
    source_chunk_id: list[str]
    documents_id: list[str] = field(default_factory=list)
    clusters: list[dict] = field(default_factory=list)
    id: str | None = None

    def __post_init__(self):
        """
        Generate a stable MD5-based identifier if not already set.

        :return: None
        """
        if not self.id:
            self.id = compute_mdhash_id(
                content=(self.entity_name + " - " + self.entity_type),
                prefix="ent-"
            )


@dataclass(slots=True)
class EntityEmbedding:
    """
    Stores vector embeddings for a single entity.

    :param id: ID corresponding to the associated :class:`Entity`.
    :param name_embedding: Embedding vector for the entity name.
    :param description_embedding: Embedding vector for the entity description.
    """
    id: str
    name_embedding: np.ndarray = None
    description_embedding: np.ndarray = None,


@dataclass(slots=True)
class Relation:
    """
    Represents a directed relation between two entities.

    :param subject_id: ID of the source (subject) entity.
    :param object_id: ID of the target (object) entity.
    :param subject_name: Display name of the subject entity.
    :param object_name: Display name of the target entity.
    :param description: Description of the relationship.
    :param relation_strength: Numerical weight of the relation (default: 1.0).
    :param source_chunk_id: Identifiers of chunks where this relation was extracted.
    :param id: Unique identifier; auto-generated if not provided.
    """
    subject_id: str
    object_id: str
    subject_name: str
    object_name: str
    description: str
    relation_strength: int | float = 1.0
    source_chunk_id: list[str] = field(default_factory=list)
    id: str | None = None

    def __post_init__(self):
        """
        Generate a stable MD5-based identifier if not already set.
        """
        if not self.id:
            self.id = compute_mdhash_id(
                content=(self.subject_id + " -> " + self.object_id),
                prefix="rel-"
            )


@dataclass(slots=True)
class RelationEmbedding:
    """
    Stores a vector embedding for a relation.

    :param id: ID corresponding to the associated :class:`Relation`.
    :param embedding: Embedding vector capturing the relation’s semantic meaning.
    """
    id: str
    embedding: np.ndarray = None,


@dataclass(slots=True)
class Community:
    """
    Represents a detected community or cluster in the knowledge graph.

    :param level: Hierarchical clustering level of the community.
    :param cluster_id: Identifier of the cluster within the given level.
    :param entities: List of :class:`Entity` objects belonging to the community.
    :param relations: List of :class:`Relation` objects connecting entities.
    :param id: Unique identifier; auto-generated if not provided.
    """
    level: int
    cluster_id: int
    entities: List[Entity]
    relations: List[Relation]
    id: str | None = None

    def __post_init__(self):
        """
        Generate a stable MD5-based identifier if not already set.
        """
        if not self.id:
            self.id = compute_mdhash_id(
                content=(str(self.level) + " - " + str(self.cluster_id)),
                prefix="com-"
            )


@dataclass(slots=True)
class CommunitySummary:
    """
    Stores a textual summary of a community.

    :param id: Unique identifier of the community.
    :param summary: Generated textual summary of the community content.
    """
    id: str
    summary: str