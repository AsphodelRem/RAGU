import asyncio
from typing import List

from ragu.graph.graph_builder_pipeline import InMemoryGraphBuilder
from ragu.graph.types import Entity, Relation, CommunitySummary
from ragu.storage.index import Index

from ragu.common.global_parameters import Settings


# TODO: implement all methods
class KnowledgeGraph:
    def __init__(
            self,
            extraction_pipeline: InMemoryGraphBuilder,
            index: Index,
            make_community_summary: bool = True,
            remove_isolated_nodes: bool = True,
            language: str = "english",
    ):
        self.pipeline = extraction_pipeline
        self.index = index
        self.make_community_summary = make_community_summary
        self.remove_isolated_nodes = remove_isolated_nodes
        self.language = language
        self.pipeline.language = language

        self._id_to_entity_map = {}
        self._id_to_relation_map = {}

        # Initialize storage folder if it doesn't exist
        Settings.init_storage_folder()

    async def build_from_docs(self, docs) -> "KnowledgeGraph":
        entities, relations, chunks = await self.pipeline.extract_graph(docs)

        # Add entities and relations
        await self.add_entity(entities)
        await self.add_relation(relations)

        if self.remove_isolated_nodes:
            await self.index.graph_backend.remove_isolated_nodes()

        # Save chunks
        await self.index.insert_chunks(chunks)
        await self.index.entity_vector_db.index_done_callback()
        await self.index.relation_vector_db.index_done_callback()

        if self.make_community_summary:
            communities, summaries = await self.high_level_build()
            await self.index.insert_communities(communities)
            await self.index.insert_community_summaries(summaries)

        await self.index.graph_backend.index_done_callback()

        return self

    async def high_level_build(self):
        communities = await self.index.graph_backend.cluster()
        summaries = await self.pipeline.get_community_summary(
            communities=communities
        )
        return communities, summaries

    # entity CRUD
    async def add_entity(self, entities: Entity | List[Entity]) -> "KnowledgeGraph":
        if isinstance(entities, Entity):
            entities = [entities]

        batch_entities_to_vdb = []
        for entity in entities:
            if await self.index.graph_backend.has_node(entity.id):
                entity_to_merge: Entity = await self.index.graph_backend.get_node(entity.id)
                entity_to_past = Entity(
                    id=entity_to_merge.id,
                    entity_name=entity_to_merge.entity_name,
                    entity_type=entity_to_merge.entity_type,
                    description=entity_to_merge.description + entity.description,
                    clusters=entity_to_merge.clusters + entity.clusters,
                    source_chunk_id=list(set(entity_to_merge.source_chunk_id + entity.source_chunk_id)),
                )
            else:
                entity_to_past = entity
            batch_entities_to_vdb.append(entity_to_past)
            await self.index.graph_backend.upsert_node(entity_to_past)
        await self.index.make_index(entities=batch_entities_to_vdb)
        return self

    async def get_entity(self, entity_id) -> Entity | None:
        if await self.index.graph_backend.has_node(entity_id):
            return await self.index.graph_backend.get_node(entity_id)
        return None

    async def delete_entity(self, entity_id) -> "KnowledgeGraph":
        self.index.graph_backend.delete_node(entity_id)
        return self

    async def update_entity(self, entity_id, new_entity) -> "KnowledgeGraph":
        ...

    # relation CRUD
    async def add_relation(self, relation: Relation | List[Relation]) -> "KnowledgeGraph":
        if isinstance(relation, Relation):
            relation = [relation]

        relations_to_past = []
        for relation in relation:
            relation_to_merge: Relation = await self.index.graph_backend.get_edge(
                relation.subject_id,
                relation.object_id
            )
            if relation_to_merge:
                relation_to_past = Relation(
                    subject_id=relation_to_merge.subject_id,
                    object_id=relation_to_merge.object_id,
                    subject_name=relation_to_merge.subject_name,
                    object_name=relation_to_merge.object_name,
                    description=relation_to_merge.description + relation.description,
                    relation_strength=sum([relation_to_merge.relation_strength, relation.relation_strength]) * 0.5,
                    source_chunk_id=list(set(relation_to_merge.source_chunk_id + relation.source_chunk_id)),
                )
            else:
                relation_to_past = relation
            relations_to_past.append(relation_to_past)
            await self.index.graph_backend.upsert_edge(relation_to_past)
        await self.index.make_index(relations=relations_to_past)

        return self

    async def get_relation(self, subject_id, object_id) -> Relation | None:
        return await self.index.graph_backend.get_edge(subject_id, object_id)

    async def delete_relation(self, subject_id, object_id) -> "KnowledgeGraph":
        await self.index.graph_backend.delete_edge(subject_id, object_id)

    async def update_relation(self, relation_id, new_relation) -> "KnowledgeGraph":
        ...

    async def get_all_entity_relations(self, entity_id) -> List[Relation] | None:
        if await self.index.graph_backend.has_node(entity_id):
            return await self.index.graph_backend.get_node_edges(entity_id)
        return None

    # summary CRUD
    def add_summary(self, summary) -> "KnowledgeGraph":
        ...

    def get_summary(self, summary_id) -> CommunitySummary | None:
        ...

    def delete_summary(self, summary_id) -> "KnowledgeGraph":
        ...

    def update_summary(self, summary_id, new_summary) -> "KnowledgeGraph":
        ...

    def find_similar_entities(self, entity) -> List[Entity]:
        ...

    def find_similar_relations(self, relation) -> List[Relation]:
        ...

    def find_similar_entity_by_query(self, query) -> List[Entity]:
        ...

    def find_similar_relation_by_query(self, query) -> List[Relation]:
        ...

    async def edge_degree(self, subject_id, object_id) -> int | None:
        if await self.index.graph_backend.has_edge(subject_id, object_id):
            return await self.index.graph_backend.get_edge_degree(subject_id, object_id)
        else:
            return None

    async def get_neighbors(self, entity_id) -> List[Entity]:
        if await self.index.graph_backend.has_node(entity_id):
            relations = await self.index.graph_backend.get_node_edges(entity_id)
            neighbors_candidates = await asyncio.gather(*[self.get_entity(relation.object_id) for relation in relations])
            return [neighbor for neighbor in neighbors_candidates if neighbor]
        else:
            return []