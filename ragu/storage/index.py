import os
from dataclasses import asdict
from typing import Dict, Optional, Type, List

from ragu.chunker.types import Chunk
from ragu.common.global_parameters import DEFAULT_FILENAMES
from ragu.common.global_parameters import Settings
from ragu.common.logger import logging
from ragu.embedder.base_embedder import BaseEmbedder
from ragu.graph.types import Entity, Relation, Community, CommunitySummary
from ragu.storage.base_storage import BaseKVStorage, BaseVectorStorage, BaseGraphStorage
from ragu.storage.graph_storage_adapters.networkx_adapter import NetworkXStorage
from ragu.storage.kv_storage_adapters.json_storage import JsonKVStorage
from ragu.storage.vdb_storage_adapters.nano_vdb import NanoVectorDBStorage
from ragu.common.logger import logger


class Index:
    """
    Index class that manages embedding storage and indexing for a knowledge graph.
    """

    def __init__(
            self,
            embedder: BaseEmbedder,
            graph_backend_storage: Type[BaseGraphStorage]=NetworkXStorage,
            kv_storage_type: Type[BaseKVStorage] = JsonKVStorage,
            vdb_storage_type: Type[BaseVectorStorage] = NanoVectorDBStorage,
            chunks_kv_storage_kwargs: Optional[Dict] = None,
            summary_kv_storage_kwargs: Optional[Dict] = None,
            communities_kv_storage_kwargs: Optional[Dict] = None,
            vdb_storage_kwargs: Optional[Dict] = None,
            graph_storage_kwargs: Optional[Dict] = None,
    ):
        """
        Initializes the Index with storage configurations and an embedding model.
        """
        # Initialize storage folder if it doesn't exist
        Settings.init_storage_folder()
        storage_folder: str = Settings.storage_folder

        self.embedder = embedder
        self.summary_kv_storage_kwargs = summary_kv_storage_kwargs or {}
        self.community_kv_storage_kwargs = communities_kv_storage_kwargs or {}
        self.chunks_kv_storage_kwargs = chunks_kv_storage_kwargs or {}
        self.vdb_storage_kwargs = vdb_storage_kwargs or {}
        self.graph_storage_kwargs = graph_storage_kwargs or {}

        # Define file paths
        self.summary_kv_storage_kwargs["filename"] = os.path.abspath(
            os.path.join(storage_folder, DEFAULT_FILENAMES["community_summary_kv_storage_name"])
        )
        self.community_kv_storage_kwargs["filename"] = os.path.abspath(
            os.path.join(storage_folder, DEFAULT_FILENAMES["community_kv_storage_name"])
        )
        self.chunks_kv_storage_kwargs["filename"] = os.path.abspath(
            os.path.join(storage_folder, DEFAULT_FILENAMES["chunks_kv_storage_name"])
        )
        self.vdb_storage_kwargs["filename"] = os.path.abspath(
            os.path.join(storage_folder, DEFAULT_FILENAMES["entity_vdb_name"])
        )
        self.graph_storage_kwargs["filename"] = os.path.abspath(
            os.path.join(storage_folder, DEFAULT_FILENAMES["knowledge_graph_storage_name"])
        )

        relation_vdb_storage_kwargs = self.vdb_storage_kwargs.copy()
        relation_vdb_storage_kwargs["filename"] = os.path.abspath(
            os.path.join(storage_folder, DEFAULT_FILENAMES["relation_vdb_name"])
        )

        # Key-value storages
        self.chunks_kv_storage = kv_storage_type(**self.chunks_kv_storage_kwargs)  # type: ignore

        self.community_summary_kv_storage = kv_storage_type(**self.summary_kv_storage_kwargs)  # type: ignore
        self.communities_kv_storage = self.community_summary_kv_storage

        self.community_kv_storage = kv_storage_type(**self.community_kv_storage_kwargs)  # type: ignore

        # Vector storage
        self.entity_vector_db = vdb_storage_type(embedder=embedder, **self.vdb_storage_kwargs)  # type: ignore
        self.relation_vector_db = vdb_storage_type(embedder=embedder, **relation_vdb_storage_kwargs)  # type: ignore

        # Graph storage
        self.graph_backend = graph_backend_storage(**self.graph_storage_kwargs)

    async def make_index(
            self,
            entities: List[Entity] = None,
            relations: List[Relation] = None,
            communities: List[Community] = None,
            summaries: List[CommunitySummary] = None,
    ) -> None:
        """
        Creates an index for the given knowledge graph by storing entities, relations,
        communities (as id lists), and community summaries (id->text).
        """
        if entities:
            await self._insert_entities_to_vdb(entities)
        if relations:
            await self._insert_relations_to_vdb(relations)
        if communities:
            await self.insert_communities(communities)
        if summaries:
            await self.insert_community_summaries(summaries)

    async def _insert_entities_to_vdb(self, entities: List[Entity]) -> None:
        """
        Inserts entities from the knowledge graph into the vector database.
        """
        if self.entity_vector_db is None:
            logging.warning("Vector database storage is not initialized.")
            return

        try:
            data_for_vdb = {
                entity.id: {
                    "entity_name": entity.entity_name,
                    "content": entity.entity_name + " - " + entity.description,
                }
                for entity in entities
            }
            await self.entity_vector_db.upsert(data_for_vdb) # type: ignore
            await self.entity_vector_db.index_done_callback()

        except Exception as e:
            logger.error(f"Failed to insert entities into vector DB: {e}")

    async def _insert_relations_to_vdb(self, relations: List[Relation]) -> None:
        """
        Inserts relations from the knowledge graph into the vector database.
        """
        if self.relation_vector_db is None:
            logger.warning("Vector database storage is not initialized.")
            return

        try:
            data_for_vdb = {
                relation.id: {
                    "subject": relation.subject_id,
                    "object": relation.object_id,
                    "content": relation.description
                }
                for relation in relations
            }
            await self.relation_vector_db.upsert(data_for_vdb) # type: ignore
            await self.relation_vector_db.index_done_callback()

        except Exception as e:
            logger.error(f"Failed to insert relations into vector DB: {e}")

    async def insert_chunks(self, chunks: List[Chunk]) -> None:
        """
        Stores raw chunks in a KV storage (id -> chunk fields).
        """
        if self.chunks_kv_storage is not None:
            data_for_kv = {}
            for chunk in chunks:
                chunk_dict = asdict(chunk)
                chunk_id = chunk_dict.pop("id")
                data_for_kv[chunk_id] = chunk_dict

            await self.chunks_kv_storage.upsert(data_for_kv)
            await self.chunks_kv_storage.index_done_callback()

    async def insert_communities(self, communities: List[Community]) -> None:
        """
        Store communities as ids only:
        community.id -> {
            "level": int,
            "cluster_id": int,
            "entity_ids": [str, ...],
            "relation_ids": [str, ...]
        }
        """
        if self.community_kv_storage is None:
            logger.warning("Community KV storage is not initialized.")
            return

        try:
            data_for_kv: Dict[str, Dict] = {}
            for c in communities:
                entity_ids = []
                seen_e = set()
                for e in c.entities:
                    eid = getattr(e, "id", str(e))
                    if eid not in seen_e:
                        entity_ids.append(eid)
                        seen_e.add(eid)

                relation_ids = []
                seen_r = set()
                for r in c.relations:
                    rid = getattr(r, "id", str(r))
                    if rid not in seen_r:
                        relation_ids.append(rid)
                        seen_r.add(rid)

                data_for_kv[c.id] = {
                    "level": c.level,
                    "cluster_id": c.cluster_id,
                    "entity_ids": entity_ids,
                    "relation_ids": relation_ids,
                }

            await self.community_kv_storage.upsert(data_for_kv)
            await self.community_kv_storage.index_done_callback()

        except Exception as e:
            logger.error(f"Failed to insert communities into KV storage: {e}")

    async def insert_community_summaries(self, summaries: List[CommunitySummary]) -> None:
        """
        Store summaries as id -> text.
        """
        if self.community_summary_kv_storage is None:
            logger.warning("Community summary KV storage is not initialized.")
            return

        try:
            data_for_kv = {s.id: s.summary for s in summaries}
            await self.community_summary_kv_storage.upsert(data_for_kv)
            await self.community_summary_kv_storage.index_done_callback()

        except Exception as e:
            logger.error(f"Failed to insert community summaries into KV storage: {e}")