import asyncio
import os
from dataclasses import asdict
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Type
)

from ragu.chunker.types import Chunk
from ragu.common.global_parameters import DEFAULT_FILENAMES
from ragu.common.global_parameters import Settings
from ragu.embedder.base_embedder import BaseEmbedder
from ragu.graph.types import (
    Entity,
    Relation,
    Community,
    CommunitySummary
)
from ragu.storage.base_storage import (
    BaseKVStorage,
    BaseVectorStorage,
    BaseGraphStorage
)
from ragu.storage.graph_storage_adapters.networkx_adapter import NetworkXStorage
from ragu.storage.kv_storage_adapters.json_storage import JsonKVStorage
from ragu.storage.vdb_storage_adapters.nano_vdb import NanoVectorDBStorage
from ragu.common.logger import logger


class Index:
    """
    Index class that manages storages for a knowledge graph.
    """

    def __init__(
            self,
            embedder: BaseEmbedder,
            graph_backend_storage: Type[BaseGraphStorage] = NetworkXStorage,
            kv_storage_type: Type[BaseKVStorage] = JsonKVStorage,
            vdb_storage_type: Type[BaseVectorStorage] = NanoVectorDBStorage,
            chunks_kv_storage_kwargs: Optional[Dict] = None,
            summary_kv_storage_kwargs: Optional[Dict] = None,
            communities_kv_storage_kwargs: Optional[Dict] = None,
            vdb_storage_kwargs: Optional[Dict] = None,
            graph_storage_kwargs: Optional[Dict] = None,
    ):
        """
        Initializes the Index.
        """
        # Initialize storage folder if it doesn't exist
        Settings.init_storage_folder()
        storage_folder: str = Settings.storage_folder

        self.embedder = embedder
        self.summary_kv_storage_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["community_summary_kv_storage_name"],
            summary_kv_storage_kwargs,
        )
        self.community_kv_storage_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["community_kv_storage_name"],
            communities_kv_storage_kwargs,
        )
        self.chunks_kv_storage_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["chunks_kv_storage_name"],
            chunks_kv_storage_kwargs,
        )
        self.vdb_storage_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["entity_vdb_name"],
            vdb_storage_kwargs,
        )
        relation_vdb_storage_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["relation_vdb_name"],
            vdb_storage_kwargs,
        )
        self.graph_storage_kwargs = self._build_storage_kwargs(
            storage_folder,
            DEFAULT_FILENAMES["knowledge_graph_storage_name"],
            graph_storage_kwargs,
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
        self.graph_backend = graph_backend_storage(**self.graph_storage_kwargs)  # type: ignore

    async def make_index(
            self,
            entities: List[Entity] = None,
            relations: List[Relation] = None,
            communities: List[Community] = None,
            summaries: List[CommunitySummary] = None,
    ) -> None:
        """
        Creates an index for the given knowledge graph. Save entities, relations, communities and community summaries.
        """
        tasks = []

        if entities:
            tasks.extend(
                [
                    self._insert_entities_to_graph(entities),
                    self._insert_entities_to_vdb(entities),
                ]
            )
        if relations:
            tasks.extend(
                [
                    self._insert_relations_to_graph(relations),
                    self._insert_relations_to_vdb(relations),
                ]
            )
        if communities:
            tasks.append(self._insert_communities(communities))
        if summaries:
            tasks.append(self._insert_summaries(summaries))

        if tasks:
            await asyncio.gather(*tasks)

    async def _insert_entities_to_graph(self, entities: List[Entity]) -> None:
        if not entities:
            return
        backend = self.graph_backend
        if self.graph_backend is None:
            logger.warning("Graph storage is not initialized.")
            return
        await self._graph_bulk_upsert(backend, entities, backend.upsert_node, "entities")

    async def _insert_relations_to_graph(self, relations: List[Relation]) -> None:
        if not relations:
            return
        backend = self.graph_backend
        if backend is None:
            logger.warning("Graph storage is not initialized.")
            return
        await self._graph_bulk_upsert(backend, relations, backend.upsert_edge, "relations")

    async def _insert_entities_to_vdb(self, entities: List[Entity]) -> None:
        """
        Inserts entities from the knowledge graph into the vector database.
        """
        if not entities:
            return

        data_for_vdb = {
            entity.id: {
                "entity_name": entity.entity_name,
                "content": entity.entity_name + " - " + entity.description,
            }
            for entity in entities
        }
        await self._vdb_upsert(self.entity_vector_db, data_for_vdb, "entities")

    async def _insert_relations_to_vdb(self, relations: List[Relation]) -> None:
        """
        Inserts relations from the knowledge graph into the vector database.
        """
        if not relations:
            return

        data_for_vdb = {
            relation.id: {
                "subject": relation.subject_id,
                "object": relation.object_id,
                "content": relation.description
            }
            for relation in relations
        }
        await self._vdb_upsert(self.relation_vector_db, data_for_vdb, "relations")

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

    async def _insert_communities(self, communities: List[Community]) -> None:
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
                data_for_kv[c.id] = {
                    "level": c.level,
                    "cluster_id": c.cluster_id,
                    "entity_ids": self._unique_ids(c.entities),
                    "relation_ids": self._unique_ids(c.relations),
                }

            await self.community_kv_storage.upsert(data_for_kv)
            await self.community_kv_storage.index_done_callback()

        except Exception as e:
            logger.error(f"Failed to insert communities into KV storage: {e}")

    async def _insert_summaries(self, summaries: List[CommunitySummary]) -> None:
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

    async def _graph_bulk_upsert(
            self,
            backend: BaseGraphStorage,
            items: Iterable[Any],
            upsert_fn: Callable[[Any], Awaitable[None]],
            artifact_label: str,
    ) -> None:
        if not items:
            return

        try:
            await backend.index_start_callback()
            await asyncio.gather(*(upsert_fn(item) for item in items))
        except Exception as e:
            logger.error(f"Failed to insert {artifact_label} into graph: {e}")
        finally:
            await backend.index_done_callback()

    async def _vdb_upsert(
            self,
            storage: Optional[BaseVectorStorage],
            payload: Dict[str, Dict],
            artifact_label: str,
    ) -> None:
        if storage is None:
            logger.warning("Vector database storage is not initialized.")
            return
        if not payload:
            return

        try:
            await storage.upsert(payload)  # type: ignore[arg-type]
        except Exception as e:
            logger.error(f"Failed to insert {artifact_label} into vector DB: {e}")
        finally:
            await storage.index_done_callback()  # type: ignore[func-returns-value]

    @staticmethod
    def _build_storage_kwargs(
            storage_folder: str,
            filename: str,
            provided_kwargs: Optional[Dict] = None,
    ) -> Dict:
        kwargs = dict(provided_kwargs or {})
        kwargs.setdefault(
            "filename",
            os.path.abspath(os.path.join(storage_folder, filename)),
        )
        return kwargs

    @staticmethod
    def _unique_ids(items: Iterable[Any]) -> List[str]:
        ids: List[str] = []
        seen = set()
        for item in items:
            item_id = getattr(item, "id", str(item))
            if item_id not in seen:
                ids.append(item_id)
                seen.add(item_id)
        return ids
