import asyncio
import os
from typing import Dict, Optional, Type

from ragu.common.embedder import BaseEmbedder
from ragu.common.global_parameters import DEFAULT_FILENAMES, storage_run_dir
from ragu.common.logger import logging
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.storage.base_storage import BaseKVStorage, BaseVectorStorage
from ragu.storage.json_storage import JsonKVStorage
from ragu.storage.nano_db import NanoVectorDBStorage
from ragu.utils.hash import compute_mdhash_id


class Index:
    """
    Index class that manages embedding storage and indexing for a knowledge graph.
    """

    def __init__(
            self,
            embedder: BaseEmbedder,
            kv_storage_type: Type[BaseKVStorage] = JsonKVStorage,
            vdb_storage_type: Type[BaseVectorStorage] = NanoVectorDBStorage,
            chunks_kv_storage_kwargs: Optional[Dict] = None,
            summary_kv_storage_kwargs: Optional[Dict] = None,
            vdb_storage_kwargs: Optional[Dict] = None,
            storage_folder: str = storage_run_dir,
    ):
        """
        Initializes the Index with storage configurations and an embedding model.
        """
        self.embedder = embedder
        self.summary_kv_storage_kwargs = summary_kv_storage_kwargs or {}
        self.chunks_kv_storage_kwargs = chunks_kv_storage_kwargs or {}
        self.vdb_storage_kwargs = vdb_storage_kwargs or {}

        # Define file paths
        self.summary_kv_storage_kwargs["filename"] = os.path.abspath(
            os.path.join(storage_folder, DEFAULT_FILENAMES["community_summary_kv_storage_name"])
        )
        self.chunks_kv_storage_kwargs["filename"] = os.path.abspath(
            os.path.join(storage_folder, DEFAULT_FILENAMES["chunks_kv_storage_name"])
        )
        self.vdb_storage_kwargs["filename"] = os.path.abspath(
            os.path.join(storage_folder, DEFAULT_FILENAMES["entity_vdb_name"])
        )

        # Key-value storages
        self.chunks_kv_storage = kv_storage_type(**self.chunks_kv_storage_kwargs)
        self.communities_kv_storage = kv_storage_type(**self.summary_kv_storage_kwargs)

        # Vector storage
        self.entity_vector_db = vdb_storage_type(embedder=embedder, **self.vdb_storage_kwargs)

    def make_index(self, knowledge_graph: KnowledgeGraph):
        """
        Creates an index for the given knowledge graph by storing entities, chunks, and summaries.

        :param knowledge_graph: The knowledge graph to index.
        """
        asyncio.run(self._insert_entities_to_vdb(knowledge_graph))

    async def _insert_entities_to_vdb(self, knowledge_graph: KnowledgeGraph) -> None:
        """
        Inserts entities from the knowledge graph into the vector database.

        :param knowledge_graph: The knowledge graph to extract entities from.
        """
        if self.entity_vector_db is None:
            logging.warning("Vector database storage is not initialized.")
            return

        try:
            data_for_vdb = {
                compute_mdhash_id(dp[0], prefix="ent-"): {
                    "entity_name": dp[0],
                    "content": dp[0] + " - " +  dp[1].get("description", ""),
                }
                for dp in knowledge_graph.graph.nodes.data()
            }
            await self.entity_vector_db.upsert(data_for_vdb)
            await self.entity_vector_db.index_done_callback()

        except Exception as e:
            logging.error(f"Failed to insert entities into vector DB: {e}")
