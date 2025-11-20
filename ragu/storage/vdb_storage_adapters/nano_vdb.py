# Based on https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/_storage/vdb_nanovectordb.py

import os
from typing import Any, Dict, List

import numpy as np
from nano_vectordb import NanoVectorDB

from ragu.common.batch_generator import BatchGenerator
from ragu.embedder.base_embedder import BaseEmbedder
from ragu.common.global_parameters import Settings
from ragu.storage.base_storage import BaseVectorStorage
from ragu.common.logger import logger


class NanoVectorDBStorage(BaseVectorStorage):
    """
    Vector storage implementation using NanoVectorDB as the backend.

    This class provides a simple vector database for storing and retrieving
    text embeddings, enabling similarity search operations such as nearest
    neighbor queries. Embeddings are generated using a provided embedder model.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        batch_size: int = 16,
        cosine_threshold: float = 0.2,
        storage_folder: str = Settings.storage_folder,
        filename: str = "data.json",
        **kwargs
    ):
        """
        Initialize the NanoVectorDB-based vector storage.

        :param embedder: The embedding model used to compute vector representations.
        :param batch_size: Number of documents to embed per batch.
        :param cosine_threshold: Minimum cosine similarity threshold for query filtering.
        :param storage_folder: Folder where the vector storage file is located.
        :param filename: Name of the JSON file containing the stored vectors.
        :param kwargs: Additional keyword arguments passed to the base class.
        """
        super().__init__(**kwargs)

        self.filename = os.path.join(storage_folder, filename)
        self.batch_size = batch_size
        self.embedder = embedder
        self.cosine_threshold = cosine_threshold
        self._client = NanoVectorDB(
            embedder.dim,
            storage_file=self.filename
        )

    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> List[Any]:
        """
        Insert or update a batch of vectorized documents in the database.

        This method computes embeddings for the provided data using the
        configured embedder, attaches them to the documents, and upserts
        them into the underlying NanoVectorDB instance.

        :param data: Dictionary mapping document IDs to their content and metadata.
        :return: List of records successfully inserted or updated.
        """
        if not data:
            logger.warning("Attempted to insert empty data into vector DB.")
            return []

        list_data = [
            {"__id__": key, **{k: v for k, v in value.items()}}
            for key, value in data.items()
        ]

        contents = [value["content"] for value in data.values()]

        batch_generator = BatchGenerator(contents, batch_size=self.batch_size)
        embeddings_list = [await self.embedder(batch) for batch in batch_generator.get_batches()]
        embeddings = np.concatenate(embeddings_list)

        for item, embedding in zip(list_data, embeddings):
            item["__vector__"] = embedding

        return self._client.upsert(datas=list_data) # type: ignore

    async def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the most similar documents in the vector database.

        Generates an embedding for the given query and performs a cosine
        similarity search against all stored vectors, returning the top
        ``k`` results exceeding the similarity threshold.

        :param query: Input text query for similarity search.
        :param top_k: Number of nearest neighbors to return.
        :return: List of dictionaries containing matched records and their distances.
        """
        embedding = await self.embedder(query)
        results = self._client.query(
            query=embedding[0],
            top_k=top_k,
            better_than_threshold=self.cosine_threshold
        )
        return [{**res, "id": res["__id__"], "distance": res["__metrics__"]} for res in results]

    async def index_start_callback(self):
        pass

    async def query_done_callback(self):
        pass

    async def index_done_callback(self) -> None:
        """
        Save the current state of the NanoVectorDB to disk.

        This method ensures that any newly inserted or updated vectors
        are persisted in the storage file.
        """
        self._client.save()


