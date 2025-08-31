# Based on https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/_storage/vdb_nanovectordb.py

import logging
import os
from typing import Any, Dict, List

import numpy as np
from nano_vectordb import NanoVectorDB

from ragu.common.batch_generator import BatchGenerator
from ragu.common.embedder import BaseEmbedder
from ragu.common.global_parameters import storage_run_dir
from ragu.storage.base_storage import BaseVectorStorage


class NanoVectorDBStorage(BaseVectorStorage):
    def __init__(
            self,
            embedder: BaseEmbedder,
            batch_size: int=16,
            cosine_threshold: float = 0.2,
            storage_folder: str = storage_run_dir,
            filename: str="data.json",
            **kwargs
    ):
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
        if not data:
            logging.warning("Attempted to insert empty data into vector DB.")
            return []

        list_data = [
            {"__id__": key, **{k: v for k, v in value.items()}}
            for key, value in data.items()
        ]

        contents = [value["content"] for value in data.values()]

        batch_generator = BatchGenerator(contents, batch_size=self.batch_size)
        embeddings_list = [self.embedder(batch) for batch in batch_generator.get_batches()]
        embeddings = np.concatenate(embeddings_list)

        for item, embedding in zip(list_data, embeddings):
            item["__vector__"] = embedding

        return self._client.upsert(datas=list_data)

    async def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        embedding = self.embedder(query)[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_threshold
        )
        return [{**res, "id": res["__id__"], "distance": res["__metrics__"]} for res in results]

    async def index_done_callback(self) -> None:
        self._client.save()
