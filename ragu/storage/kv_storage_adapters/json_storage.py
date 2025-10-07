# Based on https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/_storage/vdb_nanovectordb.py

import json
import os

from ragu.common.global_parameters import storage_run_dir
from ragu.storage.base_storage import BaseKVStorage


class JsonKVStorage(BaseKVStorage):
    def __init__(self, storage_folder: str=storage_run_dir, filename: str="kv_store.json"):
        self.filename = os.path.join(storage_folder, filename)
        if not os.path.exists(self.filename):
            self.data = {}
            with open(self.filename, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        else:
            with open(self.filename, encoding="utf-8") as f:
                self.data = json.load(f)

    async def all_keys(self) -> list[str]:
        return list(self.data.keys())

    async def index_done_callback(self):
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    async def get_by_id(self, id):
        return self.data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self.data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self.data[id].items() if k in fields}
                if self.data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self.data])

    async def upsert(self, data: dict[str, dict]):
        self.data.update(data)

    async def drop(self):
        self.data = {}
