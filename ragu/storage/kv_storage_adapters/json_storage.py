# Based on https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/_storage/vdb_nanovectordb.py

import json
import os

from ragu.common.global_parameters import Settings
from ragu.storage.base_storage import BaseKVStorage


class JsonKVStorage(BaseKVStorage):
    """
    Key-value storage implementation using a local JSON file.

    This class provides a simple persistent storage backend.
    All data is loaded into memory at initialization and written back to disk upon updates.
    """

    def __init__(self, storage_folder: str = Settings.storage_folder, filename: str = "kv_store.json"):
        """
        Initialize the JSON key-value storage.

        Creates a new JSON file if one does not exist, otherwise loads existing data.

        :param storage_folder: Path to the folder where the storage file will be located.
        :param filename: Name of the JSON file used for storage.
        """
        self.filename = os.path.join(storage_folder, filename)
        if not os.path.exists(self.filename):
            self.data = {}
            with open(self.filename, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        else:
            with open(self.filename, encoding="utf-8") as f:
                self.data = json.load(f)

    async def all_keys(self) -> list[str]:
        """
        Return a list of all keys currently stored in the JSON file.

        :return: List of keys present in the store.
        """
        return list(self.data.keys())

    async def get_by_id(self, id):
        """
        Retrieve a record by its unique identifier.

        :param id: Unique identifier key.
        :return: The stored value or ``None`` if not found.
        """
        return self.data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        """
        Retrieve multiple records by their identifiers.

        Optionally return only specified fields from each record.

        :param ids: Iterable of record IDs to fetch.
        :param fields: Optional list of fields to include in the result.
        :return: List of records or field-filtered dictionaries.
        """
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
        """
        Return a subset of keys that are not yet present in the store.

        :param data: List of keys to check.
        :return: Set of keys missing from the storage.
        """
        return set([s for s in data if s not in self.data])

    async def upsert(self, data: dict[str, dict]):
        """
        Insert or update one or more key-value pairs in the store.

        :param data: Dictionary of key-value mappings to update.
        """
        self.data.update(data)

    async def drop(self):
        """
        Remove all records from the store (in-memory only).
        """
        self.data = {}

    async def index_start_callback(self):
        pass

    async def query_done_callback(self):
        pass

    async def index_done_callback(self):
        """
        Persist the current in-memory data to disk.
        """
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
