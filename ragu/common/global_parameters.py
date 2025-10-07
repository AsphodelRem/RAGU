import os
from datetime import datetime

from loguru import logger


DEFAULT_FILENAMES = {
    "community_kv_storage_name": "kv_community.json",
    "chunks_kv_storage_name": "kv_chunks.json",
    "entity_vdb_name": "vdb_entity.json",
    "relation_vdb_name": "vdb_relation.json",
    "knowledge_graph_storage_name": "knowledge_graph.gml",
    "community_summary_kv_storage_name": "kv_community_summary.json",
}


class GlobalSettings:
    __instance = None
    _current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    _working_dir = os.path.join(os.path.join(os.getcwd(), "ragu_working_dir"), _current_time)

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super(GlobalSettings, cls).__new__(cls)
            return cls.__instance
        else:
            return cls.__instance

    @property
    def storage_folder(self):
        return self._working_dir

    @storage_folder.setter
    def storage_folder(self, path):
        self._working_dir = path

    def init_storage_folder(self):
        if not os.path.exists(self._working_dir):
            logger.info(f"Creating folder for current run: {self._working_dir}")
            os.makedirs(self._working_dir, exist_ok=True)


Settings = GlobalSettings()




