from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Generic, TypeVar, List, Set

import numpy as np

from ragu.graph.types import Entity, Relation


@dataclass
class BaseStorage(ABC):
    @abstractmethod
    async def index_start_callback(self):
        pass

    @abstractmethod
    async def index_done_callback(self):
        pass

    @abstractmethod
    async def query_done_callback(self):
        pass


@dataclass
class BaseVectorStorage(BaseStorage, ABC):
    @abstractmethod
    async def query(self, query: str, top_k: int) -> list[dict]:
        ...

    @abstractmethod
    async def upsert(self, data: dict[str, np.ndarray]):
        ...


T = TypeVar("T")

@dataclass
class BaseKVStorage(Generic[T], BaseStorage, ABC):
    @abstractmethod
    async def all_keys(self) -> List[str]:
        ...

    @abstractmethod
    async def get_by_id(self, id: str) -> Union[T, None]:
        ...

    @abstractmethod
    async def get_by_ids(self, ids: list[str], fields: Union[set[str], None] = None) -> List[Union[T, None]]:
        ...

    async def index_done_callback(self):
        pass

    @abstractmethod
    async def filter_keys(self, data: List[str]) -> Set[str]:
        ...

    @abstractmethod
    async def upsert(self, data: dict[str, T]):
        ...

    @abstractmethod
    async def drop(self):
        ...


@dataclass
class BaseGraphStorage(BaseStorage, ABC):
    @abstractmethod
    async def has_node(self, node_id: str) -> bool:
        ...

    @abstractmethod
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        ...

    @abstractmethod
    async def node_degree(self, node_id: str) -> int:
        ...

    @abstractmethod
    async def get_node(self, node_id: str) -> Entity | None:
        ...

    @abstractmethod
    async def get_edge(self, source_node_id: str, target_node_id: str) -> Relation | None:
        ...

    @abstractmethod
    async def get_node_edges(self, source_node_id: str) -> List[Relation]:
        ...

    @abstractmethod
    async def upsert_node(self, node: Entity):
        ...

    @abstractmethod
    async def upsert_edge(self, edge: Relation):
        ...

    @abstractmethod
    async def take_only_largest_component(self):
        ...

    @abstractmethod
    async def remove_isolated_nodes(self):
        ...

    @abstractmethod
    async def cluster(self, **kwargs):
        ...

    @abstractmethod
    async def index_start_callback(self):
        ...

    @abstractmethod
    async def index_done_callback(self):
        ...