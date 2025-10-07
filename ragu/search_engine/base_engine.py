from abc import ABC, abstractmethod

from pydantic import BaseModel

from ragu.common.base import RaguGenerativeModule
from ragu.utils.ragu_utils import always_get_an_event_loop


class BaseEngine(RaguGenerativeModule, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    async def a_search(self, query, *args, **kwargs):
        """
        Get relevant information from knowledge graph
        """
        pass

    @abstractmethod
    async def a_query(self, query: str) -> BaseModel:
        """
        Get answer on query from knowledge graph
        """
        pass

    async def query(self, query: str) -> BaseModel:
        """
        Get answer on query from knowledge graph
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.a_query(query)
        )

    async def search(self, query, *args, **kwargs):
        """
        Get relevant information from knowledge graph
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.a_search(query, *args, **kwargs)
        )
