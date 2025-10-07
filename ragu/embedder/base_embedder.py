from abc import abstractmethod, ABC
from typing import List


class BaseEmbedder(ABC):
    """
    Abstract base class for text embedders.
    """

    def __init__(self, dim: int | None = None):
        self.dim = dim

    @abstractmethod
    async def embed(self, texts: List[str]):
        """
        Computes embeddings for a list of text inputs.
        """
        ...

    async def __call__(self, *args, **kwargs):
        return await self.embed(*args, **kwargs)