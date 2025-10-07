import math
from typing import Any, Generator, Iterable, List, Sized


class BatchGenerator:
    """
    A utility class for generating batches of data.
    """

    def __init__(self, data: Iterable[Any] | Sized, batch_size: int):
        """
        Initializes the BatchGenerator.

        :param data: A list of strings representing the dataset.
        :param batch_size: The number of elements in each batch.
        """
        self.data = data
        self.batch_size = batch_size

    def __call__(self) -> Generator:
        yield self.get_batches()

    def get_batches(self) -> Generator:
        """
        Generates batches from the dataset.

        :return: A generator that yields batches of data.
        """
        for i in range(0, len(self.data), self.batch_size):
            yield self.data[i : i + self.batch_size]

    def __len__(self) -> int:
        """
        Returns the number of batches.

        :return: The total number of batches.
        """
        return math.ceil(len(self.data) / self.batch_size)
