from abc import ABC, abstractmethod
from typing import List

from ragu.common.register import Registrable


class Generator(ABC, Registrable):
    """
    Abstract base class for generators that create final answers based on queries and summaries.
    Must be implemented in subclasses.
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def generate_final_answer(self, query: str, community_summaries: List[str], client, *args, **kwargs):
        """
        Abstract method for generating the final answer based on the query and community summaries.
        Must be implemented in subclasses.

        :param query: The query to generate a final answer for.
        :param community_summaries: A list of community summaries related to the query.
        :param client: The client responsible for making external API requests.
        :param *args: Additional arguments for the subclass's implementation.
        :param **kwargs: Additional keyword arguments for the subclass's implementation.
        :return: The final answer generated by the subclass implementation.
        """
        pass
    
    def __call__(self, query: str, community_summaries: List[str], client, *args, **kwargs):
        """
        Calls the `generate_final_answer` method for convenience.
        
        This allows instances of the `Generator` class (and its subclasses) to be used like functions.

        :param query: The query to pass to the `generate_final_answer` method.
        :param community_summaries: A list of community summaries to pass to the method.
        :param client: The client to pass to the method.
        :param *args: Additional arguments to pass to the method.
        :param **kwargs: Additional keyword arguments to pass to the method.
        :return: The result of calling `generate_final_answer`.
        """
        return self.generate_final_answer(query, community_summaries, client, *args, **kwargs)
