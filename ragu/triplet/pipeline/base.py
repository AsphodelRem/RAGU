from abc import ABC, abstractmethod
from typing import Any, Dict


class PipelineStep(ABC):
    """
    Abstract base class for a step in the triplet extraction pipeline.
    """

    @abstractmethod
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the pipeline step.

        :param context: The context from the previous step.
        :return: The context for the next step.
        """
        pass


class BasePipeline(ABC):
    """
    Abstract base class for a triplet extraction pipeline.
    """

    def __init__(self, steps: list[PipelineStep]):
        self.steps = steps

    @abstractmethod
    async def run(self, initial_context: Dict[str, Any]) -> Any:
        """
        Runs the entire pipeline.

        :param initial_context: The initial context for the pipeline.
        :return: The final result of the pipeline.
        """
        pass
