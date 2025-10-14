from typing import Any, Dict, List, Tuple, Iterable

from ragu.triplet.base_artifact_extractor import BaseArtifactExtractor
from ragu.chunker.types import Chunk
from ragu.graph.types import Entity, Relation
from ragu.triplet.pipeline.base import PipelineStep


class Pipeline(BaseArtifactExtractor):
    """
    A pipeline for triplet extraction from text.
    """

    def __init__(self, steps: list[PipelineStep]):
        self.steps = steps

    async def extract(
        self,
        chunks: Iterable[Chunk],
        *args,
        **kwargs
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Runs the entire pipeline.

        :param chunks: The initial context for the pipeline.
        :return: The final result of the pipeline.
        """
        all_entities = []
        all_relations = []
        for chunk in chunks:
            context = {"text": chunk.content}
            for step in self.steps:
                context = await step.run(context)
            all_entities.extend(context.get("entities", []))
            all_relations.extend(context.get("relations", []))
        return all_entities, all_relations
