from typing import Any, List, Tuple

from ragu.chunker import BaseChunker
from ragu.chunker.types import Chunk
from ragu.graph.community_summarizer import CommunitySummarizer
from ragu.graph.types import CommunitySummary
from ragu.graph.types import Community
from ragu.graph.types import Entity, Relation
from ragu.graph.entity_alignment import ArtifactsDescriptionSummarizer
from ragu.llm.base_llm import BaseLLM
from ragu.triplet.base_artifact_extractor import BaseArtifactExtractor


class GraphBuilderModule:
    def run(self, entities: List[Entity] | None, relations: List[Relation] | None, **kwargs) -> Any:
        """
        For batch update (insert multiple nodes or edges when we are building the graph)
        """
        ...


class InMemoryGraphBuilder:
    def __init__(
            self,
            client: BaseLLM,
            chunker: BaseChunker,
            artifact_extractor: BaseArtifactExtractor,
            additional_pipeline: List[GraphBuilderModule] = None,
            language: str = "english",
    ):
        self.client = client
        self.chunker = chunker
        self.artifact_extractor = artifact_extractor
        self.additional_pipeline = additional_pipeline
        self.language = language

        self.artifact_summarizer = ArtifactsDescriptionSummarizer(
            client,
            use_llm_summarization=True,
            language=language
        )
        self.community_summarizer = CommunitySummarizer(self.client, language=language)

    async def extract_graph(self, documents: List[str]) -> Tuple[List[Entity], List[Relation], List[Chunk]]:
        """

        :param documents:
        :return:
        """
        # Step 1: chunking
        chunks = self.chunker(documents)

        # Step 2: extract entities and relations
        entities, relations = await self.artifact_extractor(chunks)

        # Step 3: summarize similar artifacts's description
        entities, relations = await self.artifact_summarizer.run(entities, relations)

        return entities, relations, chunks

    async def get_community_summary(self, communities: List[Community]) -> List[CommunitySummary]:
        return await self.community_summarizer.summarize(communities)

