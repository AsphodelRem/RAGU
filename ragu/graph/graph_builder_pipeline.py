import logging
from typing import Any, List, Tuple

from ragu.chunker import BaseChunker
from ragu.chunker.types import Chunk
from ragu.common.global_parameters import Settings
from ragu.embedder.base_embedder import BaseEmbedder
from ragu.graph.community_summarizer import CommunitySummarizer
from ragu.graph.types import CommunitySummary, Community, Entity, Relation
from ragu.graph.artifacts_summarizer import EntitySummarizer, RelationSummarizer
from ragu.llm.base_llm import BaseLLM
from ragu.triplet.base_artifact_extractor import BaseArtifactExtractor


class GraphBuilderModule:
    """
    Abstract interface for modules that extend the graph-building pipeline.

    Each module receives batches of entities and relations
    and can modify, enrich, or filter them before insertion into the graph.

    Typically used for:
      - normalization of entity names
      - filtering noisy relations
      - post-processing after extraction

    Methods
    -------
    run(entities, relations, **kwargs)
        Perform batch update or enrichment on provided entities and relations.
    """

    def run(
        self, entities: List[Entity] | None, relations: List[Relation] | None, **kwargs
    ) -> Any:
        """
        Process or update multiple nodes and edges during graph construction.

        :param entities: list of :class:`Entity` objects to insert or modify.
        :param relations: list of :class:`Relation` objects to insert or modify.
        :param kwargs: optional additional parameters specific to the module.
        :return: updated or enriched entities/relations or any auxiliary result.
        """
        ...


class InMemoryGraphBuilder:
    """
    High-level orchestrator for extracting and summarizing entities and relations
    directly in memory using an LLM client and supporting components.

    The pipeline consists of:
      1. **Chunking** input documents.
      2. **Entity & relation extraction** using a triplet-based artifact extractor.
      3. **Artifact summarization** for merging and deduplicating similar entities.
      4. (Optional) **Additional modules** for graph enrichment.
      5. **Community summarization** (aggregated graph-level summaries).

    When `build_only_vector_context=True`, steps 2-5 are skipped, and only chunking
    is performed. This is useful for naive vector RAG where only chunk embeddings
    are needed without knowledge graph construction.

    Parameters
    ----------
    client : BaseLLM, optional
        LLM client used for all text understanding and summarization tasks.
        Not required if build_only_vector_context=True.
    chunker : BaseChunker
        Module responsible for splitting documents into semantically meaningful chunks.
    artifact_extractor : BaseArtifactExtractor, optional
        Extracts entities and relations from text chunks (triplet-based).
        Not required if build_only_vector_context=True.
    additional_pipeline : list[GraphBuilderModule], optional
        Optional list of post-processing modules applied after main extraction.
    build_only_vector_context : bool, default=False
        If True, skip entity/relation extraction and only perform chunking.
        Use this for naive vector RAG without knowledge graph construction.
    language : str, default="english"
        Working language for summarization and extraction tasks.
    """

    def __init__(
        self,
        client: BaseLLM = None,
        chunker: BaseChunker = None,
        artifact_extractor: BaseArtifactExtractor = None,
        embedder: BaseEmbedder = None,
        use_llm_summarization: bool = True,
        use_clustering: bool = False,
        cluster_only_if_more_than: int = 128,
        llm_cache_flush_every: int = 100,
        embedder_cache_flush_every: int = 100000,
        additional_pipeline: List[GraphBuilderModule] = None,
        build_only_vector_context: bool = False,
        language: str | None = None,
    ):
        self.client = client
        self.chunker = chunker
        self.artifact_extractor = artifact_extractor
        self.additional_pipeline = additional_pipeline
        self.language = language if language else Settings.language
        self.embedder = embedder
        self.use_llm_summarization = use_llm_summarization
        self.use_clustering = use_clustering
        self.llm_cache_flush_every = llm_cache_flush_every
        self.embedder_cache_flush_every = embedder_cache_flush_every
        self.build_only_vector_context = build_only_vector_context

        if build_only_vector_context:
            # No need to create those instances => we are able not to think about its parameters
            self.entity_summarizer, self.relation_summarizer, self.community_summarizer = None, None, None
        else:
            self.entity_summarizer = EntitySummarizer(
                client,
                use_llm_summarization=use_llm_summarization,
                use_clustering=use_clustering,
                cluster_only_if_more_than=cluster_only_if_more_than,
                embedder=embedder,
                language=language,
            )
            self.relation_summarizer = RelationSummarizer(
                client,
                use_llm_summarization=use_llm_summarization,
                language=language
            )
            self.community_summarizer = CommunitySummarizer(self.client, language=language)

    async def extract_graph(
        self, documents: List[str]
    ) -> Tuple[List[Entity], List[Relation], List[Chunk]]:
        """
        Run the full extraction pipeline and produce entities, relations, and chunks.

        Steps
        -----
        1. Chunk raw documents using :class:`BaseChunker`.
        2. Extract entities and relations via :class:`BaseArtifactExtractor` (skipped if build_only_vector_context=True).
        3. Summarize or merge similar artifacts using :class:`ArtifactsDescriptionSummarizer` (skipped if build_only_vector_context=True).

        :param documents: list of input text documents.
        :return:
            A tuple ``(entities, relations, chunks)`` where
              - **entities** (:class:`list[Entity]`) — extracted and summarized entities (empty if build_only_vector_context=True).
              - **relations** (:class:`list[Relation]`) — extracted and summarized relations (empty if build_only_vector_context=True).
              - **chunks** (:class:`list[Chunk]`) — the original document chunks used for extraction.
        """
        # Step 1: chunking
        chunks = self.chunker(documents)

        # If only building vector context, skip entity/relation extraction
        if self.build_only_vector_context:
            return [], [], chunks

        # Step 2: extract entities and relations
        entities, relations = await self.artifact_extractor(chunks)

        # Step 3: summarize similar artifacts' descriptions
        entities = await self.entity_summarizer.run(entities)
        relations = await self.relation_summarizer.run(relations)

        return entities, relations, chunks

    async def get_community_summary(self, communities: List[Community]) -> List[CommunitySummary]:
        """
        Generate high-level summaries for detected communities in the graph.

        :param communities: list of :class:`Community` objects to summarize.
        :return: list of :class:`CommunitySummary` objects with aggregated information.
        """
        return await self.community_summarizer.summarize(communities)
