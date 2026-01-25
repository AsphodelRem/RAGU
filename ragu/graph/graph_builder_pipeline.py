from dataclasses import dataclass
from typing import Any, List, Tuple

from ragu.chunker import BaseChunker
from ragu.chunker.types import Chunk
from ragu.common.global_parameters import Settings
from ragu.embedder.base_embedder import BaseEmbedder
from ragu.graph.artifacts_summarizer import EntitySummarizer, RelationSummarizer
from ragu.graph.community_summarizer import CommunitySummarizer
from ragu.graph.types import CommunitySummary, Community, Entity, Relation
from ragu.llm.base_llm import BaseLLM
from ragu.triplet.base_artifact_extractor import BaseArtifactExtractor


@dataclass
class KnowledgeGraphBuilderSettings:
    """
    Configuration settings for the knowledge graph building pipeline.

    This dataclass controls various aspects of graph construction including
    summarization strategies, clustering behavior, and optimization modes.

    Attributes
    ----------
    use_llm_summarization : bool, default=True
        Enable LLM-based summarization for merging and deduplicating similar
        entity and relation descriptions.
    use_clustering : bool, default=False
        Apply clustering to group similar entities before summarization.
        Helps when number of similar entities is very large.
    build_only_vector_context : bool, default=False
        Skip entity/relation extraction and build a context only for naive (vector) RAG.
    make_community_summary : bool, default=True
        Generate high-level summaries for detected communities in the graph.
        Required for global search operations that rely on community-level context.
    remove_isolated_nodes : bool, default=True
        Remove entities that have no relations to other entities in the graph.
    vectorize_chunks : bool, default=False
        Generate and store embeddings for text chunks.
    cluster_only_if_more_than : int, default=10000
        Minimum number of entities required before clustering is applied.
    max_cluster_size : int, default=128
        Maximum number of entities per cluster during summarization.
    random_seed : int, default=42
        Random seed for reproducible clustering and community detection results.
    """
    use_llm_summarization: bool = True
    use_clustering: bool = False
    build_only_vector_context: bool = False
    make_community_summary: bool = True
    remove_isolated_nodes: bool = True
    vectorize_chunks: bool = False
    cluster_only_if_more_than: int = 10000
    max_cluster_size: int = 128
    random_seed: int = 42


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

    When `build_parameters.build_only_vector_context=True`, steps 2-5 are skipped,
    and only chunking is performed. This is useful for naive vector RAG where only
    chunk embeddings are needed without knowledge graph construction.

    Parameters
    ----------
    client : BaseLLM, optional
        LLM client used for all text understanding and summarization tasks.
        Not required if build_parameters.build_only_vector_context=True.
    chunker : BaseChunker, optional
        Module responsible for splitting documents into chunks.
    artifact_extractor : BaseArtifactExtractor, optional
        Extracts entities and relations from text chunks.
        Not required if build_parameters.build_only_vector_context=True.
    build_parameters : KnowledgeGraphBuilderSettings, optional
        Configuration settings controlling graph building behavior including
        summarization, clustering, and optimization modes.
    embedder : BaseEmbedder, optional
        Embedding model used for vectorizing entities, relations, and optionally chunks.
    llm_cache_flush_every : int, default=100
        Number of LLM calls between cache flushes to disk.
        Lower values increase I/O.
    embedder_cache_flush_every : int, default=100
        Number of embedder calls between cache flushes to disk.
        Embedder caches are typically larger, so default flush frequency is lower.
    additional_pipeline : list[GraphBuilderModule], optional
        Optional list of post-processing modules applied after main extraction.
        Used for custom normalization, filtering and others logic.
    language : str, optional
        Working language for all tasks.
        Default: inherited from global Settings.language ("english").
    """

    def __init__(
        self,
        client: BaseLLM = None,
        chunker: BaseChunker = None,
        artifact_extractor: BaseArtifactExtractor = None,
        build_parameters: KnowledgeGraphBuilderSettings = KnowledgeGraphBuilderSettings(),
        embedder: BaseEmbedder = None,
        llm_cache_flush_every: int = 100,
        embedder_cache_flush_every: int = 100,
        additional_pipeline: List[GraphBuilderModule] = None,
        language: str | None = None
    ):
        self.client = client
        self.chunker = chunker
        self.artifact_extractor = artifact_extractor
        self.additional_pipeline = additional_pipeline
        self.embedder = embedder
        self.llm_cache_flush_every = llm_cache_flush_every
        self.embedder_cache_flush_every = embedder_cache_flush_every
        self.language = language if language else Settings.language
        self.build_parameters = build_parameters

        if self.build_parameters.build_only_vector_context:
            # No need to create those instances => we are able not to think about its parameters
            self.entity_summarizer, self.relation_summarizer, self.community_summarizer = None, None, None
        else:
            self.entity_summarizer = EntitySummarizer(
                client,
                use_llm_summarization=self.build_parameters.use_llm_summarization,
                use_clustering=self.build_parameters.use_clustering,
                cluster_only_if_more_than=self.build_parameters.cluster_only_if_more_than,
                embedder=embedder,
                language=self.language,
            )
            self.relation_summarizer = RelationSummarizer(
                client,
                use_llm_summarization=self.build_parameters.use_llm_summarization,
                language=self.language
            )
            self.community_summarizer = CommunitySummarizer(self.client, language=self.language)

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
        if self.build_parameters.build_only_vector_context:
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
