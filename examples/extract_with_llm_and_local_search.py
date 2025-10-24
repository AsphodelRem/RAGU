import asyncio

from ragu.llm import OpenAIClient
from ragu.embedder import OpenAIEmbedder

from ragu.chunker import SimpleChunker
from ragu.graph import KnowledgeGraph, InMemoryGraphBuilder

from ragu.triplet import ArtifactsExtractorLLM
from ragu.storage import Index

from ragu.search_engine import LocalSearchEngine

from ragu.common.global_parameters import Settings
from ragu.utils.ragu_utils import read_text_from_files


EMBEDDER_MODEL_NAME = "..."
LLM_MODEL_NAME = "..."
BASE_URL = "..."
API_KEY = "..."


async def main():
    # Set working dir
    Settings.storage_folder = "../example_knowledge_graph"

    # Read documents
    docs = read_text_from_files("../examples/data/ru")

    # Set up chunker
    chunker = SimpleChunker(max_chunk_size=2048, overlap=0)

    # Set up LLM and embedder
    client = OpenAIClient(
        LLM_MODEL_NAME,
        BASE_URL,
        API_KEY,
        max_requests_per_second=1,
        max_requests_per_minute=60
    )
    embedder = OpenAIEmbedder(
        EMBEDDER_MODEL_NAME,
        BASE_URL,
        API_KEY,
        dim=3072,
        max_requests_per_second=1,
        max_requests_per_minute=60
    )

    # Create llm artifact extractor
    artifact_extractor = ArtifactsExtractorLLM(client=client, do_validation=True)

    # Set up pipeline
    pipeline = InMemoryGraphBuilder(client, chunker, artifact_extractor)
    index = Index(
        embedder,
        graph_storage_kwargs={"clustering_params": {"max_cluster_size": 6}}
    )

    # Build knowledge graph
    knowledge_graph = await KnowledgeGraph(
        extraction_pipeline=pipeline,
        index=index,
        make_community_summary=True,
        language="russian",
    ).build_from_docs(docs)

    # Set up search engine
    search_engine = LocalSearchEngine(
        client,
        knowledge_graph,
        embedder
    )

    # Run RAGU :)
    questions = [
        "Кто написал гимн Норвегии?",
        "Шум, издаваемый ЭТИМИ ПАУКООБРАЗНЫМИ, слышен за пять километров. Отсюда и их название.",
        "Как переводится роман 'Ка́мо гряде́ши, Го́споди?'"
    ]

    for question in questions:
        print(await search_engine.a_query(question))

if __name__ == "__main__":
    asyncio.run(main())
