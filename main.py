import asyncio
import os
from dotenv import load_dotenv

from ragu.chunker import SimpleChunker
from ragu.embedder import STEmbedder
from ragu.graph import KnowledgeGraph, InMemoryGraphBuilder
from ragu.llm import OpenAIClient
from ragu.storage import Index
from ragu.triplet.pipeline import (
    Pipeline,
    NERClient,
    NENClient,
    REClient,
    DescriptionClient,
    NERStep,
    NENStep,
    REStep,
    EntityDescriptionStep,
    RelationDescriptionStep,
)
from ragu.search_engine import LocalSearchEngine
from ragu.utils.ragu_utils import read_text_from_files

load_dotenv()

# Configuration for the vLLM server (custom_service)
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LLM_BASE_URL = "https://router.huggingface.co/v1" # Default to 8002 if not set
LLM_API_KEY = os.getenv("LLM_API_KEY")

# Configuration for other services
NER_SERVICE_BASE_URL = os.getenv("NER_SERVICE_BASE_URL", "http://localhost:8010")
RE_SERVICE_BASE_URL = os.getenv("RE_SERVICE_BASE_URL", "http://localhost:8003")
NEN_SERVICE_BASE_URL = os.getenv("NEN_SERVICE_BASE_URL", "http://localhost:8002")

async def main():
    print("Starting RAGU engine test...")

    # 1. Load Data
    print("Loading data from examples/data/ru/...")
    docs = read_text_from_files("examples/data/ru/", file_extensions=[".txt"])
    if not docs:
        print("No documents found. Please ensure 'examples/data/ru/' contains .txt files.")
        return

    # 2. Initialize LLM Client for vLLM
    print("Initializing OpenAIClient for vLLM server...")
    llm_client = OpenAIClient(
        model_name=LLM_MODEL_NAME,
        base_url=LLM_BASE_URL,
        api_token=LLM_API_KEY,
        max_requests_per_second=20,
        max_requests_per_minute=400,
    )

    # 3. Initialize Triplet Extraction Pipeline Clients and Steps
    print("Initializing Triplet Extraction Pipeline...")
    ner_client = NERClient(NER_SERVICE_BASE_URL)
    nen_client = NENClient(NEN_SERVICE_BASE_URL) # Uses vLLM
    re_client = REClient(RE_SERVICE_BASE_URL)
    description_client = DescriptionClient(NEN_SERVICE_BASE_URL) # Uses vLLM

    pipeline_steps = [
        NERStep(ner_client),
        NENStep(nen_client),
        EntityDescriptionStep(description_client),
        REStep(re_client),
        RelationDescriptionStep(description_client),
    ]
    triplet_extraction_pipeline = Pipeline(pipeline_steps)

    # 4. Initialize Chunker and Embedder
    print("Initializing Chunker and Embedder...")
    chunker = SimpleChunker(max_chunk_size=2048, overlap=0)
    embedder = STEmbedder(
        "Alibaba-NLP/gte-multilingual-base",
        trust_remote_code=True
    )

    # 5. Set up Graph Builder and Index
    print("Setting up Graph Builder and Index...")
    graph_builder = InMemoryGraphBuilder(
        client=llm_client, # This LLM client is for summarization within graph building
        chunker=chunker,
        artifact_extractor=triplet_extraction_pipeline,
        language="russian",
    )
    index = Index(
        embedder,
        graph_storage_kwargs={"clustering_params": {"max_cluster_size": 6}}
    )

    # 6. Build Knowledge Graph
    print("Building Knowledge Graph from documents (this may take a while)...")
    knowledge_graph = await KnowledgeGraph(
        extraction_pipeline=graph_builder,
        index=index,
        make_community_summary=True,
        language="russian",
    ).build_from_docs(docs)
    print("Knowledge Graph built successfully.")

    # 7. Initialize Search Engine
    print("Initializing LocalSearchEngine...")
    search_engine = LocalSearchEngine(
        client=llm_client, # This LLM client is for answering queries
        knowledge_graph=knowledge_graph,
        embedder=embedder
    )
    print("LocalSearchEngine initialized.")

    # 8. Define Questions and Query the Engine
    print("\nQuerying the RAGU engine with questions...")
    questions = [
        "Кто такой Бьёрнстьерне Бьёрнсон?",
        "Что такое Ла Фениче?",
        "Какой роман написал Генрик Сенкевич?",
        "Где обитают ревуны?",
        "Как переводится 'Камо грядеши'?",
        "Кто написал гимн Норвегии?",
    ]

    for i, question in enumerate(questions):
        print(f"\n--- Question {i+1}: {question} ---")
        answer = await search_engine.a_query(question)
        print(f"Answer: {answer}")

    print("\nRAGU engine test complete.")

if __name__ == "__main__":
    asyncio.run(main())
