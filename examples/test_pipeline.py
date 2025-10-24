import asyncio
import os
from dotenv import load_dotenv

from ragu.llm.openai_client import OpenAIClient
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
from ragu.chunker.types import Chunk

load_dotenv()

LLM_MODEL_NAME = "ragu-lm"
LLM_BASE_URL = os.getenv("NEN_SERVICE_BASE_URL", "http://localhost:8002")
LLM_API_KEY = "EMPTY"

NER_SERVICE_BASE_URL = os.getenv("NER_SERVICE_BASE_URL", "http://localhost:8010")
RE_SERVICE_BASE_URL = os.getenv("RE_SERVICE_BASE_URL", "http://localhost:8003")

async def main():
    print("Starting pipeline test...")

    # Create OpenAI client for vLLM
    llm_client = OpenAIClient(
        model_name=LLM_MODEL_NAME,
        base_url=LLM_BASE_URL,
        api_token=LLM_API_KEY,
        max_requests_per_second=20,
        max_requests_per_minute=400,
    )

    # Create clients for each service
    ner_client = NERClient(NER_SERVICE_BASE_URL)
    nen_client = NENClient(LLM_BASE_URL)
    re_client = REClient(RE_SERVICE_BASE_URL)
    description_client = DescriptionClient(LLM_BASE_URL)

    # Create the pipeline steps in the correct order
    steps = [
        NERStep(ner_client),
        NENStep(nen_client),
        EntityDescriptionStep(description_client),
        REStep(re_client),
        RelationDescriptionStep(description_client),
    ]

    # Create the pipeline
    pipeline = Pipeline(steps)

    # Run the pipeline on a sample chunk
    print("Running pipeline on a sample chunk...")
    chunk = Chunk(
        content="Главным борцом с пробками назначен заместитель министра транспорта России Николай Лямов.",
        chunk_order_idx=0,
        doc_id="test_doc"
    )
    entities, relations = await pipeline.extract([chunk])

    print("\n--- Extraction Complete ---")
    print(f"Extracted {len(entities)} entities:")
    for entity in entities:
        print(entity)
    
    print(f"\nExtracted {len(relations)} relations:")
    for relation in relations:
        print(relation)

if __name__ == "__main__":
    asyncio.run(main())
