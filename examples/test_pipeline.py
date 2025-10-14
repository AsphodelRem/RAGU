import asyncio
import os
import time
from dotenv import load_dotenv

from ragu.triplet.pipeline import (
    Pipeline,
    NERClient,
    NENClient,
    REClient,
    DescriptionClient,
    NERStep,
    NENStep,
    REStep,
    DescriptionStep,
)
from ragu.chunker.types import Chunk

load_dotenv()

async def main():
    # Wait for the services to start
    time.sleep(60)

    # Create clients for each service
    ner_client = NERClient(os.getenv("NER_SERVICE_BASE_URL"))
    nen_client = NENClient(os.getenv("NEN_SERVICE_BASE_URL"))
    re_client = REClient(os.getenv("RE_SERVICE_BASE_URL"))
    description_client = DescriptionClient(os.getenv("DESCRIPTION_SERVICE_BASE_URL"))

    # Create the pipeline steps
    steps = [
        NERStep(ner_client),
        NENStep(nen_client),
        REStep(re_client),
        DescriptionStep(description_client),
    ]

    # Create the pipeline
    pipeline = Pipeline(steps)

    # Run the pipeline
    chunk = Chunk(content="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.", chunk_order_idx=0, doc_id="test_doc")
    result = await pipeline.extract([chunk])

    print(result)

if __name__ == "__main__":
    asyncio.run(main())