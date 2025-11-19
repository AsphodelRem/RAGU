# Pipeline-based Triplet Extraction Guide

This guide explains how to use the OOP-based pipeline for triplet extraction.

## Overview

The pipeline consists of four main steps:

1.  **Named Entity Recognition (NER):** Identifies entities in the text.
2.  **Named Entity Normalization (NEN):** Normalizes the extracted entities.
3.  **Relation Extraction (RE):** Extracts relations between the normalized entities.
4.  **Description Generation:** Generates descriptions for the extracted relations, creating the final triplets.

Each step is implemented as a `PipelineStep` and can be configured to use a model running in a Docker container or a local Python function.

## Running the Pipeline

To run the pipeline, you first need to define the services for each step in the `docker-compose.yml` file. You will need to provide your own Docker images for each service.

```yaml
version: '3.8'

services:
  ner_service:
    image: your_ner_image:latest
    ports:
      - "8001:8000"

  nen_service:
    image: your_nen_image:latest
    ports:
      - "8002:8000"

  re_service:
    image: your_re_image:latest
    ports:
      - "8003:8000"

  description_service:
    image: your_description_image:latest
    ports:
      - "8004:8000"
```

Next, you need to create a `.env` file with the base URLs for each service:

```
NER_SERVICE_BASE_URL=http://localhost:8001
NEN_SERVICE_BASE_URL=http://localhost:8002
RE_SERVICE_BASE_URL=http://localhost:8003
DESCRIPTION_SERVICE_BASE_URL=http://localhost:8004
```

Once the services are running, you can use the pipeline in your Python code:

```python
import asyncio
import os

from ragu.llm.pipeline import (
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

async def main():
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
    initial_context = {"text": "Your text here"}
    result = await pipeline.run(initial_context)

    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```
