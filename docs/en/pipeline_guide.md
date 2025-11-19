# Pipeline-based Triplet Extraction Guide

This guide explains how to use the OOP-based pipeline for triplet extraction.

## Overview

The pipeline consists of several main steps, orchestrated by the `ragu.triplet.pipeline.Pipeline` class:

1.  **Named Entity Recognition (NER):** Identifies entities in the text.
2.  **Named Entity Normalization (NEN):** Normalizes the extracted entities.
3.  **Entity Description:** Generates a description for each entity based on its context.
4.  **Relation Extraction (RE):** Extracts relations between the normalized entities.
5.  **Relation Description:** Generates descriptions for the extracted relations, creating the final triplets.

Each step is implemented as a `PipelineStep` that communicates with a dedicated microservice.

## Docker Compose and Service Configuration

To run the full pipeline, you need to use the `docker-compose.yml` file located in the project root. This file defines the microservices required for the different pipeline stages.

### Discrepancy with the Project's `docker-compose.yml`

The configuration below is a generic template to illustrate the architecture. The actual `docker-compose.yml` in this repository is slightly different and more optimized:

1.  **Consolidated Services:** The `nen_service` and `description_service` are consolidated into a single `custom_service`. This service runs the `RaguTeam/RAGU-lm` model, which is capable of handling both Named Entity Normalization (NEN) and description generation for both entities and relations.
2.  **Specific Ports:** The `ner_service` in the project's actual `docker-compose.yml` uses port `8010`, not `8001`.

This consolidation is a practical optimization that reduces the number of required services. The `custom_service` is built from the local `./services` directory.

### Generic `docker-compose.yml` Structure

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

### Environment Configuration (`.env`)

You need to create a `.env` file with the base URLs for each running service. Based on the project's actual `docker-compose.yml`, the file should look like this:

```
NER_SERVICE_BASE_URL=http://localhost:8010
NEN_SERVICE_BASE_URL=http://localhost:8002
RE_SERVICE_BASE_URL=http://localhost:8003
DESCRIPTION_SERVICE_BASE_URL=http://localhost:8002
```
*Note that `NEN_SERVICE_BASE_URL` and `DESCRIPTION_SERVICE_BASE_URL` point to the same `custom_service`.*

## Models Used

The pipeline relies on a combination of models served via Docker containers.

### RaguTeam Hugging Face Models

*   **`RaguTeam/RAGU-lm`**: This is a fine-tuned model specifically for Russian language tasks. It is served by the `custom_service` and performs several key steps in the pipeline:
    *   Named Entity Normalization (NEN)
    *   Entity Description Generation
    *   Relation Description Generation

### Docker Hub Images

The following images are pulled from Docker Hub and are used for specialized NLP tasks:

*   **`mrpzzios/runne_contrastive_ner_tf:fixed`**: This image is used for the **Named Entity Recognition (NER)** step. It appears to be a private or custom-built image and is not publicly documented.
*   **`mrpzzios/bertre:1.3`**: This image is used for the **Relation Extraction (RE)** step. Similar to the NER image, it seems to be a private or custom-built model.

## Example Usage

The `examples/pipeline/` directory contains scripts that demonstrate how to use the pipeline.

*   **[examples/pipeline/test_pipeline.py](examples/pipeline/test_pipeline.py)**: A lightweight script that shows how to initialize all the clients and run the full pipeline on a single text chunk. This is useful for quick verification of the services.

*   **[examples/pipeline/build_kg_with_pipeline.py](examples/pipeline/build_kg_with_pipeline.py)**: A more comprehensive example that demonstrates the end-to-end process of building a complete Knowledge Graph from a collection of documents. It integrates the extraction pipeline with the chunker, embedder, and graph builder components.

### Basic Python Implementation

```python
import asyncio
import os
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
    EntityDescriptionStep,
    RelationDescriptionStep,
)
from ragu.chunker.types import Chunk

load_dotenv()

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
        EntityDescriptionStep(description_client),
        REStep(re_client),
        RelationDescriptionStep(description_client),
    ]

    # Create the pipeline
    pipeline = Pipeline(steps)

    # Run the pipeline on a sample chunk
    chunk = Chunk(
        content="Главным борцом с пробками назначен заместитель министра транспорта России Николай Лямов.",
        chunk_order_idx=0,
        doc_id="test_doc"
    )
    entities, relations = await pipeline.extract([chunk])

    print("--- Entities ---")
    print(entities)
    print("\n--- Relations ---")
    print(relations)


if __name__ == "__main__":
    asyncio.run(main())
```