# RAGU: Retrieval-Augmented Graph Utility

## 1. Project Overview

RAGU (Retrieval-Augmented Graph Utility) is a Python-based project designed to build, index, and perform search operations over Knowledge Graphs. It implements a pipeline that leverages multiple microservices and Large Language Models (LLMs) for triplet extraction, document chunking, and embedding-based indexing to facilitate efficient question-answering.

**Key Features:**
*   **Microservice-based Architecture:** Core NLP tasks like Named Entity Recognition (NER), Relation Extraction (RE), and language modeling are handled by separate Dockerized services.
*   **LLM-driven Pipeline:** A sequence of steps (NER, NEN, RE, Description) to extract structured information (entities, relations) from text.
*   **Knowledge Graph Construction:** Builds an in-memory knowledge graph using `networkx` from the extracted artifacts.
*   **Vector-based Search:** Utilizes `nano-vectordb` and `sentence-transformers` for efficient semantic search over the graph.
*   **Local & Global Search:** Provides mechanisms to query the constructed knowledge graph.

## 2. Core Technologies

*   **Language:** Python 3.10+
*   **LLM Serving:** vLLM (via a custom FastAPI service)
*   **Web Framework:** FastAPI
*   **Containerization:** Docker, Docker Compose
*   **Graph Database:** `networkx`
*   **Vector Database:** `nano-vectordb`
*   **LLM Client:** `openai` (configured as a generic client for local vLLM)
*   **Embeddings:** `sentence-transformers`
*   **Dependency Management:** `uv` and `setuptools`

## 3. Architecture

### 3.1. High-Level Architecture

The system is composed of several microservices managed by `docker-compose.yml`:
*   `ner_service`: A dedicated service for Named Entity Recognition.
*   `re_service`: A dedicated service for Relation Extraction.
*   `custom_service`: A custom FastAPI service that serves the `RaguTeam/RAGU-lm` model for language-model-based tasks (Normalization, Description).

### 3.2. Data Processing Pipeline

The core logic resides in a multi-step pipeline that processes text to extract a knowledge graph:
1.  **NER Step:** Extracts named entities from text chunks using `ner_service`.
2.  **NEN Step:** Normalizes the extracted entities using the `custom_service`.
3.  **RE Step:** Extracts relationships between entities using `re_service`.
4.  **Description Step:** Generates descriptions for entities and relations using the `custom_service`.

## 4. Current Status & Recent Changes

This section summarizes the state of the project after the initial analysis and recent automated improvements.

### 4.1. Initial State Analysis

*   **Inconsistencies:** The project had mismatches between documentation (`IO_FORMAT.md`, `RAGU-LM-README.md`), client-side code (`ragu/triplet/pipeline/clients.py`), and the `ragu-lm` service implementation (`services/app.py`).
*   **Configuration Issues:** Critical parameters like ports and prompts were hardcoded, and environment configuration was missing.
*   **Missing Features:** The entity description endpoint, though documented, was not implemented.
*   **Lack of Testing:** The project lacked a formal testing framework, posing risks for future development.

### 4.2. Recent Changes (Automated)

The following improvements have been implemented to address the issues above:

1.  **Service Port Correction:** The port in `services/app.py` was changed from `8002` to `8000` to align with the `docker-compose.yml` configuration.
2.  **`ragu-lm` Service Enhancement:**
    *   Added the missing `/describe` endpoint for generating entity descriptions.
    *   Corrected all prompt templates in `services/app.py` to match the official documentation, removing extra spaces and ensuring consistency.
    *   Renamed the relation description endpoint from `/describe_relations` to `/describe_relation` for clarity.
3.  **Client-Side Refactoring:**
    *   Updated `ragu/triplet/pipeline/clients.py` to correctly interface with the modified service endpoints.
    *   Removed obsolete prompt constants from the client code.
4.  **Environment Setup:**
    *   Created a `.env` file with the correct service URLs for local execution.
    *   Created an `.env_example` file to serve as a template for future developers.
5.  **Model Download Script:**
    *   Added a `services/download_model.py` script to automate the download of the `RaguTeam/RAGU-lm` model from Hugging Face into the required local directory.

## 5. Setup and Running the Project

Follow these steps to set up and run the RAGU pipeline:

### Step 1: Download the LLM Model

The `custom_service` requires the `RaguTeam/RAGU-lm` model files to be present locally. A script has been created to automate this.

Run the following commands from the project root:
```bash
cd services
python3 download_model.py
cd ..
```
This will download the model into the `services/ragu-lm/` directory.

### Step 2: Configure Environment

The project uses a `.env` file to manage service URLs. This file has already been created for you with the correct default values for a local setup.

### Step 3: Run Services

With Docker and Docker Compose installed, build and run all services in detached mode:
```bash
docker-compose up -d --build
```

### Step 4: Run the Main Pipeline

Execute the main script to run a test of the entire pipeline:
```bash
python3 main.py
```

## 6. Development Roadmap

Based on the `IMPROVEMENT_PLAN.md` and `ADVICES.md` documents, the following areas are priorities for future development. CI/CD setup is currently in the backlog.

### 6.1. Testing (Critical Priority)

*   **Task:** Implement a formal testing suite.
*   **Action:**
    1.  Create a `tests/` directory.
    2.  Implement **Contract Tests** for each service client (NER, NEN, RE, Description) to ensure they adhere to the specified data formats.
    3.  Implement **Integration Tests** to verify the end-to-end pipeline functionality.
    4.  Utilize `pytest` as the test runner.

### 6.2. Configuration & Extensibility (High Priority)

*   **Task:** Decouple configuration from code.
*   **Action:**
    1.  Move all hardcoded values (URLs, model names, prompts) from Python scripts into a centralized YAML configuration file.
    2.  Refactor the pipeline to be dynamically constructed based on this configuration file.
    3.  Implement a **Client Factory** to instantiate service clients dynamically based on the config.

### 6.3. Code Refactoring (High Priority)

*   **Task:** Improve code quality and maintainability.
*   **Action:**
    1.  Rename `OpenAIClient` to a more generic `LLMClient` to better reflect its role as a client for local vLLM.
    2.  Refactor the `Pipeline.extract` method in `ragu/triplet/pipeline/pipeline.py` to be more modular and less monolithic.

### 6.4. Monitoring (Medium Priority)

*   **Task:** Add observability to the pipeline.
*   **Action:**
    1.  Integrate `loguru` more deeply to provide structured, detailed logs for each pipeline step (including timings and data shapes).
    2.  Introduce `prometheus-client` to collect and expose key performance metrics.