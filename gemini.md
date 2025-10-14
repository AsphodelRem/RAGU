## Project: RAGU (Retrieval-Augmented Graph Utility)

**Purpose:** RAGU is a Python library that builds a knowledge graph from text documents. It uses LLMs to extract entities and relationships (triplets), chunks the text, and creates embeddings for searching. The goal is to enable question-answering over the structured knowledge graph.

**Key Features:**

*   **Knowledge Graph Construction:** Builds a graph from text, identifying entities and their relationships.
*   **LLM-based Triplet Extraction:** Uses a language model to extract entities, relationships, and their descriptions.
*   **Pipeline-based Triplet Extraction:** A new OOP-based pipeline for triplet extraction with four steps: NER, NEN, RE, and Description Generation. Each step is a component that can be implemented by a model running in a Docker container or as Python code.
*   **Chunking Strategies:** Offers simple, semantic, and smart chunking of text.
*   **Embedding and Indexing:** Creates embeddings for entities and relationships and indexes them for efficient search. It uses `nano-vectordb` for vector storage.
*   **Community Detection:** Groups related entities into communities and generates summaries for them.
*   **Search Engines:** Provides both local and global search capabilities over the knowledge graph.
    *   **Local Search:** For specific, extractive questions.
    *   **Global Search:** For broader, abstractive questions that require information from multiple parts of the graph.
*   **Storage:** Uses NetworkX for the graph structure and JSON for key-value storage.

**Development Stage:**

The project seems to be in a relatively mature stage of development. It has a clear structure, a `pyproject.toml` for packaging, documentation, and examples. The code is modular and organized into different components like `chunker`, `embedder`, `graph`, `llm`, `search_engine`, `storage`, and `triplet`. There are also TODOs in the documentation, which indicate that the project is still under active development.

## Project Tree

```
/home/solovev023/RAGU/
├───.gitignore: Specifies files and directories to be ignored by Git.
├───LICENSE: The MIT license file for the project.
├───pyproject.toml: Project metadata and build dependencies.
├───README.md: The main README file for the project.
├───uv.lock: A lock file for the uv package manager.
├───.geminiignore: Specifies files and directories to be ignored by the Gemini CLI.
├───gemini.md: This file, containing a summary of the project.
├───docker-compose.yml: Docker compose file to run the containerized services for the pipeline.
├───.env: Environment variables for the pipeline services.
├───docs/: Directory for project documentation.
│   ├───get_started.md: Getting started guide.
│   ├───workflow_ru.md: A detailed description of the project workflow in Russian.
│   ├───pipeline_guide.md: A guide to the new pipeline-based triplet extraction.
│   └───images/: Directory for images used in the documentation.
│       └───ragu.png: An image for the RAGU project.
├───examples/: Directory for project examples.
│   ├───example_of_visualization.png: An example of a knowledge graph visualization.
│   ├───The House of Romanov.png: Another example of a knowledge graph visualization.
│   └───data/: Directory for example data.
│       └───ru/: Directory for Russian language data.
│           ├───1.txt: Text file for the examples.
│           ├───2.txt: Text file for the examples.
│           ├───3.txt: Text file for the examples.
│           ├───4.txt: Text file for the examples.
│           ├───5.txt: Text file for the examples.
│           └───6.txt: Text file for the examples.
└───ragu/: The main source code directory for the RAGU library.
    ├───__init__.py: Makes the ragu directory a Python package.
    ├───chunker/: Directory for text chunking modules.
    │   ├───__init__.py: Makes the chunker directory a Python package.
    │   ├───base_chunker.py: Abstract base class for chunkers.
    │   ├───chunkers.py: Implementations of different chunking strategies.
    │   └───types.py: Data types for chunking.
    ├───common/: Directory for common utilities and base classes.
    │   ├───__init__.py: Makes the common directory a Python package.
    │   ├───base.py: Base class for generative modules.
    │   ├───batch_generator.py: A utility for generating batches of data.
    │   ├───decorator.py: Decorators for the project.
    │   ├───env.py: Environment variable management.
    │   ├───global_parameters.py: Global settings for the project.
    │   ├───logger.py: Logging configuration.
    │   └───prompts/: Directory for LLM prompts.
    │       ├───__init__.py: Makes the prompts directory a Python package.
    │       ├───default_models.py: Pydantic models for structured LLM output.
    │       ├───default_templates.py: Default prompt templates.
    │       └───prompt_storage.py: Prompt template management.
    ├───embedder/: Directory for text embedding modules.
    │   ├───__init__.py: Makes the embedder directory a Python package.
    │   ├───base_embedder.py: Abstract base class for embedders.
    │   ├───local_embedders.py: Implementation of local embedders.
    │   └───openai_embedder.py: Implementation of OpenAI embedders.
    ├───graph/: Directory for knowledge graph modules.
    │   ├───__init__.py: Makes the graph directory a Python package.
    │   ├───artifacts_summarizer.py: Summarizes and merges entities and relations.
    │   ├───community_summarizer.py: Generates summaries for graph communities.
    │   ├───graph_builder_pipeline.py: The main pipeline for building the knowledge graph.
    │   ├───knowledge_graph.py: The main KnowledgeGraph class.
    │   └───types.py: Data types for the graph.
    ├───llm/: Directory for language model clients.
    │   ├───__init__.py: Makes the llm directory a Python package.
    │   ├───base_llm.py: Abstract base class for LLM clients.
    │   ├───openai_client.py: OpenAI client implementation.
    │   └───pipeline/: Directory for the new OOP-based triplet extraction pipeline.
    │       ├───__init__.py: Makes the pipeline directory a Python package.
    │       ├───base.py: Base classes for the pipeline and its steps.
    │       ├───clients.py: Clients to communicate with the external services.
    │       ├───models.py: Pydantic models for the data that flows through the pipeline.
    │       ├───pipeline.py: The `Pipeline` class that orchestrates the steps.
    │       └───steps.py: Concrete implementations of the pipeline steps.
    ├───search_engine/: Directory for search engine modules.
    │   ├───__init__.py: Makes the search_engine directory a Python package.
    │   ├───base_engine.py: Abstract base class for search engines.
    │   ├───bfs_search.py: Breadth-first search implementation.
    │   ├───global_search.py: Global search engine implementation.
    │   ├───local_search.py: Local search engine implementation.
    │   ├───search_functional.py: Functional utilities for search.
    │   └───types.py: Data types for search.
    ├───storage/: Directory for storage modules.
    │   ├───__init__.py: Makes the storage directory a Python package.
    │   ├───base_storage.py: Abstract base classes for storage.
    │   ├───index.py: The main Index class for managing storage.
    │   ├───graph_storage_adapters/: Directory for graph storage adapters.
    │   │   └───networkx_adapter.py: NetworkX-based graph storage.
    │   ├───kv_storage_adapters/: Directory for key-value storage adapters.
    │   │   └───json_storage.py: JSON-based key-value storage.
    │   └───vdb_storage_adapters/: Directory for vector database storage adapters.
    │       └───nano_vdb.py: nano-vectordb-based vector storage.
    ├───triplet/: Directory for triplet extraction modules.
    │   ├───__init__.py: Makes the triplet directory a Python package.
    │   ├───base_artifact_extractor.py: Abstract base class for artifact extractors.
    │   ├───llm_artifact_extractor.py: LLM-based artifact extractor.
    │   └───types.py: Data types for triplets.
    └───utils/: Directory for utility functions.
        ├───__init__.py: Makes the utils directory a Python package.
        ├───ragu_utils.py: Miscellaneous utility functions.
        └───token_truncation.py: A utility for truncating text based on token count.
```
