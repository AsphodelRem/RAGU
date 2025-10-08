# RAGU:  Retrieval-Augmented Graph Utility

<h4 align="center">
  <a href="https://github.com/AsphodelRem/RAGU/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="RAGU is under the MIT license." alt="RAGU"/>
  </a>
  <img src="https://img.shields.io/badge/python->=3.10-blue">
</h4>

<h4 align="center">
  <a href="#install">Install</a> |
  <a href="#quickstart">Quickstart</a> 
</h4>


## Overview
This project provides a pipeline for building a **Knowledge Graph**, indexing it, and performing search over the indexed data. It leverages **LLM-based triplet extraction**, **semantic chunking**, and **embedding-based indexing** to enable efficient question-answering over structured knowledge.

Partially based on [nano-graphrag](https://github.com/gusye1234/nano-graphrag/tree/main)

---

## Install

```bash
pip install ragu
```

If you want to use local models (via transformers etc), run:
```bash
pip install ragu[local]
```

---

## Quickstart

### Simple example of building knowledge graph
```python
import asyncio

from ragu.chunker import SimpleChunker
from ragu.embedder import STEmbedder
from ragu.graph import KnowledgeGraph, InMemoryGraphBuilder

from ragu.llm import OpenAIClient

from ragu.storage import Index
from ragu.triplet import ArtifactsExtractorLLM
from ragu.utils.ragu_utils import read_text_from_files

LLM_MODEL_NAME = "..."
LLM_BASE_URL = "..."
LLM_API_KEY = "..."

async def main():
    # Load .txt documents from folder
    docs = read_text_from_files("/path/to/files")
    
    # Choose chunker 
    chunker = SimpleChunker(max_chunk_size=2048, overlap=0)

    # Import LLM client
    client = OpenAIClient(
        LLM_MODEL_NAME,
        LLM_BASE_URL,
        LLM_API_KEY,
        max_requests_per_second=1,
        max_requests_per_minute=60
    )

    # Set up artifacts extractor
    artifact_extractor = ArtifactsExtractorLLM(
        client=client, 
        do_validation=True
    )

    # Initialize your embedder
    embedder = STEmbedder(
        "Alibaba-NLP/gte-multilingual-base",
        trust_remote_code=True
    )
    # Set up graph storage and graph builder pipeline
    pipeline = InMemoryGraphBuilder(client, chunker, artifact_extractor)
    index = Index(
        embedder,
        graph_storage_kwargs={"clustering_params": {"max_cluster_size": 6}}
    )
    
    # Build KG
    knowledge_graph = await KnowledgeGraph(
        extraction_pipeline=pipeline,           # Pass pipeline
        index=index,                            # Pass storage
        make_community_summary=True,            # Generate community summary if you want
        language="russian",                     # You can set preferred language
    ).build_from_docs(docs)

if __name__ == "__main__":
    asyncio.run(main())
```

### Example of querying
```python
from ragu.search_engine import LocalSearchEngine

search_engine = LocalSearchEngine(
    client,
    knowledge_graph,
    embedder
)

# Find relevant local context for the query
print(await search_engine.a_search("Как переводится роман 'Ка́мо гряде́ши, Го́споди?'"))

# Or just past the query ang get final answer
print(await search_engine.a_query("Как переводится роман 'Ка́мо гряде́ши, Го́споди?'"))

# Output:
# [DefaultResponseModel(response="Роман 'Ка́мо гряде́ши, Го́споди?' переводится как 'Куда Ты идёшь, Господи?'")]
# :)
```

