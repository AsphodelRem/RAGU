# 📚 RAGU:  Retrieval-Augmented Graph Utility

## 🚀 Overview
This project provides a pipeline for building a **Knowledge Graph**, indexing it, and performing **local search** over the indexed data. It leverages **LLM-based triplet extraction**, **semantic chunking**, and **embedding-based indexing** to enable efficient question-answering over structured knowledge in Russian language.

Partially based on [nano-graphrag](https://github.com/gusye1234/nano-graphrag/tree/main)

---

## 📌 Features
- **🔗 Build** a knowledge graph from raw text using semantic chunking and different triplet extractors.
- **🔍 Search** the graph using a local search engine to answer queries.

---

## 🛠 Installation using [Poetry](https://python-poetry.org/)
Ensure you have all dependencies installed before running the script.

```bash
poetry install
```

---

## 📖 Usage Guide

<details>
  <summary>🔗 Building the Knowledge Graph</summary>
  
  1. Load the raw text data.
  2. Use **SmartSemanticChunker** to split the text into meaningful segments.
  3. Extract triplets (entities, relations and its descriptions) using **TripletLLM**.
  4. Construct the **Knowledge Graph** using **KnowledgeGraphBuilder**.
  5. Save the graph and its community summary.
  
  ```python
from ragu.utils.io_utils import read_text_from_files
from ragu.common.llm import RemoteLLM
from ragu.graph.graph_builder import KnowledgeGraphBuilder, KnowledgeGraph

from ragu.search_engine.local_search import LocalSearchEngine
from ragu.common.embedder import STEmbedder
from ragu.chunker.chunkers import SmartSemanticChunker
from ragu.triplet.triplet_makers import TripletLLM
from ragu.common.index import Index

# You can load your creditals from .env. Look into ragu/common/setting.py
LLM_MODEL_NAME = "..."
LLM_BASE_URL = "..."
LLM_API_KEY = "..."
client = RemoteLLM(LLM_MODEL_NAME, LLM_BASE_URL, LLM_API_KEY)

# Getting documnets from folders with .txt files
text = read_text_from_files('/path/to/data/folder')

# Initialize a chunker
chunker = SmartSemanticChunker(
      reranker_name="/path/to/reranker_model",
      max_chunk_length=512
)

# Initialize a triplet extractor 
artifact_extractor = TripletLLM(
      validate=False,
      # Also you can set your own entity types as a list (and others arguments)
      # entity_list_type=[your entity types],
      # batch_size=16
)

# Initialize a graph builder pipeline
graph_builder = KnowledgeGraphBuilder(
      client,
      triplet_extractor=artifact_extractor,
      chunker=chunker
)

# Run building 
knowledge_graph = graph_builder.build(text)

# Save results
knowledge_graph.save_graph("graph.gml").save_community_summary("summary.json")
  ```
  
</details>

<details>
  <summary>📌 Indexing the Graph</summary>
  You can index graph data to use it in search engines.

  1. Load the saved knowledge graph.
  2. Use **STEmbedder** (or your custom embedder) to create embeddings for the nodes.
  3. Generate an index with the **Index** class.
  
  ```python
  embedder = STEmbedder("/path/to/model", trust_remote_code=True)
  index = Index(embedder=embedder)
  index.make_index(knowledge_graph)
  ```
  
</details>

<details>
  <summary>🔍 Querying the Graph</summary>
  
  1. Initialize the **LocalSearchEngine** with the knowledge graph and index.
  2. Query the graph to retrieve relevant information.
  
  ```python
  local_search = LocalSearchEngine(
      client,
      knowledge_graph,
      embedder, 
      index
)
  
print(local_search.query("Как звали детей последнего императора Российской Империи?"))
  ```
  
</details>

---

## 📝 Notes
- Adjust chunking and triplet extraction parameters for better graph quality.
- Use a high-quality embedding model for better indexing and retrieval performance.

Happy graph-building! 🚀

