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
RAGU provides a pipeline for building a **Knowledge Graph**, and performing retrieve over the indexed data. It contains different approaches to extract structured data from raw texts to enable efficient question-answering over structured knowledge.

Partially based on [nano-graphrag](https://github.com/gusye1234/nano-graphrag/tree/main)

Our huggingface community is [here](https://huggingface.co/RaguTeam/)

---

## Install

```bash
pip install graph_ragu
```

If you want to use local models (via transformers etc), run:
```bash
pip install graph_ragu[local]
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

---

### Knowledge Graph Construction
Each text in corpus is processed to extract structured information. It consist of:

* **Entities** — textual representation, entity type, and a contextual description.
* **Relations** — textual description of the link between two entities (or a relation class), as well as its confidence/strength.

> **RAGU uses entity and relation classes from [NEREL](https://github.com/nerel-ds/NEREL).**

### Entity types

|No. | Entity type | No. | Entity type | No. | Entity type |
|---|---|---|---|---|---|
|1.| AGE |11.| FAMILY |21.| PENALTY |
|2.| AWARD |12.| IDEOLOGY |22.| PERCENT |
|3.| CITY |13.| LANGUAGE |23.| PERSON |
|4.| COUNTRY |14.| LAW |24.| PRODUCT |
|5.| CRIME |15.| LOCATION |25.| PROFESSION |
|6.| DATE |16.| MONEY |26.| RELIGION |
|7.| DISEASE |17.| NATIONALITY |27.| STATE_OR_PROV |
|8.| DISTRICT |18.| NUMBER |28.| TIME |
|9.| EVENT |19.| ORDINAL |29.| WORK_OF_ART |
|10.| FACILITY |20.| ORGANIZATION | | |

### Relation types

|No. | Relation type | No. | Relation type | No. | Relation type |
|---|---|---|---|---|---|
|1.| ABBREVIATION |18.| HEADQUARTERED_IN |35.| PLACE_RESIDES_IN |
|2.| AGE_DIED_AT |19.| IDEOLOGY_OF |36.| POINT_IN_TIME |
|3.| AGE_IS |20.| INANIMATE_INVOLVED |37.| PRICE_OF |
|4.| AGENT |21.| INCOME |38.| PRODUCES |
|5.| ALTERNATIVE_NAME |22.| KNOWS |39.| RELATIVE |
|6.| AWARDED_WITH |23.| LOCATED_IN |40.| RELIGION_OF |
|7.| CAUSE_OF_DEATH |24.| MEDICAL_CONDITION |41.| SCHOOLS_ATTENDED |
|8.| CONVICTED_OF |25.| MEMBER_OF |42.| SIBLING |
|9.| DATE_DEFUNCT_IN |26.| ORGANIZES |43.| SPOUSE |
|10.| DATE_FOUNDED_IN |27.| ORIGINS_FROM |44.| START_TIME |
|11.| DATE_OF_BIRTH |28.| OWNER_OF |45.| SUBEVENT_OF |
|12.| DATE_OF_CREATION |29.| PARENT_OF |46.| SUBORDINATE_OF |
|13.| DATE_OF_DEATH |30.| PART_OF |47.| TAKES_PLACE_IN |
|14.| END_TIME |31.| PARTICIPANT_IN |48.| WORKPLACE |
|15.| EXPENDITURE |32.| PENALIZED_AS |49.| WORKS_AS |
|16.| FOUNDED_BY |33.| PLACE_OF_BIRTH | | |
|17.| HAS_CAUSE |34.| PLACE_OF_DEATH | | |


### How it is extracted:
#### 1. Default Pipeline

File: ragu/triplet/llm_artifact_extractor.py.
A baseline pipeline that uses LLM to extract entities, relations, and their descriptions in a single step.

#### 2. [RAGU-lm](https://huggingface.co/RaguTeam/RAGU-lm) (for russian language)
A compact model (Qwen-3-0.6B) fine-tuned on the NEREL dataset.
The pipeline operates in several stages:
1. Extract unnormalized entities from text.
2. Normalize entities into canonical forms.
3. Generate entity descriptions.
4. Extract relations based on the inner product between entities.

#### 3. RAGU-lm with small models (for russian language)

A modular multi-model pipeline:

1. [runne_contrastive_ner](https://github.com/bond005/runne_contrastive_ner) — extracts entities (NER step).
2. ragu_lm — performs entity normalization.
3. ragu_lm — generates concise definitions and descriptions for entities.
4. ragu_re — extracts relation candidates. 
5. ragu_lm — refines and summarizes relations with their textual descriptions.


### Comparison
| Model                 | Dataset | F1 (Entities) | F1 (Relations) |
|-----------------------|----------|---------------|----------------|
| Qwen-2.5-14B-Instruct | NEREL | 0.32          | 0.69           |
| RAGU-lm (Qwen-3-0.6B) | NEREL | 0.6           | 0.71           |
| Small-model pipeline  | NEREL | 0.74          | 0.75           |

---

### Prompt tuning
How to know what's prompt in used
```python
search_engine = LocalSearchEngine(
    client,
    knowledge_graph,
    embedder
)

print(search_engine.get_prompts())
#
# {'local_search': PromptTemplate(template='\n**Goal**\nAnswer the query by summarizing relevant information from the context and, if necessary, well-known facts.\n\n**Instructions**\n1. If you do not know the correct answer, explicitly state that.\n2. Do not include unsupported information.\n\nQuery: {{ query }}\nContext: {{ context }}\n\nProvide the answer in the following language: {{ language }}\nReturn the result as valid JSON matching the provided schema.\n', schema=<class 'ragu.common.prompts.default_models.DefaultResponseModel'>, description='Prompt for generating a local context-based search response.')}
#

# or, if you know prompt name
print(search_engine.get_prompt("local_search")
```

You can update prompt using .update_prompt method

```python
from ragu.common.prompts import PromptTemplate

search_engine.update_prompt("prompt_name", PromptTemplate(template=..., schema=...))
```

---

### Contributors
#### **Main Idea & Inspiration**
- Ivan Bondarenko - idea, smart_chunker, NER model, ragu-lm


#### **Core Development**

- Mikhail Komarov

#### **Benchmarks & Evaluation**  
- Roman Shuvalov
- Yanya Dement'yeva 
- Alexandr Kuleshevskiy
- Nikita Kukuzey
- Stanislav Shtuka

#### **Small Models Pipeline**
- Matvey Solovyev
- Ilya Myznikov




