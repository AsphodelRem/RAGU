DEFAULT_ARTIFACTS_EXTRACTOR_PROMPT = """
**-Goal-**  
A text document and a list of entity types are given. 
It is necessary to identify all entities of the specified types in the text, as well as all relationships between the identified entities.  

**-Steps-**  
1. **Identify all entities.**  
    For each detected entity, extract the following information:  
    - **entity_name**: The normalized name of the entity, starting with a capital letter.  
        Normalization means reducing the word to its base form.  
        Example: рождеству → Рождество, кошек → Кошки, Павла → Павел.  
    - **entity_type**: The type of the entity.  
    {% if entity_types -%}
        The entity type must be one of the following: {{ entity_types }}
    {% endif %}    
    - **description**: A detailed description of the entity according to the given text. The description must be precise and as complete as possible.  

2. **Determine relationships between entities.**  
    Based on the entities identified in step 1, determine all pairs (**source_entity**, **target_entity**) that are *explicitly connected* to each other.  
    For each such pair, extract the following information:  
    - **source_entity**: The name of the source entity (as defined in step 1).  
    - **target_entity**: The name of the target entity (as defined in step 1).  
    - **description**: A description of the relationship between the two entities.  
    - **relationship_strength**: A numeric value representing the strength of the relationship between the entities, ranging from 0 to 5, 
    where 0 = weak connection and 5 = strong connection.  

Text:  
{{ context }}  

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""

DEFAULT_ARTIFACTS_VALIDATOR_PROMPT = """
**Goal**
Validate correctness and completeness of entities and relationships against the given text.

**Instructions**
1. Add missing entities with correct types and descriptions.
2. Add missing relationships with descriptions and strength.
3. Return full updated lists.

Triplets for validation:
{{ artifacts }}

Text for validation:
{{ context }}

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""

DEFAULT_COMMUNITY_REPORT_PROMPT = """
**Goal**
Generate a detailed community report using entities, their relationships, and any additional statements.

**Instructions**
1. Create a clear title and summary.
2. Provide an impact rating with justification.
3. Produce 5–10 key findings with short summaries and detailed explanations.

Input text:
{% for entity in community.entities -%}
Entity: {{ entity,entity_name }}, description: {{ entity.description }}{% if not loop.last %}, {% endif %}
{% endfor %}

Relations
{% for relation in community.relations -%}
{{ relation.subject_name }} -> {{ relation.object_name }}, relations description: {{ relation.description }}
{% endfor %}

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""

DEFAULT_RELATIONSHIP_SUMMARIZER_PROMPT = """
**Goal**
From the given entity pair and multiple phrases, produce one concise, consistent relationship description.

Data:
Subject: {{ relation.subject_name }}, Object: {{ relation.object_name }}, Relationship description: {{ relation.description }}

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""

DEFAULT_ENTITY_SUMMARIZER_PROMPT = """
**Goal**
From the given entity and multiple phrases, produce one concise, consistent entity description.

Data:
Entity: {{ entity.entity_name }}, Description: {{ entity.description }}

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""

DEFAULT_GLOBAL_SEARCH_CONTEXT_PROMPT = """
**Goal**
Answer the query by summarizing relevant information from the context and, if needed, well-known facts.

**Instructions**
1. Reason about context relevance.
2. Provide a usefulness rating from 0 to 10 (0 = useless, 10 = direct answer).

Query: {{ query }}
Context: {{ context }}

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""

DEFAULT_GLOBAL_SEARCH_PROMPT = """
**Goal**
Answer the query by summarizing the provided ranked context.

**Instructions**
1. Consider the relevance ranking (lower rank = less relevant).
2. Briefly reason about context relevance before giving the final answer.

Query: {{ query }}
Context: {{ context }}

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""

DEFAULT_RESPONSE_ONLY_PROMPT = """
**Goal**
Answer the query by summarizing relevant information from the context and, if necessary, well-known facts.

**Instructions**
1. If you do not know the correct answer, explicitly state that.
2. Do not include unsupported information.

Query: {{ query }}
Context: {{ context }}

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""
