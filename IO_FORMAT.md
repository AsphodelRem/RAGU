# NER and RE I/O Formats

This document describes the standard input and output formats for the Named Entity Recognition (NER) and Relation Extraction (RE) models used in the RAGU project.

## NER (Named Entity Recognition)

### NER Input (`NER_IN`)

The input for the NER model is a single JSON string containing the text to be processed.

**Example:**
```json
"Главным борцом с пробками назначен заместитель министра транспорта России Николай Лямов."
```

### NER Output (`NER_OUT`)

The output of the NER model is a JSON object containing the original text and the extracted entities.

- `text`: The original input string.
- `ners`: A list of extracted entities. Each entity is represented as a list with three elements:
    1.  `start_char_index` (integer): The starting character offset of the entity in the text.
    2.  `end_char_index` (integer): The ending character offset of the entity in the text.
    3.  `entity_type` (string): The type of the entity (e.g., "COUNTRY", "PERSON", "PROFESSION").

**Example:**
```json
{
  "ners": [
    [67, 73, "COUNTRY"],
    [74, 87, "PERSON"],
    [35, 73, "PROFESSION"]
  ],
  "text": "Главным борцом с пробками назначен заместитель министра транспорта России Николай Лямов."
}
```

## RE (Relation Extraction)

### RE Input (`RE_IN`)

The input for the RE model is a JSON object containing text chunks and their corresponding entities.

- `chunks`: A list of text strings (e.g., sentences or paragraphs).
- `entities_list`: A list where each element is a list of entities found in the corresponding chunk in the `chunks` list. The format for each entity is the same as in the `NER_OUT`.

**Example:**
```json
{
  "chunks": [
    "Главным борцом с пробками назначен заместитель министра транспорта России Николай Лямов.",
    "Президент Башкирии Муртаза Рахимов решил поменять главу своей администрации. Он уволил Азамата Сагитова."
  ],
  "entities_list": [
    [
      [67, 73, "COUNTRY"],
      [74, 87, "PERSON"],
      [35, 73, "PROFESSION"]
    ],
    [
      [19, 34, "PERSON"],
      [0, 18, "PROFESSION"],
      [50, 75, "PROFESSION"],
      [10, 18, "STATE_OR_PROVINCE"],
      [80, 86, "EVENT"],
      [87, 103, "PERSON"]
    ]
  ]
}
```

### RE Output (`RE_OUT`)

The output of the RE model is a JSON list of extracted relationships. Each object in the list represents a single relationship and contains the following fields:

- `source_entity` (string): The text of the source entity in the relationship.
- `target_entity` (string): The text of the target entity in the relationship.
- `relationship_type` (string): The type of the relationship (e.g., "FOUNDED_BY", "WORKPLACE").
- `relationship_description` (string or null): A natural language description of the relationship.
- `relationship_strength` (float): A confidence score for the extracted relationship, typically between 0.0 and 1.0.
- `chunk_id` (integer): The index of the chunk from the `RE_IN` `chunks` list where this relationship was found.

**Example:**
```json
[
  {
    "source_entity": "России",
    "target_entity": "Николай Лямов",
    "relationship_type": "FOUNDED_BY",
    "relationship_description": null,
    "relationship_strength": 0.04831777885556221,
    "chunk_id": 0
  },
  {
    "source_entity": "Николай Лямов",
    "target_entity": "России",
    "relationship_type": "WORKPLACE",
    "relationship_description": null,
    "relationship_strength": 0.999497652053833,
    "chunk_id": 0
  }
]
```

## RAGU-lm I/O Formats

The `RAGU-lm` model uses a prompt-based format for its tasks.

### Named Entity Normalization (NEN)

**Input:**
The input is a formatted string (prompt) that includes the unnormalized entity and the source text.

- **Prompt Template:**
  '''
  Выполните нормализацию именованной сущности, встретившейся в тексте.

  Исходная (ненормализованная) именованная сущность: {source_entity}

  Текст: {source_text}

  Нормализованная именованная сущность:
  '''
- **Parameters:**
  - `{source_entity}`: The unnormalized entity to be normalized.
  - `{source_text}`: The original text containing the entity.

**Output:**
The output is a string containing the normalized entity.

- **Example Output:**
  '''
  пресс-секретарь
  '''

### Description Generation (DG)

**Input:**
The input is a formatted string (prompt) that includes the normalized entity and the source text.

- **Prompt Template:**
  '''
  Напишите, что означает именованная сущность в тексте, то есть раскройте её смысл относительно текста.

  Именованная сущность: {normalized_entity}

  Текст: {source_text}

  Смысл именованной сущности:
  '''
- **Parameters:**
  - `{normalized_entity}`: The normalized entity for which to generate a description.
  - `{source_text}`: The original text containing the entity.

**Output:**
The output is a string containing the generated description for the entity.

- **Example Output:**
  '''
  Бывший представитель СМИ экс-президента США Билла Клинтона.
  '''