# RAGU-lm Service

This directory contains a FastAPI application that serves the RAGU-lm model for various NLP tasks.

## Endpoints

The service provides the following endpoints:

### `/extract_entities`

- **Method:** `POST`
- **Request Body:** `{"text": "..."}`
- **Response Body:** `{"extracted_entities": ["...", "..."]}`
- **Description:** Extracts named entities from the input text.

### `/nen`

- **Method:** `POST`
- **Request Body:** `{"entities": [{"name": "...", "type": "...", "start": ..., "end": ...}], "source_text": "..."}`
- **Response Body:** `{"normalized_entities": [{"name": "...", "type": "...", "start": ..., "end": ..., "normalized_name": "..."}]}`
- **Description:** Normalizes the extracted entities.

### `/describe_relations`

- **Method:** `POST`
- **Request Body:** `{"relations": [{"source_entity": "...", "target_entity": "...", "relationship_type": "..."}], "source_text": "..."}`
- **Response Body:** `{"triplets": [{"source_entity": "...", "target_entity": "...", "relationship_type": "...", "description": "..."}]}`
- **Description:** Generates descriptions for the extracted relations.

## Running the Service

To run the service, you can use the provided `Dockerfile` and `docker-compose.yml` in the root directory.

```bash
docker-compose up -d custom_service
```
