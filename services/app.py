import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
import uvicorn

app = FastAPI()

# Load the model and tokenizer
MODEL_DIR = "ragu-lm"
# Note: vLLM is primarily GPU-accelerated. Running on CPU (enforce_eager=True) might be very slow.
# Ensure your environment has the necessary vLLM CPU dependencies if not using CUDA.
llm = LLM(model=MODEL_DIR, dtype="bfloat16", enforce_eager=True)
sampling_params = SamplingParams(temperature=0.0, top_p=0.95, top_k=100, max_tokens=512)

class EntityRequest(BaseModel):
    entities: List[Dict[str, Any]]
    source_text: str

class RelationRequest(BaseModel):
    relations: List[Dict[str, Any]]
    source_text: str

@app.post("/nen")
async def normalize_entities(request_data: EntityRequest):
    normalized_entities = []
    for entity in request_data.entities:
        prompt = f"Выполните нормализацию именованной сущности, встретившейся в тексте.\n\nИсходная (ненормализованная) именованная сущность: {entity['name']}\n\nТекст: {request_data.source_text}\n\nНормализованная именованная сущность: "
        
        messages = [
            {"role": "system", "content": "Вы - эксперт в области анализа текстов и извлечения семантической информации из них."},
            {"role": "user", "content": prompt}
        ]
        
        outputs = llm.generate(messages, sampling_params)
        normalized_name = outputs[0].outputs[0].text.strip()
        
        normalized_entities.append({
            "name": entity["name"],
            "type": entity["type"],
            "start": entity["start"],
            "end": entity["end"],
            "normalized_name": normalized_name,
        })
    return {"normalized_entities": normalized_entities}

@app.post("/describe")
async def generate_descriptions(request_data: RelationRequest):
    triplets = []
    for relation in request_data.relations:
        prompt = f"Напишите, что означает именованная сущность в тексте, то есть раскройте её смысл относительно текста.\n\nИменованная сущность: {relation['source']}\n\nТекст: {request_data.source_text}\n\nСмысл именованной сущности: "
        
        messages = [
            {"role": "system", "content": "Вы - эксперт в области анализа текстов и извлечения семантической информации из них."},
            {"role": "user", "content": prompt}
        ]

        outputs = llm.generate(messages, sampling_params)
        description = outputs[0].outputs[0].text.strip()
        
        triplets.append({
            "source": relation["source"],
            "target": relation["target"],
            "type": relation["type"],
            "description": description,
        })
    return {"triplets": triplets}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)