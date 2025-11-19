import os
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

import logging
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)

app = FastAPI()

# Load the model and tokenizer
MODEL_DIR = "ragu-lm"

try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        use_fast=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype="auto",        
        device_map="auto",       
        trust_remote_code=True,
    )
    system_prompt = "Вы - эксперт в области анализа текстов и извлечения семантической информации из них."
except Exception as e:
    raise RuntimeError(f"Error loading transformers model: {e}")

class EntityRequest(BaseModel):
    entities: List[Dict[str, Any]]
    source_text: str

class RelationRequest(BaseModel):
    relations: List[Dict[str, Any]]
    source_text: str

class TextRequest(BaseModel):
    text: str

class DescribeRequest(BaseModel):
    entities: List[Dict[str, Any]]
    source_text: str

def generate_text(prompt):
    messages = [[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ]]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.95,
        top_k=100,
        temperature=0.01,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(
        gen_ids[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

@app.post("/extract_entities")
async def extract_entities(request_data: TextRequest):
    prompt = f"Распознайте все именованные сущности в тексте и выпишите их список с новой строки.\n\nТекст: {request_data.text}\n\nИменованные сущности:"
    generated_text = generate_text(prompt)
    extracted_entities = generated_text.split('\n')
    return {"extracted_entities": extracted_entities}

@app.post("/nen")
async def normalize_entities(request_data: EntityRequest):
    normalized_entities = []
    for entity in request_data.entities:
        prompt = f"Выполните нормализацию именованной сущности, встретившейся в тексте.\n\nИсходная (ненормализованная) именованная сущность: {entity['name']}\n\nТекст: {request_data.source_text}\n\nНормализованная именованная сущность:"
        normalized_name = generate_text(prompt)
        
        normalized_entities.append({
            "name": entity["name"],
            "type": entity["type"],
            "start": entity["start"],
            "end": entity["end"],
            "normalized_name": normalized_name,
        })
    return {"normalized_entities": normalized_entities}

@app.post("/describe")
async def generate_entity_descriptions(request_data: DescribeRequest):
    try:
        described_entities = []
        for entity in request_data.entities:
            prompt = f"Напишите, что означает именованная сущность в тексте, то есть раскройте её смысл относительно текста.\n\nИменованная сущность: {entity['normalized_name']}\n\nТекст: {request_data.source_text}\n\nСмысл именованной сущности:"
            description = generate_text(prompt)
            
            entity_with_description = entity.copy()
            entity_with_description["description"] = description
            described_entities.append(entity_with_description)
            
        return {"described_entities": described_entities}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/describe_relation")
async def generate_relation_description(request_data: RelationRequest):
    try:
        triplets = []
        for relation in request_data.relations:
            prompt = f"Напишите, что означает отношение между двумя именованными сущностями в тексте, то есть раскройте смысл этого отношения относительно текста (либо напишите прочерк, если между двумя именованными сущностями отсутствует отношение).\n\nПервая именованная сущность: {relation['source_entity']}\n\nВторая именованная сущность: {relation['target_entity']}\n\nТекст: {request_data.source_text}\n\nСмысл отношения между двумя именованными сущностями:"
            description = generate_text(prompt)
            
            triplets.append({
                "source_entity": relation["source_entity"],
                "target_entity": relation["target_entity"],
                "relationship_type": relation["relationship_type"],
                "description": description,
            })
        return {"triplets": triplets}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
