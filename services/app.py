import os
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from vllm.inputs import TextPrompt

import logging
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)

os.environ["VLLM_NO_PROGRESS_BAR"] = "1"

app = FastAPI()

# Load the model and tokenizer
MODEL_DIR = "ragu-lm"

# Check if GPU is available
#if torch.cuda.is_available():
#    llm_args = {
#        "model": MODEL_DIR,
#        "dtype": "bfloat16",
#        "gpu_memory_utilization": float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", 0.8)),
#        "max_model_len": int(os.environ.get("VLLM_MAX_MODEL_LEN", 4096)),
#        "max_num_seqs": int(os.environ.get("VLLM_MAX_NUM_SEQS", 4)),
#    }
#else:
#    llm_args = {
#        "model": MODEL_DIR,
#        "dtype": "auto",
#        "enforce_eager": True,
#    }

try:
#    llm = LLM(**llm_args)
    llm = LLM(model=MODEL_DIR, gpu_memory_utilization=0.9, max_model_len=4096, max_num_seqs=4)
    sampling_params = SamplingParams(
        temperature=float(os.environ.get("VLLM_TEMPERATURE", 0.0)),
        top_p=float(os.environ.get("VLLM_TOP_P", 0.95)),
        top_k=int(os.environ.get("VLLM_TOP_K", 100)),
        max_tokens=int(os.environ.get("VLLM_MAX_TOKENS", 512)),
    )
except Exception as e:
    raise RuntimeError(f"Error loading VLLM model: {e}")

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
        full_prompt = "Вы - эксперт в области анализа текстов и извлечения семантической информации из них.\n\n" + prompt
        outputs = llm.generate(TextPrompt(full_prompt))
        normalized_name = outputs[0].outputs[0].text.strip()
        
    normalized_entities.append({
            "name": entity["name"],
            "type": entity["type"],
            "start": entity["start"],
            "end": entity["end"],
            "normalized_name": normalized_name,
        })
    return {"normalized_entities": normalized_entities}
#    except Exception as e:
#        print(e)
#        raise HTTPException(status_code=500, detail=str(e))

@app.post("/describe")
async def generate_descriptions(request_data: RelationRequest):
    try:
        triplets = []
        for relation in request_data.relations:
            prompt = f"Напишите, что означает именованная сущность в тексте, то есть раскройте её смысл относительно текста.\n\nИменованная сущность: {relation['source_entity']}\n\nТекст: {request_data.source_text}\n\nСмысл именованной сущности: "
            
            full_prompt = "Вы - эксперт в области анализа текстов и извлечения семантической информации из них.\n\n" + prompt
            outputs = llm.generate(TextPrompt(full_prompt))
            description = outputs[0].outputs[0].text.strip()
            
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
    uvicorn.run(app, host="0.0.0.0", port=8002)
