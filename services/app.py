import os
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# Load the model and tokenizer
model_name = "Qwen/Qwen3-0.6B"
token = os.getenv("HUGGING_FACE_HUB_TOKEN")
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=token)

@app.route("/nen", methods=["POST"])
def normalize_entities():
    data = request.get_json()
    entities = data.get("entities", [])
    normalized_entities = []
    for entity in entities:
        prompt = f"You are an expert in named entity normalization. Normalize the following entity: '{entity['name']}'."
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=10)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        normalized_entities.append({
            "name": entity["name"],
            "type": entity["type"],
            "start": entity["start"],
            "end": entity["end"],
            "normalized_name": response.strip(),
        })
    return jsonify({"normalized_entities": normalized_entities})

@app.route("/describe", methods=["POST"])
def generate_descriptions():
    data = request.get_json()
    relations = data.get("relations", [])
    triplets = []
    for relation in relations:
        prompt = f"You are an expert in relation description. Describe the following relation: {relation['source']} -> {relation['target']} ({relation['type']})."
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        triplets.append({
            "source": relation["source"],
            "target": relation["target"],
            "type": relation["type"],
            "description": response.strip(),
        })
    return jsonify({"triplets": triplets})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
