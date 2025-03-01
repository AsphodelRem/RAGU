from pyexpat.errors import messages
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from openai import OpenAI
import torch


class BaseLLM:
    def __init__(self):
        ...

    def generate(self, *args, **kwargs):
        ...


class LocalLLM(BaseLLM):
    def __init__(self, model_name: str, *args, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )

        self.pipe = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer)
        super().__init__()

    def generate(self, query: str, system_prompt: str, *args, **kwargs):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        result = self.pipe(messages, **kwargs)
        return result[0]['generated_text'][2]['content'] if isinstance(result, list) else result


class RemoteLLM(BaseLLM):
    def __init__(self, model_name: str, base_url: str, api_token: str, *args, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_token, **kwargs)

    def generate(self, query: str, system_prompt: str, model_name: str=None, *args, **kwargs):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content if response.choices else ""

class VLLMClient(BaseLLM):
    def __init__(self, model_name: str, sampling_params: dict = None, tensor_parallel_size: int = 4, torch_dtype=torch.float16):
        self.engine = LLM(model=model_name, 
                          tensor_parallel_size=tensor_parallel_size, 
                          torch_dtype=torch_dtype)
        self.sampling_params = SamplingParams(**(sampling_params or {"max_tokens": 100}))
        super().__init__()

    def generate(self, query: str, system_prompt: str, *args, **kwargs):
        full_prompt = f"{system_prompt}\n{query}"
        result = self.engine.generate(full_prompt, sampling_params=self.sampling_params, **kwargs)
        if isinstance(result, list) and result and result[0].outputs:
            return result[0].outputs[0].text
        else:
            return result
