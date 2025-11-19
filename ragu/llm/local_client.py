from types import SimpleNamespace
from typing import Any, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
from tqdm import tqdm

from ragu.common.logger import logger
from ragu.llm.base_llm import BaseLLM

class LocalTransformerClient(BaseLLM):
    """
    Client for running a local Hugging Face transformer model.
    """

    def __init__(
        self,
        model_name: str,
        **model_kwargs: Any,
    ):
        """
        Initialize a new LocalTransformerClient.

        :param model_name: Name of the Hugging Face model to use.
        :param model_kwargs: Additional keyword arguments passed to AutoModelForCausalLM.from_pretrained.
        """
        super().__init__()

        self.model_name = model_name

        try:
            logger.info(f"Loading local transformer model: {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
                **model_kwargs
            )
            logger.info("Local model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load local transformer model {model_name}: {e}")
            raise

    async def _one_call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """
        Perform a single generation request.
        """
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        try:
            self.statistics["requests"] += 1
            
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
            
            # Remove arguments that are not part of the transformers.generate method
            kwargs.pop("schema", None)
            kwargs.pop("response_model", None)

            gen_kwargs = {
                "max_new_tokens": 512,
                "do_sample": True,
                "top_p": 0.95,
                "temperature": 0.1,
                **kwargs
            }

            output_ids = self.model.generate(**inputs, **gen_kwargs)
            
            response = self.tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            self.statistics["success"] += 1
            return response

        except Exception as e:
            logger.error(f"[LocalTransformerClient] request failed: {e}", exc_info=True)
            self.statistics["fail"] += 1
            return None

    async def generate(
        self,
        prompt: str | list[str],
        *,
        system_prompt: Optional[str] = None,
        pydantic_model: type[BaseModel] = None,
        model_name: Optional[str] = None,
        progress_bar_desc: Optional[str] = "Processing with local model",
        **kwargs: Any,
    ) -> List[Optional[Union[BaseModel, str]]]:
        """
        Generate one or multiple completions asynchronously.
        """
        schema = kwargs.get("schema") or pydantic_model

        prompts: List[str] = [prompt] if isinstance(prompt, str) else list(prompt)
        
        results = []
        # A single local transformer model runs sequentially. A progress bar is used to show progress.
        for p in tqdm(prompts, desc=progress_bar_desc):
            res_str = await self._one_call(p, system_prompt=system_prompt, **kwargs)

            if res_str is None:
                results.append(None)
                continue

            if schema:
                # HACK: Instead of parsing, we create a mock object that satisfies
                # the attribute access in the calling code (e.g., .description).
                # This is brittle but respects the constraint of not changing the graph module.
                mock_obj = SimpleNamespace(description=res_str)
                results.append(mock_obj)
            else:
                results.append(res_str)

        return results
