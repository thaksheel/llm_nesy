import numpy as np
import torch
from typing import List, Literal, Optional, Dict, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)


class LLMEncoding:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
        device: Literal["cpu", "cuda"],
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model, self.tokenizer = self.initialize()

    def initialize(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        return model, tokenizer

    def generate(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return output


if __name__ == "__main__":
    modelname = "meta-llama/Llama-2-70b-hf"
    modelname = "Qwen/Qwen2.5-7B-Instruct"
    modelname = "meta-llama/Llama-2-7b-hf"
    modelname = "meta-llama/Llama-3.1-8B-Instruct"
    modelname = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    llme = LLMEncoding(model_name=modelname, max_new_tokens=5000, device="cuda")
    out = llme.generate(prompt="why is the sky blue")
    print(out)
