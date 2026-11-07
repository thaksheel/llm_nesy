import numpy as np 
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Any, Dict
from numpy.typing import NDArray
from enum import Enum 


@dataclass
class KBData: 
    qid: int 
    query:str 
    prompt: str 
    groundtruth: bool
    depths: int  
    messages: List[Dict[str, str]] = field(default_factory=list)
    method: str = ""
    shots: str = ""


@dataclass
class Results: 
    accuracy: float 
    f1_macro: float 
    f1: NDArray 
    true_pos: NDArray 
    true_neg: NDArray 
    false_neg: NDArray 
    false_pos: NDArray 
    balanced_accuracy: NDArray 
    preds: NDArray
    trues: NDArray  
    input_token_avg: int 
    input_token: NDArray
    output_token: int 
    output_token_avg: NDArray


@dataclass
class LLMOut: 
    response: str 
    input_token:int 
    output_token: int 
    model_name: str 
    provider: str 
    reasoning_token: int = 0


class OpenModelSelection(Enum):
    tllama = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    qwen38_27b = "Qwen/Qwen3.8-27B"
    llama31_8b = "meta-llama/Llama-3.1-8B-Instruct" 
    qwen25_7b = "Qwen/Qwen2.5-7B-Instruct" 
    qwen3_8b = "Qwen/Qwen3-8B"


class CloseModelSelection(Enum):
    gpt6_luna = "gpt-6-luna"
    gpt6_sol = "gpt-6-sol"
    gpt6_astra = "gpt-6-astra"
    gpt56_sol = "gpt-5.6-sol"
    gpt56_luna = "gpt-5.6-luna"
    gpt56_terra = "gpt-5.6-terra"
    opus55 = "claude-opus-5-5"
    sonnet55 = "claude-sonnet-5-5"
    haiku55 = "claude-haiku-5-5"
