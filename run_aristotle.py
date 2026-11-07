from dataclasses import dataclass
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import Optional, Literal, List, Dict
import os
import json
from dotenv import load_dotenv

from modules.aristotle.translate_decompose import GPT3_Reasoning_Graph_Baseline

load_dotenv()


@dataclass
class Params:
    api_key: str
    model_name: str
    dataset_name: str
    split: str
    data_path: str = (
        r"C:\drive\OneDrive - Lehigh University\Lehigh\Research\LLM\symbolic\replicate\modules\aristotle\data"
    )
    save_path: str = "./results"
    demonstration_path: str = "./icl_examples"
    stop_words: str = "------"
    max_new_tokens: Optional[int] = 5000
    mode: Optional[str] = None
    base_url: Optional[str] = None
    batch_num: int = 1


params = Params(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o",
    dataset_name="LogicNLI",
    split="test",
)
rgb = GPT3_Reasoning_Graph_Baseline(args=params)
rgb.reasoning_graph_generation()

print("END")
