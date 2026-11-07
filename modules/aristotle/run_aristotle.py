import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import json
from dotenv import load_dotenv

from trans_decom import TranslateDecompose, Params

load_dotenv()


modelname = "meta-llama/Llama-2-70b-hf"
modelname = "Qwen/Qwen2.5-7B-Instruct"
modelname = "meta-llama/Llama-2-7b-hf"
modelname = "meta-llama/Llama-3.1-8B-Instruct"
modelname = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
params = Params(
    model_name=modelname,
    dataset_name="LogicNLI",
    split="test",
    api_key=None,
)
td = TranslateDecompose(params=params, device="cuda")
reasoning_outs = td.reasoning_graph_generation()

print(len(reasoning_outs))
print("END")
