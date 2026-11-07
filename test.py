import numpy as np 
import torch 
import pandas as pd 
from matplotlib import pyplot as plt 
import json 

filename = "./modules/Aristotle-main/data/LogicNLI/dev.json"
with open(filename, "r", encoding="utf-8") as f:
    logic_nli_data = json.load(f) 
outfile = "./modules/Aristotle-main/data/LogicNLI/test.json"
with open(outfile, "w", encoding="utf-8") as f:
    json.dump(logic_nli_data[:20], f, indent=4, ensure_ascii=False)

print(len(logic_nli_data))
print("END")
