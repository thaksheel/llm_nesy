import os
from dotenv import load_dotenv
import pandas as pd

from src import (
    AristotleConfig,
    Aristotle,
    LLMProvider,
    CloseModelSelection,
    OpenModelSelection,
)

load_dotenv()

llm = LLMProvider(
    provider="hf",
    model_name=OpenModelSelection.qwen25_7b.value,
    token=os.getenv("HF_TOKEN"),
    max_new_tokens=4096,
    temperature=0,
    load_in_4bit=True,
)
llm = LLMProvider(
    provider="hf",
    model_name=CloseModelSelection.gpt6_luna.value,
    token=os.getenv("OPENAI_KEY"),
    max_new_tokens=4096,
    temperature=1,
    load_in_4bit=False,
)

config = AristotleConfig(
    dataset="LogicNLI",
    data_dir="./data",
    prompt_dir="./data/prompts",
    results_dir="./exports/aristotle",
)
runs = [
    Aristotle(AristotleConfig(dataset=d), llm).run()
    for d in ("ProntoQA", "ProofWriter", "LogicNLI")
]

df = pd.concat([r.to_df() for r in runs], ignore_index=True)
df.to_excel(f"./exports/aristotle/{llm.model_name.replace('/', '_')}_aristotle.xlsx")

print(df.head())
print("END")
