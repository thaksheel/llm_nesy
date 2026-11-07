from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class AristotleConfig:
    """Settings for one Aristotle evaluation run.

    Parameters
    ----------
    dataset : {"ProntoQA", "ProofWriter", "LogicNLI"}
        Dataset to evaluate. Examples are read from ``data_dir/dataset/split.json`` and
        prompt templates from ``prompt_dir/dataset``.
    split : str, default="dev"
        Dataset split.
    search_rounds : int, default=10
        Maximum resolution steps per search path (``--search_round`` in the original code).
    limit : int or None, default=None
        Evaluate only the first `limit` examples. None evaluates all of them.
    workers : int, default=1
        Examples solved in parallel. Use more than 1 only with API models.
    resume : bool, default=True
        Keep examples already saved without an error and solve only the rest.
        Records are keyed by dataset, model and split, so use False after changing
        other settings (prompts, search rounds, temperature, ...). False starts a new
        records file.
    data_dir, prompt_dir, results_dir : str
        Folders for the datasets, the prompt templates and the outputs.
    """

    dataset: Literal["ProntoQA", "ProofWriter", "LogicNLI"]
    split: str = "dev"
    search_rounds: int = 10
    limit: Optional[int] = None
    workers: int = 1
    resume: bool = True
    data_dir: str = "data"
    prompt_dir: str = "prompts"
    results_dir: str = "results"
