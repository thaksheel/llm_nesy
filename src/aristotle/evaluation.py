import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List 
import pandas as pd 

from tqdm import tqdm

from .config import AristotleConfig
from .decomposer import Decomposer
from .models import ProviderModel
from .prover import Prover, Resolver

PROMPT_FILES = {
    "translation": "translation.txt",
    "and_or": "and_or_decomposer.txt",
    "either_or": "either_or_decomposer.txt",
    "biconditional": "logical_biconditional_decomposer.txt",
    "resolver": "logic_resolver.txt",
}
VERDICTS = {(True, False): "A", (False, True): "B", (False, False): "C", (True, True): "D"}


@dataclass
class Results:
    """Scores of one evaluation run.

    Attributes
    ----------
    dataset : str
        Dataset name.
    model : str
        Model name.
    total : int
        Number of examples scored.
    correct : int
        Examples whose prediction matches the gold label.
    errors : int
        Examples that failed with an exception. They count as wrong in `accuracy`.
    accuracy : float
        ``correct / total``.
    accuracy_excluding_errors : float
        ``correct / (total - errors)``. The original ``evaluate.py`` dropped failed
        examples, so this is the number comparable to its output.
    avg_steps : float
        Mean resolution steps (model calls in the search) per example, both paths
        together, over examples without errors.
    per_label : dict of str to float
        Accuracy for each gold label.
    records : str
        Path of the per-example records file (JSON lines).
    """

    dataset: str
    model: str
    total: int
    correct: int
    errors: int
    accuracy: float
    accuracy_excluding_errors: float
    avg_steps: float
    per_label: Dict[str, float]
    records: str 

    def to_df(self) -> pd.DataFrame:
        """One-row DataFrame of the scores, with one ``accuracy_<label>`` column per label.

        Returns
        -------
        pandas.DataFrame
            Concatenate several runs with ``pd.concat([r.to_df() for r in runs])``.
        """
        row = {key: value for key, value in asdict(self).items() if key != "per_label"}
        row.update({f"accuracy_{label}": value for label, value in self.per_label.items()})
        return pd.DataFrame([row])

    def __str__(self) -> str:
        labels = ", ".join(f"{label}: {value:.3f}" for label, value in self.per_label.items())
        return (
            f"{self.dataset} | {self.model}\n"
            f"accuracy {self.accuracy:.3f} ({self.correct}/{self.total}), errors {self.errors}, "
            f"accuracy excluding errors {self.accuracy_excluding_errors:.3f}\n"
            f"avg steps {self.avg_steps:.2f} | per label: {labels}"
        )


class Aristotle:
    """Aristotle evaluation: decompose, search and resolve, then score.

    For each example the Decomposer translates the problem to logic and normalizes its
    rules. Two proof-by-contradiction searches then run from the same clauses: the true
    path starts from the negated conjecture (a contradiction proves it True), the false
    path from the conjecture (a contradiction proves it False). The prediction follows
    Eq. (1) of the paper: True if only the true path finds a contradiction, False if only
    the false path does, Unknown if neither and Self-contradictory if both.

    Parameters
    ----------
    config : AristotleConfig
        Run settings.
    llm : LLMProvider or ProviderModel
        Language model used for translation, decomposition and resolution. An
        ``LLMProvider`` is wrapped in a ``ProviderModel``; pass a ``ProviderModel`` to
        change the system prompt or stop string. The same instance can be reused across
        datasets.

    Attributes
    ----------
    examples : list of dict
        Dataset records to evaluate.
    decomposer : Decomposer
        Translation and decomposition stage.
    prover : Prover
        Search and resolve stage.
    records_path : pathlib.Path
        JSON-lines file with one record per example, written as the run progresses.

    Examples
    --------
    >>> llm = LLMProvider(provider="openai", model_name="gpt-4o", token=key, temperature=0)
    >>> results = Aristotle(AristotleConfig(dataset="ProofWriter", workers=8), llm).run()
    >>> print(results)
    """

    def __init__(self, config: AristotleConfig, llm: Any):
        self.config = config
        self.model = llm if isinstance(llm, ProviderModel) else ProviderModel(llm)
        folder = Path(config.prompt_dir) / config.dataset
        if not folder.is_dir():
            raise FileNotFoundError(f"prompt folder not found: {folder.resolve()}")
        prompts = {
            key: (folder / name).read_text(encoding="utf-8")
            for key, name in PROMPT_FILES.items()
            if (folder / name).exists()
        }
        self.decomposer = Decomposer(self.model, prompts)
        self.prover = Prover(Resolver(self.model, prompts["resolver"]), config.search_rounds)
        data = Path(config.data_dir) / config.dataset / f"{config.split}.json"
        self.examples = json.loads(data.read_text(encoding="utf-8"))[: config.limit]
        name = self.model.name.replace("/", "_")
        self.records_path = Path(config.results_dir) / config.dataset / f"{config.dataset}_{name}_{config.split}.jsonl"

    def run(self) -> Results:
        """Solve every example not yet saved, save each record, and score the run.

        Returns
        -------
        Results
            Also written next to the records file as ``*.summary.json``.
        """
        records = self.load_records() if self.config.resume else {}
        todo = [e for e in self.examples if e["id"] not in records or records[e["id"]]["error"]]
        self.records_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self.config.resume else "w"
        with self.records_path.open(mode, encoding="utf-8") as file, ThreadPoolExecutor(self.config.workers) as pool:
            solved = pool.map(self.solve, todo) if self.config.workers > 1 else map(self.solve, todo)
            for record in tqdm(solved, total=len(todo), desc=f"Aristotle {self.config.dataset}"):
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                file.flush()
                records[record["id"]] = record
        results = self.score([records[e["id"]] for e in self.examples])
        summary = self.records_path.with_suffix(".summary.json")
        summary.write_text(json.dumps(asdict(results), indent=2), encoding="utf-8")
        return results

    def solve(self, example: dict) -> dict:
        """Decompose one example, run both search paths and record the prediction.

        Parameters
        ----------
        example : dict
            Dataset record.

        Returns
        -------
        dict
            ``id``, ``ground_truth``, ``prediction``, ``correct``, ``steps`` and ``error``,
            plus the ``problem``, ``true_path`` and ``false_path`` details when it succeeds.
        """
        record = {
            "id": example["id"],
            "ground_truth": example["answer"],
            "prediction": None,
            "correct": False,
            "steps": 0,
            "error": None,
        }
        try:
            problem = self.decomposer.decompose(example)
            premises = problem.premises()
            true_path = self.prover.prove(premises, problem.sos(negated=True))
            false_path = self.prover.prove(premises, problem.sos(negated=False))
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            return record
        prediction = VERDICTS[(true_path.contradiction, false_path.contradiction)]
        record.update(
            prediction=prediction,
            correct=prediction == example["answer"],
            steps=len(true_path.steps) + len(false_path.steps),
            problem=asdict(problem),
            true_path=asdict(true_path),
            false_path=asdict(false_path),
        )
        return record

    def load_records(self) -> Dict[str, dict]:
        """Saved records by id; the last record wins when an id was solved twice."""
        if not self.records_path.exists():
            return {}
        with self.records_path.open(encoding="utf-8") as file:
            return {record["id"]: record for record in (json.loads(line) for line in file if line.strip())}

    def score(self, records: List[dict]) -> Results:
        """Accuracy, errors, steps and per-label accuracy of `records`."""
        total = len(records)
        correct = sum(record["correct"] for record in records)
        errors = sum(record["error"] is not None for record in records)
        labels = sorted({record["ground_truth"] for record in records})
        return Results(
            dataset=self.config.dataset,
            model=self.model.name,
            total=total,
            correct=correct,
            errors=errors,
            accuracy=correct / total if total else 0.0,
            accuracy_excluding_errors=correct / (total - errors) if total > errors else 0.0,
            avg_steps=mean([r["steps"] for r in records if r["error"] is None] or [0]),
            per_label={
                label: mean(r["correct"] for r in records if r["ground_truth"] == label) for label in labels
            },
            records=str(self.records_path),
        )

