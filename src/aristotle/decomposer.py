import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .logic import (
    atom_lines,
    clean_conjecture,
    final_form,
    group_rules,
    initial_sos,
    premise_clauses,
    split_translation,
)
from .models import ProviderModel


@dataclass
class LogicProblem:
    """A problem translated to logic, with its rules in conjunctive normal form.

    Attributes
    ----------
    id : str
        Example id.
    ground_truth : str
        Gold label: A (True), B (False), C (Unknown) or D (Self-contradictory).
    facts : str
        Translated facts, one per line.
    rules : dict of str to str
        CNF clauses for each rule type that occurs (``and_or``, ``either_or``,
        ``biconditional``), one clause per line.
    conjecture : str
        Translated conjecture.
    translation : str
        Raw translation response.
    decompositions : dict of str to str
        Raw decomposition response for each rule type.
    """

    id: str
    ground_truth: str
    facts: str
    rules: Dict[str, str]
    conjecture: str
    translation: str = ""
    decompositions: Dict[str, str] = field(default_factory=dict)

    def premises(self) -> List[str]:
        """Clauses the search starts from."""
        return premise_clauses(self.facts, self.rules)

    def sos(self, negated: bool) -> str:
        """Starting clause of a search path: the negated conjecture or the conjecture."""
        return initial_sos(self.conjecture, negated)


class Decomposer:
    """Logical Decomposer: translates a problem to logic and normalizes its rules to CNF.

    Parameters
    ----------
    model : ProviderModel
        Model used for translation and decomposition.
    prompts : dict of str to str
        Prompt templates keyed by ``translation``, ``and_or``, ``either_or`` and
        ``biconditional``. A problem with a rule type that has no template fails
        (ProntoQA only has ``and_or``, as in the original code).

    Notes
    -----
    One model call translates the problem. The translated rules are grouped by type
    (either-or, biconditional, the rest) and each group is decomposed with one more call.
    """

    def __init__(self, model: ProviderModel, prompts: Dict[str, str]):
        self.model = model
        self.prompts = prompts

    def decompose(self, example: dict) -> LogicProblem:
        """Translate `example` and decompose its rules.

        Parameters
        ----------
        example : dict
            Dataset record with ``facts``, ``rules`` and ``conjecture`` (LogicNLI) or
            ``context`` and ``question`` (ProntoQA, ProofWriter), plus ``id`` and ``answer``.

        Returns
        -------
        LogicProblem
        """
        premises, conjecture = self._premises_and_conjecture(example)
        prompt = self.prompts["translation"].replace("[[PREMISES]]", premises)
        translation = self.model.generate(prompt.replace("[[CONJECTURE]]", conjecture))
        facts, rules, conjecture = split_translation(translation)
        normalized, responses = {}, {}
        for kind, lines in group_rules(rules).items():
            if lines:
                if kind not in self.prompts:
                    raise ValueError(f"no decomposer prompt for {kind} rules")
                prompt = self.prompts[kind].replace("[[PREMISES]]", "\n".join(lines))
                responses[kind] = self.model.generate(prompt)
                normalized[kind] = final_form(responses[kind])
        return LogicProblem(
            id=example["id"],
            ground_truth=example["answer"],
            facts="\n".join(atom_lines(facts)),
            rules=normalized,
            conjecture=clean_conjecture(conjecture),
            translation=translation,
            decompositions=responses,
        )

    @staticmethod
    def _premises_and_conjecture(example: dict) -> Tuple[str, str]:
        if "facts" in example:
            return "\n".join(example["facts"] + example["rules"]), example["conjecture"]
        return (
            example["context"],
            re.search(r"\?(.*)", example["question"].strip()).group(1).strip(),
        )
