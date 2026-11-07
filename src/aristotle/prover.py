from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .logic import complementary_clauses, parse_resolution
from .models import ProviderModel

Pair = Tuple[str, str]
Cache = List[Tuple[str, List[str]]]


@dataclass
class Step:
    """One resolution: the set-of-support clause, the clause it was resolved with, the
    resolvent, the resolver's sufficiency label and its raw response."""

    sos: str
    clause: str
    resolvent: str
    sufficient: bool
    response: str = ""


@dataclass
class Proof:
    """Outcome of one proof-by-contradiction search path.

    Attributes
    ----------
    contradiction : bool
        True if the resolver reported a contradiction.
    stop_reason : str
        Why the search ended: ``contradiction``, ``no complementary clause``,
        ``no untried clause pair``, ``no new clause`` or ``search rounds exhausted``.
    steps : list of Step
        Resolution steps, one model call each.
    """

    contradiction: bool
    stop_reason: str
    steps: List[Step] = field(default_factory=list)


class Resolver:
    """Logical Resolver: asks the model to resolve two clauses.

    Parameters
    ----------
    model : ProviderModel
        Model used for resolution.
    template : str
        Resolver prompt with ``[[SOS]]`` and ``[[SELECTED-CLAUSE]]`` placeholders.
    """

    def __init__(self, model: ProviderModel, template: str):
        self.model = model
        self.template = template

    def __call__(self, sos: str, clause: str) -> Step:
        prompt = self.template.replace("[[SOS]]", sos).replace("[[SELECTED-CLAUSE]]", clause)
        response = self.model.generate(prompt)
        resolvent, sufficient = parse_resolution(response)
        return Step(sos, clause, resolvent, sufficient, response)


class Prover:
    """Search Router and Resolver loop for one search path.

    Starting from a set-of-support clause, the router picks the shortest untried clause
    with a literal complementary to it, the resolver resolves the pair, and the resolvent
    becomes the next set-of-support clause. When several clauses qualify, the others are
    cached and used for backtracking if a resolution yields no new clause. The control
    flow follows the original implementation step for step.

    Parameters
    ----------
    resolver : Resolver
        Resolver used for each step.
    max_rounds : int, default=10
        Maximum number of resolution steps.
    """

    def __init__(self, resolver: Resolver, max_rounds: int = 10):
        self.resolver = resolver
        self.max_rounds = max_rounds

    def prove(self, premises: List[str], sos: str) -> Proof:
        """Search for a contradiction between `premises` and `sos`.

        Parameters
        ----------
        premises : list of str
            Clauses from the problem.
        sos : str
            Starting set-of-support clause (the negated conjecture to prove it True,
            the conjecture itself to prove it False).

        Returns
        -------
        Proof
        """
        clauses, steps, cache = premises + [sos], [], []
        clause = None
        while len(steps) < self.max_rounds:
            if clause is None:
                pair = self._route(sos, clauses, steps, cache)
                if pair is None:
                    return Proof(False, "no complementary clause", steps)
                sos, clause = pair
            if self._tried(sos, clause, steps):
                pair = self._from_cache(cache, skip=(sos, clause))
                if pair is None:
                    return Proof(False, "no untried clause pair", steps)
                sos, clause = pair
            step = self.resolver(sos, clause)
            steps.append(step)
            if step.sufficient:
                return Proof(True, "contradiction", steps)
            if step.resolvent:
                clauses.append(step.resolvent)
                sos, clause = step.resolvent, None
            else:
                pair = self._from_cache(cache)
                if pair is None:
                    return Proof(False, "no new clause", steps)
                sos, clause = pair
        return Proof(False, "search rounds exhausted", steps)

    def _route(self, sos: str, clauses: List[str], steps: List[Step], cache: Cache) -> Optional[Pair]:
        candidates = complementary_clauses(clauses, sos)
        used = {step.clause for step in steps if step.sos == sos}
        untried = sorted((c for c in candidates if c not in used), key=len)
        if not untried:
            return self._from_cache(cache)
        if len(untried) > 1:
            cache.append((sos, untried[1:]))
        return sos, untried[0]

    @staticmethod
    def _tried(sos: str, clause: str, steps: List[Step]) -> bool:
        return any(s.sos.strip() == sos.strip() and s.clause.strip() == clause.strip() for s in steps)

    @staticmethod
    def _from_cache(cache: Cache, skip: Optional[Pair] = None) -> Optional[Pair]:
        for sos, untried in cache:
            if untried and (sos, untried[0]) != skip:
                return sos, untried.pop(0)
        return None
