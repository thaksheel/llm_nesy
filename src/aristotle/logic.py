import re
from typing import Dict, List, Tuple

ATOM = re.compile(r"(\w+)\((.*?)\)")
ARGUMENTS = re.compile(r"\(([^)]+)\)")
BOOLEAN = re.compile(r"(False|True)")
OPERATOR = re.compile(r"(\\lor|\\land|\u2228|\u2227|\\wedge|\\vee)")
CONJUNCTIONS = ("\\wedge", "\\land", "\u2227")
NEGATIONS = ("\u00ac", "\\neg", "\\lnot", "eg")
HEADERS = ("Rules (others):", "Rules (biconditional):", "Rules (either_or):")
LATEX = (
    ("\\textnormal", ""),
    ("\\textrm", ""),
    ("\\\\text", "\\text"),
    ("\\text", ""),
    ("-", ""),
    ("{", ""),
    ("}", ""),
    ("\\right", ""),
    ("\\left", ""),
    ("\\newline", "\n"),
    ("$", ""),
)


def atom_lines(text: str) -> List[str]:
    return [line.strip() for line in text.split("\n") if "(" in line]


def split_translation(response: str) -> Tuple[str, str, str]:
    """Facts, rules and conjecture sections of a translation response."""
    facts = re.findall(r"Fact(.*?)Rule", response, re.DOTALL)
    rules = re.findall(r"Rule(.*?)Conjecture", response, re.DOTALL)
    _, found, conjecture = response.rpartition("Conjecture")
    conjecture = re.sub(r"^[\s:\-]*", "", conjecture.strip()) if found else ""
    return (facts[-1].strip() if facts else "", rules[-1].strip() if rules else "", conjecture)


def group_rules(rules: str) -> Dict[str, List[str]]:
    groups = {"and_or": [], "either_or": [], "biconditional": []}
    for line in rules.split("\n"):
        if "(" in line:
            kind = "either_or" if "\u2295" in line else "biconditional" if "\u21d4" in line else "and_or"
            groups[kind].append(line)
    return groups


def final_form(response: str) -> str:
    """CNF clauses that follow 'Final Form' in a decomposition response."""
    start = response.find("Final Form")
    if start >= 0:
        response = response[start:].split("Final Form:")[-1].strip()
    return "\n".join(atom_lines(response))


def clean_conjecture(conjecture: str) -> str:
    lines = conjecture.replace("\\n", "\n").split("\n")
    return "\n".join(line.split(":::")[0] for line in lines if "(" in line and not _is_header(line))


def premise_clauses(facts: str, rules: Dict[str, str]) -> List[str]:
    """Clauses the search starts from: cleaned facts and CNF rules, conjunctions split."""
    lines = []
    for line in facts.splitlines():
        if ":::" in line:
            left, right = line.split(":::", 1)
            lines.append((left if "(" in left else right).strip())
        elif "(" in line:
            lines.append(line.strip())
    for text in rules.values():
        lines.extend(line for line in text.splitlines() if line.strip())
    clauses = []
    for line in lines:
        if "(" not in line or ")" not in line or _is_header(line):
            continue
        line = "\n".join(remove_negations(part) for part in _strip_latex(line).split("\n"))
        line = next((part.strip() for part in line.split(":::", 1) if "(" in part), "")
        clauses.extend(_split_conjunction(line))
    return [clause for clause in clauses if "(" in clause and ")" in clause]


def initial_sos(conjecture: str, negated: bool) -> str:
    """Starting clause of a search path: the conjecture, or its negation."""
    sos = _atom_text(negate_conjecture(conjecture) if negated else conjecture)
    if ")" not in sos:
        sos = remove_negations(_atom_text(conjecture))
        if negated:
            sos = flip_first_boolean(sos)
    return sos


def negate_conjecture(conjecture: str) -> str:
    if "True" in conjecture:
        return conjecture.replace("True", "False")
    if "False" in conjecture:
        return conjecture.replace("False", "True")
    return "false"


def flip_first_boolean(text: str) -> str:
    return BOOLEAN.sub(lambda match: _flip(match.group()), text, count=1)


def remove_negations(text: str) -> str:
    """Rewrite negated literals as boolean flips, e.g. ¬Red(x, True) -> Red(x, False)."""
    text = ARGUMENTS.sub(_add_missing_boolean, text)
    for negation in NEGATIONS:
        while negation in text:
            start = text.find(negation)
            rest = text[start + len(negation):]
            match = BOOLEAN.search(rest)
            if match:
                rest = rest[: match.start()] + _flip(match.group()) + rest[match.end():]
            text = text[:start] + rest
    return text


def complementary_clauses(clauses: List[str], sos: str) -> List[str]:
    """Clauses with a literal complementary to the first literal of `sos` (Search Router)."""
    return [
        clause
        for clause in clauses
        if any(_complementary(term, sos) for term in OPERATOR.split(re.sub(r"[0-9.]", "", clause)))
    ]


def parse_resolution(response: str) -> Tuple[str, bool]:
    """New clause in {...} and whether the sufficiency label in [...] is True."""
    clause = re.search(r"\{(.*?)\}", response)
    label = re.search(r"\[(.*?)\]", response)
    return clause.group(1).strip() if clause else "", bool(label) and label.group(1).strip() == "True"


def _complementary(term: str, sos: str) -> bool:
    first, second = ATOM.search(term), ATOM.search(sos)
    if not first or not second or first.group(1) != second.group(1):
        return False
    args, other = first.group(2).split(","), second.group(2).split(",")
    if len(args) != len(other):
        return False
    for a, b in zip(args[:-1], other[:-1]):
        a, b = a.strip(), b.strip()
        if a != b and "x" not in (a, b):
            return False
    return {args[-1].strip(), other[-1].strip()} == {"True", "False"}


def _atom_text(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if line.strip() and "(" in line and not _is_header(line)
    )


def _add_missing_boolean(match: re.Match) -> str:
    inner = match.group(1)
    if inner.split(",")[-1].strip() in ("True", "False"):
        return f"({inner})"
    return f"({inner}, True)"


def _split_conjunction(clause: str) -> List[str]:
    for operator in CONJUNCTIONS:
        if operator in clause:
            return [part.strip() for part in clause.split(operator)]
    return [clause]


def _strip_latex(text: str) -> str:
    for old, new in LATEX:
        text = text.replace(old, new)
    return text


def _is_header(line: str) -> bool:
    return any(header in line for header in HEADERS)


def _flip(value: str) -> str:
    return "True" if value == "False" else "False"
