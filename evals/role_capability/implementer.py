"""Implementer capability battery — build-pass rate on standardized tasks.

10-task corpus: each task is a small code-completion or fix task with a
reference snippet. Live scorer compares structural elements.
"""

from __future__ import annotations

import re

from . import _common
from ._common import BatteryRunResult, Task

ROLE = "implementer"
REFERENCE_SET = "implementer_battery_v1"

_TASKS_RAW = [
    ("Write a Python function `count_words(s: str) -> int` that returns the number of words in s.", "def count_words"),
    ("Write a Python function `normalize(p: str) -> str` that strips, lowercases, and replaces spaces with hyphens.", "def normalize"),
    ("Write a SQL `CREATE TABLE events` statement with columns id (PK INT), ts (REAL), event_class (TEXT), payload (JSON).", "CREATE TABLE"),
    ("Write a regex that matches semantic version strings (e.g. 1.2.3, 0.0.1-rc1).", "\\d"),
    ("Write a shell command that finds all .py files larger than 10KB and prints their paths.", "find"),
    ("Write a Python decorator `@timed` that prints the wall-clock time of any function it wraps.", "def timed"),
    ("Write a JSONL line for a verdict event: agent=evaluator, cycle=99, verdict=APPROVE, confidence=8.", "verdict"),
    ("Write a Python function `merge_dicts(a, b)` that deep-merges b into a, b's values winning on conflict.", "def merge_dicts"),
    ("Write a curl command that POSTs JSON to http://localhost:8080/api/echo with body {\"x\": 1}.", "curl"),
    ("Write a Python contextmanager `@suppress(*exc_types)` that swallows the given exception types.", "contextmanager"),
]


TASKS: list[Task] = [
    Task(
        id=f"implementer_{i:02d}",
        prompt=("Implement the following. Return ONLY the code, no commentary.\n\n" + prompt),
        reference=ref,
    )
    for i, (prompt, ref) in enumerate(_TASKS_RAW, start=1)
]


def score(task: Task, response: str) -> float:
    text = (response or "").strip()
    if not text:
        return 0.0
    expected = task.reference or ""
    # Did the response contain the expected anchor (function def, keyword, etc.)?
    has_anchor = expected.lower() in text.lower() if expected else True
    # Is it the right shape — code-like, not prose?
    code_indicators = sum([
        1 if re.search(r"def\s+\w+|class\s+\w+", text) else 0,
        1 if re.search(r"[(){};]|->|::", text) else 0,
        1 if re.search(r"^\s+\w", text, re.M) else 0,  # indented
    ])
    return (0.5 if has_anchor else 0.0) + 0.17 * code_indicators


def run(provider: str, model: str, *, live: bool = False,
        sample: int | None = None) -> BatteryRunResult:
    return _common.run_battery(
        ROLE, TASKS, score,
        provider=provider, model=model,
        live=live, sample=sample,
    )
