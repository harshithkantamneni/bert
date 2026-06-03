"""Threshing capability battery — early-stage winnowing.

10-task corpus: each task gives a candidate dispatch and asks the
model to threshing-pass it (verdict SCOPE_STOP if out-of-scope, else
forward with annotations).
"""

from __future__ import annotations

import re

from . import _common
from ._common import BatteryRunResult, Task

ROLE = "threshing"
REFERENCE_SET = "threshing_battery_v1"

_TASKS_RAW = [
    ("Director dispatches researcher for ArXiv scan on agentic AI; mission: ML-engineering job search.", "SCOPE_STOP"),
    ("Director dispatches strategist to rank candidates from researcher_C12 findings; mission: investment thesis.", "FORWARD"),
    ("Director dispatches evaluator to verdict cross-family verdict; producer family Llama, evaluator family Llama.", "SCOPE_STOP"),
    ("Director dispatches implementer to refactor core/quota.py; mission: build harness; cycle queue includes this task.", "FORWARD"),
    ("Director dispatches researcher to scan job market data; mission: build harness.", "SCOPE_STOP"),
    ("Director dispatches strategist for 5-option ranked candidate list on bert architecture choice; falsifier set.", "FORWARD"),
    ("Director dispatches falsifier baseline; falsifier set; 14 targets; calibration window cleared.", "FORWARD"),
    ("Director dispatches evaluator on a non-contested decision (no APPROVE_WITH_CAVEATS upstream); P-VS-02 unnecessary.", "SCOPE_STOP"),
    ("Director dispatches implementer to add new bert surface; cycle queue + plan agree.", "FORWARD"),
    ("Director dispatches researcher on a topic already covered in researcher_C8.md.", "SCOPE_STOP"),
]


TASKS: list[Task] = [
    Task(
        id=f"threshing_{i:02d}",
        prompt=(
            "Threshing pass on this dispatch. Return ONE verdict from "
            "{FORWARD, SCOPE_STOP} on the first line, then 1-2 sentence "
            "rationale.\n\nDispatch:\n" + summary
        ),
        reference=ref,
    )
    for i, (summary, ref) in enumerate(_TASKS_RAW, start=1)
]


def score(task: Task, response: str) -> float:
    text = (response or "").strip().upper()
    if not text:
        return 0.0
    expected = (task.reference or "").upper()
    head = text[:120]
    if expected and expected in head:
        body_words = len(text.split())
        return 1.0 if 5 <= body_words <= 80 else 0.7
    if re.search(r"\b(FORWARD|SCOPE_STOP)\b", head):
        return 0.3
    return 0.0


def run(provider: str, model: str, *, live: bool = False,
        sample: int | None = None) -> BatteryRunResult:
    return _common.run_battery(
        ROLE, TASKS, score,
        provider=provider, model=model,
        live=live, sample=sample,
    )
