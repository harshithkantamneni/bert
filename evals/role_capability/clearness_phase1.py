"""Clearness phase 1 capability battery — non-leading clearness queries.

10-task corpus: each task gives a decision and asks the model to draft
3-5 clearness queries (questions that test the decision without
leading the answer).
"""

from __future__ import annotations

import re

from . import _common
from ._common import BatteryRunResult, Task

ROLE = "clearness_phase1"
REFERENCE_SET = "clearness_phase1_battery_v1"


_TOPICS = [
    "Adopt FalkorDB over Apache AGE for bert's KG layer",
    "Ship a Cathedral surface before the KG migration completes",
    "Lower the falsifier T8 threshold from 70% to 60% for the next 30 dispatches",
    "Replace OpenRouter with HF Router as the universal escape hatch",
    "Allow the Director to dispatch Strategist during the build phase",
    "Cache LLMLingua compression results in lab/state/llmlingua_cache.db",
    "Use Mistral as the default cross-family evaluator instead of NVIDIA Qwen",
    "Add a 9th provider (Anthropic) for non-free-tier paid runs",
    "Auto-merge skills/draft → skills/active after 5 successful sandbox runs",
    "Roll back Cerebras to qwen-3-32b once 404 is resolved",
]


TASKS: list[Task] = [
    Task(
        id=f"clearness_phase1_{i:02d}",
        prompt=(
            "Draft 3-5 clearness queries for this decision. Each query should "
            "OPEN the decision (not lead the answer). Number them 1) 2) 3).\n\n"
            "Decision:\n" + topic
        ),
        rubric_hint="3-5 numbered queries, non-leading",
    )
    for i, topic in enumerate(_TOPICS, start=1)
]


def score(task: Task, response: str) -> float:
    text = (response or "").strip()
    if not text:
        return 0.0
    # Count numbered queries
    numbered = len(re.findall(r"^\s*\d+[).]\s+\S", text, re.M))
    count_ok = 3 <= numbered <= 6
    # Penalize leading phrasings like "shouldn't we...", "isn't it obvious..."
    leading_markers = len(re.findall(
        r"\b(shouldn'?t|wouldn'?t|isn'?t|obvious|clearly|surely)\b",
        text, re.I,
    ))
    return (
        (0.6 if count_ok else 0.2)
        + max(0.0, 0.4 - 0.1 * leading_markers)
    )


def run(provider: str, model: str, *, live: bool = False,
        sample: int | None = None) -> BatteryRunResult:
    return _common.run_battery(
        ROLE, TASKS, score,
        provider=provider, model=model,
        live=live, sample=sample,
    )
