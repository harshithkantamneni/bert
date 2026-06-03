"""Clearness phase 2 capability battery — verdict + concerns.

10-task corpus: each task gives phase-1 queries + the decision, asks
for a final verdict from {APPROVE, APPROVE_WITH_CAVEATS, REJECT} with
embedded caveats when stand-aside.
"""

from __future__ import annotations

import re

from . import _common
from ._common import Task, BatteryRunResult

ROLE = "clearness_phase2"
REFERENCE_SET = "clearness_phase2_battery_v1"


_TASKS_RAW = [
    ("Decision: Adopt FalkorDB over Apache AGE.\nPhase-1 queries: 1) Operational fit? 2) License? 3) Migration cost?\nFalsifier: PostgreSQL already in stack → AGE saves ops cost.",
     "APPROVE_WITH_CAVEATS"),
    ("Decision: Ship Cathedral before KG migration.\nQueries: 1) Cathedral depends on KG? 2) Stub graph_store works?\nFalsifier: graph_store is 10-LoC stub; Cathedral needs Layer 4.",
     "REJECT"),
    ("Decision: Lower falsifier T8 from 70% to 60% for 30 dispatches.\nQueries: 1) Sample size? 2) Drift risk?\nFalsifier: A6 §11 prescribes raising threshold under uncertainty, not lowering.",
     "REJECT"),
    ("Decision: Add /summary command to bot.\nQueries: 1) Rate-limit? 2) Auth?\nFalsifier: existing /status pattern; same auth model.",
     "APPROVE"),
    ("Decision: Allow Strategist during build phase.\nQueries: 1) H-BUILD-01 still applies? 2) Mission selection due?\nFalsifier: phase H1-C4 active; no mission selection till post-C4.",
     "REJECT"),
    ("Decision: Cache LLMLingua results.\nQueries: 1) Hit rate? 2) Cache size?\nFalsifier: hit rate measurable, cache bounded.",
     "APPROVE"),
    ("Decision: Replace NVIDIA Qwen evaluator with Mistral.\nQueries: 1) Family diversity? 2) Score on evaluator battery?\nFalsifier: capability_matrix shows NVIDIA Qwen=0.88, Mistral=0.75.",
     "REJECT"),
    ("Decision: Add Anthropic provider.\nQueries: 1) Free tier? 2) Mission alignment?\nFalsifier: Anthropic is paid; bert is strict-free-tier per feedback_bert_is_proprietary.md.",
     "REJECT"),
    ("Decision: Auto-merge skills/draft → active after 5 sandbox runs.\nQueries: 1) P-005 gate? 2) Falsifier coverage?\nFalsifier: P-005 requires PI permission; auto-merge bypasses.",
     "REJECT"),
    ("Decision: Roll back to Cerebras qwen-3-32b on 404 resolve.\nQueries: 1) Detection? 2) Stability?\nFalsifier: R13 showed qwen-3-32b dropped; resolution would be Cerebras restoration.",
     "APPROVE_WITH_CAVEATS"),
]


TASKS: list[Task] = [
    Task(
        id=f"clearness_phase2_{i:02d}",
        prompt=(
            "Render a clearness phase-2 verdict for this decision + queries + "
            "falsifier. First line: ONE of {APPROVE, APPROVE_WITH_CAVEATS, REJECT}. "
            "Then 2-3 sentence rationale; if APPROVE_WITH_CAVEATS, list each "
            "caveat as a bullet.\n\n" + body
        ),
        reference=ref,
    )
    for i, (body, ref) in enumerate(_TASKS_RAW, start=1)
]


def score(task: Task, response: str) -> float:
    text = (response or "").strip().upper()
    if not text:
        return 0.0
    expected = (task.reference or "").upper()
    head = text[:120]
    if expected and expected in head:
        return 1.0
    if re.search(r"\b(APPROVE|REJECT)\b", head):
        return 0.3
    return 0.0


def run(provider: str, model: str, *, live: bool = False,
        sample: int | None = None) -> BatteryRunResult:
    return _common.run_battery(
        ROLE, TASKS, score,
        provider=provider, model=model,
        live=live, sample=sample,
    )
