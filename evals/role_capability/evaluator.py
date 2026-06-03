"""Evaluator capability battery — verdict agreement with held-out
human-rated reference.

30-task corpus: each task is a contested-decision summary with a
reference verdict (APPROVE / APPROVE_WITH_CAVEATS / REJECT / SCOPE_STOP).
Scorer (live mode) checks the model's verdict against the reference.
"""

from __future__ import annotations

import re

from . import _common
from ._common import Task, BatteryRunResult

ROLE = "evaluator"
REFERENCE_SET = "evaluator_battery_v1"


# Each task: (summary, reference_verdict)
_TASKS_RAW = [
    ("Director proposes adding a 3rd evaluator role to break ties; researcher reports 4% throughput loss; falsifier registers stand-aside.", "APPROVE_WITH_CAVEATS"),
    ("Implementer wants to bypass P-VS-02 cross-family on small commits; researcher says throughput +30%, falsifier reports drift catch rate drops 12pp.", "REJECT"),
    ("Researcher claims new arXiv pattern (KVComm) is load-bearing for token efficiency; evidence: 3 papers, mixed results.", "APPROVE_WITH_CAVEATS"),
    ("Strategist drafts roadmap to ship Atlas before Manuscript; cycle queue shows Manuscript blocks 2 downstream surfaces.", "REJECT"),
    ("Director wants to silence circuit_breaker_event for non-critical providers; researcher: 8% noise floor.", "APPROVE_WITH_CAVEATS"),
    ("Implementer ships fix for memory.search ranking bug; falsifier verifies no regression; verdict: clean.", "APPROVE"),
    ("Proposal: replace LLMLingua with hand-tuned compression heuristic; researcher: BERTScore drops 0.07.", "REJECT"),
    ("Director adds /summary command to bot; cost: low, value: PI digest visibility.", "APPROVE"),
    ("Evaluator FAIL 3rd cycle in a row on cross-family arbitration; coherence drift suspected.", "SCOPE_STOP"),
    ("Implementer wants to drop quota check_quota guard for speed; researcher: 18% RPM cap violation likelihood.", "REJECT"),
    ("Director proposes new D-NN ratification batch (5 patterns); falsifier targets cleared; falsifier registers concur.", "APPROVE"),
    ("Strategist proposes pivot from R&D mission to consumer product; cycle queue + memory log show drift; multiple stand-aside concerns.", "SCOPE_STOP"),
    ("Researcher adds A2A wire-format adapter; implementer notes 80% test coverage; falsifier: no regression.", "APPROVE"),
    ("Director shortens seasoning revival_conditions to 1 line max; researcher: information loss minimal; concern: edge case retrieval drops.", "APPROVE_WITH_CAVEATS"),
    ("Implementer ships PI permission gate bypass for hard-gate operations; falsifier: P-011 violation; severity high.", "REJECT"),
    ("Researcher claims cache hit % is the strongest cost-quality signal; data: 6 weeks live; correlation r=0.82.", "APPROVE"),
    ("Strategist proposes ship-now-iterate-later on Cathedral surface; researcher: 3 dependencies not yet stable.", "REJECT"),
    ("Director adds capability_matrix mtime falsifier (FALS-L24-01); cost: 50 LoC; value: weekly drift detection.", "APPROVE"),
    ("Implementer refactors core/agent.py iter loop; falsifier: subtle CATASTROPHIC exit on edge case.", "APPROVE_WITH_CAVEATS"),
    ("Researcher claims new model (deepseek-r1-0528) outperforms current evaluator slot on PR-VS-02 cross-family.", "APPROVE_WITH_CAVEATS"),
    ("Director loosens 80% threshold on falsifier T8 to 60% pending more data; researcher: enables drift.", "REJECT"),
    ("Strategist drafts mission pivot to ML-engineering-job-finder; cycle queue + PI notes show convergent signal.", "APPROVE"),
    ("Implementer adds emergency abort hook; falsifier: hook fires on healthy state in test; severity med.", "APPROVE_WITH_CAVEATS"),
    ("Director merges H4 Track C deferred-to-launch; researcher: Phase C3 still gated.", "APPROVE_WITH_CAVEATS"),
    ("Researcher: free-tier provider X cut RPM 20→10 effective tomorrow; cascade re-routing needed.", "APPROVE"),
    ("Implementer wants to skip Evaluator on dispatches with trust-score > 0.95; falsifier: 0.95 threshold rare; gate weak.", "REJECT"),
    ("Director enables capability_matrix consult in pick_evaluator_model; FALS-L24-03 passes.", "APPROVE"),
    ("Strategist proposes shutting down 2 of 8 providers to simplify ops; researcher: cross-family diversity drops.", "REJECT"),
    ("Implementer ships RouteLLM smart routing first-attempt layer; falsifier: cross-family rule still preserved.", "APPROVE_WITH_CAVEATS"),
    ("Director closes mission as VICTORY; PI ratifies; evaluator concurs.", "APPROVE"),
]


TASKS: list[Task] = [
    Task(
        id=f"evaluator_{i:02d}",
        prompt=(
            "Evaluate this contested decision and return ONE verdict from "
            "{APPROVE, APPROVE_WITH_CAVEATS, REJECT, SCOPE_STOP} on the first line, "
            "followed by a 2-3 sentence rationale.\n\nDecision:\n" + summary
        ),
        reference=ref,
        rubric_hint=f"expected verdict: {ref}",
    )
    for i, (summary, ref) in enumerate(_TASKS_RAW, start=1)
]


def score(task: Task, response: str) -> float:
    text = (response or "").strip().upper()
    if not text:
        return 0.0
    expected = (task.reference or "").upper()
    # Look for the expected verdict in the first 60 chars of the response.
    head = text[:120]
    if expected and expected in head:
        # Partial credit if rationale length is reasonable
        body_words = len(text.split())
        return 1.0 if 8 <= body_words <= 120 else 0.7
    # Partial credit for any valid verdict marker (proves at least the format is right)
    if re.search(r"\b(APPROVE|REJECT|SCOPE_STOP)\b", head):
        return 0.3
    return 0.0


def run(provider: str, model: str, *, live: bool = False,
        sample: int | None = None) -> BatteryRunResult:
    return _common.run_battery(
        ROLE, TASKS, score,
        provider=provider, model=model,
        live=live, sample=sample,
    )
