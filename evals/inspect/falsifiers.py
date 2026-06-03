"""Bert's 14 pre-registered falsifier targets as Inspect AI eval cases.

Each target from `tools/falsifier_baseline.py:run_all()` becomes one
Inspect AI task.

Why this matters
================

Inspect AI (UK AISI) is the de-facto research-grade eval framework
in 2026. Wiring bert's falsifier baseline through it gives:

  - reproducible eval logs (Inspect's structured output)
  - integration with the OWASP Top-10-for-Agentic-Apps 2026 suite
  - external-agent runners (Inspect can evaluate bert AS an agent,
    not just compute the baseline)
  - the eval-log-comparison tooling (drift detection across runs)

Architecture
============

Each Inspect AI Task here wraps one `TargetResult` from
falsifier_baseline. The "solver" is trivial — it just calls the
corresponding `t1_*`, `t2_*`, … function from falsifier_baseline.
The scorer translates the PASS/FAIL/INSUFFICIENT_DATA verdict into
a score Inspect AI can aggregate.

This is the structural shim, not a duplicate. The actual falsifier
logic stays in `tools/falsifier_baseline.py` so the CLI and the
Inspect entry point produce identical verdicts.

Run:
  .venv/bin/inspect eval evals/inspect/falsifiers.py
  .venv/bin/inspect eval evals/inspect/falsifiers.py@bert_falsifier_t1
  .venv/bin/inspect view  # open the local eval-log viewer
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(LAB_ROOT))

from inspect_ai import Task, task  # noqa: E402
from inspect_ai.dataset import MemoryDataset, Sample  # noqa: E402
from inspect_ai.scorer import Score, Target, accuracy, mean, scorer  # noqa: E402
from inspect_ai.solver import Generate, TaskState, solver  # noqa: E402


# Lazy import to keep startup fast and isolate failures
def _falsifier_module():
    # Ensure the lab root is on sys.path — Inspect AI's `inspect eval`
    # CLI loads this file with a different cwd / PYTHONPATH than the
    # smoke tests, so we re-insert defensively at call time.
    import sys as _sys
    lab_root_str = str(LAB_ROOT)
    if lab_root_str not in _sys.path:
        _sys.path.insert(0, lab_root_str)
    import tools.falsifier_baseline as fb
    return fb


# ── Scorer: PASS → 1.0, FAIL → 0.0, INSUFFICIENT_DATA → NaN-like ─────


@scorer(metrics=[accuracy(), mean()])
def falsifier_status_scorer():
    """Score from the TargetResult.status enum.

    PASS  → 1.0 (target met)
    FAIL  → 0.0 (target violated; investigate)
    INSUFFICIENT_DATA → 0.5 (cannot conclude; not counted toward accuracy)
    """
    async def score(state: TaskState, target: Target) -> Score:
        # The solver stashes the TargetResult on state.metadata
        result = state.metadata.get("falsifier_result")
        if result is None:
            return Score(value=0.0, answer="missing", explanation="no result")
        status = getattr(result, "status", None)
        value_map = {"PASS": 1.0, "FAIL": 0.0, "INSUFFICIENT_DATA": 0.5}
        status_str = status.value if hasattr(status, "value") else str(status)
        return Score(
            value=value_map.get(status_str, 0.0),
            answer=status_str,
            explanation=f"{result.name} {result.threshold} → {result.current_value} (n={result.sample_size})",
            metadata={
                "target_id": result.target_id,
                "pattern": result.pattern,
                "threshold": result.threshold,
                "current_value": result.current_value,
                "sample_size": result.sample_size,
                "method": result.method.value if hasattr(result.method, "value") else str(result.method),
                "notes": result.notes,
            },
        )
    return score


# ── Solver: run the named target's evaluator and stash the result ────


def _make_solver(target_fn: Callable, *, takes_window: bool = True):
    @solver
    def run_target():
        async def run(state: TaskState, generate: Generate) -> TaskState:
            # Falsifier evaluators take an int window (bert's standard
            # is 30) except t14 which is window-less.
            result = target_fn(30) if takes_window else target_fn()
            state.metadata["falsifier_result"] = result
            # Inspect AI expects an output to score against. We use the
            # status string so the eval log is human-readable.
            status = getattr(result.status, "value", str(result.status))
            state.output.completion = status
            return state
        return run
    return run_target


# ── 14 Tasks, one per falsifier target ───────────────────────────────


def _build_single(fn_name: str, label: str, *, takes_window: bool = True) -> Task:
    """Body shared by the 14 top-level @task functions below."""
    fb = _falsifier_module()
    target_fn = getattr(fb, fn_name)
    solver_fn = _make_solver(target_fn, takes_window=takes_window)
    return Task(
        dataset=MemoryDataset([Sample(input=f"Evaluate T-{label}", target="PASS")]),
        solver=solver_fn(),
        scorer=falsifier_status_scorer(),
    )


# Targets per tools/falsifier_baseline.py:run_all() — each is a top-level
# @task so Inspect AI's --task selector can address them individually.

@task
def t1(): return _build_single("t1_threshing_structural_validity", "threshing_structural_validity")
@task
def t2(): return _build_single("t2_threshing_verdict_discipline", "threshing_verdict_discipline")
@task
def t3(): return _build_single("t3_threshing_referenced_downstream", "threshing_referenced_downstream")
@task
def t4(): return _build_single("t4_clearness_phase_completion", "clearness_phase_completion")
@task
def t5(): return _build_single("t5_clearness_query_count", "clearness_query_count")
@task
def t6(): return _build_single("t6_phase2_references_phase1", "phase2_references_phase1")
@task
def t7(): return _build_single("t7_stand_aside_concerns_populated", "stand_aside_concerns_populated")
@task
def t8(): return _build_single("t8_concerns_propagation", "concerns_propagation")
@task
def t9(): return _build_single("t9_concerns_addressed", "concerns_addressed")
@task
def t10(): return _build_single("t10_concern_aging", "concern_aging")
@task
def t11(): return _build_single("t11_seasoning_queue_size", "seasoning_queue_size_bounded")
@task
def t12(): return _build_single("t12_seasoning_revival_rate", "seasoning_revival_rate")
@task
def t13(): return _build_single("t13_revival_outcome_quality", "revival_outcome_quality")
# t14 has no window param
@task
def t14(): return _build_single("t14_seasoning_entry_well_formed",
                                 "seasoning_entry_well_formed",
                                 takes_window=False)


# ── Convenience: run-all aggregate task ──────────────────────────────


@task(name="bert_falsifier_all_14")
def all_14():
    """All 14 targets in one Task; one Sample per target."""
    fb = _falsifier_module()

    @solver
    def run_all_solver():
        async def run(state: TaskState, generate: Generate) -> TaskState:
            fn_name = state.metadata["fn_name"]
            takes_window = state.metadata.get("takes_window", True)
            fn = getattr(fb, fn_name)
            result = fn(30) if takes_window else fn()
            state.metadata["falsifier_result"] = result
            status = getattr(result.status, "value", str(result.status))
            state.output.completion = status
            return state
        return run

    samples = []
    target_specs = [
        (1, "t1_threshing_structural_validity", True),
        (2, "t2_threshing_verdict_discipline", True),
        (3, "t3_threshing_referenced_downstream", True),
        (4, "t4_clearness_phase_completion", True),
        (5, "t5_clearness_query_count", True),
        (6, "t6_phase2_references_phase1", True),
        (7, "t7_stand_aside_concerns_populated", True),
        (8, "t8_concerns_propagation", True),
        (9, "t9_concerns_addressed", True),
        (10, "t10_concern_aging", True),
        (11, "t11_seasoning_queue_size", True),
        (12, "t12_seasoning_revival_rate", True),
        (13, "t13_revival_outcome_quality", True),
        (14, "t14_seasoning_entry_well_formed", False),
    ]
    for target_id, fn_name, takes_window in target_specs:
        samples.append(Sample(
            input=f"falsifier T{target_id}",
            target="PASS",
            metadata={
                "target_id": target_id,
                "fn_name": fn_name,
                "takes_window": takes_window,
            },
        ))
    return Task(
        dataset=MemoryDataset(samples),
        solver=run_all_solver(),
        scorer=falsifier_status_scorer(),
    )
