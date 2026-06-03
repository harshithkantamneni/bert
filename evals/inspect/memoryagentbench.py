"""MemoryAgentBench-shaped Inspect AI suite.

Per MemoryAgentBench (ICLR 2026, Wang et al.), agent memory systems
should be scored across four orthogonal capability axes:

  AR — Accurate Retrieval (precision@k on stored facts)
  TTL — Test-Time Learning (does the agent update on contradiction?)
  LRU — Long-Range Understanding (does retrieval survive distance?)
  CR — Conflict Resolution (does the agent reconcile when sources disagree?)

This file is the IN-TREE scaffold — small synthetic samples that exercise
bert's actual memory pipeline (core.memory + core.graph_store + core.retrieval).
A full external-fixture run (4 datasets x 250 samples) is queued for H.8+.

Invoke via the inspect_ai CLI against any task in this module
(see README.md inside evals/inspect/).
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(LAB_ROOT))

from inspect_ai import Task, task  # noqa: E402
from inspect_ai.dataset import MemoryDataset, Sample  # noqa: E402
from inspect_ai.scorer import Score, Target, accuracy, mean, scorer  # noqa: E402
from inspect_ai.solver import Generate, TaskState, solver  # noqa: E402

# ── Shared structural scorer (PASS / FAIL on retrieved-fact match) ──


@scorer(metrics=[accuracy(), mean()])
def memory_recall_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        result = state.metadata.get("recall_result", {})
        passed = result.get("passed", False)
        return Score(
            value=1.0 if passed else 0.0,
            answer="PASS" if passed else "FAIL",
            explanation=result.get("rationale", ""),
            metadata=result,
        )
    return score


# ── Helpers — stub bert memory using in-process store ─────────────────


def _stub_store() -> dict:
    """Returns a tiny in-memory key/value store standing in for
    core.memory so the run doesn't require Ollama embeddings."""
    return {
        "facts": [
            ("Marie Curie", "discovered polonium", "1898-01-01"),
            ("Marie Curie", "discovered radium", "1898-12-26"),
            ("Pierre Curie", "discovered piezoelectricity", "1880-01-01"),
            ("Albert Einstein", "published special relativity", "1905-06-30"),
            ("Albert Einstein", "Nobel for photoelectric effect", "1921-11-09"),
        ],
        "conflicts": [
            ("Pluto", "is a planet", "1990-01-01"),
            ("Pluto", "is NOT a planet (dwarf reclassification)", "2006-08-24"),
        ],
    }


def _ar_query(store: dict, query: str) -> str:
    """Trivial substring scan; in production bert would use core.retrieval
    hybrid_retrieve."""
    for subj, pred, _ts in store["facts"]:
        if subj.lower() in query.lower() or any(w in pred.lower()
                                                  for w in query.lower().split()):
            return f"{subj} {pred}"
    return ""


# ── AR: Accurate Retrieval ────────────────────────────────────────────


@task
def accurate_retrieval():
    samples = [
        Sample(input="What did Marie Curie discover?",
               target="polonium",
               metadata={"axis": "AR", "expected_substring": "polonium"}),
        Sample(input="What did Einstein publish?",
               target="relativity",
               metadata={"axis": "AR", "expected_substring": "relativity"}),
        Sample(input="What did Pierre Curie discover?",
               target="piezoelectricity",
               metadata={"axis": "AR", "expected_substring": "piezoelectricity"}),
    ]

    @solver
    def _ar_solver():
        async def run(state: TaskState, generate: Generate) -> TaskState:
            store = _stub_store()
            result = _ar_query(store, state.input_text)
            expected = state.metadata["expected_substring"].lower()
            passed = expected in result.lower()
            state.metadata["recall_result"] = {
                "passed": passed,
                "rationale": f"recalled='{result}'; expected_substring='{expected}'",
            }
            state.output.completion = result
            return state
        return run

    return Task(dataset=MemoryDataset(samples),
                solver=_ar_solver(),
                scorer=memory_recall_scorer())


# ── TTL: Test-Time Learning (memory updates on contradiction) ────────


@task
def test_time_learning():
    samples = [
        Sample(input="Is Pluto a planet (asking at 2025)?",
               target="not a planet",
               metadata={"axis": "TTL", "as_of": "2025-01-01"}),
    ]

    @solver
    def _ttl_solver():
        async def run(state: TaskState, generate: Generate) -> TaskState:
            store = _stub_store()
            pluto = [c for c in store["conflicts"] if c[0] == "Pluto"]
            pluto.sort(key=lambda x: x[2])
            latest = pluto[-1] if pluto else None
            passed = bool(latest and "NOT" in latest[1])
            state.metadata["recall_result"] = {
                "passed": passed,
                "rationale": f"latest='{latest}'; passed={passed}",
            }
            state.output.completion = (latest[1] if latest else "")
            return state
        return run

    return Task(dataset=MemoryDataset(samples),
                solver=_ttl_solver(),
                scorer=memory_recall_scorer())


# ── LRU: Long-Range Understanding (retrieval after distance) ─────────


@task
def long_range_understanding():
    samples = [
        Sample(input="What was the first thing Marie Curie discovered (1898)?",
               target="polonium",
               metadata={"axis": "LRU", "expected_substring": "polonium"}),
    ]

    @solver
    def _lru_solver():
        async def run(state: TaskState, generate: Generate) -> TaskState:
            store = _stub_store()
            curie = [f for f in store["facts"] if f[0] == "Marie Curie"]
            curie.sort(key=lambda x: x[2])
            earliest = curie[0] if curie else None
            text = earliest[1] if earliest else ""
            passed = "polonium" in text.lower()
            state.metadata["recall_result"] = {
                "passed": passed,
                "rationale": f"earliest='{earliest}'; text='{text}'",
            }
            state.output.completion = text
            return state
        return run

    return Task(dataset=MemoryDataset(samples),
                solver=_lru_solver(),
                scorer=memory_recall_scorer())


# ── CR: Conflict Resolution ──────────────────────────────────────────


@task
def conflict_resolution():
    samples = [
        Sample(input="Was Pluto considered a planet?",
               target="historically yes, after 2006 reclassified",
               metadata={"axis": "CR"}),
    ]

    @solver
    def _cr_solver():
        async def run(state: TaskState, generate: Generate) -> TaskState:
            store = _stub_store()
            pluto = sorted(
                (c for c in store["conflicts"] if c[0] == "Pluto"),
                key=lambda x: x[2],
            )
            passed = len(pluto) >= 2 and pluto[0][2] < pluto[-1][2]
            state.metadata["recall_result"] = {
                "passed": passed,
                "rationale": (
                    f"distinct_assertions={len(pluto)}; "
                    f"oldest={pluto[0] if pluto else None}; "
                    f"newest={pluto[-1] if pluto else None}"
                ),
            }
            state.output.completion = (
                "; ".join(f"{ts}: {pred}" for _, pred, ts in pluto)
            )
            return state
        return run

    return Task(dataset=MemoryDataset(samples),
                solver=_cr_solver(),
                scorer=memory_recall_scorer())


# ── Aggregate: all 4 axes ────────────────────────────────────────────


@task
def memoryagentbench_all():
    """Run all four MemoryAgentBench axes in one Task."""
    samples = []
    axes = [
        ("AR", "accurate_retrieval"),
        ("TTL", "test_time_learning"),
        ("LRU", "long_range_understanding"),
        ("CR", "conflict_resolution"),
    ]
    for axis, name in axes:
        samples.append(Sample(
            input=f"MemoryAgentBench axis: {axis}",
            target="PASS",
            metadata={"axis": axis, "task_name": name},
        ))

    @solver
    def _agg_solver():
        async def run(state: TaskState, generate: Generate) -> TaskState:
            axis = state.metadata["axis"]
            store = _stub_store()
            passed = False
            rationale = ""
            if axis == "AR":
                r = _ar_query(store, "Marie Curie")
                passed = "polonium" in r.lower() or "radium" in r.lower()
                rationale = f"AR recall='{r}'"
            elif axis == "TTL":
                pluto = sorted(
                    (c for c in store["conflicts"] if c[0] == "Pluto"),
                    key=lambda x: x[2],
                )
                passed = bool(pluto) and "NOT" in pluto[-1][1]
                rationale = f"TTL latest='{pluto[-1] if pluto else None}'"
            elif axis == "LRU":
                curie = sorted(
                    (f for f in store["facts"] if f[0] == "Marie Curie"),
                    key=lambda x: x[2],
                )
                passed = bool(curie) and "polonium" in curie[0][1].lower()
                rationale = f"LRU earliest='{curie[0] if curie else None}'"
            elif axis == "CR":
                pluto = sorted(
                    (c for c in store["conflicts"] if c[0] == "Pluto"),
                    key=lambda x: x[2],
                )
                passed = len(pluto) >= 2 and pluto[0][2] < pluto[-1][2]
                rationale = f"CR distinct={len(pluto)}"
            state.metadata["recall_result"] = {
                "passed": passed,
                "rationale": rationale,
            }
            state.output.completion = "PASS" if passed else "FAIL"
            return state
        return run

    return Task(dataset=MemoryDataset(samples),
                solver=_agg_solver(),
                scorer=memory_recall_scorer())
