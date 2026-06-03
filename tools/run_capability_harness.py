"""Capability harness runner — measures per-role per-model scores.

Per FINAL_implementation_plan_amendment_2026-05-13.md §A4 L-24.

Phase 1 deliverable. Writes one row per `{model, role, ts}` to
`lab/state/capability_matrix.jsonl`. The core/router.py and
core/subagent.py:pick_evaluator_model consult this file for first-
attempt routing.

The Phase 1 implementation seeds the matrix with the R12 May-2026
heuristic baseline so that the router has *some* signal before the
weekly cron has had a chance to run. As `evals/role_capability/*.py`
batteries are wired up (Phase 2 — typically operational, not build),
this runner orchestrates them and writes the measured rows over the
seed rows.

Usage:

  python tools/run_capability_harness.py                 # run all roles
  python tools/run_capability_harness.py --role evaluator # one role
  python tools/run_capability_harness.py --seed          # write R12 baseline only
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import capability_matrix  # noqa: E402

# R12 May-2026 heuristic baseline. One row per (role, provider, model)
# representing the heuristic capability estimate from the FINAL plan §6.1
# 5-tier cascade. Score is 0-1, derived from R12 validation notes
# (provider freshness, model size, role-stage match).
#
# When the Phase 2 batteries land, weekly cron overwrites these with
# measured scores. Until then this seed is the matrix.
R12_BASELINE: list[dict] = [
    # ── Director — top-of-stack reasoning, cross-domain orchestration
    {"role": "director", "provider": "nvidia",  "model": "meta/llama-3.3-70b-instruct", "score": 0.85, "headroom": 90},
    {"role": "director", "provider": "groq",    "model": "llama-3.3-70b-versatile",     "score": 0.83, "headroom": 70},
    {"role": "director", "provider": "gemini",  "model": "gemini-2.5-flash",            "score": 0.78, "headroom": 80},

    # ── Researcher — deep reading, citation tracking
    {"role": "researcher", "provider": "nvidia",     "model": "meta/llama-3.3-70b-instruct", "score": 0.82, "headroom": 90},
    {"role": "researcher", "provider": "openrouter", "model": "deepseek/deepseek-chat-v3:free", "score": 0.80, "headroom": 50},
    {"role": "researcher", "provider": "gemini",     "model": "gemini-2.5-flash",            "score": 0.78, "headroom": 80},

    # ── Strategist — option generation, ranking
    {"role": "strategist", "provider": "nvidia",  "model": "meta/llama-3.3-70b-instruct", "score": 0.79, "headroom": 90},
    {"role": "strategist", "provider": "groq",    "model": "llama-3.3-70b-versatile",     "score": 0.77, "headroom": 70},

    # ── Implementer — fast code, structured output
    {"role": "implementer", "provider": "cerebras", "model": "llama3.1-8b",            "score": 0.72, "headroom": 95},
    {"role": "implementer", "provider": "groq",     "model": "llama-3.3-70b-versatile", "score": 0.81, "headroom": 70},
    {"role": "implementer", "provider": "nvidia",   "model": "meta/llama-3.3-70b-instruct", "score": 0.84, "headroom": 90},

    # ── Evaluator — cross-family adversarial; non-Llama families preferred
    {"role": "evaluator", "provider": "nvidia",     "model": "qwen/qwen3-next-80b-a3b-thinking", "score": 0.88, "headroom": 85},
    {"role": "evaluator", "provider": "openrouter", "model": "deepseek/deepseek-r1-0528:free",   "score": 0.86, "headroom": 50},
    {"role": "evaluator", "provider": "gemini",     "model": "gemini-2.5-flash",                  "score": 0.80, "headroom": 80},
    {"role": "evaluator", "provider": "mistral",    "model": "mistral-small-latest",              "score": 0.75, "headroom": 80},

    # ── Threshing — Quaker discernment stage 1 (initial winnowing)
    {"role": "threshing", "provider": "nvidia", "model": "meta/llama-3.3-70b-instruct", "score": 0.80, "headroom": 90},
    {"role": "threshing", "provider": "groq",   "model": "llama-3.3-70b-versatile",     "score": 0.78, "headroom": 70},

    # ── Clearness phase 1 — Quaker queries
    {"role": "clearness_phase1", "provider": "nvidia",  "model": "meta/llama-3.3-70b-instruct", "score": 0.78, "headroom": 90},
    {"role": "clearness_phase1", "provider": "mistral", "model": "mistral-small-latest",        "score": 0.74, "headroom": 80},

    # ── Clearness phase 2 — Quaker verdict + concerns
    {"role": "clearness_phase2", "provider": "mistral", "model": "mistral-small-latest",        "score": 0.79, "headroom": 80},
    {"role": "clearness_phase2", "provider": "nvidia",  "model": "qwen/qwen3-next-80b-a3b-thinking", "score": 0.82, "headroom": 85},
]


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def seed_baseline() -> int:
    """Write R12 baseline rows to lab/state/capability_matrix.jsonl.

    Returns rows written. Each row's reference_set is 'r12_baseline_2026-05'
    so the Phase 2 batteries can identify and overwrite them.
    """
    ts = now_iso()
    written = 0
    for entry in R12_BASELINE:
        row = capability_matrix.CapabilityRow(
            ts=ts,
            role=entry["role"],
            provider=entry["provider"],
            model=entry["model"],
            score=entry["score"],
            cost_per_task_usd=0.0,  # all free-tier providers
            latency_p50_ms=0,
            latency_p95_ms=0,
            quota_headroom_pct=entry["headroom"],
            task_count=0,  # baseline is a heuristic estimate, not a measured run
            reference_set="r12_baseline_2026-05",
            notes="R12 heuristic baseline; replaced by measured score after Phase 2 batteries run",
        )
        capability_matrix.append_row(row)
        written += 1
    return written


def run_battery(role: str, *, live: bool = False,
                providers: list[tuple[str, str]] | None = None,
                sample: int | None = None) -> int:
    """Run a measurement battery for one role (Phase 2).

    Imports evals/role_capability/<role>.py, dispatches its TASKS
    against every active model in `providers`, writes one row per
    (role, provider, model) to lab/state/capability_matrix.jsonl.

    providers: list of (provider, model) tuples; defaults to the seed
    set from R12_BASELINE for that role.
    live: True = actually call the provider API; False = use the
    deterministic offline scorer in _common._seeded_score so cron can
    warm the matrix structure when network is down.
    """
    try:
        mod = __import__(f"evals.role_capability.{role}",
                         fromlist=["run", "REFERENCE_SET"])
    except ImportError as e:
        print(f"  [skip] no battery for {role!r} ({e})")
        return 0

    # Default provider set: every (provider, model) in R12_BASELINE for this role
    if providers is None:
        providers = [
            (e["provider"], e["model"]) for e in R12_BASELINE if e["role"] == role
        ]
    if not providers:
        print(f"  [skip] no providers for {role!r}")
        return 0

    from evals.role_capability import _common
    written = 0
    for provider_name, model in providers:
        try:
            result = mod.run(provider_name, model, live=live, sample=sample)
            _common.write_matrix_row(result, reference_set=mod.REFERENCE_SET)
            written += 1
            print(f"  [{role}] {provider_name}/{model}: score={result.score:.3f} "
                  f"(n={result.task_count}, {'live' if live else 'offline'})")
        except Exception as e:  # noqa: BLE001
            print(f"  [{role}] {provider_name}/{model}: FAILED ({e})")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="L-24 capability harness")
    parser.add_argument("--role", help="run battery for one role only")
    parser.add_argument("--seed", action="store_true",
                        help="write R12 heuristic baseline rows (skips battery dispatch)")
    parser.add_argument("--live", action="store_true",
                        help="dispatch through core.provider.call (consumes quota); "
                             "default is offline deterministic scoring")
    parser.add_argument("--sample", type=int,
                        help="run only the first N tasks per battery (smoke / cron)")
    args = parser.parse_args()

    start = time.time()
    if args.seed:
        n = seed_baseline()
        print(f"seeded {n} R12 baseline rows to {capability_matrix.MATRIX_PATH.relative_to(LAB_ROOT)}")
        return 0

    roles = ([args.role] if args.role
             else ["researcher", "strategist", "evaluator", "implementer",
                   "threshing", "clearness_phase1", "clearness_phase2"])
    total = 0
    for role in roles:
        print(f"role={role}:")
        total += run_battery(role, live=args.live, sample=args.sample)
    elapsed = time.time() - start
    print(f"\n{total} rows written in {elapsed:.1f}s "
          f"({'live' if args.live else 'offline'} mode)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
