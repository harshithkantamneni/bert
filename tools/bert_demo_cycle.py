"""bert demo-cycle — single-cycle live runner for the investor demo.

Wraps run_falsifier_calibration._run_one_scenario for ONE scenario, so the
demo orchestrator can show a real autonomous cycle running on stage in
the 0:45–1:30 segment of the locked flight plan.

Use only inside the demo orchestrator (or for testing the live-cycle
path). Real model providers are required (Groq + NVIDIA + Mistral keys
in env). Without keys, the cycle aborts cleanly with a clear error so
the founder isn't stuck on stage.

Usage:
  .venv/bin/python tools/bert_demo_cycle.py
  .venv/bin/python tools/bert_demo_cycle.py --scenario 1 --cycle 999
  .venv/bin/python tools/bert_demo_cycle.py --dry-run    # validate plumbing
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

import tools.run_falsifier_calibration as fc  # noqa: E402

CANDLE = "\033[38;5;215m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _print(msg: str, color: str = "") -> None:
    print(f"{color}{msg}{RESET}", flush=True)


def _ensure_keys_present() -> tuple[bool, str]:
    """Returns (ok, reason). bert demo-cycle needs at least one real
    provider key. If none present, abort cleanly with a stage-safe
    message rather than letting the dispatches 401."""
    candidates = [
        "GROQ_API_KEY",
        "NVIDIA_API_KEY",
        "MISTRAL_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
    ]
    present = [k for k in candidates if os.environ.get(k)]
    if not present:
        return False, (
            "no provider keys in env (looked for "
            + ", ".join(candidates)
            + "). Set at least GROQ_API_KEY before running."
        )
    return True, f"providers present: {', '.join(present)}"


def run_demo_cycle(
    *,
    scenario_number: int = 1,
    cycle: int = 999,
    model: str | None = None,
    dry_run: bool = False,
) -> int:
    """Run one falsifier scenario as a live demo cycle.

    Returns shell exit code: 0 on success, 1 on partial-success (some
    dispatches failed), 2 on hard failure (no keys, scenario missing,
    catastrophic error). Stage-safe: every error is printed before exit.
    """
    model = model or fc.DEFAULT_MODEL

    _print("─" * 60, CANDLE)
    _print(f" bert demo-cycle — scenario {scenario_number}, cycle {cycle}", BOLD + CANDLE)
    _print("─" * 60, CANDLE)

    # Pre-flight: scenario exists
    scenarios = fc.parse_corpus()
    matches = [s for s in scenarios if s.number == scenario_number]
    if not matches:
        _print(f"[ABORT] scenario {scenario_number} not in corpus (have 1..{len(scenarios)})", YELLOW)
        return 2
    scenario = matches[0]
    _print(f"[scenario] S{scenario.number}: {scenario.title}", DIM)
    _print(f"[model]    {model}", DIM)

    if dry_run:
        _print(f"[dry-run]  plumbing OK; would fire 5 dispatches against {model}", YELLOW)
        _print("[dry-run]  exiting before any model call", YELLOW)
        return 0

    # Pre-flight: provider keys present
    keys_ok, reason = _ensure_keys_present()
    _print(f"[keys]     {reason}", DIM)
    if not keys_ok:
        _print(f"[ABORT] {reason}", YELLOW)
        return 2

    # Run the scenario
    _print("", "")
    _print("[live]     starting 5 dispatches — researcher → strategist → threshing → clearness×2", BOLD)
    _print("[live]     each dispatch hits the real router; expect 60–120s total", DIM)
    _print("", "")
    started = time.monotonic()
    try:
        run = fc._run_one_scenario(scenario, cycle=cycle, model=model)
    except Exception as exc:  # noqa: BLE001
        _print(f"[ERROR] _run_one_scenario raised: {type(exc).__name__}: {exc}", YELLOW)
        return 2
    elapsed = time.monotonic() - started

    # Report
    _print("─" * 60, CANDLE)
    if run.success:
        _print(f" cycle {cycle} — SUCCESS in {elapsed:.1f}s", BOLD + GREEN)
    else:
        n_invalid = sum(1 for d in run.dispatches if not d.get("result_valid"))
        _print(f" cycle {cycle} — PARTIAL in {elapsed:.1f}s ({n_invalid}/5 invalid)", BOLD + YELLOW)
    _print("─" * 60, CANDLE)

    for d in run.dispatches:
        verdict = d.get("verdict", "—")
        label = d.get("label", "?")
        ok = "✓" if d.get("result_valid") else "✗"
        col = GREEN if d.get("result_valid") else YELLOW
        _print(f"  {col}{ok}{RESET}  {label:20}  verdict={verdict}", "")

    return 0 if run.success else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", type=int, default=1,
                    help="Scenario number from falsifier_corpus.md (1..10).")
    ap.add_argument("--cycle", type=int, default=999,
                    help="Cycle id to use for this demo run.")
    ap.add_argument("--model", default=None,
                    help=f"Default: {fc.DEFAULT_MODEL}")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate plumbing without firing any dispatches.")
    args = ap.parse_args()

    try:
        rc = run_demo_cycle(
            scenario_number=args.scenario,
            cycle=args.cycle,
            model=args.model,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        _print("\n[INTERRUPTED] demo cycle cancelled", YELLOW)
        return 130

    return rc


if __name__ == "__main__":
    sys.exit(main())
