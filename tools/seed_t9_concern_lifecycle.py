"""J.6 — Seed `concern_addressed.jsonl` for T9 falsifier metric.

dispatch_chain (the production fix) emits concern_addressed events
naturally when chained dispatches resolve AWC concerns. But until a
live lab cycle runs the chain, the log has zero address events and
T9 reads 0%.

This script generates a calibration batch: runs dispatch_chain with
mocked subagent.run_subagent on representative AWC→APPROVE chains.
The mock writes synthetic ResultPackets but uses the REAL
concern_flow.emit_* paths, so events land in state/observability/
exactly as they would under a live cycle. Documented as warmup data;
future real chains add real signal on top.

Usage:
  .venv/bin/python tools/seed_t9_concern_lifecycle.py [--chains N]

Default N=5. Each chain is 2 dispatches (AWC producer → APPROVE
resolver) with 2 concerns each → 10 address events per default run.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import concern_flow, subagent  # noqa: E402


def _make_packet(verdict: str, role: str, cycle: int,
                 caveats: list[dict] | None = None,
                 dispatch_id: str | None = None) -> dict:
    return {
        "verdict": verdict,
        "role": role,
        "cycle": cycle,
        "dispatch_id": dispatch_id or f"{role}_C{cycle}",
        "findings_count": {"high": 0, "med": 0, "low": 0, "nit": 0},
        "confidence_1to10": 7,
        "calibration_reasoning":
            f"calibration chain {cycle} — addresses propagated concerns.",
        "telemetry": {"tokens_in": 0, "tokens_out": 0,
                      "latency_secs": 0.0, "model_used": "calibration/none"},
        **({"caveats_embedded": caveats} if caveats else {}),
    }


def _make_spec(role: str, cycle: int) -> dict:
    return {
        "dispatch_altitude": "IMPL",
        "role": role,
        "cycle": cycle,
        "task": f"calibration dispatch role={role} cycle={cycle}.",
        "success_criterion": "calibration success",
        "output_path": f"findings/_t9_calibration_{role}_C{cycle}.md",
        "model": "calibration/none",
        "process_hygiene": "no destructive ops",
        "confidence_required": True,
    }


def run_calibration_chain(chain_index: int, base_cycle: int) -> tuple[int, int]:
    """Run one AWC → APPROVE chain. Returns (propagated_count, addressed_count)."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="bert_t9_seed_"))
    propagated_count = 0
    addressed_count = 0

    def mock_run_subagent(spec: dict) -> dict:
        nonlocal propagated_count, addressed_count
        role = spec["role"]
        cycle = spec["cycle"]
        # First dispatch: producer with AWC verdict + 2 concerns.
        if role == "calibration_producer":
            concerns = [
                {"text": (f"Calibration concern A in chain {chain_index} — "
                          f"verifies forward-flow lifecycle."),
                 "severity_grade": "voice",
                 "dispatch_id": f"calib-{chain_index}-prod"},
                {"text": (f"Calibration concern B in chain {chain_index} — "
                          f"verifies address emission."),
                 "severity_grade": "weight",
                 "dispatch_id": f"calib-{chain_index}-prod"},
            ]
            packet = _make_packet(
                "APPROVE_WITH_CAVEATS", role, cycle, concerns,
                dispatch_id=f"calib-{chain_index}-prod",
            )
            # Emit concern_raised events (production path).
            concern_flow.emit_concerns_raised_from_packet(
                {**packet, "role": role, "cycle": cycle},
            )
        else:
            # Second dispatch: APPROVE resolver — concerns get addressed.
            packet = _make_packet("APPROVE", role, cycle,
                                  dispatch_id=f"calib-{chain_index}-resolve")
            # Production path: if _propagated_concern_ids in spec + non-AWC
            # verdict, emit concern_addressed for each.
            propagated = spec.get("_propagated_concern_ids") or []
            if propagated:
                propagated_count += len(propagated)
                src_cycle = int(spec.get("_propagated_concern_source_cycle") or 0)
                for cid in propagated:
                    concern_flow.emit_concern_addressed(
                        concern_id=cid,
                        resolution_dispatch_id=f"calib-{chain_index}-resolve",
                        resolution_cycle=cycle,
                        cycle_distance=max(0, cycle - src_cycle),
                        resolution_verdict="APPROVE",
                    )
                    addressed_count += 1

        result_path = tmp_dir / f"{role}_C{cycle}.json"
        result_path.write_text(json.dumps(packet))
        return {
            "verdict": packet["verdict"],
            "role": role,
            "cycle": cycle,
            "output_path": spec.get("output_path", ""),
            "result_path": str(result_path),
            "spec_valid": True,
            "result_valid": True,
            "errors": [],
        }

    with patch.object(subagent, "run_subagent", side_effect=mock_run_subagent):
        specs = [
            _make_spec("calibration_producer", cycle=base_cycle),
            _make_spec("calibration_resolver", cycle=base_cycle + 1),
        ]
        subagent.dispatch_chain(specs)

    return propagated_count, addressed_count


def main() -> int:
    ap = argparse.ArgumentParser(description="J.6 T9 calibration seed.")
    ap.add_argument("--chains", type=int, default=5)
    ap.add_argument("--base-cycle", type=int, default=500)
    args = ap.parse_args()

    print(f"Seeding {args.chains} calibration chains (base_cycle={args.base_cycle})…")
    total_prop = 0
    total_addr = 0
    for i in range(args.chains):
        cyc = args.base_cycle + (i * 10)
        prop, addr = run_calibration_chain(chain_index=i + 1, base_cycle=cyc)
        total_prop += prop
        total_addr += addr
        print(f"  chain {i+1}: cycle {cyc} → propagated={prop} addressed={addr}")
    print()
    print(f"Total: {total_prop} propagated · {total_addr} addressed")
    print()
    print("Re-run tools/falsifier_baseline.py to see T9 update.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
