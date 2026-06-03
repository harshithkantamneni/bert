"""Smoke test for core/concern_flow.py — concern lifecycle event emission.

Verifies that concern_raised / concern_propagated / concern_addressed /
revival_proposed events fire correctly, and that derive_concern_id is
deterministic.

Tests:
  1. derive_concern_id is stable across identical inputs
  2. derive_concern_id changes with text or source change
  3. emit_concerns_raised_from_packet returns ids + writes JSONL
  4. emit_concerns_raised skips non-APPROVE_WITH_CAVEATS verdicts
  5. emit_concern_propagated writes one event with cycle_distance
  6. emit_concern_addressed writes one event with resolution_verdict
  7. emit_revival_proposed writes one event with seasoning_id

Run: `.venv/bin/python tests/_smoke_concern_lifecycle.py`
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import concern_flow, observability  # noqa: E402


def _isolated_obs() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="bert_lifecycle_"))
    observability.OBS_DIR = tmp
    return tmp


def _read(tmp: Path, event_class: str) -> list[dict]:
    p = tmp / f"{event_class}.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def test_derive_concern_id_stable() -> None:
    a = concern_flow.derive_concern_id("a concern", "researcher_C99")
    b = concern_flow.derive_concern_id("a concern", "researcher_C99")
    assert a == b
    assert a.startswith("c-")
    assert len(a) == 14  # "c-" + 12 hex chars


def test_derive_concern_id_differs_with_text_or_source() -> None:
    base = concern_flow.derive_concern_id("a concern", "researcher_C99")
    diff_text = concern_flow.derive_concern_id("a different concern", "researcher_C99")
    diff_src = concern_flow.derive_concern_id("a concern", "researcher_C100")
    assert base != diff_text
    assert base != diff_src


def test_emit_concerns_raised_from_packet() -> None:
    tmp = _isolated_obs()
    packet = {
        "role": "clearness_phase2", "cycle": 200,
        "verdict": "APPROVE_WITH_CAVEATS",
        "caveats_embedded": [
            {"text": "consider edge case X", "severity_grade": "med"},
            "the simpler string form too",
        ],
    }
    ids = concern_flow.emit_concerns_raised_from_packet(packet)
    assert len(ids) == 2
    events = _read(tmp, "concern_raised")
    assert len(events) == 2
    assert events[0]["source_cycle"] == 200
    assert events[0]["concern_id"] == ids[0]
    assert events[0]["severity_grade"] == "med"
    assert events[1]["severity_grade"] is None


def test_emit_concerns_raised_skips_non_caveats_verdicts() -> None:
    tmp = _isolated_obs()
    packet = {"role": "x", "cycle": 1, "verdict": "APPROVE", "caveats_embedded": ["x"]}
    ids = concern_flow.emit_concerns_raised_from_packet(packet)
    assert ids == []
    assert _read(tmp, "concern_raised") == []


def test_emit_concern_propagated() -> None:
    tmp = _isolated_obs()
    concern_flow.emit_concern_propagated(
        concern_id="c-abc123",
        target_dispatch_id="researcher_C100",
        target_cycle=100,
        cycle_distance=1,
    )
    events = _read(tmp, "concern_propagated")
    assert len(events) == 1
    assert events[0]["concern_id"] == "c-abc123"
    assert events[0]["cycle_distance"] == 1


def test_emit_concern_addressed() -> None:
    tmp = _isolated_obs()
    concern_flow.emit_concern_addressed(
        concern_id="c-def456",
        resolution_dispatch_id="evaluator_C103",
        resolution_cycle=103,
        cycle_distance=3,
        resolution_verdict="APPROVE",
    )
    events = _read(tmp, "concern_addressed")
    assert len(events) == 1
    assert events[0]["resolution_verdict"] == "APPROVE"
    assert events[0]["cycle_distance"] == 3


def test_emit_revival_proposed() -> None:
    tmp = _isolated_obs()
    concern_flow.emit_revival_proposed(
        seasoning_id="season-abc12345",
        proposer_dispatch_id="director_C150",
        proposer_cycle=150,
        reason="PI revisits framing",
    )
    events = _read(tmp, "revival_proposed")
    assert len(events) == 1
    assert events[0]["seasoning_id"] == "season-abc12345"
    assert events[0]["reason"] == "PI revisits framing"


def main() -> int:
    tests = [
        test_derive_concern_id_stable,
        test_derive_concern_id_differs_with_text_or_source,
        test_emit_concerns_raised_from_packet,
        test_emit_concerns_raised_skips_non_caveats_verdicts,
        test_emit_concern_propagated,
        test_emit_concern_addressed,
        test_emit_revival_proposed,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
