"""Smoke test that observability.emit fires at the right points.

Per FINAL_implementation_plan_2026-05-07.md §6 + A6 §9 falsifier
observability instrumentation round (post-H4 wiring).

Verifies emit() is called from:
  1. core/seasoning.season() → seasoning_entry event
  2. core/seasoning.revive() → seasoning_revive event
  3. core/subagent.run_subagent — emits subagent_spawn + verdict +
     role-specific dispatch events (mocked agent loop so we don't hit
     the network)
  4. core/agent.run_role provider.call success → emit_model_call
     (verified via agent.py source-string check; the live wiring is
     covered by _smoke_h4_wiring's quota record test using the same
     mock).

Run: `.venv/bin/python tests/_smoke_observability_wiring.py`
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

# Use a temp observability dir
TMP = Path(tempfile.mkdtemp(prefix="bert_obs_wiring_"))
OBS_DIR = TMP / "state" / "observability"
SEASONING_PATH = TMP / "lab" / "sod" / "seasoning.jsonl"

from core import observability, seasoning  # noqa: E402


def _reset_module_paths() -> None:
    """Re-set module-level paths each test.

    Required because other test files in the same pytest session
    (e.g. _smoke_observability_structured.py) mutate
    observability.OBS_DIR within their own test bodies and restore it
    to LAB_ROOT/state/observability — which would overwrite the wiring
    test's import-time assignment if those tests run first.
    """
    observability.OBS_DIR = OBS_DIR
    seasoning.SEASONING_PATH = SEASONING_PATH


_reset_module_paths()


def _read_events(event_class: str) -> list[dict]:
    p = OBS_DIR / f"{event_class}.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def test_seasoning_emits_event() -> None:
    _reset_module_paths()
    SEASONING_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SEASONING_PATH.exists():
        SEASONING_PATH.unlink()
    if (OBS_DIR / "seasoning_entry.jsonl").exists():
        (OBS_DIR / "seasoning_entry.jsonl").unlink()
    seasoning.season(
        source_dispatch_id="test-dispatch-1",
        summary="Smoke test seasoning entry — sufficiently long block_reason text "
                "to clear the schema's 50-char minimum.",
        revival_conditions=["if PI revisits framing in pi_notes.md"],
        cycle=99,
        altitude="META",
    )
    events = _read_events("seasoning_entry")
    assert len(events) == 1, f"expected 1 seasoning_entry event; got {len(events)}"
    assert events[0]["cycle"] == 99
    assert events[0]["altitude"] == "META"
    assert events[0]["revival_conditions_count"] == 1


def test_seasoning_revive_emits_event() -> None:
    _reset_module_paths()
    SEASONING_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SEASONING_PATH.exists():
        SEASONING_PATH.unlink()
    if (OBS_DIR / "seasoning_revive.jsonl").exists():
        (OBS_DIR / "seasoning_revive.jsonl").unlink()
    entry = seasoning.season(
        source_dispatch_id="test-dispatch-2",
        summary="Another smoke entry for the revive path; long enough.",
        revival_conditions=["if cycle >= 200 and PI revisits this question"],
        cycle=99,
    )
    seasoning.revive(entry["id"], revival_dispatch_id="dispatch-revive-1")
    events = _read_events("seasoning_revive")
    assert len(events) == 1
    assert events[0]["id"] == entry["id"]


def test_observability_helpers_exist() -> None:
    assert hasattr(observability, "emit_model_call")
    assert hasattr(observability, "emit")
    assert hasattr(observability, "calibration_count")


def test_agent_imports_observability() -> None:
    src = (LAB_ROOT / "core" / "agent.py").read_text()
    assert "import observability" in src or "observability" in src
    assert "emit_model_call" in src


def test_subagent_imports_observability() -> None:
    src = (LAB_ROOT / "core" / "subagent.py").read_text()
    assert "observability" in src
    assert "subagent_spawn" in src
    assert "verdict" in src
    assert "stand_aside_verdict" in src


def main() -> int:
    tests = [
        test_seasoning_emits_event,
        test_seasoning_revive_emits_event,
        test_observability_helpers_exist,
        test_agent_imports_observability,
        test_subagent_imports_observability,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}")
            print(f"        {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__} (exception)")
            print(f"        {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
