"""Smoke test for the autonomous-loop self-termination + UI rework.

Two halves:

  Director side — `mission-complete` cycle shape exists; bert_run
  handles it; the prompt teaches the director when to pick it.
  emit_mission_complete_event writes a top-level event so the UI
  can show a receipt.

  UI side — RunCycleControls is a single Start/Stop with a live
  cycle counter, cost delta, and a mission-complete receipt — no
  cycle slider, no max-cycles knob exposed to the user. Safety
  cap of 100 cycles applied behind the scenes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


DIRECTOR  = LAB_ROOT / "core" / "director.py"
PROMPT    = LAB_ROOT / "prompts" / "director_decision.md"
BERT_RUN  = LAB_ROOT / "tools" / "bert_run.py"
API       = LAB_ROOT / "api" / "main.py"
CTRL      = LAB_ROOT / "bert" / "v4" / "src" / "components" / "RunCycleControls.tsx"
CLIENT    = LAB_ROOT / "bert" / "v4" / "src" / "api" / "client.ts"


def _require(*paths: Path) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip("requires lab runtime artifact(s) not shipped in the public "
                    "repo: " + ", ".join(str(m) for m in missing))


# ─── Director: mission-complete cycle shape ───────────────────────


def test_director_has_mission_complete_shape() -> None:
    text = DIRECTOR.read_text()
    assert 'MISSION_COMPLETE = "mission-complete"' in text


def test_director_decision_has_is_complete_method() -> None:
    text = DIRECTOR.read_text()
    assert "def is_complete" in text
    assert "MISSION_COMPLETE.value" in text


def test_director_is_terminal_covers_mission_complete() -> None:
    """is_terminal must return True for BOTH idle and mission-complete
    so bert_run's existing break-on-terminal path covers both."""
    text = DIRECTOR.read_text()
    assert "CycleShape.IDLE.value" in text and \
           "CycleShape.MISSION_COMPLETE.value" in text


def test_director_termination_reason_includes_mission_complete() -> None:
    text = DIRECTOR.read_text()
    assert 'MISSION_COMPLETE = "mission_complete"' in text


def test_director_emits_mission_complete_event() -> None:
    text = DIRECTOR.read_text()
    assert "def emit_mission_complete_event" in text
    assert '"event_class": "mission_complete"' in text


# ─── Director prompt: teach the model when to pick it ────────────


def test_prompt_documents_mission_complete_shape() -> None:
    text = PROMPT.read_text()
    assert "`mission-complete`" in text
    # Allow either with-backticks or without around `mission_complete`
    assert "mission_complete` event" in text or "mission_complete event" in text


def test_prompt_lists_five_completion_gates() -> None:
    """Conservative: only pick mission-complete if ALL of the five
    gates hold (synthesis report, no falsifier regression, no
    objections, ≥3 cycles, named answer)."""
    text = PROMPT.read_text()
    assert "ALL of the following" in text or "ALL of the following are true" in text
    # The five gates are numbered 1-5
    for n in ("1.", "2.", "3.", "4.", "5."):
        assert n in text


def test_prompt_warns_against_premature_completion() -> None:
    text = PROMPT.read_text()
    assert "Don't pick `mission-complete`" in text


# ─── bert_run: handle the terminal mission-complete decision ──────


def test_bert_run_handles_mission_complete_decision() -> None:
    text = BERT_RUN.read_text()
    assert "is_complete()" in text
    assert "MISSION COMPLETE" in text
    assert "emit_mission_complete_event" in text


def test_bert_run_still_handles_idle_separately() -> None:
    """Backwards compat — idle still terminates and emits the
    DIRECTOR_IDLE termination reason."""
    text = BERT_RUN.read_text()
    assert "DIRECTOR_IDLE" in text


# ─── API: surface cycles_completed + mission_complete in status ──


def test_run_status_exposes_cycles_completed() -> None:
    _require(API)
    text = API.read_text()
    assert '"cycles_completed":' in text
    # Derive from stdout markers
    assert '"] ✓ success" in ln' in text


def test_run_status_exposes_mission_complete_flag() -> None:
    _require(API)
    text = API.read_text()
    assert '"mission_complete":' in text
    assert '"MISSION COMPLETE" in ln' in text


def test_client_type_includes_new_status_fields() -> None:
    _require(CLIENT)
    text = CLIENT.read_text()
    assert "cycles_completed?:" in text
    assert "mission_complete?:" in text


# ─── UI: Start/Stop autonomous controls ───────────────────────────


def test_controls_default_to_safety_cap_not_slider() -> None:
    """No cycle slider; the user just clicks Start. Cap matches
    the API's hard ceiling at 50."""
    _require(CTRL)
    text = CTRL.read_text()
    assert "SAFETY_CAP = 50" in text
    # No num input for cycle count
    assert 'type="number"' not in text
    # No "fire ${autonomousCount}" button
    assert "autonomousCount" not in text


def test_controls_render_start_mission_button() -> None:
    _require(CTRL)
    text = CTRL.read_text()
    assert "start mission" in text
    assert 'aria-label="start mission"' in text


def test_controls_render_stop_button_when_running() -> None:
    _require(CTRL)
    text = CTRL.read_text()
    assert 'aria-label="stop the autonomous loop"' in text
    assert 'method: "DELETE"' in text


def test_controls_poll_run_status() -> None:
    _require(CTRL)
    text = CTRL.read_text()
    assert "POLL_MS = 2000" in text
    assert "/api/run-cycle/${" in text


def test_controls_show_live_cycle_counter() -> None:
    _require(CTRL)
    text = CTRL.read_text()
    # The "running · cycle N" header uses phase.cycles
    assert "cycles_completed ?? 0" in text
    assert "running · cycle" in text


def test_controls_show_cost_delta() -> None:
    """Cost shown is the delta since this run started, not the
    daily total — so the PI sees what THIS mission cost."""
    _require(CTRL)
    text = CTRL.read_text()
    assert "costAtStart" in text
    assert "_costDelta" in text
    assert "spent · this run" in text


def test_controls_render_mission_complete_receipt() -> None:
    _require(CTRL)
    text = CTRL.read_text()
    assert "✓ mission complete" in text
    assert "phase.kind === \"complete\"" in text or \
           "kind: \"complete\"" in text


def test_controls_render_stopped_receipt() -> None:
    _require(CTRL)
    text = CTRL.read_text()
    assert "stopped" in text
    assert 'kind === "cancelled"' in text or 'kind: "cancelled"' in text


def test_controls_send_consent_long_run_true() -> None:
    """Safety cap = 100 implicitly requires consent_long_run; we
    send true so the API doesn't reject the run."""
    _require(CTRL)
    text = CTRL.read_text()
    assert "consent_long_run: true" in text


def test_controls_invalidate_caches_on_complete() -> None:
    """When a mission completes, the manuscript / findings / letter
    queries must refetch so the PI sees the new artifacts."""
    _require(CTRL)
    text = CTRL.read_text()
    for key in ('"events"', '"findings"', '"lab-status"', '"director-letter"'):
        assert f'queryKey: [{key}]' in text, f"missing invalidate for {key}"


def test_controls_dev_single_cycle_hidden_in_demo_mode() -> None:
    _require(CTRL)
    text = CTRL.read_text()
    assert "dev · single cycle" in text
    assert "!isDemo" in text


def main() -> int:
    tests = [
        # Director
        test_director_has_mission_complete_shape,
        test_director_decision_has_is_complete_method,
        test_director_is_terminal_covers_mission_complete,
        test_director_termination_reason_includes_mission_complete,
        test_director_emits_mission_complete_event,
        # Prompt
        test_prompt_documents_mission_complete_shape,
        test_prompt_lists_five_completion_gates,
        test_prompt_warns_against_premature_completion,
        # bert_run
        test_bert_run_handles_mission_complete_decision,
        test_bert_run_still_handles_idle_separately,
        # API
        test_run_status_exposes_cycles_completed,
        test_run_status_exposes_mission_complete_flag,
        test_client_type_includes_new_status_fields,
        # UI controls
        test_controls_default_to_safety_cap_not_slider,
        test_controls_render_start_mission_button,
        test_controls_render_stop_button_when_running,
        test_controls_poll_run_status,
        test_controls_show_live_cycle_counter,
        test_controls_show_cost_delta,
        test_controls_render_mission_complete_receipt,
        test_controls_render_stopped_receipt,
        test_controls_send_consent_long_run_true,
        test_controls_invalidate_caches_on_complete,
        test_controls_dev_single_cycle_hidden_in_demo_mode,
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
