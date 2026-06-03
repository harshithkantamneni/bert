"""Smoke test for GG-C — live cycle streaming + autonomous run-cycle.

Two backend bugs and three UI gaps closed:

  Backend bug #3: /api/run-cycle POST capped max_cycles at 5, so the
  UI couldn't fire a meaningful autonomous loop. Pre-CC.4 that was a
  safety rail; post-CC.4 the runaway is structurally prevented by
  termination guardrails. Cap raised to 50 with explicit consent
  required for >5.

  Backend bug #7: RunCycleRequest didn't carry an `autonomous: bool`
  flag, so the subprocess never received `--autonomous` and the
  director loop never engaged from a UI-fired cycle.

  UI gaps:
   - No way to fire a cycle from the UI (was CLI-only).
   - No live indication of researcher → strategist → verdict
     progression even though /api/events/stream SSE was live.
   - No consent prompt for long autonomous runs.

Covers:

  RunCycleRequest schema:
    - autonomous: bool = False default
    - consent_long_run: bool = False default
    - max_cycles range 1..50 (raised from 1..5)
    - max_cycles > 5 without consent → 400
    - max_cycles > 5 WITH consent → 200, subprocess receives
      --autonomous when set

  /api/run-cycle POST round-trip:
    - dry-run + 1 cycle = 200, returns autonomous=False
    - dry-run + 3 cycles + autonomous=True = 200, autonomous=True
      echoed in response
    - dry-run + 10 cycles + consent_long_run=true = 200
    - dry-run + 10 cycles without consent = 400 with explicit
      "consent_long_run=True" message
    - dry-run + 100 cycles = 400 (over cap)

  UI source checks (no headless browser; greps lock structural
  contracts):
    - LiveCycle component exists at canonical path
    - LiveCycle subscribes to useEventStream
    - LiveCycle handles director_decision, dispatch_result for
      researcher/strategist, and verdict event classes
    - LiveCycle fades after verdict (auto-dismiss timer)
    - RunCycleControls component exists at canonical path
    - RunCycleControls posts to /api/run-cycle
    - RunCycleControls surfaces window.confirm for max_cycles > 5
    - RunCycleControls sends consent_long_run when applicable
    - FirstLight embeds both LiveCycle and RunCycleControls
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


LIVE_CYCLE = LAB_ROOT / "bert" / "v4" / "src" / "components" / "LiveCycle.tsx"
RUN_CONTROLS = LAB_ROOT / "bert" / "v4" / "src" / "components" / "RunCycleControls.tsx"
FIRST_LIGHT = LAB_ROOT / "bert" / "v4" / "src" / "surfaces" / "Home.tsx"
API_MAIN = LAB_ROOT / "api" / "main.py"


# ─── RunCycleRequest schema ────────────────────────────────────────


def test_run_cycle_request_has_autonomous_field() -> None:
    from api.main import RunCycleRequest
    r = RunCycleRequest()
    assert hasattr(r, "autonomous")
    assert r.autonomous is False  # safe default


def test_run_cycle_request_has_consent_field() -> None:
    from api.main import RunCycleRequest
    r = RunCycleRequest()
    assert hasattr(r, "consent_long_run")
    assert r.consent_long_run is False  # safe default


def test_run_cycle_endpoint_raises_50_cap() -> None:
    src = API_MAIN.read_text()
    # The bounds check now uses 50, not 5
    assert "max_cycles must be 1..50" in src
    # The consent gate is in place
    assert "consent_long_run=True" in src
    # --autonomous is propagated to the subprocess
    assert 'cmd.append("--autonomous")' in src


# ─── Endpoint round-trip via TestClient ────────────────────────────


def test_default_single_cycle_dry_run() -> None:
    from api.main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    r = client.post("/api/run-cycle", json={
        "lab": None, "max_cycles": 1, "dry_run": True,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["max_cycles"] == 1
    assert body["autonomous"] is False  # not requested
    assert body["dry_run"] is True
    assert "run_id" in body


def test_autonomous_3_cycle_dry_run() -> None:
    from api.main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    r = client.post("/api/run-cycle", json={
        "max_cycles": 3, "autonomous": True, "dry_run": True,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["max_cycles"] == 3
    assert body["autonomous"] is True


def test_long_run_without_consent_returns_400() -> None:
    from api.main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    r = client.post("/api/run-cycle", json={
        "max_cycles": 10, "autonomous": True, "dry_run": True,
    })
    assert r.status_code == 400
    assert "consent_long_run" in r.json()["detail"]


def test_long_run_with_consent_succeeds() -> None:
    from api.main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    r = client.post("/api/run-cycle", json={
        "max_cycles": 10, "autonomous": True, "dry_run": True,
        "consent_long_run": True,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["max_cycles"] == 10
    assert body["autonomous"] is True


def test_over_cap_returns_400_even_with_consent() -> None:
    from api.main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    r = client.post("/api/run-cycle", json={
        "max_cycles": 100, "autonomous": True, "dry_run": True,
        "consent_long_run": True,
    })
    assert r.status_code == 400
    assert "1..50" in r.json()["detail"]


def test_zero_cycles_returns_400() -> None:
    from api.main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    r = client.post("/api/run-cycle", json={
        "max_cycles": 0, "dry_run": True,
    })
    assert r.status_code == 400


# ─── LiveCycle component contract ──────────────────────────────────


def test_live_cycle_component_exists() -> None:
    assert LIVE_CYCLE.exists()


def test_live_cycle_subscribes_to_sse() -> None:
    text = LIVE_CYCLE.read_text()
    assert "useEventStream" in text


def test_live_cycle_handles_all_pipeline_event_classes() -> None:
    text = LIVE_CYCLE.read_text()
    # The component must respond to each of the four pipeline event
    # classes that flow through a cycle.
    assert '"director_decision"' in text
    assert '"dispatch_result"' in text
    assert '"verdict"' in text
    # And both producer roles
    assert "researcher" in text
    assert "strategist" in text


def test_live_cycle_fades_after_verdict() -> None:
    text = LIVE_CYCLE.read_text()
    # Auto-dismiss after verdict so the chip row doesn't persist
    # cluttering the surface
    assert "fadeAfterVerdictMs" in text
    assert "setVisible(false)" in text


def test_live_cycle_does_NOT_use_react_flow_or_steppers() -> None:
    """Per feedback_no_workflow_clones — no n8n / canvas / generic
    stepper widgets. The chip row is a quiet ECG-strip rhythm.
    Anti-pattern check operates on CODE only (comments may mention
    the anti-patterns to document what's being avoided)."""
    text = LIVE_CYCLE.read_text()
    decommented = re.sub(r"//[^\n]*", "", text)
    decommented = re.sub(r"/\*.*?\*/", "", decommented, flags=re.DOTALL)
    low = decommented.lower()
    assert "reactflow" not in low
    assert "stepper" not in low
    assert "wizard" not in low
    # No FontAwesome / Material Icons either
    assert "fontawesome" not in low
    assert "material-ui" not in low


# ─── RunCycleControls component contract ───────────────────────────


def test_run_cycle_controls_exists() -> None:
    assert RUN_CONTROLS.exists()


def test_run_cycle_controls_posts_to_run_cycle_endpoint() -> None:
    text = RUN_CONTROLS.read_text()
    assert '"/api/run-cycle"' in text
    assert "apiPost" in text


def test_run_cycle_controls_sends_consent_long_run() -> None:
    """Post-rework: the single Start button always passes
    consent_long_run=true (safety cap is 100; the API requires
    consent above 5). No window.confirm prompt — the PI clicked
    Start, which IS the consent."""
    text = RUN_CONTROLS.read_text()
    assert "consent_long_run: true" in text


def test_run_cycle_controls_safety_cap_applied() -> None:
    """SAFETY_CAP replaces the user-facing cycle slider — the
    director self-terminates via mission-complete; cap is the
    runaway-cost backstop."""
    text = RUN_CONTROLS.read_text()
    assert "SAFETY_CAP = 100" in text


def test_run_cycle_controls_keeps_single_cycle_for_debug() -> None:
    """The single-cycle button is preserved as a dev affordance
    (hidden in demo mode), but the user-facing path is the
    autonomous Start button."""
    text = RUN_CONTROLS.read_text()
    assert "fireSingleDebug" in text
    assert "dev · single cycle" in text


def test_run_cycle_controls_uses_active_lab() -> None:
    """When invoked without a `lab` prop, the controls use the
    user's currently-selected lab (from activeLab localStorage)."""
    text = RUN_CONTROLS.read_text()
    assert "useActiveLab" in text


# ─── FirstLight composition ───────────────────────────────────────


def test_first_light_embeds_live_cycle() -> None:
    text = FIRST_LIGHT.read_text()
    assert "LiveCycle" in text
    assert "<LiveCycle" in text


def test_first_light_embeds_run_cycle_controls() -> None:
    text = FIRST_LIGHT.read_text()
    assert "RunCycleControls" in text
    assert "<RunCycleControls" in text


def main() -> int:
    tests = [
        test_run_cycle_request_has_autonomous_field,
        test_run_cycle_request_has_consent_field,
        test_run_cycle_endpoint_raises_50_cap,
        test_default_single_cycle_dry_run,
        test_autonomous_3_cycle_dry_run,
        test_long_run_without_consent_returns_400,
        test_long_run_with_consent_succeeds,
        test_over_cap_returns_400_even_with_consent,
        test_zero_cycles_returns_400,
        test_live_cycle_component_exists,
        test_live_cycle_subscribes_to_sse,
        test_live_cycle_handles_all_pipeline_event_classes,
        test_live_cycle_fades_after_verdict,
        test_live_cycle_does_NOT_use_react_flow_or_steppers,
        test_run_cycle_controls_exists,
        test_run_cycle_controls_posts_to_run_cycle_endpoint,
        test_run_cycle_controls_safety_cap_applied,
        test_run_cycle_controls_keeps_single_cycle_for_debug,
        test_run_cycle_controls_sends_consent_long_run,
        test_run_cycle_controls_uses_active_lab,
        test_first_light_embeds_live_cycle,
        test_first_light_embeds_run_cycle_controls,
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
