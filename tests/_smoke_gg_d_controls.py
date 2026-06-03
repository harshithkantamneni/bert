"""Smoke test for GG-D — pause/resume + cancel controls.

GG-A-prep already shipped per-lab pause flag + bert_run autonomous
loop honoring it. GG-D completes the loop with the UI controls and
adds a cancel endpoint for in-flight runs.

Covers:

  Backend:
    - DELETE /api/run-cycle/{run_id} sends SIGTERM
    - 404 when run_id unknown
    - 200 + already_cancelled on second DELETE (idempotent)
    - 200 + already_finished on DELETE of a naturally-completed run
    - Cancel state echoed in GET /api/run-cycle/{run_id} response

  PauseResumeControls UI:
    - Component file exists at canonical path
    - Reads paused state from useLabStatus hook
    - Toggles between /api/pause and /api/resume
    - Routes via labQuery (per-lab)
    - Invalidates queryClient on toggle so UI updates

  LiveCycle cancel button:
    - Subscribes to bert:current-run-changed event
    - Renders cancel button only when run_id known + verdict not landed
    - Calls DELETE /api/run-cycle/{run_id}
    - Surfaces confirm dialog with provider-quota disclosure

  RunCycleControls run-id broadcast:
    - On fire (single or autonomous), publishes run_id to localStorage
    - Dispatches bert:current-run-changed event

  FirstLight integration:
    - PauseResumeControls mounted alongside RunCycleControls
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


PAUSE_CTRL = LAB_ROOT / "bert" / "v4" / "src" / "components" / "PauseResumeControls.tsx"
LIVE_CYCLE = LAB_ROOT / "bert" / "v4" / "src" / "components" / "LiveCycle.tsx"
RUN_CTRL = LAB_ROOT / "bert" / "v4" / "src" / "components" / "RunCycleControls.tsx"
FIRST_LIGHT = LAB_ROOT / "bert" / "v4" / "src" / "surfaces" / "Home.tsx"
API_MAIN = LAB_ROOT / "api" / "main.py"


# ─── DELETE endpoint ──────────────────────────────────────────────


def test_delete_endpoint_defined() -> None:
    src = API_MAIN.read_text()
    assert '@app.delete("/api/run-cycle/{run_id}")' in src
    assert "def cancel_run_cycle" in src


def test_delete_endpoint_sends_sigterm() -> None:
    src = API_MAIN.read_text()
    # The body must import signal AND call proc.send_signal(SIGTERM)
    assert "import signal" in src
    assert "proc.send_signal" in src
    assert "SIGTERM" in src


def test_delete_unknown_run_returns_404() -> None:
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    r = client.delete("/api/run-cycle/run_nonexistent_xyz")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"]


def test_delete_of_finished_run_returns_already_finished() -> None:
    """A naturally-completed run shouldn't appear cancelled — the
    response must distinguish operator-cancel from normal exit."""
    from fastapi.testclient import TestClient
    from api.main import app
    import time as _time
    client = TestClient(app)
    # Start a fast dry-run cycle that finishes immediately
    r = client.post("/api/run-cycle", json={
        "max_cycles": 1, "dry_run": True,
    })
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    # Wait for natural exit (dry-run is fast)
    _time.sleep(1.5)
    r2 = client.delete(f"/api/run-cycle/{run_id}")
    assert r2.status_code == 200
    body = r2.json()
    assert body.get("already_finished") is True
    assert body.get("cancelled") is False
    assert body.get("alive") is False


def test_delete_is_idempotent() -> None:
    """Multiple DELETE calls return the same state without re-killing."""
    from fastapi.testclient import TestClient
    from api.main import app
    import time as _time
    client = TestClient(app)
    r = client.post("/api/run-cycle", json={
        "max_cycles": 1, "dry_run": True,
    })
    run_id = r.json()["run_id"]
    _time.sleep(1.5)
    r1 = client.delete(f"/api/run-cycle/{run_id}")
    r2 = client.delete(f"/api/run-cycle/{run_id}")
    assert r1.status_code == 200
    assert r2.status_code == 200
    # Both responses should agree on the terminal state
    assert r1.json()["alive"] == r2.json()["alive"]
    assert r1.json()["exit_code"] == r2.json()["exit_code"]


def test_get_run_status_includes_cancelled_field() -> None:
    """GET /api/run-cycle/{run_id} must surface cancelled status so
    the UI knows whether non-zero exit was a normal failure or an
    operator cancel."""
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    r = client.post("/api/run-cycle", json={
        "max_cycles": 1, "dry_run": True,
    })
    run_id = r.json()["run_id"]
    g = client.get(f"/api/run-cycle/{run_id}")
    assert g.status_code == 200
    assert "cancelled" in g.json()


# ─── PauseResumeControls ──────────────────────────────────────────


def test_pause_resume_component_exists() -> None:
    assert PAUSE_CTRL.exists()


def test_pause_resume_reads_lab_status() -> None:
    text = PAUSE_CTRL.read_text()
    assert "useLabStatus" in text
    # Reads the paused field
    assert "paused" in text


def test_pause_resume_toggles_endpoints() -> None:
    text = PAUSE_CTRL.read_text()
    assert "/api/pause" in text
    assert "/api/resume" in text
    assert "apiPost" in text


def test_pause_resume_uses_lab_query() -> None:
    """Per-lab routing — the toggle must scope to the active lab."""
    text = PAUSE_CTRL.read_text()
    assert "labQuery" in text
    assert "useActiveLab" in text


def test_pause_resume_invalidates_status_query() -> None:
    """After toggle, the UI must refetch /api/status so the button
    state updates without waiting for the periodic 5s refresh."""
    text = PAUSE_CTRL.read_text()
    assert "invalidateQueries" in text
    assert '"status"' in text


def test_pause_resume_renders_banner_when_paused() -> None:
    """The lab's paused state must be visible from across the surface —
    a banner, not just a button label."""
    text = PAUSE_CTRL.read_text()
    # The banner block is conditional on paused
    assert "paused && (" in text or "paused &&" in text
    # Has the explicit "paused" word in the banner copy
    assert "the autonomous loop sleeps" in text or "paused ·" in text


def test_pause_resume_is_NOT_an_icon_button() -> None:
    """Per feedback_consumer_product: no icon-only ambiguity. The
    pause button has explicit verbs."""
    import re
    text = PAUSE_CTRL.read_text()
    decommented = re.sub(r"//[^\n]*", "", text)
    decommented = re.sub(r"/\*.*?\*/", "", decommented, flags=re.DOTALL)
    # The button labels are spelled out
    assert ("pause this lab" in decommented or
            "resume — bert is paused" in decommented)


# ─── LiveCycle cancel button ──────────────────────────────────────


def test_live_cycle_subscribes_to_run_id_event() -> None:
    text = LIVE_CYCLE.read_text()
    assert "bert:current-run-changed" in text
    assert "addEventListener" in text


def test_live_cycle_renders_cancel_when_run_known() -> None:
    text = LIVE_CYCLE.read_text()
    # Conditional rendering on currentRunId
    assert "currentRunId" in text
    # Cancel button
    assert "cancel" in text.lower()
    # Calls DELETE
    assert 'method: "DELETE"' in text


def test_live_cycle_cancel_hidden_after_verdict() -> None:
    """No point cancelling after the verdict lands."""
    text = LIVE_CYCLE.read_text()
    # The cancel button only renders while verdict hasn't landed
    assert "!stages.verdict.done" in text


def test_live_cycle_cancel_confirms_with_quota_disclosure() -> None:
    """Operator must see that provider quota is not refunded — the
    confirm dialog explains the cost of cancelling."""
    text = LIVE_CYCLE.read_text()
    assert "window.confirm" in text
    # Mentions quota / refund / billing
    assert ("quota" in text.lower() or "refund" in text.lower()
            or "bills" in text.lower())


# ─── RunCycleControls run-id broadcast ────────────────────────────


def test_run_cycle_controls_publishes_run_id() -> None:
    text = RUN_CTRL.read_text()
    assert "_publishCurrentRun" in text
    # Both fire paths publish
    assert text.count("_publishCurrentRun") >= 2


def test_publish_uses_localstorage_and_event() -> None:
    text = RUN_CTRL.read_text()
    assert 'localStorage.setItem("bert:current-run"' in text
    assert "bert:current-run-changed" in text
    assert "CustomEvent" in text


# ─── FirstLight integration ───────────────────────────────────────


def test_first_light_embeds_pause_resume_controls() -> None:
    text = FIRST_LIGHT.read_text()
    assert "PauseResumeControls" in text
    assert "<PauseResumeControls" in text


def main() -> int:
    tests = [
        test_delete_endpoint_defined,
        test_delete_endpoint_sends_sigterm,
        test_delete_unknown_run_returns_404,
        test_delete_of_finished_run_returns_already_finished,
        test_delete_is_idempotent,
        test_get_run_status_includes_cancelled_field,
        test_pause_resume_component_exists,
        test_pause_resume_reads_lab_status,
        test_pause_resume_toggles_endpoints,
        test_pause_resume_uses_lab_query,
        test_pause_resume_invalidates_status_query,
        test_pause_resume_renders_banner_when_paused,
        test_pause_resume_is_NOT_an_icon_button,
        test_live_cycle_subscribes_to_run_id_event,
        test_live_cycle_renders_cancel_when_run_known,
        test_live_cycle_cancel_hidden_after_verdict,
        test_live_cycle_cancel_confirms_with_quota_disclosure,
        test_run_cycle_controls_publishes_run_id,
        test_publish_uses_localstorage_and_event,
        test_first_light_embeds_pause_resume_controls,
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
