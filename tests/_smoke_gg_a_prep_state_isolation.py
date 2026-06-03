"""Smoke test for GG-A-prep — multi-lab state isolation refactor.

The bug being fixed: every state write in api/main.py went to a
module-global STATE_DIR bound to the repo's lab/ directory. So
`/api/pause?lab=foo` paused the SUPERVISOR lab, not foo. Same for
steers, pins, suppressions, notes, approvals, vetoes, blessings.

This phase introduces StatePaths + _state_paths(lab) and threads
`lab: str | None = Query(None)` through every writing endpoint. The
default (no lab param) preserves pre-GG behavior — writes go to the
supervisor lab — so any existing caller continues to work.

The bert_run.py autonomous loop also gained a paused-flag check
between iterations (bug #2 fix promoted from GG-D since it shares
the isolation contract).

Covers:
  - StatePaths exports every per-lab state path bundle
  - _state_paths(None) → supervisor lab paths
  - _state_paths("nonexistent") → raises 404 (via _resolve_lab_path)
  - _load_overrides / _save_overrides accept a paths kwarg
  - Every writing endpoint accepts `lab:` Query param
  - Endpoint signatures contain `lab: str | None = Query(None)`
  - bert_run.py autonomous loop polls <lab_path>/state/paused
  - In-process TestClient: pause/resume with lab=None writes to
    supervisor; pause/resume with lab=<existing> writes to that lab's
    state dir
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


# ─── Module shape ──────────────────────────────────────────────────


def test_statepaths_class_exported() -> None:
    from api import main
    assert hasattr(main, "StatePaths")
    assert hasattr(main, "_state_paths")


def test_statepaths_has_all_state_fields() -> None:
    from api.main import StatePaths
    from pathlib import Path as _P
    sp = StatePaths(_P("/tmp/fake-lab"))
    for fname in ("state_dir", "pi_actions", "blessings", "vetoes",
                  "steers", "asks", "pi_overrides", "paused_flag",
                  "notes_dir", "approvals_dir", "letters_dir",
                  "voice_steers_dir", "dev_pending"):
        assert hasattr(sp, fname), f"StatePaths missing {fname!r}"
        # Every path must be under <lab_path>/state/
        assert "/tmp/fake-lab/state" in str(getattr(sp, fname))


def test_state_paths_lab_none_resolves_to_supervisor() -> None:
    from api.main import _state_paths, LAB_PATH
    sp = _state_paths(None)
    assert sp.lab_path == LAB_PATH
    assert sp.state_dir == LAB_PATH / "state"


def test_state_paths_nonexistent_lab_raises_404() -> None:
    from fastapi import HTTPException
    from api.main import _state_paths
    try:
        _state_paths("definitely-no-such-lab-xyz")
        raise AssertionError("expected HTTPException")
    except HTTPException as e:
        assert e.status_code == 404


# ─── Overrides helper signature ────────────────────────────────────


def test_load_overrides_accepts_paths_kwarg() -> None:
    from api.main import _load_overrides, _state_paths
    sp = _state_paths(None)
    # Should not raise; either returns existing or empty defaults
    overrides = _load_overrides(sp)
    assert isinstance(overrides, dict)
    assert "pinned" in overrides and "suppressed" in overrides


def test_save_overrides_writes_to_paths_pi_overrides() -> None:
    from api.main import _save_overrides
    from api.main import StatePaths
    from pathlib import Path as _P
    tmp = _P(tempfile.mkdtemp())
    try:
        sp = StatePaths(tmp)
        _save_overrides({"pinned": ["a"], "suppressed": ["b"]}, sp)
        assert sp.pi_overrides.exists()
        data = json.loads(sp.pi_overrides.read_text())
        assert data["pinned"] == ["a"]
        assert data["suppressed"] == ["b"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── Endpoint signatures now take `lab:` ───────────────────────────


def test_endpoints_take_lab_query_param() -> None:
    src = (LAB_ROOT / "api" / "main.py").read_text()
    # Each of these endpoint handlers must accept lab: str | None = Query
    endpoint_functions = [
        "def pause(", "def resume(", "def steer(", "def ask(",
        "def bless(", "def veto(", "def pin(", "def unpin(",
        "def suppress(", "def unsuppress(", "def approve(",
        "def write_note(", "def get_note(", "def list_asks_for(",
        "def list_approvals(", "def get_overrides(",
        "def list_pending(", "def get_latest_letter(",
    ]
    for fn in endpoint_functions:
        # Find the function declaration and check its signature span
        idx = src.find(fn)
        assert idx >= 0, f"endpoint {fn} not found"
        # Read until the closing paren of the signature
        sig_end = src.index("):", idx)
        sig = src[idx:sig_end]
        assert "lab: str | None = Query(None)" in sig, (
            f"endpoint {fn} missing `lab: str | None = Query(None)` "
            f"in signature: {sig!r}"
        )


# ─── bert_run.py honors paused flag ────────────────────────────────


def test_bert_run_polls_paused_flag_per_lab() -> None:
    src = (LAB_ROOT / "tools" / "bert_run.py").read_text()
    # The flag lives under <lab_path>/state/paused per the refactor
    assert 'paused_flag = lab_path / "state" / "paused"' in src
    # Loop polls with sleep cadence (don't busy-wait)
    assert "while paused_flag.exists():" in src
    # Honors Ctrl-C while paused
    assert 'interrupted["caught"]' in src
    # 5-second poll cadence
    assert "time.sleep(5)" in src


# ─── In-process roundtrip ──────────────────────────────────────────


def test_pause_resume_supervisor_via_testclient() -> None:
    from fastapi.testclient import TestClient
    from api.main import app, LAB_PATH
    client = TestClient(app)
    paused_flag = LAB_PATH / "state" / "paused"
    # Ensure clean start
    if paused_flag.exists():
        paused_flag.unlink()
    try:
        r = client.post("/api/pause", json={"reason": "smoke"})
        assert r.status_code == 200
        body = r.json()
        assert body["paused"] is True
        assert body["lab"] == "(default)"
        assert paused_flag.exists()
        r2 = client.post("/api/resume")
        assert r2.status_code == 200
        assert r2.json()["paused"] is False
        assert not paused_flag.exists()
    finally:
        if paused_flag.exists():
            paused_flag.unlink()


def test_pause_with_lab_param_writes_to_per_lab_state() -> None:
    """Create a temp lab under ~/.bert/labs/, pause it, verify the
    paused flag lands in THAT lab's state dir (not supervisor's)."""
    from fastapi.testclient import TestClient
    from api.main import app, LAB_PATH
    client = TestClient(app)

    user_labs = Path.home() / ".bert" / "labs"
    user_labs.mkdir(parents=True, exist_ok=True)
    test_lab_name = "gg-a-prep-smoke-test"
    test_lab = user_labs / test_lab_name
    try:
        # Scaffold a minimal lab
        (test_lab / "sor").mkdir(parents=True, exist_ok=True)
        (test_lab / "sor" / "events.jsonl").write_text("")
        (test_lab / "state").mkdir(parents=True, exist_ok=True)
        (test_lab / "seed_brief.md").write_text("# x")

        supervisor_paused = LAB_PATH / "state" / "paused"
        lab_paused = test_lab / "state" / "paused"
        # Clean any prior state
        for p in (supervisor_paused, lab_paused):
            if p.exists():
                p.unlink()

        r = client.post(f"/api/pause?lab={test_lab_name}",
                         json={"reason": "per-lab smoke"})
        assert r.status_code == 200
        body = r.json()
        assert body["paused"] is True
        assert body["lab"] == test_lab_name

        # CRITICAL: per-lab paused flag must land in the lab's state
        # dir, NOT the supervisor's. This is the actual bug GG-A-prep
        # fixes.
        assert lab_paused.exists(), (
            f"per-lab paused flag missing at {lab_paused}"
        )
        assert not supervisor_paused.exists(), (
            "supervisor was paused when only the test lab was — "
            "the isolation refactor regressed"
        )

        # Resume the test lab; supervisor must remain untouched
        r2 = client.post(f"/api/resume?lab={test_lab_name}")
        assert r2.status_code == 200
        assert not lab_paused.exists()
        assert not supervisor_paused.exists()
    finally:
        shutil.rmtree(test_lab, ignore_errors=True)
        for p in (LAB_PATH / "state" / "paused",):
            if p.exists():
                p.unlink()


def test_steer_with_lab_param_writes_to_per_lab_state() -> None:
    from fastapi.testclient import TestClient
    from api.main import app, LAB_PATH
    client = TestClient(app)

    user_labs = Path.home() / ".bert" / "labs"
    user_labs.mkdir(parents=True, exist_ok=True)
    test_lab_name = "gg-a-prep-steer-test"
    test_lab = user_labs / test_lab_name
    try:
        (test_lab / "sor").mkdir(parents=True, exist_ok=True)
        (test_lab / "sor" / "events.jsonl").write_text("")
        (test_lab / "state").mkdir(parents=True, exist_ok=True)
        (test_lab / "seed_brief.md").write_text("# x")

        r = client.post(f"/api/steer?lab={test_lab_name}",
                         json={"text": "refocus on memory", "modality": "typed"})
        assert r.status_code == 200
        body = r.json()
        assert "steer_id" in body
        assert body["lab"] == test_lab_name

        lab_steers = test_lab / "state" / "steers.jsonl"
        supervisor_steers = LAB_PATH / "state" / "steers.jsonl"
        # Count supervisor lines before (it may have accumulated)
        sup_lines_before = (
            len(supervisor_steers.read_text().splitlines())
            if supervisor_steers.exists() else 0
        )

        assert lab_steers.exists(), (
            f"per-lab steers.jsonl missing at {lab_steers}"
        )
        steer_entries = [
            json.loads(line) for line in lab_steers.read_text().splitlines()
            if line.strip()
        ]
        assert any("refocus on memory" in e.get("text", "")
                   for e in steer_entries)

        sup_lines_after = (
            len(supervisor_steers.read_text().splitlines())
            if supervisor_steers.exists() else 0
        )
        assert sup_lines_after == sup_lines_before, (
            "supervisor steers.jsonl grew when steering went to a "
            "named lab — isolation broken"
        )
    finally:
        shutil.rmtree(test_lab, ignore_errors=True)


def main() -> int:
    tests = [
        test_statepaths_class_exported,
        test_statepaths_has_all_state_fields,
        test_state_paths_lab_none_resolves_to_supervisor,
        test_state_paths_nonexistent_lab_raises_404,
        test_load_overrides_accepts_paths_kwarg,
        test_save_overrides_writes_to_paths_pi_overrides,
        test_endpoints_take_lab_query_param,
        test_bert_run_polls_paused_flag_per_lab,
        test_pause_resume_supervisor_via_testclient,
        test_pause_with_lab_param_writes_to_per_lab_state,
        test_steer_with_lab_param_writes_to_per_lab_state,
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
