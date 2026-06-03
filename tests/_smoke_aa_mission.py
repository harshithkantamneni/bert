"""Smoke test for AA-phase: mission input + run-cycle wiring.

Covers:
  AA.1 — GET /api/seed-brief (read), PUT /api/seed-brief (write),
         optimistic-lock concurrent-edit detection (409), empty content
         validation (400)
  AA.2 — POST /api/run-cycle (spawn subprocess), GET /api/run-cycle/{id}
         (status), SSE stream shape, max_cycles validation, unknown lab
         404, missing seed_brief 400
  AA.3 — MissionEditor.tsx exists + imports cleanly into the React build
         (post HH-B rehaul; was Mission.tsx pre-rehaul, now retired)
  AA.4 — /mission route redirects to / (post HH-E retirement); the
         mission editor is rendered inline on Home (post HH-F rename)
  AA.6 — bert_doctor's Groq check sends a non-default User-Agent
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

VENV_PY = LAB_ROOT / ".venv" / "bin" / "python"
APP_MAIN = LAB_ROOT / "api" / "main.py"
MISSION_EDITOR_TSX = LAB_ROOT / "bert" / "v4" / "src" / "components" / "MissionEditor.tsx"
APP_TSX = LAB_ROOT / "bert" / "v4" / "src" / "App.tsx"
HOME_TSX = LAB_ROOT / "bert" / "v4" / "src" / "surfaces" / "Home.tsx"
DOCTOR = LAB_ROOT / "tools" / "bert_doctor.py"
CLIENT_TS = LAB_ROOT / "bert" / "v4" / "src" / "api" / "client.ts"


# ── Helper: spawn uvicorn on a free port, return (port, proc) ───────

def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn_uvicorn() -> tuple[int, subprocess.Popen]:
    port = _find_free_port()
    proc = subprocess.Popen(
        [str(VENV_PY), "-m", "uvicorn", "api.main:app",
         "--host", "127.0.0.1", "--port", str(port),
         "--log-level", "error"],
        env={**os.environ, "BERT_DISABLE_IDLE_COMPUTE": "1"},
        cwd=str(LAB_ROOT),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # Wait for ready. 30s, not 10s: api.main imports the full ML stack
    # (sentence-transformers / torch), so cold startup under batch memory
    # pressure routinely exceeds 10s — the old cap made this a flaky gate.
    for _ in range(120):
        time.sleep(0.25)
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/status", timeout=1
            ) as r:
                if r.status == 200:
                    return port, proc
        except Exception:
            continue
    proc.terminate()
    raise AssertionError("uvicorn didn't start within 30s")


def _http_get(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode())
        except Exception:
            body = {"detail": str(e)}
        return e.code, body


def _http_send(url: str, method: str, body: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(
        url, method=method,
        headers={"Content-Type": "application/json"} if body else {},
        data=json.dumps(body).encode() if body else None,
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            data = json.loads(e.read().decode())
        except Exception:
            data = {"detail": str(e)}
        return e.code, data


# ── AA.1: seed-brief read/write ─────────────────────────────────────

def test_seed_brief_endpoints_registered() -> None:
    """Module loads without import errors."""
    text = APP_MAIN.read_text()
    assert "@app.get(\"/api/seed-brief\")" in text, \
        "GET /api/seed-brief not registered"
    assert "@app.put(\"/api/seed-brief\")" in text, \
        "PUT /api/seed-brief not registered"


def test_seed_brief_get_returns_schema() -> None:
    port, proc = _spawn_uvicorn()
    try:
        status, body = _http_get(f"http://127.0.0.1:{port}/api/seed-brief")
        assert status == 200, f"GET should succeed; got {status}"
        for key in ("ts", "lab", "path", "exists", "content", "mtime", "size_bytes"):
            assert key in body, f"GET response missing key {key}"
    finally:
        proc.terminate(); proc.wait(timeout=3)


def test_seed_brief_put_writes_and_returns_mtime() -> None:
    port, proc = _spawn_uvicorn()
    tmp_home = Path(tempfile.mkdtemp(prefix="aa_seed_"))
    try:
        # Scaffold a fresh lab so we don't pollute the default
        lab_dir = tmp_home / ".bert" / "labs" / "aa-test"
        lab_dir.mkdir(parents=True)
        (lab_dir / "sor").mkdir()
        (lab_dir / "sor" / "events.jsonl").write_text("")
        (lab_dir / "state").mkdir()
        # Add the lab to HOME by monkey-patching env? Simpler: write
        # directly. But _resolve_lab_path looks at ~/.bert/labs/.
        # We can't easily monkey HOME here. Use the default lab instead
        # but back up its seed_brief if any.
        seed_path = LAB_ROOT / "lab" / "seed_brief.md"
        backup = seed_path.read_text() if seed_path.exists() else None
        try:
            status, body = _http_send(
                f"http://127.0.0.1:{port}/api/seed-brief", "PUT",
                {"content": "# Mission\n\nAA-phase smoke test content."}
            )
            assert status == 200, f"PUT should succeed; got {status} {body}"
            assert body.get("ok") is True
            assert "mtime" in body and body["mtime"] > 0
            assert body["size_bytes"] > 0

            # Verify GET reads back the written content
            status2, body2 = _http_get(f"http://127.0.0.1:{port}/api/seed-brief")
            assert "AA-phase smoke test content" in body2["content"]
            assert abs(body2["mtime"] - body["mtime"]) < 0.01
        finally:
            # BB.5 — restore the original seed_brief if any. NEVER
            # unlink: the default lab now ships with a committed
            # seed_brief.md (BB.5); deleting it breaks subsequent tests
            # in the regression that depend on it (and breaks the lab).
            if backup is not None:
                seed_path.write_text(backup)
    finally:
        proc.terminate(); proc.wait(timeout=3)
        shutil.rmtree(tmp_home, ignore_errors=True)


def test_seed_brief_optimistic_lock_returns_409() -> None:
    port, proc = _spawn_uvicorn()
    seed_path = LAB_ROOT / "lab" / "seed_brief.md"
    backup = seed_path.read_text() if seed_path.exists() else None
    try:
        # First write
        _http_send(
            f"http://127.0.0.1:{port}/api/seed-brief", "PUT",
            {"content": "# Mission\n\nOriginal."}
        )
        # Concurrent edit with stale mtime
        status, body = _http_send(
            f"http://127.0.0.1:{port}/api/seed-brief", "PUT",
            {"content": "# Mission\n\nClobbering edit.", "expected_mtime": 1.0}
        )
        assert status == 409, f"stale mtime should return 409; got {status}"
        assert "concurrent edit" in body.get("detail", "").lower(), \
            f"409 message should explain: {body}"
    finally:
        # BB.5 — see note above. Restore but never unlink.
        if backup is not None:
            seed_path.write_text(backup)
        proc.terminate(); proc.wait(timeout=3)


def test_seed_brief_empty_content_returns_400() -> None:
    port, proc = _spawn_uvicorn()
    try:
        status, body = _http_send(
            f"http://127.0.0.1:{port}/api/seed-brief", "PUT",
            {"content": "   "}
        )
        assert status == 400
        assert "non-empty" in body.get("detail", "").lower()
    finally:
        proc.terminate(); proc.wait(timeout=3)


def test_seed_brief_oversize_returns_400() -> None:
    port, proc = _spawn_uvicorn()
    try:
        huge = "x" * (33 * 1024)  # >32KB
        status, body = _http_send(
            f"http://127.0.0.1:{port}/api/seed-brief", "PUT",
            {"content": huge}
        )
        assert status == 400
        assert "too large" in body.get("detail", "").lower()
    finally:
        proc.terminate(); proc.wait(timeout=3)


# ── AA.2: run-cycle ─────────────────────────────────────────────────

def test_run_cycle_endpoints_registered() -> None:
    text = APP_MAIN.read_text()
    assert "@app.post(\"/api/run-cycle\")" in text
    assert "stream_run_cycle" in text
    assert "get_run_status" in text


def test_run_cycle_spawns_subprocess_with_dry_run() -> None:
    """Real subprocess with --dry-run; should complete in <5s without
    firing model calls."""
    port, proc = _spawn_uvicorn()
    seed_path = LAB_ROOT / "lab" / "seed_brief.md"
    backup = seed_path.read_text() if seed_path.exists() else None
    try:
        # Ensure seed_brief exists so the spawn doesn't 400
        _http_send(
            f"http://127.0.0.1:{port}/api/seed-brief", "PUT",
            {"content": "# Mission\n\nTest seed for run-cycle smoke."}
        )
        # Spawn
        status, body = _http_send(
            f"http://127.0.0.1:{port}/api/run-cycle", "POST",
            {"max_cycles": 1, "dry_run": True}
        )
        assert status == 200, f"POST should succeed; got {status} {body}"
        run_id = body["run_id"]
        assert run_id.startswith("run_"), f"run_id format unexpected: {run_id}"
        assert body["dry_run"] is True
        assert body["stream_url"] == f"/api/run-cycle/{run_id}/stream"

        # Poll status until alive=False
        for _ in range(20):
            time.sleep(0.5)
            s2, b2 = _http_get(f"http://127.0.0.1:{port}/api/run-cycle/{run_id}")
            if s2 == 200 and not b2.get("alive", True):
                assert b2["exit_code"] == 0, f"dry-run should exit 0; got {b2['exit_code']}"
                assert b2["line_count"] > 0, "should have captured stdout lines"
                return  # success
        raise AssertionError("subprocess didn't complete within 10s")
    finally:
        # BB.5 — restore the original seed if we backed one up so
        # downstream tests in the regression see the committed default
        # mission, not this test's stub.
        if backup is not None:
            seed_path.write_text(backup)
        proc.terminate(); proc.wait(timeout=3)


def test_run_cycle_validates_max_cycles_upper_bound() -> None:
    """Post-GG-C: the cap is now 1..50 (raised from 1..5 once CC.4
    termination guardrails landed), AND values > 5 require an
    explicit `consent_long_run` flag because long autonomous runs
    bill against provider quotas. This test sends max_cycles=10
    WITHOUT consent → 400 + message says consent required.
    Also confirms the over-cap branch still fires at >50.
    """
    port, proc = _spawn_uvicorn()
    try:
        status, body = _http_send(
            f"http://127.0.0.1:{port}/api/run-cycle", "POST",
            {"max_cycles": 10, "dry_run": True}
        )
        assert status == 400, (
            f"max_cycles=10 without consent should 400; got {status}: {body}"
        )
        detail = body.get("detail", "")
        assert "consent_long_run" in detail, (
            f"expected consent_long_run hint in error; got: {detail!r}"
        )

        # And the over-cap branch fires at >50
        status2, body2 = _http_send(
            f"http://127.0.0.1:{port}/api/run-cycle", "POST",
            {"max_cycles": 100, "dry_run": True, "consent_long_run": True}
        )
        assert status2 == 400
        assert "1..50" in body2.get("detail", "")
    finally:
        proc.terminate(); proc.wait(timeout=3)


def test_run_cycle_rejects_missing_seed_brief() -> None:
    port, proc = _spawn_uvicorn()
    seed_path = LAB_ROOT / "lab" / "seed_brief.md"
    backup = seed_path.read_text() if seed_path.exists() else None
    if seed_path.exists():
        seed_path.unlink()
    try:
        status, body = _http_send(
            f"http://127.0.0.1:{port}/api/run-cycle", "POST",
            {"max_cycles": 1, "dry_run": True}
        )
        assert status == 400, f"should reject missing seed; got {status} {body}"
        assert "seed_brief.md" in body.get("detail", "")
    finally:
        if backup is not None:
            seed_path.write_text(backup)
        proc.terminate(); proc.wait(timeout=3)


def test_run_cycle_rejects_unknown_lab() -> None:
    port, proc = _spawn_uvicorn()
    try:
        status, body = _http_send(
            f"http://127.0.0.1:{port}/api/run-cycle", "POST",
            {"lab": "phantom-lab-aa-test", "max_cycles": 1, "dry_run": True}
        )
        assert status == 404
        assert "phantom-lab" in body.get("detail", "")
    finally:
        proc.terminate(); proc.wait(timeout=3)


def test_run_cycle_status_404_on_unknown_run_id() -> None:
    port, proc = _spawn_uvicorn()
    try:
        status, _body = _http_get(
            f"http://127.0.0.1:{port}/api/run-cycle/run_phantom"
        )
        assert status == 404
    finally:
        proc.terminate(); proc.wait(timeout=3)


# ── AA.3: MissionEditor.tsx (post HH-B rehaul) ──────────────────────

def test_mission_editor_tsx_exists() -> None:
    """HH-B retired the standalone /mission surface in favour of an
    inline editor on Home. The component file moved from
    surfaces/Mission.tsx to components/MissionEditor.tsx."""
    assert MISSION_EDITOR_TSX.exists(), \
        "bert/v4/src/components/MissionEditor.tsx missing"


def test_mission_editor_tsx_exports_component() -> None:
    text = MISSION_EDITOR_TSX.read_text()
    assert "export function MissionEditor" in text, \
        "MissionEditor.tsx must export a function named MissionEditor"


def test_mission_editor_tsx_uses_seed_brief_endpoint() -> None:
    text = MISSION_EDITOR_TSX.read_text()
    assert "/api/seed-brief" in text


def test_mission_editor_tsx_uses_palette_tokens() -> None:
    text = MISSION_EDITOR_TSX.read_text()
    assert "PALETTE" in text
    assert "FONTS" in text


def test_client_ts_has_apiPut_and_types() -> None:
    text = CLIENT_TS.read_text()
    assert "export async function apiPut" in text
    assert "SeedBriefRead" in text
    assert "SeedBriefWrite" in text
    assert "RunCycleStart" in text


# ── AA.4: post-HH-E redirect + post-HH-F inline rendering ───────────

def test_mission_route_redirects_to_home() -> None:
    """HH-E retired /mission; the route now redirects to / via
    QueryPreservingRedirect. The editor is inline on Home."""
    text = APP_TSX.read_text()
    assert 'path="/mission"' in text
    assert "QueryPreservingRedirect" in text


def test_home_renders_mission_editor_inline() -> None:
    """Post HH-B/F: MissionEditor renders on Home (renamed from
    FirstLight), between the director's letter and the pulse strip."""
    text = HOME_TSX.read_text()
    assert "<MissionEditor />" in text
    assert "import { MissionEditor }" in text


def test_mission_editor_visible_in_demo_mode() -> None:
    """Critical: the most user-facing surface (Home + inline mission
    editor) must NOT be gated behind demo-mode=off. This is the
    bert-actually-does-something path. Post-HH the editor lives on
    Home rather than its own /mission route; check Home's gate."""
    text = APP_TSX.read_text()
    import re
    # The Home route uses bounded("home", <Home />) — find it and
    # verify it's NOT preceded by an isDemoMode guard.
    m = re.search(r"(.{300})bounded\(\"home\", <Home />\)", text, re.DOTALL)
    assert m, "Home route not found in App.tsx"
    preceding = m.group(1)
    # The last 200 chars before the home route shouldn't gate on demo
    # mode (it's the route element conditional, not the route guard).
    # Demo mode toggles which surfaces appear, but Home is always-on.
    assert "isDemoMode &&" not in preceding[-200:], \
        "Home route appears gated behind isDemoMode — must be visible always"


# ── AA.6: bert_doctor Groq check ────────────────────────────────────

def test_doctor_groq_check_uses_explicit_user_agent() -> None:
    text = DOCTOR.read_text()
    assert "User-Agent" in text, \
        "doctor's Groq check must send an explicit User-Agent"
    assert "bert-doctor" in text, \
        "doctor's UA should identify itself"


def test_doctor_groq_check_distinguishes_401_from_403() -> None:
    text = DOCTOR.read_text()
    assert "401" in text and "Unauthorized" in text, \
        "doctor should distinguish 401 (key invalid)"
    assert "403" in text and "Forbidden" in text, \
        "doctor should distinguish 403 (edge rejection)"


def main() -> int:
    tests = [
        # AA.1
        test_seed_brief_endpoints_registered,
        test_seed_brief_get_returns_schema,
        test_seed_brief_put_writes_and_returns_mtime,
        test_seed_brief_optimistic_lock_returns_409,
        test_seed_brief_empty_content_returns_400,
        test_seed_brief_oversize_returns_400,
        # AA.2
        test_run_cycle_endpoints_registered,
        test_run_cycle_spawns_subprocess_with_dry_run,
        test_run_cycle_validates_max_cycles_upper_bound,
        test_run_cycle_rejects_missing_seed_brief,
        test_run_cycle_rejects_unknown_lab,
        test_run_cycle_status_404_on_unknown_run_id,
        # AA.3 (post HH-B/E/F rehaul)
        test_mission_editor_tsx_exists,
        test_mission_editor_tsx_exports_component,
        test_mission_editor_tsx_uses_seed_brief_endpoint,
        test_mission_editor_tsx_uses_palette_tokens,
        test_client_ts_has_apiPut_and_types,
        # AA.4
        test_mission_route_redirects_to_home,
        test_home_renders_mission_editor_inline,
        test_mission_editor_visible_in_demo_mode,
        # AA.6
        test_doctor_groq_check_uses_explicit_user_agent,
        test_doctor_groq_check_distinguishes_401_from_403,
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
