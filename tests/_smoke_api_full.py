"""Smoke: exercise the full api/main.py surface via TestClient.

api/main.py was the single biggest coverage gap (640 missing, 54%). This
drives every route's handler — GET reads (bare + ?lab=test01), POST
writes with valid bodies scoped to the test01 lab (state writes land in
gitignored lab/state), id-routes with valid + unknown ids (404 paths),
and the .well-known agent endpoints. The goal is handler-line coverage;
each call asserts the handler ran (returned an int status, no 500 on the
read paths).
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app)
LAB = "test01"

# Read-only GET routes (safe). Hit bare + ?lab= where lab-scoped.
GET_ROUTES = [
    # NOTE: /api/events/stream is SSE — a blocking GET never returns, so
    # it's excluded here (covering it needs a streaming client).
    "/api/status", "/api/events", "/api/agents",
    "/api/choreography", "/api/quota", "/api/diagnostics", "/api/pending",
    "/api/approvals", "/api/overrides", "/api/loom", "/api/findings",
    "/api/letters/latest", "/api/graph", "/api/retrieval", "/api/compaction",
    "/api/quality-report", "/api/eval-scorecard", "/api/labs", "/api/demo-mode",
    "/api/artifact-acceptance", "/api/token-redundancy", "/api/seed-brief",
    "/api/memory-tiers", "/api/mcp-replay", "/api/signing/local-rekor",
    "/api/idle-compute", "/api/semantic-cache", "/api/onboarding/credentials-status",
    "/.well-known/agent.json", "/.well-known/agntcy-directory.json",
    "/.well-known/agent.json.sig",
]


def test_get_routes_no_server_error():
    failures = []
    for route in GET_ROUTES:
        for url in (route, f"{route}{'&' if '?' in route else '?'}lab={LAB}"):
            r = client.get(url)
            if r.status_code >= 500:
                failures.append((url, r.status_code))
    assert not failures, f"GET routes 5xx'd: {failures}"


def test_event_and_finding_by_id_paths():
    # an unknown id → 404 (covers the not-found branch); a real one if present
    assert client.get("/api/events/nonexistent_xyz").status_code in (200, 404)
    assert client.get("/api/findings/nonexistent_xyz").status_code in (200, 404)
    assert client.get(f"/api/notes/evt_xyz?lab={LAB}").status_code in (200, 404)
    assert client.get(f"/api/asks/tgt_xyz?lab={LAB}").status_code in (200, 404)


def test_governance_post_handlers_run():
    # Each writes to test01's state (gitignored). We assert the handler
    # ran (not 5xx) — covers the write-path logic.
    posts = [
        ("/api/ask", {"target_id": "t1", "question": "why this approach?"}),
        ("/api/steer", {"text": "prefer the cited-source path"}),
        ("/api/pause", {"reason": "smoke test"}),
        ("/api/resume", {}),
    ]
    for path, body in posts:
        r = client.post(f"{path}?lab={LAB}", json=body)
        assert r.status_code < 500, f"{path} → {r.status_code}: {r.text[:200]}"


def test_decision_governance_paths():
    # seed a decision (dev), then bless/veto it; covers seed + bless + veto
    client.post("/api/dev/seed-decision")
    for path in ("/api/bless/dec_smoke", "/api/veto/dec_smoke"):
        r = client.post(f"{path}?lab={LAB}", json={"rationale": "x", "reason": "x"})
        assert r.status_code < 500
    client.post("/api/dev/clear-decisions")


def test_pin_suppress_note_handlers():
    for path in ("/api/pin/evt_smoke", "/api/unpin/evt_smoke",
                 "/api/suppress/evt_smoke", "/api/unsuppress/evt_smoke"):
        assert client.post(f"{path}?lab={LAB}").status_code < 500
    r = client.post(f"/api/notes/evt_smoke?lab={LAB}", json={"text": "a smoke note"})
    assert r.status_code < 500


def test_approval_handlers():
    client.post("/api/dev/seed-approval")
    r = client.post(f"/api/approve/appr_smoke?lab={LAB}",
                    json={"choice": "yes", "rationale": "ok"})
    assert r.status_code < 500


def test_credential_endpoints_validation_and_status():
    # invalid body → 422 (covers validation); a fake provider key → handled
    assert client.post("/api/onboarding/test-credential", json={}).status_code == 422
    r = client.post("/api/onboarding/test-credential",
                    json={"provider": "groq", "key": "sk-definitely-invalid-smoke"})
    assert r.status_code < 500  # handler runs (likely returns ok:false)


def test_a2a_send_and_task_get():
    r = client.post("/a2a/v0/tasks/send", json={"task": {}, "message": {"text": "hi"}})
    assert r.status_code < 500
    assert client.get("/a2a/v0/tasks/unknown_task_xyz").status_code in (200, 404)


def test_a2a_observability_event():
    r = client.post("/a2a/v0/observability/event", json={"event_class": "model_call", "lab": LAB})
    assert r.status_code < 500


def test_run_cycle_dry_run_via_api():
    # recheck 2026-05-28 — also assert the POST actually writes a DURABLE
    # registry record (the record_start wiring is best-effort try/except, so it
    # could silently no-op; this proves it fires end-to-end, not just in the
    # run_registry unit test).
    import shutil
    import tempfile
    import time as _t

    from core import run_registry as rr
    tmp = Path(tempfile.mkdtemp())
    orig = rr.RUNS_DIR
    try:
        rr.RUNS_DIR = tmp
        r = client.post("/api/run-cycle", json={"max_cycles": 1, "dry_run": True})
        assert r.status_code == 200, r.text[:200]
        run_id = r.json().get("run_id", "")
        assert run_id.startswith("run_")
        # give the POST handler's record_start a beat (it's synchronous, but the
        # dry-run subprocess may race the durable write under load)
        rec = None
        for _ in range(20):
            rec = rr.get(run_id)
            if rec is not None:
                break
            _t.sleep(0.1)
        assert rec is not None, "POST /api/run-cycle did not write a durable run record"
        assert rec.run_id == run_id and rec.pid > 0
    finally:
        rr.RUNS_DIR = orig
        shutil.rmtree(tmp, ignore_errors=True)


def test_signing_checkpoint_merkle():
    r = client.post("/api/signing/checkpoint-merkle", json={})
    assert r.status_code < 500


def test_save_credentials_endpoint():
    # The handler validates each key against the provider's /models endpoint
    # and writes accepted keys to config.CRED_PATH. We fake httpx so groq
    # "validates" (200) and others "reject" (401), and redirect CRED_PATH to
    # a temp file so the real ~/.bert-lab/credentials.json is never touched.
    import shutil
    import tempfile

    import httpx

    from core import config as _cfg
    orig_client, orig_cred = httpx.Client, _cfg.CRED_PATH
    tmpdir = Path(tempfile.mkdtemp())

    class _Resp:
        def __init__(self, status):
            self.status_code = status
            self.text = "ok"

    class _Client:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def get(self, url, **k):
            return _Resp(200 if "groq" in url else 401)
    try:
        _cfg.CRED_PATH = tmpdir / "credentials.json"
        httpx.Client = lambda *a, **k: _Client()
        r = client.post("/api/onboarding/save-credentials", json={"credentials": {
            "GROQ_API_KEY": "valid-key",     # validates → saved
            "MISTRAL_API_KEY": "bad-key",    # 401 → not saved
            "BOGUS_KEY": "x",                # unknown env var → skipped
            "NVIDIA_API_KEY": "",            # empty → skipped
        }})
        assert r.status_code == 200, r.text[:200]
        body = r.json()
        assert "per_key" in body or "saved" in body
        assert (tmpdir / "credentials.json").exists()   # validated key persisted
    finally:
        httpx.Client = orig_client
        _cfg.CRED_PATH = orig_cred
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_sse_generator_bounded():
    # /api/events/stream is an infinite async generator; drive it bounded with
    # asyncio.sleep no-op'd + time.monotonic advanced so the tick branch fires,
    # and EVENTS_PATH pointed at a temp file to hit the new-events + shrink paths.
    import asyncio
    import json as _j
    import shutil
    import tempfile

    from api import main as M
    ev = Path(tempfile.mkdtemp()) / "events.jsonl"
    ev.write_text("")
    orig = (M.EVENTS_PATH, M.asyncio.sleep, M.time.monotonic)
    counter = {"t": 0.0}

    async def _nosleep(*a, **k):
        return None
    def _mono():
        counter["t"] += 10.0
        return counter["t"]
    try:
        M.EVENTS_PATH = ev
        M.asyncio.sleep = _nosleep
        M.time.monotonic = _mono

        async def drive():
            gen = M._sse_generator()
            first = await gen.__anext__()
            assert first.startswith(":connected")
            ev.write_text(_j.dumps({"id": "e1", "event_class": "finding"}) + "\nbad\n")
            for _ in range(3):          # event yield + tick branches
                await gen.__anext__()
            ev.write_text("")           # shrink → position reset branch
            await gen.__anext__()
            await gen.aclose()
        asyncio.run(drive())
    finally:
        M.EVENTS_PATH, M.asyncio.sleep, M.time.monotonic = orig
        shutil.rmtree(ev.parent, ignore_errors=True)


def test_create_lab_endpoint():
    # POST /api/labs scaffolds via bert_init._scaffold_lab — mock it + redirect
    # LABS_DIR to a temp tree so no real ~/.bert/labs entry is created.
    import shutil
    import tempfile

    from tools import bert_init as _bi
    tmp = Path(tempfile.mkdtemp())
    orig_dir, orig_scaffold = _bi.LABS_DIR, _bi._scaffold_lab
    try:
        _bi.LABS_DIR = tmp
        made = tmp / "my_smoke_lab"
        _bi._scaffold_lab = lambda answers, **k: (made.mkdir(parents=True, exist_ok=True), made)[1]
        body = {"name": "My Smoke Lab", "archetype": "research", "provider": "groq",
                "autonomy": "collaborator", "mission": "investigate X", "focus_areas": ["a"]}
        r = client.post("/api/labs", json=body)
        assert r.status_code < 500, r.text[:200]
        # conflict path: pre-create a non-empty slug dir → 409
        conflict = tmp / "conflict_lab"
        conflict.mkdir()
        (conflict / "x.txt").write_text("x")
        r2 = client.post("/api/labs", json={**body, "name": "Conflict Lab"})
        assert r2.status_code in (200, 409, 422)
    finally:
        _bi.LABS_DIR, _bi._scaffold_lab = orig_dir, orig_scaffold
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_status_durable_fallback():
    # Sprint 4 B — a run recorded only in the durable registry (not in the
    # in-memory _RUN_REGISTRY, e.g. after an API restart) is still queryable.
    import shutil
    import tempfile

    from core import run_registry as rr
    tmp = Path(tempfile.mkdtemp())
    orig = rr.RUNS_DIR
    try:
        rr.RUNS_DIR = tmp
        rr.record_start("run_durable_smoke", pid=2_000_000_000, lab="test01")
        r = client.get("/api/run-cycle/run_durable_smoke")
        assert r.status_code == 200, r.text[:200]
        body = r.json()
        assert body["from"] == "durable_registry" and body["lab"] == "test01"
        # a truly-unknown run is still a 404
        assert client.get("/api/run-cycle/no_such_run_xyz").status_code == 404
    finally:
        rr.RUNS_DIR = orig
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    tests = [
        test_get_routes_no_server_error,
        test_event_and_finding_by_id_paths,
        test_governance_post_handlers_run,
        test_decision_governance_paths,
        test_pin_suppress_note_handlers,
        test_approval_handlers,
        test_credential_endpoints_validation_and_status,
        test_a2a_send_and_task_get,
        test_a2a_observability_event,
        test_run_cycle_dry_run_via_api,
        test_signing_checkpoint_merkle,
        test_save_credentials_endpoint,
        test_sse_generator_bounded,
        test_create_lab_endpoint,
        test_run_status_durable_fallback,
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
