"""Behavioral integration test for the agent.py lifecycle wiring.

Replaces / augments the structural source-string checks in
`_smoke_lifecycle_wiring.py`. Quality-first: a passing source-string
test is worthless if the actual call doesn't fire the intended event.
This test mocks the heavy components (provider.call, brief_assembler,
consolidator, evaluator, indexer daemon), invokes the real
agent.run_role, and asserts behavioral outcomes:

  - hooks.fire() actually executes registered scripts with payload
    on stdin and the right argv
  - observability.emit() writes events to OBS_DIR with the right
    event_class and field shape
  - watchdog session opens + closes for both subagent and top-level
  - quota.record_call records every successful provider response
  - the order: RoleStart → ModelCall → ToolUse* → RoleEnd

Run: `.venv/bin/python tests/_smoke_lifecycle_integration.py`
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest import mock

LAB_ROOT_REAL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT_REAL))

from core import (  # noqa: E402
    agent,
    brief_assembler,
    consolidator,
    hooks,
    observability,
    quota,
    watchdog,
)
from core import (
    session as session_mod,
)
from core.types import (  # noqa: E402
    PermissionMode,
    ProviderResponse,
)


def _setup_temp_lab(tmp: Path) -> dict:
    """Repoint all module-level paths into a temp tree. Returns a dict
    of original values for teardown."""
    orig = {
        "agent.LAB_ROOT": agent.LAB_ROOT,
        "agent.SESSION_EXIT": agent.SESSION_EXIT,
        "agent.SESSION_START": agent.SESSION_START,
        "agent._INDEXER_DAEMON": agent._INDEXER_DAEMON,
        "observability.OBS_DIR": observability.OBS_DIR,
        "quota.QUOTA_DB": quota.QUOTA_DB,
        "watchdog.WATCHDOG_DB": watchdog.WATCHDOG_DB,
        "hooks.HOOKS_DIR": hooks.HOOKS_DIR,
        "brief_assembler.LAB_ROOT": brief_assembler.LAB_ROOT,
        "brief_assembler.MEMORIES_DIR": brief_assembler.MEMORIES_DIR,
        "brief_assembler.STATE_DIR": brief_assembler.STATE_DIR,
        "brief_assembler.BRIEF_PATH": brief_assembler.BRIEF_PATH,
        "session_mod.LOGS_DIR": session_mod.LOGS_DIR,
        "consolidator.LAB_ROOT": consolidator.LAB_ROOT,
        "consolidator.MEMORIES_DIR": consolidator.MEMORIES_DIR,
        "consolidator.LAST_RUN_PATH": consolidator.LAST_RUN_PATH,
    }
    agent.LAB_ROOT = tmp
    agent.SESSION_EXIT = tmp / "state" / "session_exit.md"
    agent.SESSION_START = tmp / "state" / "session_start.md"
    agent._INDEXER_DAEMON = None  # forces autostart attempt (will fail/no-op gracefully on temp)
    observability.OBS_DIR = tmp / "state" / "observability"
    quota.QUOTA_DB = tmp / "lab" / "state" / "quota.db"
    watchdog.WATCHDOG_DB = tmp / "lab" / "state" / "watchdog.db"
    hooks.HOOKS_DIR = tmp / "hooks"
    brief_assembler.LAB_ROOT = tmp
    brief_assembler.MEMORIES_DIR = tmp / "memories"
    brief_assembler.STATE_DIR = tmp / "state"
    brief_assembler.BRIEF_PATH = tmp / "memories" / "context_brief.md"
    session_mod.LOGS_DIR = tmp / "logs"
    consolidator.LAB_ROOT = tmp
    consolidator.MEMORIES_DIR = tmp / "memories"
    consolidator.LAST_RUN_PATH = tmp / "lab" / "state" / "consolidator.last_run.json"
    # Pre-create needed files
    (tmp / "memories" / "governance").mkdir(parents=True, exist_ok=True)
    (tmp / "memories" / "current.md").write_text("## §Current Program\n\ntest")
    (tmp / "memories" / "log.md").write_text("## D-1\nplaceholder\n")
    (tmp / "memories" / "procedures.md").write_text("# procedures")
    (tmp / "memories" / "governance" / "constitutional.md").write_text(
        "# bert constitutional preamble\n\nPlaceholder for tests.\n"
    )
    # role prompt directory
    (tmp / "prompts").mkdir(exist_ok=True)
    (tmp / "prompts" / "researcher.md").write_text("# Researcher\nTest role.")
    (tmp / "state").mkdir(exist_ok=True)
    return orig


def _restore_lab(orig: dict) -> None:
    for key, val in orig.items():
        mod_name, attr = key.split(".", 1)
        mod = {
            "agent": agent, "observability": observability, "quota": quota,
            "watchdog": watchdog, "hooks": hooks, "brief_assembler": brief_assembler,
            "session_mod": session_mod, "consolidator": consolidator,
        }[mod_name]
        setattr(mod, attr, val)


def _stop_response(text: str = "ok") -> ProviderResponse:
    """Synthetic ProviderResponse for tests that mock provider.call directly.
    Used by tests that don't need to exercise quota.record_call /
    structured_output / etc. — for full-fidelity tests that need
    provider.call's body to execute, mock httpx.Client instead via
    `_httpx_returning(...)`."""
    return ProviderResponse(
        text=text, tool_calls=[], finish_reason="stop",
        usage_prompt_tokens=42, usage_completion_tokens=8,
        model="test-model", provider="test-provider", elapsed_ms=120,
    )


def _fake_httpx_response(json_body: dict, status_code: int = 200):
    """Build a mock httpx.Response shape for httpx.Client.post return."""
    resp = mock.MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    resp.text = json.dumps(json_body)
    resp.headers = {}
    return resp


def _httpx_canned_chat_completion(text: str = "hello",
                                   prompt_tokens: int = 42,
                                   completion_tokens: int = 8,
                                   tool_calls: list | None = None):
    """OpenAI-compatible chat.completions response shape, suitable for
    httpx.Client.post() return-value. Routes through provider._parse_response
    so the post-success quota.record_call wiring fires."""
    body: dict = {
        "model": "test-model",
        "choices": [{
            "message": {"content": text, "tool_calls": tool_calls or []},
            "finish_reason": "tool_use" if tool_calls else "stop",
        }],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }
    return _fake_httpx_response(body)


def _patch_httpx_returning(response):
    """Returns a mock.patch context that makes httpx.Client.post return
    the given response. Used so the real core.provider.call body
    executes (including quota.record_call + observability emit)."""
    patcher = mock.patch("httpx.Client")
    return patcher, response


class _HttpxCtx:
    """Context manager that wires httpx.Client to return canned bodies
    in order, then call core.config.load to return a fake cfg with the
    test's API key. Lets agent.run_role exercise the real provider.call
    body."""
    def __init__(self, *responses):
        self._responses = list(responses)
        self._client_patcher = None
        self._cfg_patcher = None

    def __enter__(self):
        self._client_patcher = mock.patch("httpx.Client")
        mc = self._client_patcher.__enter__()
        client_inst = mc.return_value.__enter__.return_value
        # side_effect cycles through the canned responses
        responses_iter = iter(self._responses)
        def post(*a, **k):
            try:
                return next(responses_iter)
            except StopIteration:
                return _httpx_canned_chat_completion()
        client_inst.post.side_effect = post

        self._cfg_patcher = mock.patch("core.config.load")
        cl = self._cfg_patcher.__enter__()
        cfg = mock.MagicMock()
        cfg.get.return_value = "fake-key"
        cfg.max_tokens_default = 100
        cfg.context_usage_cap = 0.7
        cfg.permission_mode = PermissionMode.DEFAULT
        cl.return_value = cfg
        return self

    def __exit__(self, *exc):
        self._cfg_patcher.__exit__(*exc)
        self._client_patcher.__exit__(*exc)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ── tests ────────────────────────────────────────────────────────────


def test_subagent_run_fires_lifecycle_observability_quota_watchdog() -> None:
    """is_subagent=True path: minimal work, no top-level brief/consolidator.
    Must still fire RoleStart/ModelCall/RoleEnd hooks + emit_model_call +
    quota.record_call + watchdog session record.

    Mocks at httpx.Client level (not provider.call) so the real
    provider.call body executes, including its post-success
    quota.record_call wiring. Quality-first: test the real lifecycle,
    not a synthetic short-circuit."""
    tmp = Path(tempfile.mkdtemp(prefix="bert_lifecycle_"))
    orig = _setup_temp_lab(tmp)
    try:
        # Drop capture scripts
        for ev in ("RoleStart", "ModelCall", "RoleEnd"):
            d = hooks.HOOKS_DIR / ev
            d.mkdir(parents=True, exist_ok=True)
            cap = d / "capture.sh"
            cap.write_text(
                f"#!/usr/bin/env bash\n"
                f"cat > {tmp}/captured_{ev}.json\n"
            )
            cap.chmod(0o755)

        with _HttpxCtx(_httpx_canned_chat_completion(text="hello")):
            rc = agent.run_role(
                "researcher", cycle=999, task="test integration",
                provider_name="nvidia", model="test-model",
                max_iterations=2, is_subagent=True,
            )
        assert rc == 0, f"expected rc=0; got {rc}"

        # Hooks: RoleStart + ModelCall + RoleEnd captured
        for ev in ("RoleStart", "ModelCall", "RoleEnd"):
            cap_file = tmp / f"captured_{ev}.json"
            assert cap_file.exists(), f"hook {ev} did not fire"
            payload = json.loads(cap_file.read_text())
            assert payload.get("role") == "researcher"
            assert payload.get("cycle") == 999

        # Observability: model_call event
        events = _read_jsonl(observability.OBS_DIR / "model_call.jsonl")
        assert len(events) >= 1
        assert events[0]["model"] == "test-model"
        assert events[0]["input_tokens"] == 42
        assert events[0]["output_tokens"] == 8

        # Quota: record_call ran. P.2 — the test originally asserted
        # rows[0][0] == "nvidia" but the run picks up additional
        # incidental quota events from agent setup (mistral validation
        # probes, etc.) so the ordering isn't guaranteed. Check that
        # nvidia is recorded somewhere in the event stream.
        with sqlite3.connect(quota.QUOTA_DB) as conn:
            rows = conn.execute("SELECT provider FROM events").fetchall()
        assert len(rows) >= 1
        providers_recorded = {r[0] for r in rows}
        assert "nvidia" in providers_recorded, (
            f"expected 'nvidia' in recorded providers; got {providers_recorded}"
        )

        # Watchdog: session opened + closed
        with sqlite3.connect(watchdog.WATCHDOG_DB) as conn:
            rows = conn.execute(
                "SELECT role, cycle, ended_ts FROM sessions WHERE cycle=999"
            ).fetchall()
        assert len(rows) >= 1
        for role, cycle, ended_ts in rows:
            assert role == "researcher" and cycle == 999
            assert ended_ts is not None, "session must be closed in finally block"
    finally:
        _restore_lab(orig)


def test_top_level_run_fires_brief_session_consolidator_evaluator() -> None:
    """is_subagent=False: top-level extras must fire — brief_assembler,
    session.start_session/end_session, consolidator.consolidate after
    evaluator. Mock the heavy bits to keep the test fast."""
    tmp = Path(tempfile.mkdtemp(prefix="bert_lifecycle_top_"))
    orig = _setup_temp_lab(tmp)
    try:
        # Disable the indexer daemon for the test (autostart would need
        # a real fs-watcher Observer; we don't need it for this test).
        agent._INDEXER_DAEMON = "DUMMY"  # short-circuits _ensure_running

        # Capture EvaluatorVerdict + RoleEnd
        for ev in ("EvaluatorVerdict", "RoleEnd"):
            d = hooks.HOOKS_DIR / ev
            d.mkdir(parents=True, exist_ok=True)
            cap = d / "capture.sh"
            cap.write_text(
                f"#!/usr/bin/env bash\ncat > {tmp}/captured_{ev}.json\n"
            )
            cap.chmod(0o755)

        # Mock heavy bits so the test runs fast
        from core.evaluator import CycleEvaluation
        with _HttpxCtx(_httpx_canned_chat_completion()), \
             mock.patch("core.brief_assembler.assemble_brief") as mb, \
             mock.patch("core.consolidator.consolidate") as mc, \
             mock.patch("core.evaluator.evaluate_cycle") as me:
            me.return_value = CycleEvaluation(cycle=12345)
            rc = agent.run_role(
                "researcher", cycle=12345, task="top-level test",
                provider_name="nvidia", model="test-model",
                max_iterations=2, is_subagent=False,
            )

        assert rc == 0
        assert mb.called, "brief_assembler.assemble_brief must fire at top-level"
        assert mc.called, "consolidator.consolidate must fire at top-level"
        assert me.called, "evaluator.evaluate_cycle must fire at top-level"

        # Session log was created (start_session + end_session both fired)
        session_files = list((tmp / "logs").glob("session_*.jsonl"))
        assert len(session_files) >= 1
        events = _read_jsonl(session_files[0])
        kinds = [e.get("kind") for e in events]
        assert "session_start" in kinds
        assert "session_end" in kinds

        # EvaluatorVerdict + RoleEnd hooks fired
        for ev in ("EvaluatorVerdict", "RoleEnd"):
            cap_file = tmp / f"captured_{ev}.json"
            assert cap_file.exists(), f"hook {ev} did not fire"
    finally:
        _restore_lab(orig)


def test_lifecycle_event_order() -> None:
    """RoleStart fires BEFORE ModelCall fires BEFORE RoleEnd. The order
    matters for downstream consumers (e.g., a hook that opens a span on
    RoleStart and closes it on RoleEnd needs strict ordering)."""
    tmp = Path(tempfile.mkdtemp(prefix="bert_order_"))
    orig = _setup_temp_lab(tmp)
    try:
        order_log = tmp / "order.log"
        for ev in ("RoleStart", "ModelCall", "RoleEnd"):
            d = hooks.HOOKS_DIR / ev
            d.mkdir(parents=True, exist_ok=True)
            cap = d / "capture.sh"
            cap.write_text(
                f"#!/usr/bin/env bash\necho {ev} >> {order_log}\n"
            )
            cap.chmod(0o755)

        with _HttpxCtx(_httpx_canned_chat_completion()):
            agent.run_role(
                "researcher", cycle=100, task="order test",
                provider_name="nvidia", model="x",
                max_iterations=1, is_subagent=True,
            )
        assert order_log.exists()
        order = order_log.read_text().splitlines()
        # Expected: RoleStart, ModelCall, RoleEnd (single iteration)
        assert order[0] == "RoleStart", f"RoleStart not first; got {order}"
        assert order[-1] == "RoleEnd", f"RoleEnd not last; got {order}"
        # ModelCall(s) between
        assert "ModelCall" in order[1:-1]
    finally:
        _restore_lab(orig)


def test_hooks_failure_doesnt_break_loop() -> None:
    """Quality-first: hook failures must NOT propagate. Drop a hook that
    crashes; verify run_role still returns 0 + provider.call still fired."""
    tmp = Path(tempfile.mkdtemp(prefix="bert_hookfail_"))
    orig = _setup_temp_lab(tmp)
    try:
        d = hooks.HOOKS_DIR / "RoleStart"
        d.mkdir(parents=True, exist_ok=True)
        bad = d / "boom.sh"
        bad.write_text("#!/usr/bin/env bash\necho 'I am broken' >&2; exit 7\n")
        bad.chmod(0o755)

        with _HttpxCtx(_httpx_canned_chat_completion()):
            rc = agent.run_role(
                "researcher", cycle=1, task="hook fail test",
                provider_name="nvidia", model="x",
                max_iterations=1, is_subagent=True,
            )
        assert rc == 0, "run_role must return 0 even if hook crashed"
        # Confirm the model_call observability event still landed —
        # provider.call ran successfully in spite of the crashing hook.
        events = _read_jsonl(observability.OBS_DIR / "model_call.jsonl")
        assert len(events) >= 1, "model_call event must fire even if RoleStart hook crashed"
    finally:
        _restore_lab(orig)


def main() -> int:
    tests = [
        test_subagent_run_fires_lifecycle_observability_quota_watchdog,
        test_top_level_run_fires_brief_session_consolidator_evaluator,
        test_lifecycle_event_order,
        test_hooks_failure_doesnt_break_loop,
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
            import traceback
            print(f"  FAIL  {t.__name__} (exception)")
            print(f"        {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1
    print(f"\nAll {len(tests)} integration tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
