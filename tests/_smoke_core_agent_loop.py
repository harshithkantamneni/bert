"""Smoke: core/agent.py run_role loop driven by a scripted provider (62%→higher).

Stubs the one network seam (provider.call) to return a scripted sequence of
ProviderResponses, so the real agent loop runs offline as a sub-agent
(is_subagent=True skips the session/evaluator/consolidator finalize): the
model-call → emit → tool-dispatch → permission-gate → _execute_tool →
tool-result path, the stop exit, the provider-error CATASTROPHIC exit, and
the max-iterations-exhausted exit. Observability/session writes are stubbed
to no-ops so nothing touches the real lab state.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import agent  # noqa: E402
from core.types import ProviderResponse, ToolCall  # noqa: E402

_CONSTITUTIONAL = LAB_ROOT / "memories" / "governance" / "constitutional.md"


def _require_constitutional():
    if not _CONSTITUTIONAL.exists():
        pytest.skip("requires lab runtime artifact not shipped in the public repo")


class _MP:
    def __init__(self):
        self._u = []
    def setattr(self, obj, name, val):
        self._u.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)
    def undo(self):
        for o, n, v in reversed(self._u):
            setattr(o, n, v)
        self._u.clear()


def _resp(*, finish, text="", tool_calls=None):
    return ProviderResponse(
        text=text, tool_calls=tool_calls or [], finish_reason=finish,
        usage_prompt_tokens=50, usage_completion_tokens=20,
        usage_thinking_tokens=0, usage_cached_tokens=0,
        model="nvidia/test", provider="nvidia", elapsed_ms=120)


def _scripted(responses):
    seq = list(responses)
    def _call(provider_name, messages, **kw):
        return seq.pop(0) if seq else _resp(finish="stop", text="done")
    return _call


def _silence(monkeypatch):
    # Stub the advisory side-effecting seams so the loop touches no real state.
    monkeypatch.setattr(agent.observability, "emit", lambda *a, **k: None)
    monkeypatch.setattr(agent.observability, "emit_model_call", lambda *a, **k: None)
    monkeypatch.setattr(agent.log, "append_session_event", lambda *a, **k: None)


def test_stop_immediately(monkeypatch):
    _require_constitutional()
    _silence(monkeypatch)
    monkeypatch.setattr(agent.provider, "call", _scripted([_resp(finish="stop", text="done")]))
    sink: dict = {}
    rc = agent.run_role("researcher", cycle=1, task="do a thing", is_subagent=True,
                        telemetry_sink=sink, max_iterations=5)
    assert rc == 0
    assert sink["tokens_in"] == 50 and sink["model_used"] == "nvidia/test"


def test_tool_use_then_stop(monkeypatch):
    _require_constitutional()
    _silence(monkeypatch)
    monkeypatch.setattr(agent.provider, "call", _scripted([
        _resp(finish="tool_use",
              tool_calls=[ToolCall(id="t1", name="Bash", arguments={"command": "echo hi"})]),
        _resp(finish="stop", text="finished"),
    ]))
    sink: dict = {}
    rc = agent.run_role("researcher", cycle=2, task="run a tool", is_subagent=True,
                        telemetry_sink=sink, max_iterations=5)
    assert rc == 0
    assert sink["tokens_in"] == 100   # two model calls accumulated


def test_provider_error_is_catastrophic(monkeypatch):
    _silence(monkeypatch)
    monkeypatch.setattr(agent.provider, "call",
                        _scripted([_resp(finish="error", text="[bert] provider down")]))
    rc = agent.run_role("researcher", cycle=3, task="x", is_subagent=True, max_iterations=5)
    assert rc == 1   # CATASTROPHIC exit


def test_quota_error_fails_over_then_stops(monkeypatch):
    # A 429/quota error fails OVER to another lane instead of dying.
    from core import provider_fallback
    _require_constitutional()
    _silence(monkeypatch)
    monkeypatch.setattr(provider_fallback, "_has_credential", lambda p: True)
    monkeypatch.setattr(agent.provider, "call", _scripted([
        _resp(finish="error", text="[bert] rate-limited (429) by gemini after 5 attempts"),
        _resp(finish="stop", text="done on the fallback lane"),
    ]))
    rc = agent.run_role("researcher", cycle=4, task="x", is_subagent=True, max_iterations=5)
    assert rc == 0   # recovered on the fallback, NOT catastrophic


def test_unknown_provider_fails_over(monkeypatch):
    # The writer scenario: router resolves to anthropic-cli (host tier1) which the
    # standard executor can't call -> "unknown provider" -> must fail OVER, not die.
    from core import provider_fallback
    _require_constitutional()
    _silence(monkeypatch)
    monkeypatch.setattr(provider_fallback, "_has_credential", lambda p: True)
    monkeypatch.setattr(agent.provider, "call", _scripted([
        _resp(finish="error", text="[bert] unknown provider: anthropic-cli"),
        _resp(finish="stop", text="done on a real cloud lane"),
    ]))
    rc = agent.run_role("writer", cycle=6, task="x", is_subagent=True, max_iterations=5)
    assert rc == 0   # recovered on a cloud lane


def test_quota_error_no_fallback_is_catastrophic(monkeypatch):
    from core import provider_fallback
    _silence(monkeypatch)
    monkeypatch.setattr(provider_fallback, "_has_credential", lambda p: False)  # no lane
    monkeypatch.setattr(agent.provider, "call", _scripted([
        _resp(finish="error", text="[bert] rate-limited (429) by gemini after 5 attempts"),
    ]))
    rc = agent.run_role("researcher", cycle=5, task="x", is_subagent=True, max_iterations=5)
    assert rc == 1   # no fallback available -> still catastrophic


def test_max_iterations_exhausted(monkeypatch):
    _require_constitutional()
    _silence(monkeypatch)
    # always tool_use → loop never hits a stop → exhausts max_iterations
    def _always_tool(provider_name, messages, **kw):
        return _resp(finish="tool_use",
                     tool_calls=[ToolCall(id="t", name="Bash", arguments={"command": "echo x"})])
    monkeypatch.setattr(agent.provider, "call", _always_tool)
    rc = agent.run_role("researcher", cycle=4, task="loop", is_subagent=True, max_iterations=2)
    assert rc == 0   # CONTEXT_FULL is not catastrophic


def test_execute_tool_catches_any_handler_exception(monkeypatch):
    # A tool handler raising ANY exception (e.g. AttributeError from a malformed
    # model arg) must degrade to a tool-error result, never crash the loop.
    from core import tool_registry
    from core.types import PermissionMode

    def _boom(**kwargs):
        raise AttributeError("'str' object has no attribute 'get'")

    tool_registry.register_function(
        name="_boom_tool", description="raises", parameters_schema={"type": "object"},
        handler=_boom, permission_mode=PermissionMode.AUTO)
    res = agent._execute_tool(ToolCall(id="b", name="_boom_tool", arguments={"x": "str"}))
    assert res.error is not None
    assert "AttributeError" in res.content   # returned to the model, not raised


def test_stop_without_deliverable_nudges_then_accepts_after_write(monkeypatch):
    # The bug: model summarizes in chat + stops WITHOUT writing its required
    # deliverable. The loop must nudge it to actually Write, then accept once the
    # file exists.
    import tempfile

    from core import lab_context
    _require_constitutional()
    _silence(monkeypatch)
    calls = {"n": 0}

    def _script(provider_name, messages, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return _resp(finish="stop", text="here is a brief summary")  # no write
        if calls["n"] == 2:
            return _resp(finish="tool_use", tool_calls=[ToolCall(
                id="w", name="Write",
                arguments={"file_path": "findings/x.md", "content": "# finding\nbody"})])
        return _resp(finish="stop", text="done")  # file now exists

    monkeypatch.setattr(agent.provider, "call", _script)
    with tempfile.TemporaryDirectory() as d:
        tok = lab_context.set_active_lab_path(Path(d))
        try:
            rc = agent.run_role("literature_hunter", cycle=7, task="x", is_subagent=True,
                                output_path="findings/x.md", max_iterations=8)
        finally:
            lab_context.reset_active_lab_path(tok)
        assert rc == 0
        assert calls["n"] == 3   # stop(nudge) -> write -> stop(accept)
        assert (Path(d) / "findings" / "x.md").exists()


def test_undersized_deliverable_nudges_with_verification_failures(monkeypatch):
    # File EXISTS but fails the verification spec (too short) -> nudge with the
    # exact failure -> model rewrites a compliant one -> accept.
    import tempfile

    from core import lab_context
    _require_constitutional()
    _silence(monkeypatch)
    long = "# Heading\n" + ("words " * 40)   # >50 chars, has a header
    seq = [
        _resp(finish="tool_use", tool_calls=[ToolCall(
            id="w1", name="Write",
            arguments={"file_path": "findings/y.md", "content": "hi"})]),  # too short
        _resp(finish="stop", text="done?"),                                # premature
        _resp(finish="tool_use", tool_calls=[ToolCall(
            id="w2", name="Write",
            arguments={"file_path": "findings/y.md", "content": long})]),  # compliant
        _resp(finish="stop", text="done"),
    ]
    monkeypatch.setattr(agent.provider, "call", _scripted(seq))
    vspec = {"output_required": True, "min_chars": 50}
    with tempfile.TemporaryDirectory() as d:
        tok = lab_context.set_active_lab_path(Path(d))
        try:
            rc = agent.run_role("writer", cycle=11, task="x", is_subagent=True,
                                output_path="findings/y.md", verification_spec=vspec,
                                max_iterations=10)
        finally:
            lab_context.reset_active_lab_path(tok)
        assert rc == 0
        assert len((Path(d) / "findings" / "y.md").read_text()) >= 50  # compliant now


def test_stop_with_deliverable_present_accepts_immediately(monkeypatch):
    import tempfile

    from core import lab_context
    _require_constitutional()
    _silence(monkeypatch)
    calls = {"n": 0}

    def _script(provider_name, messages, **kw):
        calls["n"] += 1
        return _resp(finish="stop", text="done")

    monkeypatch.setattr(agent.provider, "call", _script)
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "findings").mkdir()
        (Path(d) / "findings" / "x.md").write_text("already written")
        tok = lab_context.set_active_lab_path(Path(d))
        try:
            rc = agent.run_role("literature_hunter", cycle=8, task="x", is_subagent=True,
                                output_path="findings/x.md", max_iterations=5)
        finally:
            lab_context.reset_active_lab_path(tok)
        assert rc == 0
        assert calls["n"] == 1   # deliverable present -> no nudge, accept first stop


def test_completion_nudge_is_bounded(monkeypatch):
    # Model never writes -> nudges are capped, loop gives up (does NOT spin to
    # max_iterations).
    import tempfile

    from core import lab_context
    _require_constitutional()
    _silence(monkeypatch)
    calls = {"n": 0}

    def _always_stop(provider_name, messages, **kw):
        calls["n"] += 1
        return _resp(finish="stop", text="summary, still not writing")

    monkeypatch.setattr(agent.provider, "call", _always_stop)
    with tempfile.TemporaryDirectory() as d:
        tok = lab_context.set_active_lab_path(Path(d))
        try:
            agent.run_role("literature_hunter", cycle=9, task="x", is_subagent=True,
                           output_path="findings/missing.md", max_iterations=10)
        finally:
            lab_context.reset_active_lab_path(tok)
        assert calls["n"] == agent._MAX_COMPLETION_NUDGES + 1  # bounded, not 10


def main() -> int:
    tests = [
        test_stop_immediately,
        test_tool_use_then_stop,
        test_provider_error_is_catastrophic,
        test_quota_error_fails_over_then_stops,
        test_unknown_provider_fails_over,
        test_quota_error_no_fallback_is_catastrophic,
        test_max_iterations_exhausted,
        test_execute_tool_catches_any_handler_exception,
        test_stop_without_deliverable_nudges_then_accepts_after_write,
        test_undersized_deliverable_nudges_with_verification_failures,
        test_stop_with_deliverable_present_accepts_immediately,
        test_completion_nudge_is_bounded,
    ]
    for t in tests:
        mp = _MP()
        try:
            kwargs = {"monkeypatch": mp} if "monkeypatch" in inspect.signature(t).parameters else {}
            t(**kwargs)
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1
        finally:
            mp.undo()
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
