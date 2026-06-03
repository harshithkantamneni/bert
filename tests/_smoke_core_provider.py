"""Smoke: core/provider.py — provider HTTP shim (was 52%).

Pure helpers (_retry_after_seconds, _serialize_messages, _parse_response)
run directly; call() + probe_models() run against a faked httpx.Client
with config.load / quota.record_call / observability.emit stubbed so no
network, no real credentials, no quota-DB writes. retry_max=0 keeps the
retry branches from sleeping. Covers: unknown-provider, missing-credential,
success, gemini max_tokens floor, HTTP 4xx, retryable-exhausted (circuit
breaker), network-error, and every probe_models branch.
"""

from __future__ import annotations

import inspect
import json
import sys
import types
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import provider as pv  # noqa: E402


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


class _Resp:
    def __init__(self, status=200, body=None, headers=None, text=""):
        self.status_code = status
        self._body = body or {}
        self.headers = headers or {}
        self.text = text or json.dumps(self._body)
    def json(self):
        return self._body


class _Client:
    def __init__(self, resp=None, raise_exc=None):
        self._resp = resp
        self._raise = raise_exc
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def post(self, *a, **k):
        if self._raise:
            raise self._raise
        return self._resp
    def get(self, *a, **k):
        if self._raise:
            raise self._raise
        return self._resp


def _fake_httpx(resp=None, raise_exc=None):
    return lambda *a, **k: _Client(resp, raise_exc)


def _stub_deps(mp, creds=None):
    mp.setattr(pv.config, "load", lambda: creds if creds is not None else {})
    from core import cost_ledger, observability, quota
    mp.setattr(quota, "record_call", lambda *a, **k: None)
    mp.setattr(observability, "emit", lambda *a, **k: None)
    mp.setattr(cost_ledger, "record", lambda *a, **k: {})  # no real ledger write


# ── pure helpers ──────────────────────────────────────────────────────

def test_retry_after_seconds():
    assert pv._retry_after_seconds({"retry-after": "5"}) == 5.0
    assert pv._retry_after_seconds({"Retry-After": "100"}) == 60.0   # capped
    assert pv._retry_after_seconds({}) is None
    assert pv._retry_after_seconds({"retry-after": "soon"}) is None  # HTTP-date → None


def test_serialize_messages():
    msg = types.SimpleNamespace(
        role="assistant", content="hi",
        tool_calls=[types.SimpleNamespace(id="t1", name="Read", arguments={"file_path": "x"})],
        tool_call_id=None, name=None)
    tool_msg = types.SimpleNamespace(
        role="tool", content="result", tool_calls=None, tool_call_id="t1", name="Read")
    out = pv._serialize_messages([msg, tool_msg, {"role": "user", "content": "raw"}])
    assert out[0]["tool_calls"][0]["function"]["name"] == "Read"
    assert out[1]["tool_call_id"] == "t1" and out[1]["name"] == "Read"
    assert out[2] == {"role": "user", "content": "raw"}   # dict passthrough


def test_parse_response_text_and_usage():
    raw = {
        "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4,
                  "prompt_tokens_details": {"cached_tokens": 3},
                  "completion_tokens_details": {"reasoning_tokens": 2}},
        "model": "m1",
    }
    r = pv._parse_response("nvidia", raw)
    assert r.text == "hello" and r.usage_prompt_tokens == 12
    assert r.usage_cached_tokens == 3 and r.usage_thinking_tokens == 2


def test_parse_response_tool_calls():
    raw = {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "tc1", "function": {"name": "Bash", "arguments": '{"command":"ls"}'}},
        {"function": {}},  # malformed → skipped
    ]}}], "model": "m"}
    r = pv._parse_response("groq", raw)
    assert r.finish_reason == "tool_use"
    assert len(r.tool_calls) == 1 and r.tool_calls[0].name == "Bash"
    assert r.tool_calls[0].arguments == {"command": "ls"}


def test_parse_response_malformed():
    r = pv._parse_response("nvidia", {})   # no choices → error
    assert r.finish_reason == "error" and "failed to parse" in r.text


# ── call() ────────────────────────────────────────────────────────────

def test_call_unknown_provider():
    r = pv.call("no_such_provider", [])
    assert r.finish_reason == "error" and "unknown provider" in r.text


def test_call_missing_credential(monkeypatch):
    _stub_deps(monkeypatch, creds={})  # no NVIDIA_API_KEY
    r = pv.call("nvidia", [{"role": "user", "content": "hi"}])
    assert r.finish_reason == "error" and "missing credential" in r.text


def test_call_success(monkeypatch):
    _stub_deps(monkeypatch, creds={"NVIDIA_API_KEY": "k"})
    resp = _Resp(200, {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 1, "completion_tokens": 1}, "model": "m"})
    monkeypatch.setattr(pv.httpx, "Client", _fake_httpx(resp))
    r = pv.call("nvidia", [{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "X"}}], max_tokens=64)
    assert r.text == "ok" and r.finish_reason == "stop"


def test_call_gemini_max_tokens_floor(monkeypatch):
    _stub_deps(monkeypatch, creds={"GOOGLE_AI_API_KEY": "k"})
    resp = _Resp(200, {"choices": [{"message": {"content": "g"}, "finish_reason": "stop"}], "model": "m"})
    monkeypatch.setattr(pv.httpx, "Client", _fake_httpx(resp))
    r = pv.call("gemini", [{"role": "user", "content": "hi"}], max_tokens=None)
    assert r.text == "g"


def test_call_http_400(monkeypatch):
    _stub_deps(monkeypatch, creds={"NVIDIA_API_KEY": "k"})
    monkeypatch.setattr(pv.httpx, "Client", _fake_httpx(_Resp(400, {}, text="bad request")))
    r = pv.call("nvidia", [{"role": "user", "content": "hi"}])
    assert r.finish_reason == "error" and "400" in r.text


def test_call_retryable_exhausted(monkeypatch):
    _stub_deps(monkeypatch, creds={"NVIDIA_API_KEY": "k"})
    monkeypatch.setattr(pv.httpx, "Client", _fake_httpx(_Resp(429, {}, text="slow down")))
    r = pv.call("nvidia", [{"role": "user", "content": "hi"}], retry_max=0)
    assert r.finish_reason == "error" and "rate-limited" in r.text


def test_call_network_error(monkeypatch):
    _stub_deps(monkeypatch, creds={"NVIDIA_API_KEY": "k"})
    monkeypatch.setattr(pv.httpx, "Client",
                        _fake_httpx(raise_exc=pv.httpx.TimeoutException("slow")))
    r = pv.call("nvidia", [{"role": "user", "content": "hi"}], retry_max=0)
    assert r.finish_reason == "error" and "network error" in r.text


# ── probe_models() ────────────────────────────────────────────────────

def test_probe_unknown():
    ok, ids, err = pv.probe_models("no_such_provider")
    assert ok is False and "unknown provider" in err


def test_probe_missing_credential(monkeypatch):
    _stub_deps(monkeypatch, creds={})
    ok, ids, err = pv.probe_models("nvidia")
    assert ok is False and "missing" in err


def test_probe_success_and_failures(monkeypatch):
    _stub_deps(monkeypatch, creds={"NVIDIA_API_KEY": "k"})
    monkeypatch.setattr(pv.httpx, "Client",
                        _fake_httpx(_Resp(200, {"data": [{"id": "m1"}, {"id": "m2"}]})))
    ok, ids, err = pv.probe_models("nvidia")
    assert ok and ids == ["m1", "m2"]
    # non-200
    monkeypatch.setattr(pv.httpx, "Client", _fake_httpx(_Resp(500, {}, text="oops")))
    ok2, _, err2 = pv.probe_models("nvidia")
    assert ok2 is False and "500" in err2
    # network error
    monkeypatch.setattr(pv.httpx, "Client",
                        _fake_httpx(raise_exc=pv.httpx.NetworkError("down")))
    ok3, _, err3 = pv.probe_models("nvidia")
    assert ok3 is False and "NetworkError" in err3


def main() -> int:
    tests = [
        test_retry_after_seconds,
        test_serialize_messages,
        test_parse_response_text_and_usage,
        test_parse_response_tool_calls,
        test_parse_response_malformed,
        test_call_unknown_provider,
        test_call_missing_credential,
        test_call_success,
        test_call_gemini_max_tokens_floor,
        test_call_http_400,
        test_call_retryable_exhausted,
        test_call_network_error,
        test_probe_unknown,
        test_probe_missing_credential,
        test_probe_success_and_failures,
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
