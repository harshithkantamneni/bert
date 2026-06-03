"""Smoke + TDD: core/tool_synthesizer.py — sandboxed tool synthesis (Sprint 6 #30).

AC: generated tools are sandboxed + reviewed before use. Pipeline:
  synthesize(spec) -> LLM generates source + smoke test
  static_safety_scan(source) -> AST defense-in-depth (NOT the primary control)
  sandbox_validate(source, smoke) -> run in core.sandbox (real containment)
  propose(candidate) -> write state/tools_pending_pi.md review gate
  install(name, source) -> ONLY after PI bless: write lib + register in registry

Layers proven network-free:
  - static_safety_scan / _parse_generated: pure (AST + JSON parse).
  - sandbox_validate: real subprocess via core.sandbox, pinned TRUSTED in tests.
  - propose / install: filesystem + registry, tmp-isolated.
  - synthesize: provider stubbed.

Invariant under test: a synthesized tool is NEVER registered/callable before
install() (the PI-blessed step). The sandbox is the real containment; the AST
scan only flags foot-guns for PI attention, it does not gate the pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import tool_synthesizer as ts  # noqa: E402
from core.sandbox import Tier  # noqa: E402
from core.types import ProviderResponse  # noqa: E402

_SAFE_SRC = (
    "def add_two(**kwargs):\n"
    "    return {'sum': int(kwargs.get('a', 0)) + int(kwargs.get('b', 0))}\n"
)
_SAFE_SMOKE = (
    "assert add_two(a=2, b=3)['sum'] == 5\n"
    "print('ok')\n"
)


def _spec():
    return ts.ToolSpec(
        name="add_two",
        description="Add two integers a and b.",
        params_schema={"type": "object", "properties": {
            "a": {"type": "integer"}, "b": {"type": "integer"}}},
        returns="{'sum': int}",
        implementation_hint="return a+b",
    )


# ── static safety scan (pure AST) ────────────────────────────────────


def test_scan_flags_eval_and_exec():
    res = ts.static_safety_scan("def f(**k):\n    return eval('1+1')\n")
    assert res.safe is False
    assert any("eval" in v for v in res.violations)


def test_scan_flags_dangerous_import():
    res = ts.static_safety_scan("import subprocess\ndef f(**k):\n    return 1\n")
    assert res.safe is False
    assert any("subprocess" in v for v in res.violations)


def test_scan_flags_os_system_attr():
    res = ts.static_safety_scan("import os\ndef f(**k):\n    return os.system('ls')\n")
    assert res.safe is False
    assert any("os.system" in v for v in res.violations)


def test_scan_clean_source_is_safe():
    res = ts.static_safety_scan(_SAFE_SRC)
    assert res.safe is True
    assert res.violations == []


def test_scan_syntax_error_is_unsafe():
    res = ts.static_safety_scan("def f(:\n")  # not valid python
    assert res.safe is False
    assert any("syntax" in v.lower() for v in res.violations)


# ── parse generated payload (pure) ───────────────────────────────────


def test_parse_generated_valid():
    obj = {"source": _SAFE_SRC, "smoke_test": _SAFE_SMOKE}
    parsed = ts._parse_generated(obj)
    assert parsed is not None
    assert parsed[0] == _SAFE_SRC and parsed[1] == _SAFE_SMOKE


def test_parse_generated_missing_source_is_none():
    assert ts._parse_generated({"smoke_test": "x"}) is None
    assert ts._parse_generated({"source": ""}) is None


# ── synthesize (stubbed provider) ────────────────────────────────────


def _stub(mp, *, body):
    from core import provider as prov

    def fake_call(provider_name, messages, **kw):
        return ProviderResponse(
            text=json.dumps(body), tool_calls=[], finish_reason="stop",
            usage_prompt_tokens=10, usage_completion_tokens=10,
            usage_thinking_tokens=0, usage_cached_tokens=0, model="stub",
            provider=provider_name, elapsed_ms=1)

    mp.setattr(prov, "call", fake_call)


def test_synthesize_returns_candidate(monkeypatch):
    _stub(monkeypatch, body={"source": _SAFE_SRC, "smoke_test": _SAFE_SMOKE})
    cand = ts.synthesize(_spec(), cascade=[("groq", "m1")])
    assert cand.method == "llm-v1"
    assert cand.error is None
    assert "add_two" in cand.source
    assert cand.scan.safe is True


def test_synthesize_unavailable_does_not_crash(monkeypatch):
    from core import provider as prov

    def dead(provider_name, messages, **kw):
        return ProviderResponse(
            text="[bert] down", tool_calls=[], finish_reason="error",
            usage_prompt_tokens=0, usage_completion_tokens=0,
            usage_thinking_tokens=0, usage_cached_tokens=0, model="x",
            provider=provider_name, elapsed_ms=1)

    monkeypatch.setattr(prov, "call", dead)
    cand = ts.synthesize(_spec(), cascade=[("groq", "m1")])
    assert cand.method == "unavailable"
    assert cand.error is not None
    assert cand.source == ""


# ── sandbox validation (real subprocess, pinned TRUSTED) ─────────────


def test_sandbox_validate_passing_tool(tmp_path):
    res = ts.sandbox_validate(_SAFE_SRC, _SAFE_SMOKE, name="add_two",
                              tier=Tier.TRUSTED, work_dir=tmp_path)
    assert res.exit_code == 0
    assert "ok" in res.stdout


def test_sandbox_validate_failing_tool(tmp_path):
    bad_smoke = "assert add_two(a=1, b=1)['sum'] == 99\n"
    res = ts.sandbox_validate(_SAFE_SRC, bad_smoke, name="add_two",
                              tier=Tier.TRUSTED, work_dir=tmp_path)
    assert res.exit_code != 0


# ── name validation (path-traversal hardening) ──────────────────────


def test_valid_name_accepts_identifier_rejects_traversal():
    assert ts.is_valid_tool_name("add_two") is True
    assert ts.is_valid_tool_name("_x9") is True
    assert ts.is_valid_tool_name("../evil") is False
    assert ts.is_valid_tool_name("a/b") is False
    assert ts.is_valid_tool_name("a.b") is False
    assert ts.is_valid_tool_name("9bad") is False
    assert ts.is_valid_tool_name("") is False
    assert ts.is_valid_tool_name("x" * 65) is False


def test_sandbox_validate_rejects_traversal_name(tmp_path):
    try:
        ts.sandbox_validate(_SAFE_SRC, _SAFE_SMOKE, name="../evil",
                            tier=Tier.TRUSTED, work_dir=tmp_path)
        raise AssertionError("expected ValueError for traversal name")
    except ValueError:
        pass


def test_install_rejects_traversal_name(tmp_path):
    try:
        ts.install("../evil", _SAFE_SRC, "d", {}, lib_dir=tmp_path)
        raise AssertionError("expected ValueError for traversal name")
    except ValueError:
        pass


# ── review gate + install (filesystem + registry) ────────────────────


def test_propose_writes_review_gate_and_does_not_register(tmp_path):
    from core import tool_registry
    cand = ts.SynthesisCandidate(
        spec=_spec(), source=_SAFE_SRC, smoke_test=_SAFE_SMOKE,
        scan=ts.static_safety_scan(_SAFE_SRC), sandbox=None, method="llm-v1")
    gate = tmp_path / "tools_pending_pi.md"
    pid = ts.propose(cand, proposals_path=gate, pending_dir=tmp_path / "pending")
    assert pid.startswith("tool-")
    body = gate.read_text()
    assert "add_two" in body and "pending" in body
    # The crucial AC invariant: NOT registered/callable before install.
    assert tool_registry.get("add_two") is None


def test_install_writes_lib_and_registers(tmp_path):
    from core import tool_registry
    assert tool_registry.get("install_me_tool") is None
    src = _SAFE_SRC.replace("add_two", "install_me_tool")
    path = ts.install("install_me_tool", src, _spec().description,
                      _spec().params_schema, lib_dir=tmp_path)
    assert path.exists()
    td = tool_registry.get("install_me_tool")
    assert td is not None
    assert td.source == "creator"
    out = td.handler(a=4, b=5)
    assert out["sum"] == 9


# ── standalone runner ────────────────────────────────────────────────


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


def main() -> int:
    import inspect
    import tempfile
    tests = [
        test_scan_flags_eval_and_exec,
        test_scan_flags_dangerous_import,
        test_scan_flags_os_system_attr,
        test_scan_clean_source_is_safe,
        test_scan_syntax_error_is_unsafe,
        test_parse_generated_valid,
        test_parse_generated_missing_source_is_none,
        test_valid_name_accepts_identifier_rejects_traversal,
        test_sandbox_validate_rejects_traversal_name,
        test_install_rejects_traversal_name,
        test_synthesize_returns_candidate,
        test_synthesize_unavailable_does_not_crash,
        test_sandbox_validate_passing_tool,
        test_sandbox_validate_failing_tool,
        test_propose_writes_review_gate_and_does_not_register,
        test_install_writes_lib_and_registers,
    ]
    mp = _MP()
    for t in tests:
        params = inspect.signature(t).parameters
        try:
            if "monkeypatch" in params:
                t(mp)
            elif "tmp_path" in params:
                with tempfile.TemporaryDirectory() as d:
                    t(Path(d))
            else:
                t()
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
