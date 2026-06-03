"""Smoke: core/agent.py helpers — _execute_tool + _load_system_prompt (52%).

The full run_role loop is covered by the live E2E smokes (it needs a real
provider). Here we cover the two pure-ish helpers directly against the real
tool_registry + repo prompt files: tool dispatch (unknown / str-result /
dict-result / oversized-truncation / handler-error) and system-prompt
assembly (real role + missing-role fallback).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import (
    agent,  # noqa: E402
    )
from core.types import ToolCall  # noqa: E402

_CONSTITUTIONAL = LAB_ROOT / "memories" / "governance" / "constitutional.md"


def test_execute_tool_unknown():
    r = agent._execute_tool(ToolCall(id="t1", name="NoSuchTool", arguments={}))
    assert r.error == "unknown_tool" and "unknown tool" in r.content


def test_execute_tool_str_result(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("hello world")
    r = agent._execute_tool(ToolCall(id="t2", name="Read", arguments={"file_path": str(f)}))
    assert r.error is None and "hello world" in r.content


def test_execute_tool_dict_result():
    # Bash returns a dict → json-serialized content (cheap, no embedder;
    # memory_search would load the real model + index the corpus → too slow
    # for the coverage timeout, and it's covered network-free in _smoke_core_tools).
    r = agent._execute_tool(ToolCall(id="t3", name="Bash", arguments={"command": "echo hi"}))
    assert r.error is None and "exit_code" in r.content


def test_execute_tool_handler_error():
    # Bad kwargs → TypeError caught → wrapped tool error
    r = agent._execute_tool(ToolCall(id="t4", name="Read", arguments={"not_a_param": 1}))
    assert r.error is not None and "tool error" in r.content


def test_execute_tool_truncates_oversized(tmp_path):
    big = tmp_path / "big.txt"
    big.write_text("x" * 20000)
    r = agent._execute_tool(ToolCall(id="t5", name="Read", arguments={"file_path": str(big)}))
    assert r.truncated is True and "truncated" in r.content


def test_load_system_prompt():
    if not _CONSTITUTIONAL.exists():
        pytest.skip("requires lab runtime artifact not shipped in the public repo")
    sp = agent._load_system_prompt("researcher")
    assert isinstance(sp, str) and len(sp) > 100   # constitutional + role prompt
    # missing role → fallback marker, never crashes
    missing = agent._load_system_prompt("zzz_nonexistent_role")
    assert "role prompt missing" in missing


def main() -> int:
    import inspect
    import shutil
    import tempfile
    tests = [
        test_execute_tool_unknown,
        test_execute_tool_str_result,
        test_execute_tool_dict_result,
        test_execute_tool_handler_error,
        test_execute_tool_truncates_oversized,
        test_load_system_prompt,
    ]
    for t in tests:
        td = Path(tempfile.mkdtemp())
        try:
            kwargs = {"tmp_path": td} if "tmp_path" in inspect.signature(t).parameters else {}
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
            shutil.rmtree(td, ignore_errors=True)
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
