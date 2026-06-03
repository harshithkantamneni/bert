"""Smoke test for core/hooks.py — event-driven hook runner.

Tests:
  1. KNOWN_EVENTS includes the expected event classes
  2. register() writes script + chmods executable
  3. list_hooks() filters by event correctly
  4. fire() runs registered hooks in lex order
  5. fire() passes payload to hook on stdin
  6. fire() captures non-zero exit + stderr in outcome
  7. fire() respects timeout
  8. fire() on unknown event returns empty report
  9. Python hooks dispatch via venv interpreter

Run: `.venv/bin/python tests/_smoke_hooks.py`
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import hooks  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="bert_hooks_smoke_"))
hooks.HOOKS_DIR = TMP / "hooks"


def _reset() -> None:
    if hooks.HOOKS_DIR.exists():
        import shutil
        shutil.rmtree(hooks.HOOKS_DIR)


def test_known_events_include_lifecycle() -> None:
    for ev in ["PreToolUse", "PostToolUse", "RoleStart", "EvaluatorVerdict",
               "ApprovalRequested", "SeasoningEntry"]:
        assert ev in hooks.KNOWN_EVENTS, f"missing {ev}"


def test_register_writes_executable() -> None:
    _reset()
    p = hooks.register("PreToolUse", "log-tool", "echo hello")
    assert p.exists()
    import os
    assert os.access(p, os.X_OK), "registered hook should be executable"
    assert p.read_text().startswith("#!/usr/bin/env bash"), "shebang prepended"


def test_register_rejects_unknown_event() -> None:
    _reset()
    try:
        hooks.register("UnknownEvent", "x", "echo x")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_list_hooks_filter() -> None:
    _reset()
    hooks.register("PreToolUse", "a", "echo A")
    hooks.register("PreToolUse", "b", "echo B")
    hooks.register("PostToolUse", "c", "echo C")
    pre = hooks.list_hooks("PreToolUse")
    post = hooks.list_hooks("PostToolUse")
    assert len(pre) == 2 and len(post) == 1
    # lex order
    assert pre[0].name < pre[1].name


def test_fire_runs_in_lex_order() -> None:
    _reset()
    # Two hooks that both append their name to a marker file
    marker = TMP / "marker.txt"
    hooks.register("PreToolUse", "01-first",
                   f"echo first >> {marker}")
    hooks.register("PreToolUse", "02-second",
                   f"echo second >> {marker}")
    rep = hooks.fire("PreToolUse", {"k": "v"}, timeout_secs=5)
    assert rep.all_passed, f"all hooks should pass; outcomes={rep.outcomes}"
    assert len(rep.outcomes) == 2
    text = marker.read_text()
    assert text.index("first") < text.index("second"), "lex order violated"


def test_fire_passes_payload_via_stdin() -> None:
    _reset()
    out_file = TMP / "stdin_capture.txt"
    hooks.register("PostToolUse", "echo-stdin",
                   f"cat > {out_file}")
    rep = hooks.fire("PostToolUse", {"tool": "Bash", "ok": True})
    assert rep.all_passed
    assert out_file.exists()
    payload = json.loads(out_file.read_text())
    assert payload == {"tool": "Bash", "ok": True}


def test_fire_captures_failures() -> None:
    _reset()
    hooks.register("RoleStart", "fail-hook",
                   "echo on-stderr >&2; exit 7")
    rep = hooks.fire("RoleStart", {})
    assert not rep.all_passed
    assert rep.fail_count == 1
    o = rep.outcomes[0]
    assert o.exit_code == 7
    assert "on-stderr" in o.stderr


def test_fire_respects_timeout() -> None:
    _reset()
    hooks.register("Stop", "slow", "sleep 5")
    import time
    t0 = time.monotonic()
    rep = hooks.fire("Stop", {}, timeout_secs=1)
    elapsed = time.monotonic() - t0
    assert rep.outcomes[0].timed_out
    assert rep.outcomes[0].exit_code == 124
    assert elapsed < 3.0, f"timeout not enforced; took {elapsed:.1f}s"


def test_fire_unknown_event_empty_report() -> None:
    _reset()
    rep = hooks.fire("MysteryEvent", {})
    assert rep.outcomes == []
    assert rep.all_passed  # vacuously


def test_python_hook_dispatch() -> None:
    _reset()
    out_file = TMP / "py_capture.txt"
    py = (
        "import sys, json\n"
        "p = json.load(sys.stdin)\n"
        f"open('{out_file}','w').write(p['x'])\n"
    )
    hooks.register("ModelCall", "py-hook", py, lang="py")
    rep = hooks.fire("ModelCall", {"x": "captured"})
    assert rep.all_passed, f"outcomes={rep.outcomes}"
    assert out_file.read_text() == "captured"


def main() -> int:
    tests = [
        test_known_events_include_lifecycle,
        test_register_writes_executable,
        test_register_rejects_unknown_event,
        test_list_hooks_filter,
        test_fire_runs_in_lex_order,
        test_fire_passes_payload_via_stdin,
        test_fire_captures_failures,
        test_fire_respects_timeout,
        test_fire_unknown_event_empty_report,
        test_python_hook_dispatch,
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
            print(f"  FAIL  {t.__name__} (exception)")
            print(f"        {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
