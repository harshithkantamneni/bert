"""Smoke + TDD: Write/Read/Edit lab-relative path containment (Sprint 7 hardening #7).

The adversarial bug-hunt flagged that a lab-RELATIVE path containing `..` escapes
the active lab root (e.g. finalize output_path='../../../etc/passwd' resolves under
active_lab/../../../ and writes outside the lab). The three file tools share
_resolve_relative_path, so the guard lives there: a relative path must resolve to
within the active root, else it's rejected. ABSOLUTE paths remain a documented
escape hatch (the tool schema allows them, tests use tmp paths, and an agent with
Bash can already write anywhere) — only the surprising relative-escape is closed.
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import lab_context, tools  # noqa: E402


def test_write_relative_escape_rejected(tmp_path):
    lab = tmp_path / "lab"
    lab.mkdir()
    tok = lab_context.set_active_lab_path(lab)
    try:
        out = tools._write("../escape.txt", "pwned")
        assert "error" in out.lower()
        # the escape target (one level up, still inside the tmp tree) is untouched
        assert not (tmp_path / "escape.txt").exists()
    finally:
        lab_context.reset_active_lab_path(tok)


def test_write_relative_in_lab_ok(tmp_path):
    tok = lab_context.set_active_lab_path(tmp_path)
    try:
        out = tools._write("findings/ok.md", "# ok")
        assert out.startswith("wrote")
        assert (tmp_path / "findings" / "ok.md").read_text() == "# ok"
    finally:
        lab_context.reset_active_lab_path(tok)


def test_write_nested_relative_ok(tmp_path):
    tok = lab_context.set_active_lab_path(tmp_path)
    try:
        out = tools._write("a/b/c.md", "x")
        assert out.startswith("wrote")
    finally:
        lab_context.reset_active_lab_path(tok)


def test_write_sneaky_midpath_escape_rejected(tmp_path):
    lab = tmp_path / "lab"
    lab.mkdir()
    tok = lab_context.set_active_lab_path(lab)
    try:
        out = tools._write("findings/../../escape.txt", "x")
        assert "error" in out.lower()
        assert not (tmp_path / "escape.txt").exists()
    finally:
        lab_context.reset_active_lab_path(tok)


def test_write_absolute_path_still_allowed(tmp_path):
    # Absolute paths are a documented escape hatch (tests + schema rely on it).
    target = tmp_path / "abs" / "note.txt"
    out = tools._write(str(target), "hello")
    assert out.startswith("wrote")
    assert target.read_text() == "hello"


def test_read_relative_escape_rejected(tmp_path):
    tok = lab_context.set_active_lab_path(tmp_path)
    try:
        out = tools._read("../../etc/passwd")
        assert "error" in out.lower()
    finally:
        lab_context.reset_active_lab_path(tok)


def test_edit_relative_escape_rejected(tmp_path):
    tok = lab_context.set_active_lab_path(tmp_path)
    try:
        out = tools._edit("../escape.txt", "a", "b")
        assert out["ok"] is False and out["error"]
    finally:
        lab_context.reset_active_lab_path(tok)


def main() -> int:
    import inspect
    import tempfile
    tests = [
        test_write_relative_escape_rejected,
        test_write_relative_in_lab_ok,
        test_write_nested_relative_ok,
        test_write_sneaky_midpath_escape_rejected,
        test_write_absolute_path_still_allowed,
        test_read_relative_escape_rejected,
        test_edit_relative_escape_rejected,
    ]
    for t in tests:
        try:
            if "tmp_path" in inspect.signature(t).parameters:
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
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
