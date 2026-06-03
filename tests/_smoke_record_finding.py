"""Smoke + TDD: record_finding — macro-op fusion (memory v3+).

v3+ macro-op fusion: 25% of agent tool calls are adjacent Write+memory_create
pairs (write the finding artifact, then log it). record_finding fuses that pair
into ONE atomic tool: write the finding into the active lab's findings/, record
its lineage in the finding + the emitted finding event, and append a one-line
summary to memories/log.md. Active-lab-aware (like Write, not memory.create which
routes to the repo root) and traversal-contained.
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import lab_context, tools  # noqa: E402


def test_record_finding_writes_finding_and_log(tmp_path):
    tok = lab_context.set_active_lab_path(tmp_path)
    try:
        out = tools._record_finding(
            content="# RWKV beats BM25\nlinear attention wins on throughput.",
            name="rwkv-throughput", summary="RWKV wins on throughput",
            lineage=["prior-arch-survey"])
        assert out["ok"] is True
        # finding landed in the ACTIVE lab's findings/ (not the repo root)
        fpath = tmp_path / "findings" / "rwkv-throughput.md"
        assert fpath.exists()
        body = fpath.read_text()
        assert "linear attention wins" in body
        assert "prior-arch-survey" in body  # lineage recorded in the finding
        # the log got a one-line summary entry
        log = (tmp_path / "memories" / "log.md").read_text()
        assert "RWKV wins on throughput" in log
        assert out["lineage"] == ["prior-arch-survey"]
    finally:
        lab_context.reset_active_lab_path(tok)


def test_record_finding_no_summary_skips_log(tmp_path):
    tok = lab_context.set_active_lab_path(tmp_path)
    try:
        out = tools._record_finding(content="body", name="f1")
        assert out["ok"] is True
        assert out["log_path"] is None
        assert not (tmp_path / "memories" / "log.md").exists()
    finally:
        lab_context.reset_active_lab_path(tok)


def test_record_finding_appends_not_overwrites_log(tmp_path):
    tok = lab_context.set_active_lab_path(tmp_path)
    try:
        tools._record_finding(content="a", name="f1", summary="first finding")
        tools._record_finding(content="b", name="f2", summary="second finding")
        log = (tmp_path / "memories" / "log.md").read_text()
        assert "first finding" in log and "second finding" in log
    finally:
        lab_context.reset_active_lab_path(tok)


def test_record_finding_emits_finding_event_with_lineage(tmp_path):
    tok = lab_context.set_active_lab_path(tmp_path)
    try:
        tools._record_finding(content="x", name="f1", lineage=["p1", "p2"])
        events = (tmp_path / "sor" / "events.jsonl")
        assert events.exists()
        import json
        rows = [json.loads(ln) for ln in events.read_text().splitlines() if ln.strip()]
        finding_rows = [r for r in rows if r.get("event_class") == "finding"]
        assert finding_rows and finding_rows[-1]["lineage"] == ["p1", "p2"]
    finally:
        lab_context.reset_active_lab_path(tok)


def test_record_finding_rejects_traversal(tmp_path):
    tok = lab_context.set_active_lab_path(tmp_path)
    try:
        out = tools._record_finding(content="x", name="../../escape")
        assert out["ok"] is False
        assert not (tmp_path.parent / "escape.md").exists()
    finally:
        lab_context.reset_active_lab_path(tok)


def test_record_finding_registered():
    import core.tools  # noqa: F401
    from core import tool_registry
    assert tool_registry.get("record_finding") is not None


def main() -> int:
    import inspect
    import tempfile
    tests = [
        test_record_finding_writes_finding_and_log,
        test_record_finding_no_summary_skips_log,
        test_record_finding_appends_not_overwrites_log,
        test_record_finding_emits_finding_event_with_lineage,
        test_record_finding_rejects_traversal,
        test_record_finding_registered,
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
