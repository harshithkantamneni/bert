"""Smoke test for F8 + F9 — Write tool emits finding events.

F8 (core/tools.py): writes to <active_lab>/findings/<name>.md auto-
emit a canvas-shape `finding` event to the active lab's events.jsonl.
F9 (tools/bert_run.py): researcher + strategist output_paths point at
findings/ so each successful cycle publishes 2 findings the
Manuscript surface can render.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


def test_write_to_findings_emits_finding_event() -> None:
    """A Write to findings/foo.md in an active lab triggers an
    event_class=finding append to <lab>/sor/events.jsonl."""
    from core.lab_context import reset_active_lab_path, set_active_lab_path
    from core.tools import _write

    with tempfile.TemporaryDirectory() as td:
        lab = Path(td) / "scratch-lab"
        lab.mkdir(parents=True)
        token = set_active_lab_path(lab)
        try:
            _write("findings/bert_run_C7_researcher.md",
                   "# Finding\n\nReal-shaped prose. Bert's first finding "
                   "on this scratch lab.")
            events_path = lab / "sor" / "events.jsonl"
            assert events_path.exists(), "events.jsonl not created"
            lines = events_path.read_text().splitlines()
            findings = [json.loads(l) for l in lines
                        if "finding" in l]
            assert findings, "no finding event written"
            ev = findings[-1]
            assert ev["event_class"] == "finding"
            assert ev["source_path"] == "findings/bert_run_C7_researcher.md"
            assert ev["agent"] == "researcher"
            assert ev["cycle"] == 7
            assert "Real-shaped prose" in ev["content"]
            assert "#finding" in ev["tags"]
        finally:
            reset_active_lab_path(token)


def test_write_to_non_findings_path_does_not_emit() -> None:
    """Writes to drafts/ or any other dir should NOT emit a finding."""
    from core.lab_context import reset_active_lab_path, set_active_lab_path
    from core.tools import _write

    with tempfile.TemporaryDirectory() as td:
        lab = Path(td) / "scratch-lab"
        lab.mkdir(parents=True)
        token = set_active_lab_path(lab)
        try:
            _write("drafts/foo.md", "# Draft\n\nNot a finding.")
            events_path = lab / "sor" / "events.jsonl"
            assert not events_path.exists() or "finding" not in events_path.read_text(), \
                "draft write should not emit finding event"
        finally:
            reset_active_lab_path(token)


def test_write_to_findings_archive_does_not_emit() -> None:
    """findings/archive/* are historical — no live finding event."""
    from core.lab_context import reset_active_lab_path, set_active_lab_path
    from core.tools import _write

    with tempfile.TemporaryDirectory() as td:
        lab = Path(td) / "scratch-lab"
        lab.mkdir(parents=True)
        token = set_active_lab_path(lab)
        try:
            _write("findings/archive/foo.md", "Old finding.")
            events_path = lab / "sor" / "events.jsonl"
            assert not events_path.exists() or "finding" not in events_path.read_text(), \
                "archive write should not emit finding event"
        finally:
            reset_active_lab_path(token)


def test_write_to_findings_non_md_does_not_emit() -> None:
    """Only .md files in findings/ count as findings."""
    from core.lab_context import reset_active_lab_path, set_active_lab_path
    from core.tools import _write

    with tempfile.TemporaryDirectory() as td:
        lab = Path(td) / "scratch-lab"
        lab.mkdir(parents=True)
        token = set_active_lab_path(lab)
        try:
            _write("findings/data.json", '{"x":1}')
            events_path = lab / "sor" / "events.jsonl"
            assert not events_path.exists() or "finding" not in events_path.read_text(), \
                "non-.md write should not emit finding event"
        finally:
            reset_active_lab_path(token)


def test_bert_run_writes_findings_not_drafts() -> None:
    """F9 — per-role output_paths point at findings/, not drafts/.
    Verifies bert_run._run_one_cycle's spec construction.

    Sprint 1 generalized the hardcoded researcher/strategist paths to a
    single roster-driven `findings/bert_run_C{cycle}_{role}.md` form
    (the roster is now schema-derived, not a fixed researcher→strategist
    pair). The F9 invariant — findings/ not drafts/ — is unchanged."""
    import inspect

    from tools import bert_run as br
    src = inspect.getsource(br._run_one_cycle)
    # Generalized roster-driven output path (covers researcher,
    # strategist, literature_hunter, writer, analyst, ... — any role)
    assert "findings/bert_run_C{cycle}_{role}.md" in src
    # And explicitly NOT drafts/ for cycle output paths
    assert "drafts/bert_run_C{cycle}" not in src


def test_finding_id_is_deterministic() -> None:
    """Same source_path → same id, so re-writes update rather than
    duplicate. Matches canvas_watcher's _hash_id convention."""
    from core.lab_context import reset_active_lab_path, set_active_lab_path
    from core.tools import _write

    with tempfile.TemporaryDirectory() as td:
        lab = Path(td) / "scratch-lab"
        lab.mkdir(parents=True)
        token = set_active_lab_path(lab)
        try:
            _write("findings/same.md", "# A\n\nOne.")
            _write("findings/same.md", "# A\n\nTwo (updated).")
            events_path = lab / "sor" / "events.jsonl"
            lines = events_path.read_text().splitlines()
            events = [json.loads(l) for l in lines]
            findings = [e for e in events if e.get("event_class") == "finding"]
            assert len(findings) == 2, \
                "expected 2 events (both write attempts emit)"
            assert findings[0]["id"] == findings[1]["id"], \
                "same path → same id (deterministic hash)"
        finally:
            reset_active_lab_path(token)


def main() -> int:
    tests = [
        test_write_to_findings_emits_finding_event,
        test_write_to_non_findings_path_does_not_emit,
        test_write_to_findings_archive_does_not_emit,
        test_write_to_findings_non_md_does_not_emit,
        test_bert_run_writes_findings_not_drafts,
        test_finding_id_is_deterministic,
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
