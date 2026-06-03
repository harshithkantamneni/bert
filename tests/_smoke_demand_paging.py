"""Smoke + TDD: demand paging — page out archived findings from the index.

v3+ demand paging: 56% of findings are never referenced. The safe, bounded slice
implemented here is the PAGE-OUT half: archived/superseded findings (which the
consolidator moves to */archive/ as stale) must not be embedded or surface in
live search. _list_corpus_files now excludes any */archive/* path, so:
  - new archived findings are never indexed (no wasted embedding), and
  - already-indexed findings that get archived are GC'd on the next re-index
    (orphan removal in _index_corpus, since they leave the corpus listing).

The riskier DEFER-PAGE-IN half (BM25-gated lazy embedding of never-referenced
LIVE findings) restructures the vector stage with retrieval-quality + determinism
implications — deferred to a measured step (the stale single-lab evidence doesn't
justify forcing it). This slice is correctness-positive on its own: live search
should not return historical/archived findings.
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import memory  # noqa: E402


def _corpus(tmp_path):
    (tmp_path / "findings").mkdir()
    (tmp_path / "findings" / "live.md").write_text("# live finding")
    (tmp_path / "findings" / "archive").mkdir()
    (tmp_path / "findings" / "archive" / "old.md").write_text("# archived finding")
    (tmp_path / "memories").mkdir()
    (tmp_path / "memories" / "note.md").write_text("# live memory")
    (tmp_path / "memories" / "archive").mkdir()
    (tmp_path / "memories" / "archive" / "stale.md").write_text("# archived memory")
    return [tmp_path / "memories", tmp_path / "findings"]


def test_archive_excluded_from_corpus(monkeypatch, tmp_path):
    monkeypatch.setattr(memory, "INDEX_DIRS", _corpus(tmp_path))
    names = {p.name for p in memory._list_corpus_files()}
    assert "live.md" in names and "note.md" in names      # live content indexed
    assert "old.md" not in names and "stale.md" not in names  # archived paged out


def test_nested_archive_excluded(monkeypatch, tmp_path):
    dirs = _corpus(tmp_path)
    deep = tmp_path / "findings" / "archive" / "2026-05"
    deep.mkdir()
    (deep / "deep.md").write_text("# deeply archived")
    monkeypatch.setattr(memory, "INDEX_DIRS", dirs)
    names = {p.name for p in memory._list_corpus_files()}
    assert "deep.md" not in names


def test_non_archive_subdir_still_indexed(monkeypatch, tmp_path):
    dirs = _corpus(tmp_path)
    sub = tmp_path / "findings" / "2026-05"   # a date subdir, NOT archive
    sub.mkdir()
    (sub / "dated.md").write_text("# dated live finding")
    monkeypatch.setattr(memory, "INDEX_DIRS", dirs)
    names = {p.name for p in memory._list_corpus_files()}
    assert "dated.md" in names  # only `archive` is excluded, not all subdirs


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
        test_archive_excluded_from_corpus,
        test_nested_archive_excluded,
        test_non_archive_subdir_still_indexed,
    ]
    mp = _MP()
    for t in tests:
        params = inspect.signature(t).parameters
        try:
            kwargs = {}
            tmpctx = None
            if "monkeypatch" in params:
                kwargs["monkeypatch"] = mp
            if "tmp_path" in params:
                tmpctx = tempfile.TemporaryDirectory()
                kwargs["tmp_path"] = Path(tmpctx.name)
            t(**kwargs)
            if tmpctx:
                tmpctx.cleanup()
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
