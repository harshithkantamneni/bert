"""TDD for WS0b: core.memory must scope its DB + index dirs to the ACTIVE lab
(via lab_context) so a benchmark corpus is retrieved from the corpus, not from
bert's global memory.db. With no active lab it falls back to the module
constants (DB_PATH/INDEX_DIRS/LAB_ROOT) — fully backward-compatible."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import lab_context  # noqa: E402
from core import memory as mem  # noqa: E402


def _with_lab(lab: Path, fn):
    tok = lab_context.set_active_lab_path(lab)
    try:
        return fn()
    finally:
        lab_context.reset_active_lab_path(tok)


def test_db_path_follows_active_lab(tmp_path):
    assert _with_lab(tmp_path, lambda: mem._db_path()) == tmp_path / "memory.db"
    # No active lab -> module constant (back-compat with the existing suite).
    assert mem._db_path() == mem.DB_PATH


def test_index_dirs_follow_active_lab(tmp_path):
    dirs = _with_lab(tmp_path, lambda: mem._index_dirs())
    assert dirs == [tmp_path / "memories", tmp_path / "findings"]
    assert mem._index_dirs() == mem.INDEX_DIRS


def test_create_scopes_to_active_lab(tmp_path):
    res = _with_lab(tmp_path, lambda: mem.create("findings/note.md", "hello world"))
    assert res["ok"] is True
    assert (tmp_path / "findings" / "note.md").read_text() == "hello world"
    # It must NOT have written under the repo root.
    assert not (mem.LAB_ROOT / "findings" / "note.md").exists() or \
        mem.LAB_ROOT == tmp_path


def test_create_rejects_outside_active_lab(tmp_path):
    res = _with_lab(tmp_path, lambda: mem.create("state/x.md", "x"))
    assert res["ok"] is False                     # only memories/ or findings/


def test_ingest_corpus_writes_shards_under_active_lab(tmp_path):
    src = tmp_path / "src"
    (src / "pkg").mkdir(parents=True)
    (src / "a.py").write_text("def alpha(): return 1\n")
    (src / "pkg" / "b.md").write_text("# Beta\nsome docs\n")
    lab = tmp_path / "lab"
    lab.mkdir()
    n = _with_lab(lab, lambda: mem.ingest_corpus(src, eager_index=False))
    assert n == 2
    corpus = lab / "findings" / "corpus"
    written = sorted(p.name for p in corpus.rglob("*.md"))
    assert "a.py.md" in written and "b.md.md" in written


def main() -> int:
    import inspect
    import tempfile
    tests = [test_db_path_follows_active_lab, test_index_dirs_follow_active_lab,
             test_create_scopes_to_active_lab, test_create_rejects_outside_active_lab,
             test_ingest_corpus_writes_shards_under_active_lab]
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
