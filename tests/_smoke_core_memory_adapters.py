"""Smoke: core/memory_adapters/{code_repo,document_corpus}.py (32%/59%).

code_repo runs for real against a temp lab (migrations auto-create the
files/symbols schema): pure regex symbol extraction + ingest (file +
recursive dir + missing + unsupported) + LIKE search (+ filters) + stats
+ get/delete + export. document_corpus stubs the heavy delegates
(core.memory.create, core.retrieval.hybrid_retrieve, core.memory.search,
core.graph_store.neighbors) so its source-routing + search-fallback +
related + file-based get/delete/stats/export logic runs network-free.
"""

from __future__ import annotations

import inspect
import shutil
import sys
import tempfile
import types
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core.memory_adapters import code_repo as cr  # noqa: E402
from core.memory_adapters import document_corpus as dc  # noqa: E402


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


_PY = "import os\n\ndef foo(x):\n    return x\n\nclass Bar:\n    def baz(self):\n        pass\n"


# ── code_repo: pure extraction ────────────────────────────────────────

def test_detect_language():
    assert cr.detect_language(Path("a.py")) == "python"
    assert cr.detect_language(Path("a.ts")) == "typescript"
    assert cr.detect_language(Path("a.md")) == "markdown"
    assert cr.detect_language(Path("a.zzz")) == "unknown"


def test_extract_symbols_regex():
    syms = cr.extract_symbols_regex(_PY, "python", Path("sample.py"))
    names = {s.name for s in syms}
    assert {"foo", "Bar"} <= names
    assert any(s.kind == "class" for s in syms)
    # markdown sections
    md = cr.extract_symbols_regex("# Title\n## Sub\nbody\n", "markdown", Path("d.md"))
    assert any(s.kind == "section" for s in md)
    # unknown language → nothing
    assert cr.extract_symbols_regex("whatever", "cobol", Path("x.cob")) == []
    # dispatch wrapper
    assert cr.extract_symbols(_PY, "python", Path("s.py"))


# ── code_repo: real adapter against temp lab ──────────────────────────

def test_code_repo_ingest_search_stats(tmp_path):
    a = cr.CodeRepoAdapter(tmp_path)
    src = tmp_path / "sample.py"
    src.write_text(_PY)
    res = a.ingest(src)
    assert res.items_added == 3 and not res.warnings
    hits = a.search("foo")
    assert hits and hits[0].score == 1.0   # exact name match ranks top
    assert a.search("") == []               # empty query
    assert a.search("Bar", filters={"kind": "class"})  # kind filter
    assert a.search("foo", filters={"language": "python"})
    # stats + export
    assert a.stats().items_total == 3
    manifest = a.export_for_packet()
    assert isinstance(manifest, dict)


def test_code_repo_ingest_dir_and_edges(tmp_path):
    a = cr.CodeRepoAdapter(tmp_path)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m.py").write_text(_PY)
    (tmp_path / "pkg" / "notes.txt").write_text("not code")      # unsupported ext → skipped
    (tmp_path / "pkg" / "__pycache__").mkdir()
    (tmp_path / "pkg" / "__pycache__" / "c.py").write_text(_PY)  # excluded dir
    res = a.ingest(tmp_path / "pkg")
    assert res.items_added == 3            # only m.py counts
    # missing path → warning
    assert a.ingest(tmp_path / "nope").warnings
    # unsupported source type → warning
    assert a.ingest(12345).warnings


# ── document_corpus: stubbed delegates ────────────────────────────────

def test_code_repo_get_delete_related(tmp_path):
    a = cr.CodeRepoAdapter(tmp_path)
    (tmp_path / "sample.py").write_text(_PY)
    a.ingest(tmp_path / "sample.py")
    # search scoring tiers: substring-in-name (0.8), in-qname (0.6), in-sig (0.3)
    assert any(h.score == 0.8 for h in a.search("fo"))
    assert any(h.score == 0.6 for h in a.search("sample"))
    assert any(h.score == 0.3 for h in a.search("def"))
    # get by symbol id (ids 1..3 from ingest)
    got = a.get("1")
    assert got and "name" in got
    assert a.get("") is None
    assert a.get("99999") is None        # no such row
    assert a.get("not-an-int") is None   # ValueError → None
    # related (symbol_refs empty → [], but exercises the SQL path)
    assert a.related("1") == []
    assert a.related("") == []
    assert a.related("not-an-int") == []
    # delete by id
    assert a.delete("1") is True
    assert a.delete("") is False
    assert a.delete("99999") is False
    assert a.delete("not-an-int") is False


def test_document_corpus_ingest(monkeypatch, tmp_path):
    from core import memory as _mem
    monkeypatch.setattr(_mem, "create", lambda p, c: {"ok": True})
    a = dc.DocumentCorpusAdapter(tmp_path)
    (tmp_path / "findings").mkdir()
    f = tmp_path / "findings" / "doc.md"
    f.write_text("# Doc\n\nbody\n")
    assert a.ingest(f).items_added == 1
    assert a.ingest({"text": "raw text", "metadata": {"filename": "x.md"}}).items_added == 1
    assert a.ingest(12345).warnings                 # unsupported
    assert a.ingest(tmp_path / "missing.md").warnings  # not found


def test_document_corpus_search_paths(monkeypatch, tmp_path):
    from core import memory as _mem
    from core import retrieval as _ret
    a = dc.DocumentCorpusAdapter(tmp_path)
    fake = types.SimpleNamespace(
        id="find_1", final_score=0.9, text="chunk text",
        metadata={"path": "findings/x.md", "chunk_idx": 0}, sources=["vector"])
    monkeypatch.setattr(_ret, "hybrid_retrieve", lambda *a, **k: [fake])
    res = a.search("query", method="hybrid")
    assert res and res[0].id == "find_1"
    # explicit vector method
    monkeypatch.setattr(_mem, "search", lambda q, k=8: [
        {"path": "m.md", "chunk_idx": 0, "content": "c", "distance": 0.2}])
    rv = a.search("q", method="vector")
    assert rv and rv[0].source_path == "m.md"
    # hybrid raises → vector fallback
    def _boom(*a, **k):
        raise RuntimeError("hybrid down")
    monkeypatch.setattr(_ret, "hybrid_retrieve", _boom)
    rf = a.search("q", method="hybrid")
    assert rf and rf[0].source_path == "m.md"


def test_document_corpus_related_get_delete(monkeypatch, tmp_path):
    from core import graph_store as gs
    a = dc.DocumentCorpusAdapter(tmp_path)
    monkeypatch.setattr(gs, "neighbors", lambda nid, edge_type=None: [
        {"id": "n2", "edge_type": "cites", "label": "Paper Two"}])
    rel = a.related("find_1")
    assert rel and rel[0].id == "n2"
    # get/delete (file-based)
    (tmp_path / "findings").mkdir()
    (tmp_path / "findings" / "doc.md").write_text("hello content")
    got = a.get("findings/doc.md")
    assert got and got["content"] == "hello content"
    assert a.get("findings/doc.md#3")["chunk_idx"] == 3
    assert a.get("") is None
    assert a.get("findings/missing.md") is None
    assert a.delete("findings/doc.md") is True       # soft-delete → .deleted/
    assert a.delete("findings/missing.md") is False
    assert a.delete("") is False


def test_document_corpus_stats_export(tmp_path):
    a = dc.DocumentCorpusAdapter(tmp_path)
    # fresh lab → 0 items, note present
    s0 = a.stats()
    assert s0.items_total == 0 and s0.notes
    (tmp_path / "findings").mkdir()
    (tmp_path / "findings" / "a.md").write_text("x")
    s1 = a.stats()
    assert s1.items_total == 1
    assert isinstance(a.export_for_packet(), dict)


def main() -> int:
    tests = [
        test_detect_language,
        test_extract_symbols_regex,
        test_code_repo_ingest_search_stats,
        test_code_repo_ingest_dir_and_edges,
        test_code_repo_get_delete_related,
        test_document_corpus_ingest,
        test_document_corpus_search_paths,
        test_document_corpus_related_get_delete,
        test_document_corpus_stats_export,
    ]
    for t in tests:
        mp = _MP()
        td = Path(tempfile.mkdtemp())
        try:
            params = inspect.signature(t).parameters
            kwargs = {}
            if "tmp_path" in params:
                kwargs["tmp_path"] = td
            if "monkeypatch" in params:
                kwargs["monkeypatch"] = mp
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
            shutil.rmtree(td, ignore_errors=True)
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
