"""Smoke: core/memory.py — sqlite-vec memory store (was 17%).

Exercises the REAL indexing + vector-search plumbing (chunk → embed →
vec0 insert → KNN query → stats/cli) against a temp DB + temp corpus.
Only the torch embedder is faked: `_get_embedder` returns a deterministic
numpy stub (mem.EMBED_DIM-d, L2-normalized) — the bm25 traffic tool exists
precisely because loading the real embedder is memory-heavy on this
machine, so faking just the model keeps the smoke fast + RAM-safe while
the sqlite-vec math (struct pack, vec0 MATCH/k, distance ordering) is all
real. Every module path constant is monkeypatched to the temp tree.
"""

from __future__ import annotations

import inspect
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import memory as mem  # noqa: E402


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


class _FakeEmbedder:
    """Deterministic embedder (mem.EMBED_DIM-d) — same text → same unit vector."""
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(mem.EMBED_DIM).astype("float32")
            if normalize_embeddings:
                v = v / (np.linalg.norm(v) + 1e-9)
            out.append(v)
        return np.array(out, dtype="float32")


_CORPUS = {
    "memories/decisions.md": (
        "# Decisions\n\nWe chose sqlite-vec for the memory store because it "
        "embeds in-process with no server.\n\nHybrid retrieval fuses BM25 with "
        "dense vectors via reciprocal rank fusion.\n"
    ),
    "findings/survey.md": (
        "# Survey\n\nCross-encoder rerankers improve nDCG at a latency cost.\n\n"
        "Personalized PageRank traverses a token graph from seed nodes.\n"
    ),
}


def _setup(mp, tmp):
    (tmp / "memories").mkdir(parents=True, exist_ok=True)
    (tmp / "findings").mkdir(parents=True, exist_ok=True)
    for rel, body in _CORPUS.items():
        (tmp / rel).write_text(body, encoding="utf-8")
    mp.setattr(mem, "LAB_ROOT", tmp)
    mp.setattr(mem, "DB_PATH", tmp / "memory.db")
    mp.setattr(mem, "INDEX_DIRS", [tmp / "memories", tmp / "findings"])
    mp.setattr(mem, "_conn", None)            # reopen against temp DB
    mp.setattr(mem, "_embedder", _FakeEmbedder())
    mp.setattr(mem, "_get_embedder", lambda: mem._embedder)
    mem._invalidate_index_corpus_cache()


# ── pure ──────────────────────────────────────────────────────────────

def test_chunk_short_and_long():
    assert mem._chunk("one para") == ["one para"]
    multi = mem._chunk("a\n\nb\n\nc")
    assert isinstance(multi, list) and multi
    # a paragraph longer than max_chars hard-splits with overlap
    big = mem._chunk("x" * 4000, max_chars=1000, overlap=100)
    assert len(big) >= 4


def test_create_path_validation(monkeypatch, tmp_path):
    monkeypatch.setattr(mem, "LAB_ROOT", tmp_path)
    ok = mem.create("findings/note.md", "body text")
    assert ok["ok"] and ok["bytes"] == len("body text")
    assert (tmp_path / "findings" / "note.md").exists()
    # disallowed dir
    bad = mem.create("state/config.md", "x")
    assert bad["ok"] is False and "memories/ or findings/" in bad["error"]
    # outside lab root (absolute /tmp path that isn't under LAB_ROOT)
    out = mem.create("/etc/hosts", "x")
    assert out["ok"] is False


# ── index / search / stats / cli (fake embedder, real sqlite-vec) ─────

def test_index_then_search(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    n = mem._index_corpus()
    assert n >= 2, f"expected ≥2 chunks indexed, got {n}"
    hits = mem.search("sqlite-vec memory store", k=3)
    assert hits and {"path", "chunk_idx", "content", "distance"} <= hits[0].keys()
    assert all(isinstance(h["distance"], float) for h in hits)
    # empty query short-circuits
    assert mem.search("") == []
    assert mem.search("   ") == []


def test_skip_indexer_env(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(mem.os, "environ", {**mem.os.environ, "BERT_SKIP_INDEXER": "1"})
    assert mem._index_corpus() == 0


def test_list_corpus_and_invalidate(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    files = mem._list_corpus_files()
    assert len(files) == 2 and all(f.suffix == ".md" for f in files)
    mem._invalidate_index_corpus_cache()  # no crash, clears cache


def test_stats(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    mem._index_corpus()
    s = mem.stats()
    assert s["chunks_indexed"] >= 2 and s["files_indexed"] >= 2
    assert s["embedding_dim"] == mem.EMBED_DIM


def test_cli_ops(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    assert mem.cli("index", []) == 0
    assert mem.cli("stats", []) == 0
    assert mem.cli("search", ["sqlite vector store"]) == 0
    assert mem.cli("search", []) == 1          # usage error
    assert mem.cli("bogus_op", []) == 1        # unsupported op


def test_chunk_multi_paragraph_append():
    # two paragraphs each below max but together over → flush + start new
    chunks = mem._chunk("aaaaaa\n\nbbbbbb", max_chars=10, overlap=2)
    assert len(chunks) == 2 and chunks == ["aaaaaa", "bbbbbb"]


def test_ttl_cache_and_orphan_and_reindex(monkeypatch, tmp_path):
    import os
    import time
    _setup(monkeypatch, tmp_path)
    assert mem._index_corpus() >= 2
    # 2nd walk: no changes → sets TTL cache (line 253-256)
    assert mem._index_corpus() == 0
    # 3rd walk within TTL → short-circuits via cache (line 208)
    assert mem._index_corpus() == 0
    # orphan GC: delete a file on disk, re-walk → drops its chunks
    mem._invalidate_index_corpus_cache()
    (tmp_path / "findings" / "survey.md").unlink()
    assert mem._index_corpus() == 0
    assert mem.stats()["files_indexed"] == 1
    # reindex changed file: rewrite + bump mtime → delete-prior + re-embed
    mem._invalidate_index_corpus_cache()
    dec = tmp_path / "memories" / "decisions.md"
    dec.write_text("# Decisions\n\nReplaced body content entirely.\n")
    future = time.time() + 100
    os.utime(dec, (future, future))
    assert mem._index_corpus() >= 1
    # empty-content file → chunks empty → skipped (no crash)
    mem._invalidate_index_corpus_cache()
    (tmp_path / "memories" / "blank.md").write_text("   \n\n   \n")
    os.utime(tmp_path / "memories" / "blank.md", (future, future))
    assert mem._index_corpus() >= 0


def test_index_conn_error_returns_zero(monkeypatch, tmp_path):
    import sqlite3
    _setup(monkeypatch, tmp_path)
    def _boom():
        raise sqlite3.OperationalError("db locked")
    monkeypatch.setattr(mem, "_get_conn", _boom)
    assert mem._index_corpus() == 0  # graceful degradation, not a crash


def main() -> int:
    tests = [
        test_chunk_short_and_long,
        test_create_path_validation,
        test_index_then_search,
        test_skip_indexer_env,
        test_list_corpus_and_invalidate,
        test_stats,
        test_cli_ops,
        test_chunk_multi_paragraph_append,
        test_ttl_cache_and_orphan_and_reindex,
        test_index_conn_error_returns_zero,
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
            mem._conn = None  # ensure no temp conn leaks to next test
            shutil.rmtree(td, ignore_errors=True)
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
