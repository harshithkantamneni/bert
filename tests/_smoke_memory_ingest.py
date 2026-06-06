"""TDD for the memory_ingest MCP tool + incremental ingest + source registry +
auto-resync (design: docs/design/2026-06-05-ingest-mcp-tool.md).

These tests cover the new SHARDING / REGISTRY / TOOL-CONTRACT behavior, which is
pure filesystem + JSON and needs no embedding model or sqlite-vec. We force
BERT_SKIP_INDEXER=1 so eager_index is a no-op (the embed path is existing,
unchanged behavior covered elsewhere)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("BERT_SKIP_INDEXER", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import lab_context  # noqa: E402
from core import memory as mem  # noqa: E402
from tools.mcp import bert_lab  # noqa: E402


def _with_lab(lab: Path, fn):
    tok = lab_context.set_active_lab_path(lab)
    try:
        return fn()
    finally:
        lab_context.reset_active_lab_path(tok)


def _mklab(tmp: Path) -> Path:
    lab = tmp / "lab"
    lab.mkdir(parents=True, exist_ok=True)
    return lab


def _bump_mtime(f: Path) -> None:
    """Set mtime well into the future so the source > shard comparison is
    unambiguous regardless of filesystem mtime granularity."""
    future = time.time() + 100
    os.utime(f, (future, future))


# ── A. incremental ingest + GC ──────────────────────────────────────


def test_reingest_skips_unchanged(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("def a(): return 1\n")
    lab = _mklab(tmp_path)
    r1 = _with_lab(lab, lambda: mem.ingest_corpus_report(src, eager_index=False))
    assert r1["written"] == 1, r1
    r2 = _with_lab(lab, lambda: mem.ingest_corpus_report(src, eager_index=False))
    assert r2["written"] == 0 and r2["skipped"] == 1, r2


def test_reingest_reshards_changed(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    f = src / "a.py"
    f.write_text("def a(): return 1\n")
    lab = _mklab(tmp_path)
    _with_lab(lab, lambda: mem.ingest_corpus_report(src, eager_index=False))
    f.write_text("def a(): return 2\n")
    _bump_mtime(f)
    r = _with_lab(lab, lambda: mem.ingest_corpus_report(src, eager_index=False))
    assert r["written"] == 1, r
    shard = lab / "findings" / "corpus" / "a.py.md"
    assert "return 2" in shard.read_text()


def test_reingest_gcs_deleted_source(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("x = 1\n")
    (src / "b.py").write_text("y = 2\n")
    lab = _mklab(tmp_path)
    _with_lab(lab, lambda: mem.ingest_corpus_report(src, eager_index=False))
    shard_b = lab / "findings" / "corpus" / "b.py.md"
    assert shard_b.exists()
    (src / "b.py").unlink()
    r = _with_lab(lab, lambda: mem.ingest_corpus_report(src, eager_index=False))
    assert r["removed"] == 1, r
    assert not shard_b.exists()


# ── B. source registry ──────────────────────────────────────────────


def test_register_and_list_sources_dedup(tmp_path):
    lab = _mklab(tmp_path)
    src = tmp_path / "proj"
    src.mkdir()
    _with_lab(lab, lambda: mem.register_ingest_source(src))
    _with_lab(lab, lambda: mem.register_ingest_source(src))  # idempotent
    srcs = _with_lab(lab, lambda: mem.list_ingest_sources())
    assert len(srcs) == 1, srcs
    assert Path(srcs[0]["source"]) == src.resolve()


# ── C. resync ───────────────────────────────────────────────────────


def test_resync_picks_up_out_of_band_change(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    f = src / "a.py"
    f.write_text("def a(): return 1\n")
    lab = _mklab(tmp_path)
    _with_lab(lab, lambda: mem.ingest_corpus_report(src, eager_index=False))
    _with_lab(lab, lambda: mem.register_ingest_source(src))
    f.write_text("def a(): return 99\n")
    _bump_mtime(f)
    rep = _with_lab(lab, lambda: mem.resync_sources(force=True, eager_index=False))
    assert rep["written"] >= 1, rep
    assert "return 99" in (lab / "findings" / "corpus" / "a.py.md").read_text()


# ── F. safety: exclusions + caps ────────────────────────────────────


def test_excludes_vcs_and_vendor_dirs(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "keep.py").write_text("k = 1\n")
    for d in (".git", "node_modules", "__pycache__", ".venv"):
        sub = src / d
        sub.mkdir()
        (sub / "junk.py").write_text("j = 1\n")
    lab = _mklab(tmp_path)
    r = _with_lab(lab, lambda: mem.ingest_corpus_report(src, eager_index=False))
    assert r["written"] == 1, r
    names = sorted(p.name for p in (lab / "findings" / "corpus").rglob("*.md"))
    assert names == ["keep.py.md"], names


def test_max_files_cap_reports_truncated(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(5):
        (src / f"f{i}.py").write_text(f"v = {i}\n")
    lab = _mklab(tmp_path)
    r = _with_lab(
        lab, lambda: mem.ingest_corpus_report(src, eager_index=False, max_files=3)
    )
    assert r["truncated"] is True, r
    assert r["written"] == 3, r


# ── D. memory_ingest MCP tool ───────────────────────────────────────


def test_memory_ingest_tool_happy(tmp_path):
    src = tmp_path / "proj"
    src.mkdir()
    (src / "a.py").write_text("def a(): return 1\n")
    lab = _mklab(tmp_path)
    res = bert_lab._t_memory_ingest({"lab": str(lab), "source": str(src)})
    assert res["ok"] is True, res
    assert res["files_ingested"] == 1, res
    srcs = _with_lab(lab, lambda: mem.list_ingest_sources())
    assert len(srcs) == 1
    assert (lab / "findings" / "corpus" / "a.py.md").exists()


def test_memory_ingest_tool_bad_source(tmp_path):
    lab = _mklab(tmp_path)
    res = bert_lab._t_memory_ingest(
        {"lab": str(lab), "source": str(tmp_path / "does_not_exist")}
    )
    assert res["ok"] is False and "source" in res["error"].lower(), res


def test_memory_ingest_tool_bad_lab(tmp_path):
    src = tmp_path / "proj"
    src.mkdir()
    res = bert_lab._t_memory_ingest(
        {"lab": "no_such_lab_xyz_123", "source": str(src)}
    )
    assert res["ok"] is False and "lab" in res["error"].lower(), res


# ── E. auto-resync inside memory_search ─────────────────────────────


def test_search_autoresyncs_registered_source(tmp_path):
    src = tmp_path / "proj"
    src.mkdir()
    f = src / "a.py"
    f.write_text("def alpha_marker(): return 1\n")
    lab = _mklab(tmp_path)
    bert_lab._t_memory_ingest({"lab": str(lab), "source": str(src)})
    # change the source out of band, then search for a marker only present AFTER
    f.write_text("def beta_marker(): return 2\n")
    _bump_mtime(f)
    res = bert_lab._t_memory_search({"lab": str(lab), "query": "beta_marker"})
    assert res.get("results"), res  # non-empty only if the resync re-sharded


def main() -> int:
    import inspect
    import tempfile

    tests = [
        obj
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    ]
    failed = 0
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
            failed += 1
        except Exception as e:  # noqa: BLE001
            import traceback

            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
