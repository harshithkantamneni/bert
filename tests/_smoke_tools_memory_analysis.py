"""Smoke: tools/memory.py (Memory class) + tools/analyze_per_alpha.py (both 0%).

tools/memory.py: the 6-op sandboxed memory protocol against a temp root —
path-traversal/escape rejection, view (file / dir / range / missing),
create, str_replace (ok / not-found / not-unique), insert, delete (archives
to history/, collision-rename), rename (ok / dest-exists), _nearby_files.

tools/analyze_per_alpha.py: all_retrieval_events (temp OBS_DIR), slice_window
(window filter + bad-ts skip), zipfian_stats (empty + populated incl. cache
hit-rate).
"""

from __future__ import annotations

import importlib
import inspect
import json
import shutil
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "tools"))

import tools.memory as mem_tool  # noqa: E402


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


# ── tools/memory.py ───────────────────────────────────────────────────

def test_memory_resolve_sandbox(tmp_path):
    m = mem_tool.Memory(tmp_path)
    assert m._resolve("a/b.md").is_relative_to(m.root)
    try:
        m._resolve("../escape.md"); raise SystemExit("no raise")
    except ValueError:
        pass
    try:
        m._resolve("/etc/passwd"); raise SystemExit("no raise")
    except ValueError:
        pass


def test_memory_crud(tmp_path):
    m = mem_tool.Memory(tmp_path)
    m.create("notes/a.md", "line1\nline2\nline3\n")
    assert m.view("notes/a.md") == "line1\nline2\nline3\n"
    assert m.view("notes/a.md", view_range=(2, 2)) == "line2"
    assert "a.md" in m.view("notes")              # dir listing
    # missing → FileNotFoundError with nearby hint
    try:
        m.view("notes/missing.md"); raise SystemExit("no raise")
    except FileNotFoundError:
        pass
    assert "a.md" in m._nearby_files("notes/x.md")


def test_memory_str_replace_insert(tmp_path):
    m = mem_tool.Memory(tmp_path)
    m.create("f.md", "alpha beta gamma\n")
    m.str_replace("f.md", "beta", "BETA")
    assert "BETA" in m.view("f.md")
    # not found
    try:
        m.str_replace("f.md", "zzz", "q"); raise SystemExit("no raise")
    except ValueError:
        pass
    # not unique
    m.create("dup.md", "x x x\n")
    try:
        m.str_replace("dup.md", "x", "y"); raise SystemExit("no raise")
    except ValueError:
        pass
    # insert
    m.create("g.md", "L0\nL1\n")
    m.insert("g.md", 1, "INSERTED")
    assert "INSERTED" in m.view("g.md")


def test_memory_delete_and_rename(tmp_path):
    m = mem_tool.Memory(tmp_path)
    m.create("d.md", "bye")
    m.delete("d.md")
    assert not (tmp_path / "d.md").exists()
    assert list((tmp_path / "history").rglob("d.md"))   # archived
    m.delete("never_existed.md")                          # no-op, no crash
    # collision archive (create + delete same name twice)
    m.create("d.md", "again")
    m.delete("d.md")
    # rename
    m.create("old.md", "data")
    m.rename("old.md", "new.md")
    assert m.view("new.md") == "data"
    m.create("occupied.md", "x")
    try:
        m.rename("new.md", "occupied.md"); raise SystemExit("no raise")
    except ValueError:
        pass


# ── tools/analyze_per_alpha.py ────────────────────────────────────────

def test_analyze_per_alpha(monkeypatch, tmp_path):
    apa = importlib.import_module("analyze_per_alpha")
    monkeypatch.setattr(apa, "OBS_DIR", tmp_path)
    rows = [
        {"ts": "2026-05-01T10:00:00Z", "query": "vector db"},
        {"ts": "2026-05-01T11:00:00Z", "query": "vector db"},
        {"ts": "2026-05-02T10:00:00Z", "query": "bm25"},
        {"ts": "bad-ts", "query": "x"},
    ]
    (tmp_path / "retrieval.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    events = apa.all_retrieval_events()
    assert len(events) >= 3
    win = apa.slice_window(events, "2026-05-01T00:00:00+00:00", "2026-05-01T23:59:59+00:00")
    assert all(e.get("ts", "").startswith("2026-05-01") for e in win)
    stats = apa.zipfian_stats(events)
    assert stats["n_total"] >= 3 and stats["n_unique"] >= 2
    assert "top1_pct" in stats and "cache_hit_rate" in stats
    assert apa.zipfian_stats([]) == {"n_total": 0, "n_unique": 0}
    # add events inside one of main()'s hardcoded windows (α=1.0) so the
    # populated print branch runs, then drive main() under suppressed stdout
    inwin = [{"ts": "2026-05-26T19:10:30+00:00", "query": f"q{i % 3}"} for i in range(12)]
    with (tmp_path / "retrieval.jsonl").open("a") as f:
        for r in inwin:
            f.write(json.dumps(r) + "\n")
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        apa.main()


def main() -> int:
    tests = [
        test_memory_resolve_sandbox,
        test_memory_crud,
        test_memory_str_replace_insert,
        test_memory_delete_and_rename,
        test_analyze_per_alpha,
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
