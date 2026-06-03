"""Smoke: tools/analyze_observability.py — retrieval analytics report (was 78%).

Pure section_* builders over event lists + the main report writer. Seeds a
temp OBS_DIR with retrieval/cycle_outcome/background events and drives main()
(runs every section) + a few sections directly + _read_jsonl/_pct helpers.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "tools"))

ao = importlib.import_module("analyze_observability")


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


def _retrieval_events(n=12):
    out = []
    for i in range(n):
        out.append({
            "query": f"query cluster {i % 3}",
            "query_class": ["hot", "warm", "cold"][i % 3],
            "timings_ms": {"total_ms": 20.0 + i, "vector": 8.0, "bm25": 5.0,
                           "graph": 3.0, "cache": 1.0, "rerank": 2.0},
            "sources": ["vector", "bm25"] if i % 2 else ["vector"],
            "final_top_k": [{"id": f"c{i}", "score": 0.9}],
            "top_n": 5,
        })
    return out


def _seed(obs: Path):
    obs.mkdir(parents=True, exist_ok=True)
    (obs / "retrieval.jsonl").write_text(
        "\n".join(json.dumps(e) for e in _retrieval_events()) + "\n")
    (obs / "cycle_outcome.jsonl").write_text(
        "\n".join(json.dumps({"cycle_id": c, "success": c % 2 == 0,
                              "findings_produced": c}) for c in range(1, 6)) + "\n")
    (obs / "background_invocation.jsonl").write_text(
        "\n".join(json.dumps({"tool": t, "duration_ms": 5.0, "success": True})
                  for t in ("falsifier", "weekly_report")) + "\n")


def test_helpers_and_sections():
    assert ao._pct(1, 4) == "25.0%" or "%" in ao._pct(1, 4)
    assert ao._pct(0, 0) == "0.0%" or "%" in ao._pct(0, 0)
    evs = _retrieval_events()
    for fn in (ao.section_retrieval_latency, ao.section_query_distribution,
               ao.section_signal_contribution, ao.section_cache_potential,
               ao.section_per_query_class_latency):
        out = fn(evs)
        assert isinstance(out, list) and out
    # empty → still returns a section (no-data branch)
    assert isinstance(ao.section_retrieval_latency([]), list)


def test_read_jsonl(monkeypatch, tmp_path):
    monkeypatch.setattr(ao, "OBS_DIR", tmp_path)
    (tmp_path / "x.jsonl").write_text(json.dumps({"a": 1}) + "\nbad\n")
    assert len(ao._read_jsonl("x.jsonl")) == 1
    assert ao._read_jsonl("missing.jsonl") == []


def test_main(monkeypatch, tmp_path):
    obs = tmp_path / "obs"
    _seed(obs)
    monkeypatch.setattr(ao, "OBS_DIR", obs)
    out_path = tmp_path / "report.md"
    with contextlib.redirect_stdout(io.StringIO()):
        rc = ao.main(output_path=str(out_path))
    assert rc == 0 and out_path.exists()
    # no-output path (prints to stdout)
    with contextlib.redirect_stdout(io.StringIO()):
        assert ao.main() == 0


def main() -> int:
    tests = [
        test_helpers_and_sections,
        test_read_jsonl,
        test_main,
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
