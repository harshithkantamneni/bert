"""Smoke: bm25 CLI + index_stats + skill_executor retry failure-mode.

Residual-branch coverage: bm25._cli (rebuild/search/stats/usage) +
index_stats + needs_rebuild against a synthetic chunks DB, and the
skill_executor retry handler in _apply_failure_mode (a step whose tool
fails, with a `retry` failure_mode).
"""

from __future__ import annotations

import contextlib
import io
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import bm25, skill_dsl, skill_executor  # noqa: E402

_CHUNKS = [
    "BM25 is a sparse lexical retrieval method using term frequencies.",
    "Hybrid retrieval fuses dense vectors with BM25 via RRF.",
    "Cross-encoder rerankers improve nDCG at the cost of latency.",
]


def _seed(lab: Path) -> None:
    lab.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(lab / "memory.db")
    con.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, path TEXT)")
    con.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, content TEXT, doc_id INTEGER)")
    con.execute("INSERT INTO documents VALUES (1, 'findings/s.md')")
    for i, c in enumerate(_CHUNKS, start=1):
        con.execute("INSERT INTO chunks VALUES (?, ?, 1)", (i, c))
    con.commit()
    con.close()


def test_bm25_index_stats_and_cli():
    with tempfile.TemporaryDirectory() as td:
        lab = Path(td) / "lab"
        _seed(lab)
        bm25.build_index(lab)
        stats = bm25.index_stats(lab)
        assert isinstance(stats, dict)
        assert bm25.needs_rebuild(lab) is False
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            assert bm25._cli(["x"]) == 2                              # usage
            assert bm25._cli(["x", "build", str(lab)]) == 0
            assert bm25._cli(["x", "search", str(lab), "bm25 retrieval"]) == 0
            assert bm25._cli(["x", "search", str(lab), "bm25", "3"]) == 0   # explicit k
            assert bm25._cli(["x", "search", str(lab)]) == 2          # search usage
            assert bm25._cli(["x", "bogus", str(lab)]) == 2          # unknown


def _parse(text: str):
    fd, p = tempfile.mkstemp(suffix=".md")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return skill_dsl.parse_skill_file(Path(p))


def test_skill_executor_retry_failure_mode():
    skill = _parse(
        '---\nname: retryer\nversion: "1.0"\ndescription: retry demo\n'
        'inputs:\n  x: {type: string, required: true}\n'
        'tools_required: [flaky]\n'
        'failure_modes:\n  - condition: "flaky fails"\n    handler: "retry max_retries=2"\n'
        'steps:\n  - id: s\n    tool: flaky\n    args: {x: "{{x}}"}\n    capture: out\n---\n# retryer\n'
    )
    calls = {"n": 0}

    def invoker(name, args):
        calls["n"] += 1
        raise RuntimeError("flaky fails")   # always fails → retries exhaust
    ctx = skill_executor.ExecutionContext(tool_invoker=invoker, skill_registry={})
    result = skill_executor.execute_skill(skill, {"x": "v"}, ctx)
    # the retry handler should have re-attempted the failing step
    assert result is not None and calls["n"] >= 2


def main() -> int:
    tests = [
        test_bm25_index_stats_and_cli,
        test_skill_executor_retry_failure_mode,
    ]
    for t in tests:
        try:
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
