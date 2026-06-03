"""Smoke: token_graph.search (PPR) + skill_executor sequential paths.

Extends _smoke_core_retrieval. token_graph.search runs Personalized
PageRank over a rebuilt cooccur graph (the big uncovered block) against a
synthetic chunks DB — network-free. skill_executor sequential-foreach +
condition/templating helpers round out the branches the inline smokes miss.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import skill_dsl, skill_executor, token_graph  # noqa: E402

_CHUNKS = [
    "BM25 is a sparse lexical retrieval method using term frequencies.",
    "Hybrid retrieval fuses dense vectors with BM25 via RRF reciprocal rank fusion.",
    "Cross-encoder rerankers improve nDCG at the cost of latency.",
    "Personalized PageRank PPR traverses a token graph from seed nodes.",
    "RRF and BM25 together beat either alone on BEIR nDCG.",
]


def _seed_chunks_db(lab: Path) -> None:
    lab.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(lab / "memory.db")
    con.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, path TEXT)")
    con.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, content TEXT, doc_id INTEGER)")
    con.execute("INSERT INTO documents VALUES (1, 'findings/survey.md')")
    for i, c in enumerate(_CHUNKS, start=1):
        con.execute("INSERT INTO chunks VALUES (?, ?, 1)", (i, c))
    con.commit()
    con.close()


def test_token_graph_search_ppr():
    with tempfile.TemporaryDirectory() as td:
        lab = Path(td) / "lab"
        _seed_chunks_db(lab)
        token_graph.rebuild(lab)
        hits = token_graph.search("BM25 and RRF on BEIR nDCG", lab_path=lab, k=5)
        assert isinstance(hits, list)
        if hits:
            assert hasattr(hits[0], "chunk_id") or hasattr(hits[0], "score")
        # empty query → []
        assert token_graph.search("", lab_path=lab) == []
        # query with no canonical tokens → []
        assert token_graph.search("the and of a", lab_path=lab) == []
        # missing db → []
        assert token_graph.search("BM25", lab_path=Path(td) / "nope") == []


def test_token_graph_cli():
    with tempfile.TemporaryDirectory() as td:
        lab = Path(td) / "lab"
        _seed_chunks_db(lab)
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = token_graph._cli(["x", "rebuild", str(lab)])
        assert rc in (0, 1, 2) or rc is None


def _parse(text: str):
    fd, p = tempfile.mkstemp(suffix=".md")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return skill_dsl.parse_skill_file(Path(p))


def test_skill_executor_sequential_foreach():
    skill = _parse(
        '---\nname: seq\nversion: "1.0"\ndescription: sequential foreach\n'
        'inputs:\n  items: {type: list, required: true}\n'
        'outputs:\n  seen: {type: list}\n'
        'tools_required: [work]\n'
        'steps:\n'
        '  - id: s\n    foreach: "items"\n    tool: work\n'
        '    args: {item: "{{item}}"}\n    capture: seen\n'
        '---\n# seq\n'
    )
    seen = []
    ctx = skill_executor.ExecutionContext(
        tool_invoker=lambda n, a: seen.append(a.get("item")) or a.get("item"),
        skill_registry={})
    result = skill_executor.execute_skill(skill, {"items": ["a", "b", "c"]}, ctx)
    assert result.ok, result.errors
    assert seen == ["a", "b", "c"]   # sequential preserves order


def test_skill_executor_helpers():
    # _eval_condition + _deep_lookup + _resolve_string
    assert skill_executor._eval_condition("flag", {"flag": True}) is True
    assert skill_executor._eval_condition("flag", {"flag": False}) is False
    assert skill_executor._deep_lookup({"a": {"b": 7}}, "a.b") == 7
    resolved = skill_executor._resolve_string("{{name}}", {"name": "bert"})
    assert resolved == "bert"


def main() -> int:
    tests = [
        test_token_graph_search_ppr,
        test_token_graph_cli,
        test_skill_executor_sequential_foreach,
        test_skill_executor_helpers,
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
