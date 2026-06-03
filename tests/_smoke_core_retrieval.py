"""Smoke: core retrieval + skill-executor advanced paths.

Pushes three partially-covered core modules toward exemplary:
  - core/bm25.py        (51%) — build_index + search against a synthetic
    chunks DB, tokenize/stem, needs_rebuild, index_stats
  - core/token_graph.py (38%) — rebuild graph from chunks + extract_tokens
    + PPR seed/query
  - core/skill_executor.py (50%) — the if_ conditional, fallback:<skill>,
    and foreach_parallel branches not hit by the existing inline smokes

bm25/token_graph read <lab>/memory.db (table chunks(id, content, doc_id)
+ documents(id, path)); we seed a temp one so the real index/graph build
runs without touching the repo DB.
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import bm25, skill_dsl, skill_executor, token_graph  # noqa: E402

_CHUNKS = [
    "Vector databases compare on recall, latency, and RAM footprint.",
    "BM25 is a sparse lexical retrieval method using term frequencies.",
    "Hybrid retrieval fuses dense vector search with BM25 via RRF.",
    "Cross-encoder rerankers improve nDCG at the cost of latency.",
    "Personalized PageRank traverses a token graph from seed nodes.",
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


def test_bm25_tokenize_and_stem():
    toks = bm25.tokenize("Running runners ran quickly through latencies")
    assert toks and all(isinstance(t, str) for t in toks)
    # stemming collapses inflections; stopword drop removes 'through'
    assert bm25._stem("running") == bm25._stem("runs")
    raw = bm25.tokenize("the and of running", drop_stopwords=False, stem=False)
    assert "the" in raw  # stopwords retained when disabled


def test_bm25_build_and_search():
    with tempfile.TemporaryDirectory() as td:
        lab = Path(td) / "lab"
        _seed_chunks_db(lab)
        assert bm25.needs_rebuild(lab) is True
        stats = bm25.build_index(lab)
        assert len(stats.get("chunk_ids", [])) >= 5
        hits = bm25.search("bm25 sparse retrieval", lab_path=lab, k=3)
        assert isinstance(hits, list)
        assert hits, "expected BM25 hits for an in-corpus query"
        # after build, a fresh check shouldn't need a rebuild
        assert bm25.needs_rebuild(lab) is False


def test_token_graph_extract_and_rebuild():
    toks = token_graph.extract_tokens("BM25 and RRF improve nDCG@10 on BEIR")
    assert isinstance(toks, list)
    with tempfile.TemporaryDirectory() as td:
        lab = Path(td) / "lab"
        _seed_chunks_db(lab)
        result = token_graph.rebuild(lab)
        assert result.get("chunks_scanned", 0) >= 5
        seeds = token_graph.seed_tokens_from_query("vector databases recall")
        assert isinstance(seeds, list)


def _parse(text: str):
    import os
    fd, p = tempfile.mkstemp(suffix=".md")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return skill_dsl.parse_skill_file(Path(p))


def test_executor_if_conditional_skips_step():
    skill = _parse(
        '---\nname: cond\nversion: "1.0"\ndescription: conditional\n'
        'inputs:\n  go: {type: bool, required: true}\n'
        'tools_required: [identity]\n'
        'steps:\n'
        '  - id: maybe\n    if_: "go"\n    tool: identity\n    args: {value: "ran"}\n    capture: out\n'
        '---\n# cond\n'
    )
    calls = []
    ctx = skill_executor.ExecutionContext(
        tool_invoker=lambda n, a: calls.append(a) or "ran", skill_registry={})
    # go=False → step skipped
    r_skip = skill_executor.execute_skill(skill, {"go": False}, ctx)
    assert r_skip.ok and not calls, "if_ false should skip the step"
    # go=True → step runs
    r_run = skill_executor.execute_skill(skill, {"go": True}, ctx)
    assert r_run.ok and calls, "if_ true should run the step"


def test_executor_fallback_to_sub_skill():
    fallback_skill = _parse(
        '---\nname: rescue\nversion: "1.0"\ndescription: rescue\n'
        'inputs:\n  x: {type: string}\n'
        'tools_required: [identity]\n'
        'steps:\n  - id: r\n    tool: identity\n    args: {value: "rescued"}\n    capture: out\n---\n# r\n'
    )
    primary = _parse(
        '---\nname: primary\nversion: "1.0"\ndescription: primary\n'
        'inputs:\n  x: {type: string, required: true}\n'
        'tools_required: [boom]\n'
        'failure_modes:\n  - condition: "boom fails"\n    handler: "fallback:rescue"\n'
        'steps:\n  - id: s\n    tool: boom\n    args: {x: "{{x}}"}\n    capture: out\n---\n# primary\n'
    )

    def invoker(name, args):
        if name == "boom":
            raise RuntimeError("boom fails hard")
        return "rescued"

    ctx = skill_executor.ExecutionContext(
        tool_invoker=invoker, skill_registry={"rescue": fallback_skill})
    result = skill_executor.execute_skill(primary, {"x": "v"}, ctx)
    # fallback handler should route to the rescue sub-skill
    assert result is not None


def test_executor_foreach_parallel():
    skill = _parse(
        '---\nname: par\nversion: "1.0"\ndescription: parallel\n'
        'inputs:\n  items: {type: list, required: true}\n'
        'outputs:\n  seen: {type: list}\n'
        'tools_required: [work]\n'
        'steps:\n'
        '  - id: p\n    foreach_parallel: "items"\n    foreach_max_concurrent: 2\n'
        '    tool: work\n    args: {item: "{{item}}"}\n    capture: seen\n'
        '---\n# par\n'
    )
    seen = []
    ctx = skill_executor.ExecutionContext(
        tool_invoker=lambda n, a: seen.append(a.get("item")) or a.get("item"),
        skill_registry={})
    result = skill_executor.execute_skill(skill, {"items": ["a", "b", "c", "d"]}, ctx)
    assert result.ok, result.errors
    assert sorted(x for x in seen if x) == ["a", "b", "c", "d"]


def main() -> int:
    tests = [
        test_bm25_tokenize_and_stem,
        test_bm25_build_and_search,
        test_token_graph_extract_and_rebuild,
        test_executor_if_conditional_skips_step,
        test_executor_fallback_to_sub_skill,
        test_executor_foreach_parallel,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
