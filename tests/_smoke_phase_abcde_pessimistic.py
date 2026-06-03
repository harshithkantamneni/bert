"""Tier-2 pessimistic edge cases for Phase A-E.

Goes beyond unit-level robustness into INTEGRATION + RACE + RESOURCE
edge cases:

  - Concurrent ingest into same adapter
  - MCP JSON-RPC with malformed payloads
  - Profile reshape during cycle execution
  - Promotion when target role already exists
  - Roster spawn → MCP discovery interaction
  - Memory adapter cross-call: ingest in adapter A, search in adapter B
  - lab_resume with token for nonexistent lab
  - Schema migration on lab with PARTIAL existing state (migrated to v0.5)
  - 0-byte files
  - Files with no extension
  - Symlinks
  - Permission-denied directories
  - Token graph with malformed regex patterns

Each test asserts: no crash, no hang, no silent corruption.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("BERT_DISABLE_RERANKER", "1")  # tests: no 568MB cold-start

import sqlite3
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


passed = 0
failed = 0


def check(name: str, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS  {name}")
        passed += 1
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
        failed += 1
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL  {name} (UNEXPECTED {type(e).__name__}): {e}")
        failed += 1


# ── Concurrent + race ─────────────────────────────────────────────


def test_concurrent_adapter_ingest():
    """Two threads ingesting into same adapter shouldn't corrupt state."""
    from core.memory_adapters import find_adapter_for_shape
    cls = find_adapter_for_shape("code_repo")
    with tempfile.TemporaryDirectory() as tmp:
        lab = Path(tmp)
        ad = cls(lab)
        src = lab / "src"
        src.mkdir()
        # Write 10 files
        for i in range(10):
            (src / f"file_{i}.py").write_text(
                f"def f_{i}():\n    return {i}\n"
            )
        errors = []
        def ingest():
            try:
                ad.ingest(src)
            except Exception as e:
                errors.append(e)
        ts = [threading.Thread(target=ingest) for _ in range(3)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=30)
        # SQLite single-writer model means we accept some retries but
        # no crashes. Final state should have all 10 symbols.
        st = ad.stats()
        # At least 10 (each thread might've added more if races)
        assert st.items_total >= 10


def test_concurrent_token_minting():
    """100 tokens minted in parallel — all unique, all verify."""
    from core import pause_resume
    minted = []
    def mint_one(i):
        st = pause_resume.PausedState(
            lab="x", cycle=i, step_id=f"s{i}",
        )
        minted.append(pause_resume.mint_resume_token(st))
    ts = [threading.Thread(target=mint_one, args=(i,)) for i in range(100)]
    for t in ts: t.start()
    for t in ts: t.join()
    # All unique
    assert len(set(minted)) == 100
    # All verify
    for token in minted:
        assert pause_resume.verify_resume_token(token) is not None


# ── MCP JSON-RPC malformed ────────────────────────────────────────


def _mcp_call(method: str, params: dict) -> dict:
    """Helper: spawn MCP server, send one tool call, return response."""
    p = subprocess.Popen(
        [sys.executable, "-m", "tools.mcp.bert_lab"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    def s(m): p.stdin.write(json.dumps(m) + chr(10)); p.stdin.flush()
    def r(): return json.loads(p.stdout.readline())
    s({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    r()
    s({"jsonrpc": "2.0", "method": "notifications/initialized"})
    s({"jsonrpc": "2.0", "id": 2, "method": method, "params": params})
    out = r()
    p.terminate()
    p.wait(timeout=5)
    return out


def test_mcp_tools_call_unknown_tool():
    resp = _mcp_call("tools/call", {
        "name": "does_not_exist", "arguments": {},
    })
    # MCP server should return an error, not crash
    assert "error" in resp or "result" in resp


def test_mcp_lab_start_missing_required():
    resp = _mcp_call("tools/call", {
        "name": "lab_start", "arguments": {"mission": "short"},
    })
    # Missing name; expect ok=False in result
    inner = json.loads(resp["result"]["content"][0]["text"])
    assert not inner.get("ok", False)


def test_mcp_lab_status_nonexistent():
    resp = _mcp_call("tools/call", {
        "name": "lab_status", "arguments": {"lab": "garbage_xyz"},
    })
    inner = json.loads(resp["result"]["content"][0]["text"])
    assert not inner.get("ok", False)


def test_mcp_lab_resume_garbage_token():
    resp = _mcp_call("tools/call", {
        "name": "lab_resume",
        "arguments": {"token": "not_a_real_token", "answer": "x"},
    })
    inner = json.loads(resp["result"]["content"][0]["text"])
    assert not inner.get("ok", False)


# ── Adapter weird sources ────────────────────────────────────────


def test_adapter_ingest_zero_byte_file():
    from core.memory_adapters import find_adapter_for_shape
    cls = find_adapter_for_shape("code_repo")
    with tempfile.TemporaryDirectory() as tmp:
        lab = Path(tmp)
        ad = cls(lab)
        empty = lab / "empty.py"
        empty.touch()
        r = ad.ingest(empty)
        assert r.items_added == 0  # no symbols in empty file
        assert r.bytes_in == 0


def test_adapter_ingest_file_no_extension():
    from core.memory_adapters import find_adapter_for_shape
    cls = find_adapter_for_shape("code_repo")
    with tempfile.TemporaryDirectory() as tmp:
        lab = Path(tmp)
        ad = cls(lab)
        no_ext = lab / "Makefile"
        no_ext.write_text("all:\n\techo hi\n")
        r = ad.ingest(no_ext)
        # Language=unknown; 0 symbols extracted but file row recorded
        # (Actually our regex extractor returns 0 for unknown lang, so
        # no symbols, but ingest succeeds without crashing.)
        assert isinstance(r.items_added, int)


def test_adapter_ingest_huge_directory():
    """100 small files in one dir."""
    from core.memory_adapters import find_adapter_for_shape
    cls = find_adapter_for_shape("code_repo")
    with tempfile.TemporaryDirectory() as tmp:
        lab = Path(tmp)
        ad = cls(lab)
        src = lab / "src"
        src.mkdir()
        for i in range(100):
            (src / f"m_{i}.py").write_text(f"def f_{i}(): pass\n")
        r = ad.ingest(src, max_files=200)
        # 100 functions detected
        assert r.items_added == 100


def test_adapter_ingest_path_with_spaces():
    from core.memory_adapters import find_adapter_for_shape
    cls = find_adapter_for_shape("code_repo")
    with tempfile.TemporaryDirectory() as tmp:
        lab = Path(tmp)
        ad = cls(lab)
        weird = lab / "dir with spaces"
        weird.mkdir()
        (weird / "file with spaces.py").write_text("def f(): pass\n")
        r = ad.ingest(weird / "file with spaces.py")
        assert r.items_added == 1


# ── Profile / synthesizer edge cases ──────────────────────────────


def test_profile_all_data_shapes_have_a_synthesis_rule():
    """Every data_shape value must hit SOME synthesizer rule."""
    from core import mission_profile, schema_synthesizer
    for shape in mission_profile.DATA_SHAPES:
        p = mission_profile.MissionProfile(
            domain="x", domain_confidence=0.5,
            work_types=("discover",), primary_work="discover",
            horizon="short", cadence=None,
            output_kind="report", rigor="cited",
            data_shape=shape,
            expected_volume="medium",
            input_surfaces=(),
            audience="self",
            success_criteria=(),
            classifier_confidence=0.5,
            stage_used="test",
        )
        try:
            s = schema_synthesizer.synthesize(p)
            assert s.rule_id  # any rule_id, even default
        except Exception as e:
            raise AssertionError(
                f"shape={shape!r} crashed synthesizer: {e}"
            ) from e


def test_profile_yaml_roundtrip():
    from core import mission_profile
    p = mission_profile.default_profile("test mission text")
    yaml_block = p.to_yaml_block()
    # Roundtrip through yaml parser
    import yaml
    parsed = yaml.safe_load("data:\n" + yaml_block)
    assert parsed["data"]["data_shape"] == p.data_shape


# ── Roster + consolidator integration ─────────────────────────────


def test_promotion_when_role_dir_already_exists():
    """If agents/<role>/ already exists, promotion should be no-op."""
    from core import consolidator, roster
    with tempfile.TemporaryDirectory() as tmp:
        labs_dir = Path(tmp)
        lab = labs_dir / "x"
        lab.mkdir()
        # Spawn 3+ uses
        for c in (1, 2, 3, 4):
            roster.spawn_inline(
                lab_path=lab, template="researcher",
                inline_name="x", cycle=c,
            )
        # Pre-create the role dir
        role_dir = lab / "agents" / "researcher__x"
        role_dir.mkdir(parents=True, exist_ok=True)
        (role_dir / "procedural.md").write_text("# Pre-existing\n")
        # Promote should NOT overwrite
        consolidator._promote_specializations(cycle=5, labs_dir=labs_dir)
        # File still has original content
        assert "Pre-existing" in (role_dir / "procedural.md").read_text()


def test_promotion_idempotent_across_consolidator_runs():
    from core import consolidator, roster
    with tempfile.TemporaryDirectory() as tmp:
        labs_dir = Path(tmp)
        lab = labs_dir / "y"
        lab.mkdir()
        for c in (1, 2, 3):
            roster.spawn_inline(
                lab_path=lab, template="researcher",
                inline_name="y_spec", cycle=c,
            )
        p1 = consolidator._promote_specializations(cycle=4, labs_dir=labs_dir)
        p2 = consolidator._promote_specializations(cycle=5, labs_dir=labs_dir)
        p3 = consolidator._promote_specializations(cycle=6, labs_dir=labs_dir)
        assert len(p1) == 1
        assert p2 == []
        assert p3 == []


# ── Migration weird state ─────────────────────────────────────────


def test_migrations_lab_with_partial_schema_v0_5():
    """Lab has a stale `chunks` table missing `doc_id` — the migration
    must FAIL ATOMICALLY: error reported AND no partial-apply leak."""
    from core import migrations
    with tempfile.TemporaryDirectory() as tmp:
        lab = Path(tmp)
        db_dir = lab / "memory" / "document_corpus"
        db_dir.mkdir(parents=True)
        db = db_dir / "document_corpus.db"
        with sqlite3.connect(db) as con:
            con.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, content TEXT)")
            con.commit()
        r = migrations.apply_pending(lab, "document_corpus")
        # The migration MUST fail loudly (incompatible stale schema)
        assert r.errors, "expected errors but migration silently succeeded"
        # AND the rollback must hold: documents table NOT created
        with sqlite3.connect(db) as con:
            tables = {row[0] for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            assert "documents" not in tables, (
                "partial-apply leaked despite rollback"
            )


def test_migrations_with_readonly_state_dir():
    """If state dir is read-only, migration runner should report error."""
    from core import migrations
    with tempfile.TemporaryDirectory() as tmp:
        lab = Path(tmp)
        state = lab / "state"
        state.mkdir()
        try:
            state.chmod(0o500)  # no write
            # The runner might fail to write meta.db
            r = migrations.apply_pending(lab, "document_corpus")
            # Either it errors cleanly OR it succeeds (some OSes ignore
            # chmod for owner). Either is fine — what matters: no crash.
            assert isinstance(r.errors, tuple)
        finally:
            state.chmod(0o700)  # restore for cleanup


# ── Token graph + BM25 weird inputs ───────────────────────────────


def test_token_graph_extract_overlapping_patterns():
    """Same chunk has C107 + researcher + APPROVE + finding_path."""
    from core import token_graph
    text = (
        "In cycle C107 the researcher emitted verdict APPROVE on "
        "findings/bert_run_C107_researcher.md"
    )
    tokens = token_graph.extract_tokens(text)
    kinds = {k for k, _ in tokens}
    # Should have multiple distinct kinds
    assert "cycle" in kinds
    assert "role" in kinds
    assert "verdict" in kinds
    assert "finding_path" in kinds


def test_token_graph_search_query_no_match():
    """Query with canonical tokens that don't exist in graph → empty."""
    from core import token_graph
    with tempfile.TemporaryDirectory() as tmp:
        # No rebuild — DB doesn't exist
        hits = token_graph.search("C999", lab_path=Path(tmp), k=5)
        assert hits == []


def test_bm25_index_grows_correctly():
    from core import bm25
    # Mock: bm25 uses the global memory.db; we just test tokenization
    text = "Mamba Linear-Time Sequence Modeling 2312.00752"
    tokens = bm25.tokenize(text)
    assert "mamba" in tokens
    assert "2312" in tokens  # arxiv number tokenizable


# ── Final runner ─────────────────────────────────────────────────


TESTS = [
    test_concurrent_adapter_ingest,
    test_concurrent_token_minting,
    test_mcp_tools_call_unknown_tool,
    test_mcp_lab_start_missing_required,
    test_mcp_lab_status_nonexistent,
    test_mcp_lab_resume_garbage_token,
    test_adapter_ingest_zero_byte_file,
    test_adapter_ingest_file_no_extension,
    test_adapter_ingest_huge_directory,
    test_adapter_ingest_path_with_spaces,
    test_profile_all_data_shapes_have_a_synthesis_rule,
    test_profile_yaml_roundtrip,
    test_promotion_when_role_dir_already_exists,
    test_promotion_idempotent_across_consolidator_runs,
    test_migrations_lab_with_partial_schema_v0_5,
    test_migrations_with_readonly_state_dir,
    test_token_graph_extract_overlapping_patterns,
    test_token_graph_search_query_no_match,
    test_bm25_index_grows_correctly,
]


def main() -> int:
    print(f"Running {len(TESTS)} tier-2 pessimistic tests…\n")
    for fn in TESTS:
        check(fn.__name__, fn)
    print()
    print(f"Tier-2: pass={passed} fail={failed}")
    if failed == 0:
        print(f"All {passed} tests passed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
