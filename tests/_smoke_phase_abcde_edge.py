"""Adversarial edge-case test suite for Phase A-E shipped code.

Goes pessimistic: every "what if the user does this wrong" case I can
think of for the new primitives. Each test asserts EITHER (a) it
works correctly OR (b) it fails CLEANLY with a useful error — never
crashes / hangs / silently corrupts.

Categories:
  1. Mission classifier — garbage in, no crash
  2. Cycle budget — out-of-range, type errors
  3. Schema synthesizer — unknown profiles, empty rules
  4. Migration runner — corrupted meta.db, concurrent applies
  5. Token / pause_resume — tampered, expired
  6. Roster — duplicate spawn, missing template
  7. Adapter — corrupt DB, missing dirs
  8. Retrieval — empty query, very long query
  9. Drift / reshape — no events, cross-shape reject
 10. CLI — bad args, missing files

Run via:
  .venv/bin/python tests/_smoke_phase_abcde_edge.py
"""

from __future__ import annotations

import json
import sqlite3
import os
os.environ.setdefault("BERT_DISABLE_RERANKER", "1")  # tests: no 568MB cold-start

import subprocess
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


# ── Helpers ──────────────────────────────────────────────────────


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


# ── 1. Mission classifier — pessimistic inputs ─────────────────


def test_classifier_empty_mission():
    from core import mission_profile
    p = mission_profile.classify_mission("", use_llm=False)
    assert p.data_shape in mission_profile.DATA_SHAPES
    # Empty mission → heuristic fallback. The heuristic has SOME confidence
    # (it can still safely-default to document_corpus / discover); the LLM
    # confidence path stays at 1.0 when LLM succeeds. We just require it's
    # NOT pretending to be a high-confidence LLM result.
    assert 0.0 <= p.classifier_confidence < 0.6


def test_classifier_whitespace_only():
    from core import mission_profile
    p = mission_profile.classify_mission("   \n  \t  ", use_llm=False)
    assert p.data_shape in mission_profile.DATA_SHAPES


def test_classifier_extremely_long_mission():
    """30K-char mission should not crash."""
    from core import mission_profile
    p = mission_profile.classify_mission("blah " * 6000, use_llm=False)
    assert p.data_shape in mission_profile.DATA_SHAPES


def test_classifier_unicode_mission():
    from core import mission_profile
    p = mission_profile.classify_mission(
        "调查变压器的替代方案 — 研究非Transformer架构 🤖", use_llm=False,
    )
    assert p.data_shape in mission_profile.DATA_SHAPES


def test_classifier_injection_attempt():
    """User puts code-like content in mission. Don't execute it."""
    from core import mission_profile
    p = mission_profile.classify_mission(
        "'; DROP TABLE chunks; --", use_llm=False,
    )
    assert p.data_shape in mission_profile.DATA_SHAPES


def test_stage0_path_extraction_with_dotdot():
    """Don't follow ../../ paths blindly."""
    from core import mission_profile
    hints = mission_profile.stage0_precheck(
        "refactor repo at /../../etc/passwd"
    )
    # The regex captures the path, but we don't expand or follow it
    # — that's the caller's responsibility. Just verify no crash.
    assert isinstance(hints, dict)


# ── 2. Cycle budget — boundary conditions ─────────────────────


def test_budget_negative_int():
    from core import cycle_budget
    try:
        cycle_budget.resolve_budget(-1)
        raise AssertionError("should reject negative")
    except ValueError:
        pass


def test_budget_zero():
    from core import cycle_budget
    try:
        cycle_budget.resolve_budget(0)
        raise AssertionError("should reject 0")
    except ValueError:
        pass


def test_budget_huge_int():
    from core import cycle_budget
    try:
        cycle_budget.resolve_budget(10000)
        raise AssertionError("should reject 10000 (>50 cap)")
    except ValueError:
        pass


def test_budget_unknown_preset():
    from core import cycle_budget
    try:
        cycle_budget.resolve_budget("ultra_deep")
        raise AssertionError("should reject unknown preset")
    except ValueError:
        pass


def test_novelty_score_on_nonexistent_lab():
    from core import cycle_budget
    score = cycle_budget.novelty_score(
        Path("/tmp/nonexistent_lab_xyz"), 1,
    )
    assert score == 0.0


def test_saturation_window_too_large():
    from core import cycle_budget
    # current_cycle=2 with window=10 → not enough history → False
    sat, scores = cycle_budget.is_saturated(
        Path("/tmp"), current_cycle=2, window=10,
    )
    assert sat is False
    assert scores == []


def test_saturation_invalid_window():
    from core import cycle_budget
    try:
        cycle_budget.is_saturated(Path("/tmp"), current_cycle=1, window=0)
        raise AssertionError("should reject window=0")
    except ValueError:
        pass


# ── 3. Schema synthesizer — robustness ────────────────────────


def test_synth_rules_load():
    from core import schema_synthesizer as ss
    rules = ss._load_rules()
    assert len(rules) >= 4
    # Last rule must be wildcard default
    assert rules[-1]["id"] == "default"


def test_synth_unknown_profile_falls_through_to_default():
    from core import mission_profile, schema_synthesizer
    # Bizarre profile — should still hit the default rule
    p = mission_profile.MissionProfile(
        domain="nonexistent_domain", domain_confidence=0.1,
        work_types=("discover",), primary_work="discover",
        horizon="short", cadence=None,
        output_kind="report", rigor="cited",
        data_shape="multimodal",  # uncommon shape
        expected_volume="medium",
        input_surfaces=(),
        audience="self",
        success_criteria=(),
        classifier_confidence=0.1,
        stage_used="default_fallback",
    )
    s = schema_synthesizer.synthesize(p)
    assert s.rule_id == "default"


def test_scaffold_knowledge_idempotent():
    from core import schema_synthesizer, mission_profile
    with tempfile.TemporaryDirectory() as tmp:
        lab = Path(tmp)
        p = mission_profile.default_profile("test")
        s = schema_synthesizer.synthesize(p)
        first = schema_synthesizer.scaffold_knowledge_files(lab, s)
        second = schema_synthesizer.scaffold_knowledge_files(lab, s)
        assert len(first) > 0
        assert len(second) == 0  # idempotent


# ── 4. Migration runner ───────────────────────────────────────


def test_migrations_status_on_fresh_lab():
    from core import migrations
    with tempfile.TemporaryDirectory() as tmp:
        st = migrations.status(Path(tmp), "document_corpus")
        assert st.current_version == 0
        assert st.available_version >= 1


def test_migrations_apply_idempotent():
    from core import migrations
    with tempfile.TemporaryDirectory() as tmp:
        lab = Path(tmp)
        r1 = migrations.apply_pending(lab, "document_corpus")
        assert len(r1.applied) >= 1
        r2 = migrations.apply_pending(lab, "document_corpus")
        assert len(r2.applied) == 0


def test_migrations_unknown_adapter():
    from core import migrations
    with tempfile.TemporaryDirectory() as tmp:
        # No errors expected for unknown adapter — just empty result
        st = migrations.status(Path(tmp), "nonexistent_adapter")
        assert st.current_version == 0
        assert st.available_version == 0


def test_migrations_corrupt_meta_db():
    """If meta.db is corrupted, runner should report an error, not crash."""
    from core import migrations
    with tempfile.TemporaryDirectory() as tmp:
        lab = Path(tmp)
        (lab / "state").mkdir()
        # Write garbage where meta.db expects sqlite
        (lab / "state" / "bert_meta.db").write_bytes(b"not a sqlite db")
        try:
            migrations.status(lab, "document_corpus")
        except sqlite3.DatabaseError:
            # Acceptable — clean error
            pass


# ── 5. Token / pause_resume tampering ──────────────────────────


def test_resume_token_tampered():
    from core import pause_resume
    st = pause_resume.PausedState(
        lab="x", cycle=1, step_id="s",
    )
    token = pause_resume.mint_resume_token(st)
    # Flip a byte in the payload
    parts = token.rsplit(".", 1)
    tampered = parts[0][:-2] + "AA" + "." + parts[1]
    assert pause_resume.verify_resume_token(tampered) is None


def test_resume_token_empty():
    from core import pause_resume
    assert pause_resume.verify_resume_token("") is None
    assert pause_resume.verify_resume_token("nothex.nothex") is None


def test_resume_token_expired():
    """Token with past expiry rejected."""
    from core import pause_resume
    import time
    st = pause_resume.PausedState(
        lab="x", cycle=1, step_id="s",
        created_at_ts=time.time() - 100000,
        expires_at_ts=time.time() - 50000,  # past
    )
    token = pause_resume.mint_resume_token(st)
    assert pause_resume.verify_resume_token(token) is None


def test_resume_state_complex_payload():
    """Complex saved_state with nested dicts + lists round-trips."""
    from core import pause_resume
    st = pause_resume.PausedState(
        lab="x", cycle=1, step_id="s",
        saved_state={
            "list": [1, 2, 3],
            "nested": {"a": "b"},
            "special": "weird unicode 🤖",
        },
    )
    token = pause_resume.mint_resume_token(st)
    v = pause_resume.verify_resume_token(token)
    assert v is not None
    assert v.saved_state["list"] == [1, 2, 3]
    assert v.saved_state["nested"]["a"] == "b"


# ── 6. Roster ────────────────────────────────────────────────


def test_roster_spawn_unknown_template():
    from core import roster
    with tempfile.TemporaryDirectory() as tmp:
        r = roster.spawn_inline(
            lab_path=Path(tmp), template="no_such_template",
            inline_name="x",
        )
        assert not r["ok"]


def test_roster_spawn_same_inline_many_times():
    """Same inline_name N times in N cycles → use_count=N."""
    from core import roster
    with tempfile.TemporaryDirectory() as tmp:
        lab = Path(tmp)
        for c in range(1, 11):
            roster.spawn_inline(
                lab_path=lab, template="researcher",
                inline_name="lit_hunter", cycle=c,
            )
        cands = roster.candidates_for_promotion(lab, threshold=3)
        assert len(cands) == 1
        assert cands[0].use_count == 10


def test_roster_corrupt_tracker_json():
    """Corrupt _spawn_tracker.json should not crash spawn."""
    from core import roster
    with tempfile.TemporaryDirectory() as tmp:
        lab = Path(tmp)
        agents = lab / "agents"
        agents.mkdir()
        (agents / "_spawn_tracker.json").write_text("{ corrupt")
        # Should fall back to empty tracker
        r = roster.spawn_inline(
            lab_path=lab, template="researcher",
            inline_name="x", cycle=1,
        )
        assert r["ok"]


def test_roster_mark_promoted_unknown():
    from core import roster
    with tempfile.TemporaryDirectory() as tmp:
        result = roster.mark_promoted(
            Path(tmp), "researcher", "nonexistent",
        )
        assert result is False


# ── 7. Adapter robustness ─────────────────────────────────────


def test_adapter_get_nonexistent_id():
    from core.memory_adapters import find_adapter_for_shape
    cls = find_adapter_for_shape("document_corpus")
    with tempfile.TemporaryDirectory() as tmp:
        ad = cls(Path(tmp))
        assert ad.get("nonexistent/path.md") is None


def test_adapter_stats_on_empty_lab():
    from core.memory_adapters import find_adapter_for_shape
    cls = find_adapter_for_shape("document_corpus")
    with tempfile.TemporaryDirectory() as tmp:
        ad = cls(Path(tmp))
        s = ad.stats()
        assert s.items_total == 0
        # Fresh lab: empty, OK to flag as degraded
        assert s.health in ("ok", "degraded")


def test_adapter_search_empty_query():
    from core.memory_adapters import find_adapter_for_shape
    cls = find_adapter_for_shape("document_corpus")
    with tempfile.TemporaryDirectory() as tmp:
        ad = cls(Path(tmp))
        results = ad.search("", k=5)
        assert isinstance(results, list)


def test_code_adapter_unknown_extension():
    """File with extension we don't recognize — just record file row, no symbols."""
    from core.memory_adapters import find_adapter_for_shape
    cls = find_adapter_for_shape("code_repo")
    with tempfile.TemporaryDirectory() as tmp:
        lab = Path(tmp)
        ad = cls(lab)
        # Create a .xyz file
        src_dir = lab / "src"
        src_dir.mkdir()
        (src_dir / "thing.xyz").write_text("some content here")
        r = ad.ingest(src_dir / "thing.xyz")
        # File ingested (language='unknown'); 0 symbols extracted
        assert r.items_added == 0
        assert r.bytes_in > 0


def test_code_adapter_huge_file():
    """1MB file should ingest without hanging."""
    from core.memory_adapters import find_adapter_for_shape
    cls = find_adapter_for_shape("code_repo")
    with tempfile.TemporaryDirectory() as tmp:
        lab = Path(tmp)
        ad = cls(lab)
        big = lab / "big.py"
        big.write_text("def f():\n    pass\n" * 50_000)
        r = ad.ingest(big)
        assert r.items_added > 0  # 50k function defs detected


# ── 8. Retrieval ─────────────────────────────────────────────


def test_bm25_empty_query():
    from core import bm25
    hits = bm25.search("", lab_path=Path("/tmp"))
    assert hits == []


def test_token_graph_query_with_no_canonical_tokens():
    from core import token_graph
    hits = token_graph.search(
        "the quick brown fox jumped over the lazy dog",
        lab_path=Path("/tmp/nonexistent"), k=5,
    )
    assert hits == []  # no canonical tokens + nonexistent DB


def test_extract_tokens_empty_text():
    from core import token_graph
    assert token_graph.extract_tokens("") == []
    assert token_graph.extract_tokens(None) == []


def test_extract_tokens_giant_text():
    from core import token_graph
    text = "C107 " * 10000 + "researcher " * 10000
    tokens = token_graph.extract_tokens(text)
    # Dedup ensures bounded output
    assert len(tokens) <= 20


# ── 9. Drift / reshape ───────────────────────────────────────


def test_drift_on_lab_with_no_events():
    from core import profile_drift
    with tempfile.TemporaryDirectory() as tmp:
        lab = Path(tmp)
        (lab / "sor").mkdir()
        r = profile_drift.drift_score(lab)
        assert r.score == 0.0
        assert "no" in r.recommendation.lower() or "need" in r.recommendation.lower()


def test_reshape_no_lab_yaml():
    from core import profile_drift
    with tempfile.TemporaryDirectory() as tmp:
        result = profile_drift.within_shape_reshape(
            Path(tmp), {"primary_work": "decide"},
        )
        assert not result["ok"]


def test_reshape_cross_shape_rejected():
    """L-11: cross-shape reshape must be rejected in v1."""
    from core import profile_drift
    with tempfile.TemporaryDirectory() as tmp:
        lab = Path(tmp)
        (lab / "lab.yaml").write_text("""mission_profile:
  data_shape: document_corpus
  primary_work: discover
""")
        result = profile_drift.within_shape_reshape(
            lab, {"data_shape": "code_repo"},
        )
        assert not result["ok"]
        assert "Cross-shape" in result["error"]


# ── 10. CLI ──────────────────────────────────────────────────


def _run_cli(args: list[str]) -> tuple[int, str, str]:
    """Run bert_cli.py with args; return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, str(LAB_ROOT / "tools" / "bert_cli.py"), *args],
        capture_output=True, text=True, timeout=30,
    )
    return result.returncode, result.stdout, result.stderr


def test_cli_no_args():
    rc, out, err = _run_cli([])
    assert rc != 0  # argparse forces a subcommand


def test_cli_bad_subcommand():
    rc, out, err = _run_cli(["does_not_exist"])
    assert rc != 0


def test_cli_lab_list_works():
    rc, out, err = _run_cli(["lab", "list"])
    assert rc == 0
    assert "labs" in out.lower() or len(out) > 0


def test_cli_lab_status_nonexistent():
    rc, out, err = _run_cli(["lab", "status", "nonexistent_lab_xyz"])
    assert rc != 0  # exit code 3 — lab not found


def test_cli_lab_start_short_mission():
    rc, out, err = _run_cli(["lab", "start", "x", "too short"])
    # Mission < 20 chars rejected
    assert rc != 0


def test_cli_lab_reshape_garbage_kv():
    rc, out, err = _run_cli([
        "lab", "reshape", "test01", "this_is_not_a_kv_pair",
    ])
    assert rc != 0


# ── Runner ──────────────────────────────────────────────────


TESTS = [
    test_classifier_empty_mission,
    test_classifier_whitespace_only,
    test_classifier_extremely_long_mission,
    test_classifier_unicode_mission,
    test_classifier_injection_attempt,
    test_stage0_path_extraction_with_dotdot,
    test_budget_negative_int,
    test_budget_zero,
    test_budget_huge_int,
    test_budget_unknown_preset,
    test_novelty_score_on_nonexistent_lab,
    test_saturation_window_too_large,
    test_saturation_invalid_window,
    test_synth_rules_load,
    test_synth_unknown_profile_falls_through_to_default,
    test_scaffold_knowledge_idempotent,
    test_migrations_status_on_fresh_lab,
    test_migrations_apply_idempotent,
    test_migrations_unknown_adapter,
    test_migrations_corrupt_meta_db,
    test_resume_token_tampered,
    test_resume_token_empty,
    test_resume_token_expired,
    test_resume_state_complex_payload,
    test_roster_spawn_unknown_template,
    test_roster_spawn_same_inline_many_times,
    test_roster_corrupt_tracker_json,
    test_roster_mark_promoted_unknown,
    test_adapter_get_nonexistent_id,
    test_adapter_stats_on_empty_lab,
    test_adapter_search_empty_query,
    test_code_adapter_unknown_extension,
    test_code_adapter_huge_file,
    test_bm25_empty_query,
    test_token_graph_query_with_no_canonical_tokens,
    test_extract_tokens_empty_text,
    test_extract_tokens_giant_text,
    test_drift_on_lab_with_no_events,
    test_reshape_no_lab_yaml,
    test_reshape_cross_shape_rejected,
    test_cli_no_args,
    test_cli_bad_subcommand,
    test_cli_lab_list_works,
    test_cli_lab_status_nonexistent,
    test_cli_lab_start_short_mission,
    test_cli_lab_reshape_garbage_kv,
]


def main() -> int:
    print(f"Running {len(TESTS)} adversarial edge-case tests…\n")
    for fn in TESTS:
        check(fn.__name__, fn)
    print()
    print(f"All {len(TESTS)} edge-case tests: pass={passed} fail={failed}")
    if failed == 0:
        print(f"All {passed} tests passed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
