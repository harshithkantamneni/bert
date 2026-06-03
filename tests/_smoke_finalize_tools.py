"""Smoke + TDD: core/finalize_tools.py + the registry->invoker bridge (task #68).

Proves the finalize_project tool suite is real (reads the SoR ledger + findings,
provider cascade for synth/holes, pure sha256/ledger ops) AND that
finalize_project runs END-TO-END through skill_executor against the REAL
tool_registry — the engine that was missing in production. The two LLM tools +
the grader's judges are stubbed via provider.call so the whole flow is
network-free; the data + pure tools run for real against temp fixtures.
"""

from __future__ import annotations

import inspect
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

import core.tools  # noqa: E402,F401 — registers the full tool suite


@pytest.fixture
def tmp(tmp_path: Path) -> Path:
    """Per-test temp lab dir. Named `tmp` to match the standalone main()
    runner below, which injects a fresh tempdir under the same name."""
    return tmp_path
from core import (  # noqa: E402  # noqa: E402
    finalize_tools,
    lab_context,
    quality,
    skill_executor,
    skill_registry,
    tool_registry,
)
from core.types import ProviderResponse  # noqa: E402

_NEW_TOOLS = ["identity", "list_findings", "read_ledger_rows",
              "assemble_evidence_bundle", "synthesize_artifact_body",
              "analyze_evidence_holes", "sha256_envelope", "validate_ledger_row",
              "append_jsonl_atomic", "finalize_ready_check"]


# ── pure / data tools (network-free) ─────────────────────────────────


def test_identity():
    assert finalize_tools._identity(value=42) == 42
    assert finalize_tools._identity(value={"a": 1}) == {"a": 1}


def test_all_tools_registered_and_invoker():
    reg = {t.name for t in tool_registry.all_tools()}
    for t in _NEW_TOOLS:
        assert t in reg, f"{t} not registered"
    inv = tool_registry.make_invoker()
    assert inv("identity", {"value": "x"}) == "x"
    try:
        inv("no_such_tool", {})
        raise AssertionError("expected KeyError for unknown tool")
    except KeyError:
        pass


def test_list_findings_walks_dir(tmp):
    (tmp / "findings").mkdir()
    (tmp / "findings" / "a.md").write_text("# A\nfinding a")
    (tmp / "findings" / "b.md").write_text("# B\nfinding b")
    tok = lab_context.set_active_lab_path(tmp)
    try:
        out = finalize_tools._list_findings(dir="findings/", min_quality=0.5)
        paths = sorted(f["path"] for f in out["files"])
        assert paths == ["findings/a.md", "findings/b.md"]
        assert all(f["quality_score"] >= 0.5 for f in out["files"])
    finally:
        lab_context.reset_active_lab_path(tok)


def test_read_ledger_rows_filters(tmp):
    led = tmp / "events.jsonl"
    led.write_text(
        json.dumps({"event_class": "finding", "cycle": 1, "content": "x"}) + "\n" +
        json.dumps({"event_class": "decision", "cycle": 2, "payload": {"d": 1}}) + "\n" +
        json.dumps({"event_class": "retrieval", "cycle": 3}) + "\n")
    out = finalize_tools._read_ledger_rows(path=str(led), event_types=["decision"])
    assert len(out["rows"]) == 1
    assert out["rows"][0]["event_type"] == "decision" and out["rows"][0]["cycle_id"] == 2


def test_assemble_evidence_bundle(tmp):
    (tmp / "f.md").write_text("# Finding\nbody text")
    tok = lab_context.set_active_lab_path(tmp)
    try:
        out = finalize_tools._assemble_evidence_bundle(
            findings=[{"path": "f.md", "quality_score": 0.9}],
            ledger_rows=[{"event_type": "decision", "cycle_id": 5, "payload": {}}],
            objective="test")
        assert out["count"] == 2
        kinds = {e["type"] for e in out["evidence"]}
        assert kinds == {"finding", "ledger"}
        assert 5 in out["cycles_covered"]
        fe = next(e for e in out["evidence"] if e["type"] == "finding")
        assert "body text" in fe["content"]
    finally:
        lab_context.reset_active_lab_path(tok)


def test_sha256_envelope_deterministic():
    a = finalize_tools._sha256_envelope(artifact="art", gaps="gaps", grade="B",
                                        components={"x": 0.5})
    b = finalize_tools._sha256_envelope(artifact="art", gaps="gaps", grade="B",
                                        components={"x": 0.5})
    assert a["hash"] == b["hash"] and len(a["hash"]) == 64
    # changing the artifact changes the hash
    c = finalize_tools._sha256_envelope(artifact="art2", gaps="gaps", grade="B",
                                        components={"x": 0.5})
    assert c["hash"] != a["hash"]
    assert a["envelope"]["grade"] == "B"


def test_validate_ledger_row():
    assert finalize_tools._validate_ledger_row(
        event_type="artifact_accepted", cycle_id=0, agent="x", payload={})["ok"] is True
    bad = finalize_tools._validate_ledger_row(event_type="", cycle_id="nope",
                                              agent="", payload=None)
    assert bad["ok"] is False and len(bad["errors"]) == 4


def test_append_jsonl_atomic(tmp):
    led = tmp / "out.jsonl"
    r1 = finalize_tools._append_jsonl_atomic(
        path=str(led), row={"event_type": "e", "cycle_id": 1, "payload": {"a": 1}})
    assert r1["offset"] == 0 and r1["row_id"].startswith("evt-1-e-") and r1["appended_at"]
    r2 = finalize_tools._append_jsonl_atomic(
        path=str(led), row={"event_type": "e", "cycle_id": 2, "payload": {}})
    assert r2["offset"] > 0  # appended after the first row
    assert len(led.read_text().strip().splitlines()) == 2


def test_finalize_ready_check(tmp):
    gaps = tmp / "gaps.md"
    gaps.write_text("# gaps")
    tok = lab_context.set_active_lab_path(tmp)
    try:
        assert finalize_tools._finalize_ready_check(
            grade="B", gaps_path="gaps.md", ledger_row_id="evt-1") is True
        assert finalize_tools._finalize_ready_check(
            grade="F", gaps_path="gaps.md", ledger_row_id="evt-1") is False
        assert finalize_tools._finalize_ready_check(
            grade="A", gaps_path="missing.md", ledger_row_id="evt-1") is False
        assert finalize_tools._finalize_ready_check(
            grade="A", gaps_path="gaps.md", ledger_row_id=None) is False
    finally:
        lab_context.reset_active_lab_path(tok)


# ── LLM tools (provider stubbed) ─────────────────────────────────────


def _stub_provider(mp, *, judge=4):
    from core import provider as prov

    def fake(provider_name, messages, **kw):
        sp = messages[0]["content"].lower()
        if "artifact synthesizer" in sp:
            body = {"body": "# Final\n\nClaim X[^0].\n\n[^0]: f.md",
                    "citations_used": 1, "uncited_evidence": []}
        elif "gap auditor" in sp:
            body = {"gaps_md": "# Gaps\n- only one benchmark", "gap_count": 1,
                    "unanswered_questions": ["edge cases?"], "honest_score": 0.8}
        else:  # grader judge
            body = dict.fromkeys(quality.DIMENSIONS, judge)
            body["rationale"] = "ok"
        return ProviderResponse(text=json.dumps(body), tool_calls=[],
                                finish_reason="stop", usage_prompt_tokens=10,
                                usage_completion_tokens=10, usage_thinking_tokens=0,
                                usage_cached_tokens=0, model="stub",
                                provider=provider_name, elapsed_ms=1)

    mp.setattr(prov, "call", fake)


def test_synthesize_artifact_body(monkeypatch):
    _stub_provider(monkeypatch)
    out = finalize_tools._synthesize_artifact_body(
        evidence=[{"source_path": "f.md", "content": "x"}], objective="o",
        cascade=[("groq", "m")])
    assert out["citations_used"] == 1 and out["word_count"] > 0
    assert "Final" in out["body"]


def test_analyze_evidence_holes(monkeypatch):
    _stub_provider(monkeypatch)
    out = finalize_tools._analyze_evidence_holes(
        evidence=[{"source_path": "f.md"}], artifact="# Final", objective="o",
        cascade=[("groq", "m")])
    assert out["gap_count"] == 1 and abs(out["honest_score"] - 0.8) < 1e-9
    assert "Gaps" in out["gaps_md"]


# ── the crown jewel: finalize_project end-to-end through the executor ─


def test_finalize_project_e2e_real_registry(monkeypatch, tmp):
    # Engine proof (task #68): finalize_project runs end-to-end against the REAL
    # tool_registry via make_invoker — gather -> synthesize -> disclose -> sign ->
    # record_in_ledger — producing a graded, signed, ready artifact. The two LLM
    # tools + grader judges are stubbed; everything else runs for real.
    _stub_provider(monkeypatch, judge=4)  # judges score 4 -> weighted 0.8 -> grade B
    (tmp / "findings").mkdir()
    (tmp / "findings" / "r1.md").write_text("# R1\nbert beats BM25 on scifact")
    tok = lab_context.set_active_lab_path(tmp)
    try:
        skill_registry.load_all(force_reload=True)
        reg = skill_registry.snapshot()
        ctx = skill_executor.ExecutionContext(
            tool_invoker=tool_registry.make_invoker(), skill_registry=reg)
        fp = reg["finalize_project"]
        result = skill_executor.execute_skill(fp, {
            "objective": "Audit retrieval quality",
            "output_path": "final.md",
        }, ctx)
        assert result.ok, f"errors={result.errors} steps={result.steps_executed}"
        assert result.outputs["grade"] == "B"
        assert len(result.outputs["signed_hash"]) == 64
        assert result.outputs["ready"] is True
        # real side effects landed in the temp lab
        assert (tmp / "final.md").exists()
        assert (tmp / "gaps.md").exists()
        assert (tmp / "lab" / "sor" / "events.jsonl").exists()  # ledger row written
        # composition order preserved
        order = result.steps_executed
        assert order.index("gather") < order.index("synthesize") < order.index("sign")
    finally:
        lab_context.reset_active_lab_path(tok)


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


def main() -> int:
    tests = [
        test_identity, test_all_tools_registered_and_invoker,
        test_list_findings_walks_dir, test_read_ledger_rows_filters,
        test_assemble_evidence_bundle, test_sha256_envelope_deterministic,
        test_validate_ledger_row, test_append_jsonl_atomic,
        test_finalize_ready_check, test_synthesize_artifact_body,
        test_analyze_evidence_holes, test_finalize_project_e2e_real_registry,
    ]
    for t in tests:
        mp = _MP()
        td = Path(tempfile.mkdtemp(prefix="bert_fin_"))
        try:
            params = inspect.signature(t).parameters
            kwargs = {}
            if "monkeypatch" in params:
                kwargs["monkeypatch"] = mp
            if "tmp" in params:
                kwargs["tmp"] = td
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
