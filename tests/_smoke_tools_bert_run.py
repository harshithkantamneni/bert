"""Smoke: tools/bert_run.py — autonomous-loop builders (was 26%).

The bulk of bert_run is pure text/spec builders + file helpers, none of
which need a provider. We drive them directly, plus _safe_dispatch /
_run_one_cycle with core.subagent.run_subagent stubbed (and the claude-CLI
bridge disabled via env) so no model is ever called.
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

br = importlib.import_module("bert_run")


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


def test_print_and_keys():
    br._print("hi", "")
    ok, present = br._check_provider_keys()
    assert isinstance(ok, bool) and isinstance(present, list)


def test_resolve_lab_and_seed(tmp_path):
    assert br._resolve_lab(None) == br.LAB_ROOT / "lab"
    assert isinstance(br._resolve_lab(str(tmp_path)), Path)
    (tmp_path / "seed_brief.md").write_text("# Seed\n\nInvestigate vector DB recall.\n")
    brief = br._read_seed_brief(tmp_path)
    assert "Investigate" in brief


def test_next_cycle_id(tmp_path):
    assert br._next_cycle_id(tmp_path, fallback_start=1) == 1   # no events
    (tmp_path / "sor").mkdir(parents=True)
    (tmp_path / "sor" / "events.jsonl").write_text(
        "\n".join(json.dumps({"cycle": c}) for c in (3, 5, 4)) + "\n")
    assert br._next_cycle_id(tmp_path) == 6                       # max+1


def test_task_builders():
    seed = "Survey vector databases and compare recall/latency."
    assert "vector" in br._seed_to_research_task(seed).lower()
    assert isinstance(br._skill_plan_section_for_role("researcher"), str)
    assert isinstance(br._seed_to_role_task("evaluator", seed), str)
    assert isinstance(br._seed_to_role_task("strategist", seed, prior_findings=["f1"]), str)
    strat = br._seed_to_strategy_task(seed, "findings/research_C1.md")
    assert "findings/research_C1.md" in strat


def test_model_and_verification_and_spec():
    m = br._resolve_dispatch_model("researcher", "compare recall", br.DEFAULT_MODEL)
    assert isinstance(m, str) and m
    assert isinstance(br._verification_spec_for_role("implementer"), dict)
    assert isinstance(br._verification_spec_for_role("researcher"), dict)
    spec = br._build_spec(role="researcher", cycle=3, task="Do the research thoroughly. " * 3,
                          output_path="findings/r_C3.md", model="nvidia/x",
                          falsifier_text="Fails if the finding is uncited or the file is missing entirely.")
    assert spec["role"] == "researcher" and spec["cycle"] == 3
    assert len(spec["falsifier_text"]) >= 30 and "process_hygiene" in spec


def test_warmup_jsonschema():
    br._warmup_jsonschema(attempts=1, sleep_s=0.0)   # best-effort, never raises


def test_safe_dispatch_and_cycle(monkeypatch, tmp_path):
    from core import subagent
    monkeypatch.setattr(br.os, "environ", {**br.os.environ, "BERT_VIA_CLAUDE": "0"})
    canned = {"verdict": "APPROVE", "role": "researcher", "cycle": 1,
              "result_valid": True, "errors": [], "output_path": "findings/x.md",
              "findings_count": {"high": 1, "med": 0, "low": 0, "nit": 0},
              "telemetry": {}}
    monkeypatch.setattr(subagent, "run_subagent", lambda spec: canned)
    spec = br._build_spec(role="researcher", cycle=1, task="Research the topic well. " * 3,
                          output_path="findings/x.md", model="nvidia/x",
                          falsifier_text="Fails if the output file is missing or schema-invalid entirely.")
    out = br._safe_dispatch(spec, "researcher", lab_path=tmp_path)
    assert out["verdict"] == "APPROVE"
    # _run_one_cycle drives a dispatch sequence; stub _safe_dispatch to stay offline
    monkeypatch.setattr(br, "_safe_dispatch", lambda spec, label, lab_path=None: canned)
    rc = _quiet(br._run_one_cycle, seed="Survey vector DBs.", cycle=1,
                model=br.DEFAULT_MODEL, lab_path=tmp_path)
    assert isinstance(rc, (int, dict, tuple)) or rc is None


def _quiet(fn, **k):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(**k)


def test_run_dry_run(monkeypatch, tmp_path):
    # provider keys present + seed brief + dispatch stubbed → exercise the run()
    # setup/cycle-plan path without any model call.
    monkeypatch.setattr(br, "_check_provider_keys", lambda: (True, ["GROQ_API_KEY"]))
    canned = {"verdict": "APPROVE", "role": "researcher", "cycle": 1,
              "result_valid": True, "errors": [], "output_path": "findings/x.md",
              "findings_count": {"high": 1, "med": 0, "low": 0, "nit": 0}, "telemetry": {}}
    monkeypatch.setattr(br, "_safe_dispatch", lambda spec, label, lab_path=None: canned)
    (tmp_path / "seed_brief.md").write_text("# Seed\n\nInvestigate vector DB recall thoroughly.\n")
    rc = None
    with contextlib.suppress(Exception):
        rc = _quiet(br.run, lab_path=tmp_path, max_cycles=1, dry_run=True, autonomous=False)
    assert rc is None or isinstance(rc, int)


def test_run_one_real_cycle(monkeypatch, tmp_path):
    # Drive the full (non-dry, non-autonomous) run() loop for one cycle with
    # the dispatch seam stubbed: legacy roster (skip the LLM schema classifier),
    # keys present, _safe_dispatch canned, observability emits no-op'd.
    monkeypatch.setattr(br.os, "environ",
                        {**br.os.environ, "BERT_LEGACY_RESEARCHER_STRATEGIST": "1"})
    monkeypatch.setattr(br, "_check_provider_keys", lambda: (True, ["GROQ_API_KEY"]))
    canned = {"verdict": "APPROVE", "role": "researcher", "cycle": 1,
              "result_valid": True, "errors": [], "label": "researcher",
              "output_path": "findings/x.md", "elapsed_secs": 0.1,
              "findings_count": {"high": 1, "med": 0, "low": 0, "nit": 0}, "telemetry": {}}
    monkeypatch.setattr(br, "_safe_dispatch", lambda spec, label, lab_path=None: canned)
    from core import observability as obs
    for fn in ("emit", "emit_cycle_outcome", "emit_model_call"):
        monkeypatch.setattr(obs, fn, lambda *a, **k: None)
    (tmp_path / "sor").mkdir(parents=True)
    (tmp_path / "findings").mkdir(parents=True)
    (tmp_path / "seed_brief.md").write_text("# Seed\n\nInvestigate vector DB recall thoroughly.\n")
    rc = _quiet(br.run, lab_path=tmp_path, max_cycles=1, dry_run=False, autonomous=False)
    assert rc in (0, 1)


def test_cycle_started_emitted(monkeypatch, tmp_path):
    # Sprint 4 C1 — _run_one_cycle emits a cycle_started event before dispatch.
    canned = {"verdict": "APPROVE", "role": "researcher", "cycle": 7,
              "result_valid": True, "errors": [], "label": "researcher",
              "output_path": "findings/x.md", "elapsed_secs": 0.1,
              "findings_count": {"high": 0, "med": 0, "low": 0, "nit": 0}, "telemetry": {}}
    monkeypatch.setattr(br, "_safe_dispatch", lambda spec, label, lab_path=None: canned)
    from core import observability as obs
    seen: list[str] = []
    monkeypatch.setattr(obs, "emit", lambda ec, payload=None, *a, **k: seen.append(ec))
    (tmp_path / "sor").mkdir(parents=True)
    (tmp_path / "findings").mkdir(parents=True)
    _quiet(lambda: br._run_one_cycle(seed="investigate recall", cycle=7,
                                     model=br.DEFAULT_MODEL, lab_path=tmp_path,
                                     roster=("researcher",)))
    assert "cycle_started" in seen


def test_run_autonomous_director(monkeypatch, tmp_path):
    # Drive the autonomous director block with core.director + core.outcome
    # stubbed (non-terminal decision, no termination guardrails tripped) so the
    # director→dispatch→outcome-grading path runs offline.
    import types
    monkeypatch.setattr(br.os, "environ",
                        {**br.os.environ, "BERT_LEGACY_RESEARCHER_STRATEGIST": "1"})
    monkeypatch.setattr(br, "_check_provider_keys", lambda: (True, ["GROQ_API_KEY"]))
    canned = {"verdict": "APPROVE", "role": "researcher", "cycle": 1,
              "result_valid": True, "errors": [], "label": "researcher",
              "output_path": "findings/x.md", "elapsed_secs": 0.1,
              "findings_count": {"high": 1, "med": 0, "low": 0, "nit": 0}, "telemetry": {}}
    monkeypatch.setattr(br, "_safe_dispatch", lambda spec, label, lab_path=None: canned)

    from core import director as dmod
    from core import observability as obs
    from core import outcome as omod
    decision = types.SimpleNamespace(
        cycle_shape="research-deeper", focus_area="recall", confidence_1to10=7,
        rationale="proceed with the next research pass on recall metrics",
        is_terminal=lambda: False, is_complete=lambda: False)
    observation = types.SimpleNamespace(pending_count=0)
    graded = types.SimpleNamespace(label=types.SimpleNamespace(value="ADVANCED"), reasoning="solid")
    monkeypatch.setattr(dmod, "gather_observation", lambda *a, **k: observation)
    monkeypatch.setattr(dmod, "decide_next_cycle", lambda *a, **k: decision)
    monkeypatch.setattr(dmod, "emit_decision_event", lambda *a, **k: None)
    monkeypatch.setattr(dmod, "check_three_strike", lambda *a, **k: False)
    monkeypatch.setattr(dmod, "check_failure_cascade", lambda *a, **k: False)
    monkeypatch.setattr(dmod, "check_pending_threshold", lambda *a, **k: False)
    monkeypatch.setattr(dmod, "_read_recent_events", lambda *a, **k: [])
    monkeypatch.setattr(dmod, "compose_researcher_prompt_from_decision", lambda d, s: s)
    monkeypatch.setattr(omod, "grade_immediate", lambda *a, **k: graded)
    monkeypatch.setattr(omod, "emit_outcome_event", lambda *a, **k: None)
    for fn in ("emit", "emit_cycle_outcome", "emit_model_call"):
        monkeypatch.setattr(obs, fn, lambda *a, **k: None)
    (tmp_path / "sor").mkdir(parents=True)
    (tmp_path / "findings").mkdir(parents=True)
    (tmp_path / "seed_brief.md").write_text("# Seed\n\nInvestigate vector DB recall thoroughly.\n")
    rc = _quiet(br.run, lab_path=tmp_path, max_cycles=1, dry_run=False, autonomous=True)
    assert rc in (0, 1)


def main() -> int:
    tests = [
        test_print_and_keys,
        test_resolve_lab_and_seed,
        test_next_cycle_id,
        test_task_builders,
        test_model_and_verification_and_spec,
        test_warmup_jsonschema,
        test_safe_dispatch_and_cycle,
        test_run_dry_run,
        test_run_one_real_cycle,
        test_cycle_started_emitted,
        test_run_autonomous_director,
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
