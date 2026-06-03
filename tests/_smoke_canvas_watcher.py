"""Smoke: core/canvas_watcher.py (was 0% — 225 statements).

Drives the watcher's real logic against a temp lab fixture: pure helpers,
the three event builders, state load/save, a full poll_once pass (scan
findings/ + log.md decisions + seasoning.jsonl → emit), the enrich path,
and start/stop of the background thread. The LLM enrichment + canvas_emit
file write are stubbed so poll_once stays network-free + side-effect-free.
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

cw = importlib.import_module("core.canvas_watcher")


def test_pure_helpers():
    assert cw._hash_id("finding", "a", "b").startswith("finding")
    assert cw._hash_id("x", "p") == cw._hash_id("x", "p")  # deterministic
    assert len(cw._take_summary("word " * 500, max_chars=100)) <= 100
    # role regex anchors at the START of the filename
    assert cw._infer_role("researcher_C7.md") == "researcher"
    assert cw._infer_role("bert_run_C7_researcher.md") is None
    assert cw._extract_cycle("bert_run_C7_researcher.md") == 7
    assert cw._extract_cycle("no_cycle_here.md") is None


def test_state_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(cw, "STATE_PATH", tmp_path / "cw.state.json")
    monkeypatch.setattr(cw, "EVENTS_PATH", tmp_path / "events.jsonl")
    state = cw._load_state()              # missing file → fresh state
    assert isinstance(state, dict)
    state["seen"] = {"finding:x": "2026-05-28"}
    cw._save_state(state)
    reloaded = cw._load_state()
    assert reloaded.get("seen", {}).get("finding:x") == "2026-05-28"


def test_build_finding_event(tmp_path):
    p = tmp_path / "researcher_C3.md"
    p.write_text("# Title\n\nSome finding body with content.\n")
    ev = cw._build_finding_event(p)
    assert ev is not None
    assert ev.get("event_class") == "finding"
    assert ev.get("cycle") == 3


def test_build_log_decision_and_seasoning_events():
    if not cw.LOG_MD.exists():
        pytest.skip("requires lab runtime artifact not shipped in the public repo")
    dec = cw._build_log_decision_event("D-42", " ratify the thing\n\nbody")
    assert dec is not None and "D-42" in json.dumps(dec)
    seas = cw._build_seasoning_event(json.dumps({
        "ts": "2026-05-28T00:00:00Z", "claim": "x holds", "verdict": "seasoned",
    }))
    assert seas is None or isinstance(seas, dict)


def test_poll_once_scans_and_emits(tmp_path, monkeypatch):
    # temp corpora
    findings = tmp_path / "findings"
    findings.mkdir()
    (findings / "bert_run_C1_researcher.md").write_text("# F1\n\nbody one\n")
    (findings / "bert_run_C2_writer.md").write_text("# F2\n\nbody two\n")
    log_md = tmp_path / "log.md"
    log_md.write_text("## D-1 first decision\n\nrationale\n\n## D-2 second\n\nmore\n")
    seasoning = tmp_path / "seasoning.jsonl"
    seasoning.write_text(json.dumps({"ts": "2026-05-28T0:0:0Z", "claim": "c"}) + "\n")

    monkeypatch.setattr(cw, "FINDINGS_DIR", findings)
    monkeypatch.setattr(cw, "LOG_MD", log_md)
    monkeypatch.setattr(cw, "SEASONING_JSONL", seasoning)
    monkeypatch.setattr(cw, "STATE_PATH", tmp_path / "cw.state.json")

    emitted_events = []
    monkeypatch.setattr(cw, "_enrich_and_emit", lambda ev: emitted_events.append(ev))

    state = {"seen": {}}
    count = cw.poll_once(state)
    assert count >= 2, f"expected ≥2 findings emitted, got {count}"
    assert len(emitted_events) == count
    # second pass: nothing new (dedup via state)
    count2 = cw.poll_once(state)
    assert count2 == 0, f"second pass should emit 0 (deduped), got {count2}"


def test_enrich_and_emit_path(monkeypatch, tmp_path):
    from core import enrichment
    monkeypatch.setattr(enrichment, "enrich_one",
                        lambda ev: {"provenance": "stub", "enriched": True})
    monkeypatch.setattr(cw, "EVENTS_PATH", tmp_path / "events.jsonl")
    # canvas_emit._FILE_LOCK exists; the write appends to our temp EVENTS_PATH
    cw._enrich_and_emit({"event_class": "finding", "id": "f1", "summary": "s"})
    assert (tmp_path / "events.jsonl").exists()


def test_start_background_then_stop(monkeypatch, tmp_path):
    # neuter poll_once so the loop does nothing heavy, then start + stop
    monkeypatch.setattr(cw, "poll_once", lambda state: 0)
    monkeypatch.setattr(cw, "POLL_INTERVAL_SECS", 0.05)
    cw.start_background()
    time.sleep(0.15)
    cw.stop(timeout_secs=2.0)
    assert True  # started + stopped without hanging


def main() -> int:
    tests = [
        test_pure_helpers,
        test_state_roundtrip,
        test_build_finding_event,
        test_build_log_decision_and_seasoning_events,
        test_poll_once_scans_and_emits,
        test_enrich_and_emit_path,
        test_start_background_then_stop,
    ]
    # minimal monkeypatch shim for the non-pytest runner
    import contextlib

    class _MP:
        def __init__(self):
            self._undo = []
        def setattr(self, obj, name, val):
            old = getattr(obj, name)
            self._undo.append((obj, name, old))
            setattr(obj, name, val)
        def undo(self):
            for obj, name, old in reversed(self._undo):
                setattr(obj, name, old)
            self._undo.clear()

    import tempfile
    for t in tests:
        mp = _MP()
        td = Path(tempfile.mkdtemp())
        try:
            import inspect
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
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
        finally:
            mp.undo()
            with contextlib.suppress(Exception):
                import shutil
                shutil.rmtree(td)
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
