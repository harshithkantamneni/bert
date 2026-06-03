"""Smoke: enrich_events_jsonl + validate_provider_limits + measure_ollama (all 0%).

All network/LLM-free:
  - enrich_events_jsonl.enrich_all: temp events.jsonl, _enrich_one stubbed
    (llm / heuristic / skip), serial + parallel workers, dry-run, missing-file.
  - validate_provider_limits: _within_tolerance (pure), _format_report (pure),
    _probe_provider with prov.probe_models + record_probe stubbed.
  - measure_ollama_prefix_cache: _ns_to_ms (pure), _measure_pair + measure with
    _call_ollama stubbed to canned ollama timing dicts.
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "tools"))


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


# ── enrich_events_jsonl ───────────────────────────────────────────────

def test_enrich_all(monkeypatch, tmp_path):
    eej = importlib.import_module("enrich_events_jsonl")
    events = tmp_path / "events.jsonl"
    rows = [{"id": "a", "k": "llm"}, {"id": "b", "k": "heuristic"},
            {"id": "c", "k": "skip"}, {"id": "d", "tags": ["#done"]}]
    events.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def fake_enrich(ev, provider=None, model=None):
        if ev.get("k") == "skip":
            return None
        return {"tags": ["#x"], "lineage": [],
                "provenance": "llm" if ev.get("k") == "llm" else "heuristic"}
    monkeypatch.setattr(eej, "_enrich_one", fake_enrich)

    # dry run returns counts without enriching
    dr = eej.enrich_all(events_path=events, dry_run=True)
    assert dr["targeted"] == 3 and dr["total_events"] == 4
    # real run (serial)
    st = eej.enrich_all(events_path=events, dry_run=False, workers=1)
    assert st["enriched_llm"] == 1 and st["enriched_heuristic"] == 1 and st["skipped"] == 1
    # parallel path (re_enrich so there are targets again)
    st2 = eej.enrich_all(events_path=events, dry_run=False, workers=3, re_enrich=True)
    assert st2["targeted"] == 4
    # missing file
    assert "error" in eej.enrich_all(events_path=tmp_path / "nope.jsonl")


# ── validate_provider_limits ──────────────────────────────────────────

def test_within_tolerance():
    vpl = importlib.import_module("validate_provider_limits")
    assert vpl._within_tolerance(None, 100) is True       # no ceiling
    assert vpl._within_tolerance(100, None) is True        # can't observe
    assert vpl._within_tolerance(100, 102) is True         # within tolerance
    assert vpl._within_tolerance(100, 5000) is False       # way off


def test_format_report_and_probe(monkeypatch):
    vpl = importlib.import_module("validate_provider_limits")
    from core.quota import PROVIDER_LIMITS
    name = next(iter(PROVIDER_LIMITS))
    limits = PROVIDER_LIMITS[name]
    ok = vpl.ProbeResult(provider=name, reachable=True, latency_ms=42, declared=limits)
    bad = vpl.ProbeResult(provider="x", reachable=False, latency_ms=0, declared=limits, error="boom")
    report = vpl._format_report([ok, bad], since="2026-05-01")
    assert "Provider limits validation report" in report and "✓ OK" in report and "✗" in report
    # _probe_provider with mocked probe + record
    monkeypatch.setattr(vpl.prov, "probe_models", lambda n: (True, ["m1"], ""))
    monkeypatch.setattr(vpl, "record_probe", lambda *a, **k: None)
    res = vpl._probe_provider(name, limits, vpl.config.load())
    assert isinstance(res, vpl.ProbeResult)
    # unknown provider → unreachable
    res2 = vpl._probe_provider("no_such_provider", limits, vpl.config.load())
    assert res2.reachable is False


# ── measure_ollama_prefix_cache ───────────────────────────────────────

def _canned_ollama(host, model, prefix, delta, timeout=120.0):
    return {
        "prompt_eval_count": 100, "prompt_eval_duration": 50_000_000,  # 50ms
        "load_duration": 1_000_000, "eval_count": 32, "eval_duration": 80_000_000,
        "total_duration": 130_000_000,
    }


def test_measure_ollama(monkeypatch):
    mo = importlib.import_module("measure_ollama_prefix_cache")
    assert mo._ns_to_ms(1_000_000) == 1.0
    assert mo._ns_to_ms(0) == 0.0
    monkeypatch.setattr(mo, "_call_ollama", _canned_ollama)
    pair = mo._measure_pair("h", "m", "prefix", "a", "b")
    assert pair["cold"]["prompt_eval_ms"] == 50.0 and pair["warm"]["eval_count"] == 32
    out = mo.measure(iterations=2)
    assert out["ok"] is True and out["iterations"] == 2
    assert "summary" in out


def test_measure_ollama_unreachable(monkeypatch):
    mo = importlib.import_module("measure_ollama_prefix_cache")
    def _boom(*a, **k):
        raise RuntimeError("ollama down")
    monkeypatch.setattr(mo, "_call_ollama", _boom)
    out = mo.measure(iterations=1)
    assert out["ok"] is False and "error" in out


def test_enrich_main(monkeypatch, tmp_path):
    eej = importlib.import_module("enrich_events_jsonl")
    events = tmp_path / "events.jsonl"
    events.write_text(json.dumps({"id": "a"}) + "\n")
    monkeypatch.setattr(sys, "argv", ["x", "--events", str(events), "--dry-run"])
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        assert eej.main() == 0


def test_validate_main(monkeypatch, tmp_path):
    vpl = importlib.import_module("validate_provider_limits")
    monkeypatch.setattr(vpl.prov, "probe_models", lambda n: (True, ["m1"], ""))
    monkeypatch.setattr(vpl, "record_probe", lambda *a, **k: None)
    monkeypatch.setattr(vpl, "LAB_ROOT", tmp_path)
    monkeypatch.setattr(vpl, "REPORT_PATH", tmp_path / "report.md")
    monkeypatch.setattr(sys, "argv", ["x"])
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        rc = vpl.main()
    assert rc == 0 and (tmp_path / "report.md").exists()


def test_measure_main(monkeypatch):
    mo = importlib.import_module("measure_ollama_prefix_cache")
    monkeypatch.setattr(mo, "_call_ollama", _canned_ollama)
    monkeypatch.setattr(sys, "argv", ["x", "--iterations", "2", "--json"])
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        assert mo.main() == 0


def test_measure_verdict_tiers():
    mo = importlib.import_module("measure_ollama_prefix_cache")
    assert mo._verdict(6.0, 5.0)["rating"] == "excellent"
    assert mo._verdict(3.0, 2.5)["rating"] == "good"
    assert mo._verdict(1.2, 1.1)["rating"] == "poor"


def test_measure_main_human_readable(monkeypatch):
    mo = importlib.import_module("measure_ollama_prefix_cache")
    monkeypatch.setattr(mo, "_call_ollama", _canned_ollama)
    monkeypatch.setattr(sys, "argv", ["x", "--iterations", "1"])  # no --json
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        assert mo.main() == 0


def main() -> int:
    tests = [
        test_enrich_all,
        test_enrich_main,
        test_within_tolerance,
        test_format_report_and_probe,
        test_validate_main,
        test_measure_ollama,
        test_measure_ollama_unreachable,
        test_measure_main,
        test_measure_verdict_tiers,
        test_measure_main_human_readable,
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
            import shutil
            shutil.rmtree(td, ignore_errors=True)
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
