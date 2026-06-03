"""TDD: _run_one_cycle must route by effort. Trivial lookups get ONE direct
cheap-tier answer (no ritual, no 2nd role); deep tasks get the full roster.
Stubs _safe_dispatch so no real model is called."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import tools.bert_run as br  # noqa: E402


def _install_capture(monkeypatch):
    seen = []

    def fake_dispatch(spec, label, lab_path=None):
        seen.append({"spec": spec, "label": label})
        return {"role": spec["role"], "cycle": spec["cycle"], "verdict": "APPROVE",
                "result_valid": True, "telemetry": {}}

    monkeypatch.setattr(br, "_safe_dispatch", fake_dispatch)
    return seen


def test_trivial_seed_runs_single_direct_dispatch(monkeypatch):
    monkeypatch.setenv("BERT_EFFORT_TRIAGE", "on")
    seen = _install_capture(monkeypatch)
    out = br._run_one_cycle(seed="What's the default port for PostgreSQL?",
                            cycle=1, model=br.DEFAULT_MODEL,
                            roster=("researcher", "strategist"))
    assert len(seen) == 1                       # NOT the 2-role roster
    spec = seen[0]["spec"]
    assert spec["model"].endswith("haiku") or "haiku" in spec["model"]  # cheap tier
    vs = spec.get("verification_spec") or {}
    assert vs.get("min_chars", 9999) <= 100     # light spec, no 1500 ritual floor
    assert not vs.get("required_headers")       # no headers mandated
    assert out.get("effort") == "trivial"


def test_deep_seed_runs_full_roster(monkeypatch):
    monkeypatch.setenv("BERT_EFFORT_TRIAGE", "on")
    seen = _install_capture(monkeypatch)
    br._run_one_cycle(
        seed="Compare Kafka, RabbitMQ and Redis Streams and recommend one.",
        cycle=2, model=br.DEFAULT_MODEL, roster=("researcher", "strategist"))
    assert len(seen) == 2                        # full roster, full ritual


def test_triage_off_ablation_runs_full_roster(monkeypatch):
    monkeypatch.setenv("BERT_EFFORT_TRIAGE", "off")
    seen = _install_capture(monkeypatch)
    br._run_one_cycle(seed="What's the default port for PostgreSQL?",
                      cycle=3, model=br.DEFAULT_MODEL,
                      roster=("researcher", "strategist"))
    assert len(seen) == 2                        # ablation: triage disabled


class _MP:
    def __init__(self):
        self._u, self._e = [], []

    def setattr(self, obj, name, val):
        self._u.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)

    def setenv(self, k, v):
        self._e.append((k, os.environ.get(k)))
        os.environ[k] = v

    def undo(self):
        for o, n, v in reversed(self._u):
            setattr(o, n, v)
        for k, v in reversed(self._e):
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        self._u.clear()
        self._e.clear()


def main() -> int:
    import inspect
    tests = [test_trivial_seed_runs_single_direct_dispatch,
             test_deep_seed_runs_full_roster,
             test_triage_off_ablation_runs_full_roster]
    mp = _MP()
    for t in tests:
        try:
            t(mp) if "monkeypatch" in inspect.signature(t).parameters else t()
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
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
