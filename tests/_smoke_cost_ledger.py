"""Smoke + TDD: core/cost_ledger.py — per-model_call cost ledger (Sprint 4 C2).

Appends one priced row per model_call to state/observability/cost.jsonl
(free-tier providers = $0 but tokens still tracked for attribution; host/BYO
models priced from core/library/model_prices.yaml). Thinking tokens tracked
as a separate column (criterion 25). summarize() rolls up per (provider,model).
"""

from __future__ import annotations

import inspect
import shutil
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import cost_ledger as cl  # noqa: E402


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


def test_record_free_tier_zero_cost(monkeypatch, tmp_path):
    monkeypatch.setattr(cl, "LEDGER_PATH", tmp_path / "cost.jsonl")
    row = cl.record(provider="groq", model="llama-3.3-70b-versatile",
                    input_tokens=1000, output_tokens=500, lab="demo", cycle=1)
    assert row["usd_estimate"] == 0.0          # free-tier
    assert row["input_tokens"] == 1000 and row["output_tokens"] == 500
    assert (tmp_path / "cost.jsonl").exists()


def test_record_priced_model(monkeypatch, tmp_path):
    monkeypatch.setattr(cl, "LEDGER_PATH", tmp_path / "cost.jsonl")
    # a host/BYO priced model → non-zero estimate
    row = cl.record(provider="anthropic", model="claude-opus-4-7",
                    input_tokens=1000, output_tokens=1000, thinking_tokens=200)
    assert row["usd_estimate"] > 0.0
    assert row["thinking_tokens"] == 200        # criterion 25 — tracked separately


def test_price_for():
    free = cl._price_for("nvidia", "meta/llama-3.3-70b-instruct")
    assert free == (0.0, 0.0)
    priced = cl._price_for("anthropic", "claude-opus-4-7")
    assert priced[0] > 0 and priced[1] > priced[0]
    # unknown → default (free)
    assert cl._price_for("mystery", "mystery-model") == (0.0, 0.0)


def test_summarize(monkeypatch, tmp_path):
    monkeypatch.setattr(cl, "LEDGER_PATH", tmp_path / "cost.jsonl")
    cl.record(provider="groq", model="m1", input_tokens=100, output_tokens=50)
    cl.record(provider="groq", model="m1", input_tokens=200, output_tokens=80)
    cl.record(provider="anthropic", model="claude-opus-4-7",
              input_tokens=1000, output_tokens=1000)
    summ = cl.summarize()
    assert summ["totals"]["calls"] == 3
    key = "groq/m1"
    assert summ["by_model"][key]["calls"] == 2
    assert summ["by_model"][key]["input_tokens"] == 300
    assert summ["totals"]["usd_estimate"] > 0.0   # opus row priced


def test_summarize_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(cl, "LEDGER_PATH", tmp_path / "nope.jsonl")
    summ = cl.summarize()
    assert summ["totals"]["calls"] == 0


def test_history_usd_filters(monkeypatch, tmp_path):
    monkeypatch.setattr(cl, "LEDGER_PATH", tmp_path / "cost.jsonl")
    cl.record(provider="anthropic", model="claude-opus-4-7",
              input_tokens=1000, output_tokens=1000, lab="A")
    cl.record(provider="anthropic", model="claude-opus-4-7",
              input_tokens=2000, output_tokens=2000, lab="A")
    cl.record(provider="groq", model="g", input_tokens=100, output_tokens=50, lab="B")
    all_usd = cl.history_usd()
    assert len(all_usd) == 3
    opus = cl.history_usd(model="claude-opus-4-7")
    assert len(opus) == 2 and all(u > 0 for u in opus)
    lab_b = cl.history_usd(lab="B")
    assert len(lab_b) == 1


def test_ledger_feeds_cost_estimator_end_to_end(monkeypatch, tmp_path):
    # The integration claim "ledger feeds the cost-estimate-with-CI" (criterion
    # 22), now ENCODED as a test (recheck 2026-05-28): ledger rows → history_usd
    # → cost_estimator.estimate(history=...) → the CI is history-based, not the
    # default band.
    from core import cost_estimator as ce
    monkeypatch.setattr(cl, "LEDGER_PATH", tmp_path / "cost.jsonl")
    for _ in range(5):  # ≥ MIN_HISTORY so history overrides the default band
        cl.record(provider="anthropic", model="claude-opus-4-7",
                  input_tokens=1000, output_tokens=1000)
    history = cl.history_usd(model="claude-opus-4-7")
    assert len(history) == 5
    est = ce.estimate(0.05, history=history)
    assert est.n_samples == 5                 # estimator consumed the ledger history
    assert "default" not in est.basis.lower() or est.n_samples >= ce.MIN_HISTORY


def main() -> int:
    tests = [
        test_record_free_tier_zero_cost,
        test_record_priced_model,
        test_price_for,
        test_summarize,
        test_summarize_empty,
        test_history_usd_filters,
        test_ledger_feeds_cost_estimator_end_to_end,
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
