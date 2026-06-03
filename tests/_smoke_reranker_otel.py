"""Smoke: core/reranker.py (54%) + tools/check_otel_setup.py (0%).

reranker: status / is_available / rerank / get_cross_encoder_rerank_fn with
the CrossEncoder model faked (a stub .predict), so the real ranking plumbing
runs without loading torch. check_otel_setup: check_sdk / check_exporter
(env-driven) / emit_probe / main — all direct, no network.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import io
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "tools"))

from core import reranker  # noqa: E402


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


class _FakeCE:
    def predict(self, pairs, show_progress_bar=False):
        # higher score for longer passages — deterministic
        return [float(len(p[1])) for p in pairs]


def test_reranker_status_available(monkeypatch):
    # Pin a fake model so is_available()/_ensure_loaded() short-circuit at the
    # "model already loaded" check and NEVER load the real 568 MB CrossEncoder
    # (which OOM-stalls this host and blew the coverage timeout).
    monkeypatch.setattr(reranker, "_model", _FakeCE())
    st = reranker.status()
    assert hasattr(st, "loaded") and hasattr(st, "model")
    assert reranker.is_available() is True


def test_reranker_rerank(monkeypatch):
    import types
    monkeypatch.setattr(reranker, "_model", _FakeCE())
    monkeypatch.setattr(reranker, "_load_failed", False)
    monkeypatch.setattr(reranker.os, "environ",
                        {k: v for k, v in reranker.os.environ.items()
                         if k != "BERT_DISABLE_RERANKER"})
    scores = reranker.rerank("query", ["short", "a much longer passage here"])
    assert len(scores) == 2 and scores[1] > scores[0]
    # empty inputs short-circuit
    assert reranker.rerank("", ["p"]) == []
    assert reranker.rerank("q", []) == []
    # the convenience fn wrapper: candidates are objects with .text
    fn = reranker.get_cross_encoder_rerank_fn()
    assert callable(fn)
    cands = [types.SimpleNamespace(text="one", rerank_score=0.0),
             types.SimpleNamespace(text="two much longer passage", rerank_score=0.0)]
    out = fn("query", cands)
    assert out[1].rerank_score >= out[0].rerank_score
    assert fn("query", []) == []        # empty short-circuit


def test_reranker_disabled(monkeypatch):
    monkeypatch.setattr(reranker.os, "environ", {"BERT_DISABLE_RERANKER": "1"})
    assert reranker.get_cross_encoder_rerank_fn() is None


def test_reranker_unavailable(monkeypatch):
    # no model + load disabled → rerank returns [] gracefully
    monkeypatch.setattr(reranker, "_model", None)
    monkeypatch.setattr(reranker, "_ensure_loaded", lambda: False)
    assert reranker.rerank("q", ["p1", "p2"]) == []


def test_check_otel(monkeypatch):
    otel = importlib.import_module("check_otel_setup")
    sdk = otel.check_sdk()
    assert "sdk_present" in sdk
    # exporter: env-driven both ways
    monkeypatch.setattr(otel.os, "environ", {})
    assert otel.check_exporter()["endpoint_set"] is False
    monkeypatch.setattr(otel.os, "environ",
                        {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4318",
                         "OTEL_SERVICE_NAME": "bert-test"})
    exp = otel.check_exporter()
    assert exp["endpoint_set"] is True and exp["service"] == "bert-test"
    # probe + main (no network; emit is a no-op without a collector)
    assert "emit_ok" in otel.emit_probe()
    with contextlib.redirect_stdout(io.StringIO()):
        assert otel.main() in (0, 1)


def main() -> int:
    tests = [
        test_reranker_status_available,
        test_reranker_rerank,
        test_reranker_disabled,
        test_reranker_unavailable,
        test_check_otel,
    ]
    for t in tests:
        mp = _MP()
        try:
            kwargs = {"monkeypatch": mp} if "monkeypatch" in inspect.signature(t).parameters else {}
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
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
