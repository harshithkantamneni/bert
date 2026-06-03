"""Smoke + TDD: core/prewarm.py — pin retrieval models at MCP start (memory v3+).

v3+ priority-2 cold-start fix: the embedder (every search) and reranker (every
search unless BERT_DISABLE_RERANKER) are lazy-loaded on the FIRST query, paying a
measured p99.9 = 8.6s / max = 46s cold-start tail. prewarm() loads them at server
start (background daemon thread by default) so the first real query is warm.

Best-effort: a load failure (no model cached / offline) is logged, never raised —
pre-warming must not break server startup. Loaders are stubbed here (no real model
load). The MCP server's real entry (serve()) calls prewarm(); make_server() must
NOT (tests build servers constantly).
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "tools"))

from core import prewarm  # noqa: E402


def test_prewarm_sync_calls_both_loaders(monkeypatch):
    calls = []
    from core import memory, reranker
    monkeypatch.setattr(memory, "_get_embedder", lambda: calls.append("embed"))
    monkeypatch.setattr(memory, "_embedder", object())  # appears loaded
    monkeypatch.setattr(reranker, "_ensure_loaded", lambda: calls.append("rerank") or True)
    monkeypatch.delenv("BERT_DISABLE_RERANKER", raising=False)
    t = prewarm.prewarm(background=False)
    assert t is None  # sync mode returns no thread
    assert "embed" in calls and "rerank" in calls


def test_prewarm_background_returns_thread(monkeypatch):
    from core import memory, reranker
    monkeypatch.setattr(memory, "_get_embedder", lambda: None)
    monkeypatch.setattr(memory, "_embedder", object())
    monkeypatch.setattr(reranker, "_ensure_loaded", lambda: True)
    monkeypatch.delenv("BERT_DISABLE_RERANKER", raising=False)
    t = prewarm.prewarm(background=True)
    assert t is not None
    t.join(timeout=5)
    assert not t.is_alive()


def test_prewarm_embedder_failure_is_best_effort(monkeypatch):
    from core import memory

    def boom():
        raise RuntimeError("no model cached")

    monkeypatch.setattr(memory, "_get_embedder", boom)
    # must NOT raise
    ok = prewarm.prewarm_embedder()
    assert ok is False


def test_prewarm_reranker_skipped_when_disabled(monkeypatch):
    from core import reranker
    called = []
    monkeypatch.setattr(reranker, "_ensure_loaded", lambda: called.append(1) or True)
    monkeypatch.setenv("BERT_DISABLE_RERANKER", "1")
    ok = prewarm.prewarm_reranker()
    assert ok is False  # skipped
    assert called == []


def test_prewarm_reranker_loads_when_enabled(monkeypatch):
    from core import reranker
    monkeypatch.setattr(reranker, "_ensure_loaded", lambda: True)
    monkeypatch.delenv("BERT_DISABLE_RERANKER", raising=False)
    assert prewarm.prewarm_reranker() is True


# ── MCP server wiring ────────────────────────────────────────────────


def test_serve_prewarms_then_serves(monkeypatch):
    import importlib
    bl = importlib.import_module("tools.mcp.bert_lab")
    seen = {}

    class _FakeSrv:
        def serve_stdio(self):
            seen["served"] = True
            return 0

    monkeypatch.setattr(bl, "make_server", lambda: _FakeSrv())
    monkeypatch.setattr(prewarm, "prewarm", lambda **kw: seen.setdefault("prewarmed", True))
    rc = bl.serve()
    assert rc == 0
    assert seen.get("prewarmed") is True and seen.get("served") is True


def test_make_server_does_not_prewarm(monkeypatch):
    # Building a server (what tests do constantly) must NOT load models.
    import importlib
    bl = importlib.import_module("tools.mcp.bert_lab")
    triggered = []
    monkeypatch.setattr(prewarm, "prewarm", lambda **kw: triggered.append(1))
    bl.make_server()
    assert triggered == []


class _MP:
    def __init__(self):
        self._u = []
        self._env = []

    def setattr(self, obj, name, val):
        self._u.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)

    def setenv(self, k, v):
        import os
        self._env.append((k, os.environ.get(k)))
        os.environ[k] = v

    def delenv(self, k, raising=False):
        import os
        self._env.append((k, os.environ.get(k)))
        os.environ.pop(k, None)

    def undo(self):
        import os
        for o, n, v in reversed(self._u):
            setattr(o, n, v)
        for k, v in reversed(self._env):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._u.clear()
        self._env.clear()


def main() -> int:
    import inspect
    tests = [
        test_prewarm_sync_calls_both_loaders,
        test_prewarm_background_returns_thread,
        test_prewarm_embedder_failure_is_best_effort,
        test_prewarm_reranker_skipped_when_disabled,
        test_prewarm_reranker_loads_when_enabled,
        test_serve_prewarms_then_serves,
        test_make_server_does_not_prewarm,
    ]
    mp = _MP()
    for t in tests:
        try:
            if "monkeypatch" in inspect.signature(t).parameters:
                t(mp)
            else:
                t()
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
