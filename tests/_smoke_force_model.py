"""Smoke + TDD: BERT_FORCE_MODEL escape hatch in bert_run._resolve_dispatch_model.

The per-role router can pick a provider whose free-tier quota is exhausted (e.g.
gemini 429), and a single dispatch retries that one model rather than falling
back across providers. BERT_FORCE_MODEL lets an operator (or a data-gen / test
run) pin a known-good 'provider/model' for every dispatch, bypassing the router.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "tools"))

br = importlib.import_module("tools.bert_run")


def test_force_model_overrides_router(monkeypatch):
    monkeypatch.setenv("BERT_FORCE_MODEL", "groq/llama-3.3-70b-versatile")
    # even if the router would pick something else, the forced model wins
    out = br._resolve_dispatch_model("literature_hunter", "find papers", "nvidia/x")
    assert out == "groq/llama-3.3-70b-versatile"


def test_no_force_uses_router_or_default(monkeypatch):
    monkeypatch.delenv("BERT_FORCE_MODEL", raising=False)
    # router may resolve or fall back to the default; either way it's a
    # provider/model string, never the forced sentinel.
    out = br._resolve_dispatch_model("literature_hunter", "find papers", "nvidia/x")
    assert "/" in out


def test_blank_force_ignored(monkeypatch):
    monkeypatch.setenv("BERT_FORCE_MODEL", "   ")
    out = br._resolve_dispatch_model("writer", "write", "nvidia/x")
    assert out != "   " and "/" in out


class _MP:
    def __init__(self):
        self._env = []

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
        for k, v in reversed(self._env):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._env.clear()


def main() -> int:
    import inspect
    tests = [test_force_model_overrides_router, test_no_force_uses_router_or_default,
             test_blank_force_ignored]
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
