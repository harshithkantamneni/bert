"""Smoke: tools/bert_doctor.py — demo-readiness health checks (was 63%).

run_all_checks(with_network=False) exercises every offline check_* in one
call against the real repo state; overall_exit_code is checked across
synthetic fail/warn/ok results; render_text (verbose + color/no-color) and
render_json are exercised; main() is driven for text/json/verbose argv
(offline — no --with-network so no provider pings).
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "tools"))

import tools.bert_doctor as bd  # noqa: E402


def test_run_all_checks_offline():
    results = bd.run_all_checks(with_network=False)
    assert isinstance(results, list) and len(results) > 8
    assert all(hasattr(r, "level") and r.level in ("ok", "warn", "fail") for r in results)
    assert bd.overall_exit_code(results) in (0, 1, 2)


def test_overall_exit_code():
    ok = bd.CheckResult("a", "ok", "fine")
    warn = bd.CheckResult("b", "warn", "meh")
    fail = bd.CheckResult("c", "fail", "broken", fix_hint="do x")
    assert bd.overall_exit_code([ok]) == 0
    assert bd.overall_exit_code([ok, warn]) == 1
    assert bd.overall_exit_code([ok, warn, fail]) == 2


def test_render_text_and_json():
    results = [
        bd.CheckResult("ok-check", "ok", "good"),
        bd.CheckResult("warn-check", "warn", "careful", fix_hint="warn fix"),
        bd.CheckResult("fail-check", "fail", "bad", fix_hint="fail fix"),
    ]
    txt = bd.render_text(results, verbose=True, use_color=True)
    assert "bert doctor" in txt and "fix:" in txt
    txt2 = bd.render_text(results, verbose=False, use_color=False)
    assert "BLOCKED" in txt2          # a fail present → blocked
    parsed = json.loads(bd.render_json(results))
    assert parsed["exit_code"] == 2 and parsed["summary"]["fail"] == 1


def test_render_text_go_and_warn():
    assert "GO" in bd.render_text([bd.CheckResult("a", "ok", "x")], use_color=False)
    warn_txt = bd.render_text([bd.CheckResult("a", "warn", "x", fix_hint="h")],
                              verbose=True, use_color=False)
    assert "GO with warnings" in warn_txt


def test_individual_checks():
    # a free high port should be available
    r = bd.check_port_available(54999)
    assert r.level in ("ok", "warn", "fail")
    for fn in (bd.check_python_version, bd.check_venv_exists,
               bd.check_required_deps, bd.check_credentials_mode,
               bd.check_host_detection, bd.check_model_cards_present):
        res = fn()
        assert isinstance(res, bd.CheckResult)


def test_check_groq_reachable(monkeypatch):
    import urllib.request
    # no key → skipped warn (network-free)
    monkeypatch.setattr(bd, "os", bd.os)  # anchor for _MP restore
    monkeypatch.setattr(bd.os, "environ", {})
    assert bd.check_groq_reachable().level == "warn"
    # key present + mocked 200 → ok
    monkeypatch.setattr(bd.os, "environ", {"GROQ_API_KEY": "gsk_test"})

    class _Resp:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    assert bd.check_groq_reachable().level == "ok"
    # urlopen raises → handled (warn/fail, never crashes)
    def _boom(*a, **k):
        raise urllib.error.URLError("down")
    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    assert bd.check_groq_reachable().level in ("warn", "fail")
    # with_network=True now runs the (mocked) network check too
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    res = bd.run_all_checks(with_network=True)
    assert any(r.name.startswith("Groq") for r in res)


def test_main_variants(monkeypatch):
    for argv in (["x"], ["x", "--json"], ["x", "--verbose", "--no-color"]):
        monkeypatch.setattr(sys, "argv", argv)
        with contextlib.redirect_stdout(io.StringIO()):
            rc = bd.main()
        assert rc in (0, 1, 2)


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
    import inspect
    tests = [
        test_run_all_checks_offline,
        test_overall_exit_code,
        test_render_text_and_json,
        test_render_text_go_and_warn,
        test_individual_checks,
        test_check_groq_reachable,
        test_main_variants,
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
