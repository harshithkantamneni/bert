"""Smoke: file-based generator/migration tools (all were 0%).

Each tool uses module-level path constants; we monkeypatch them to temp
trees so the real logic runs without touching the repo (migrate_to_pace_
layers literally moves files — it runs against a throwaway tree here).
install_nightly's launchctl subprocess is mocked.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import sys
import tempfile
import types
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


def _run(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return fn()


def test_generate_now_page(monkeypatch, tmp_path):
    m = importlib.import_module("generate_now_page")
    monkeypatch.setattr(m, "NOW_PATH", tmp_path / "now.md")
    rc = _run(m.main)
    assert (rc in (0, 1)) or rc is None
    # it should have written the now page
    assert (tmp_path / "now.md").exists()


def test_export_for_web(monkeypatch, tmp_path):
    m = importlib.import_module("export_for_web")
    monkeypatch.setattr(m, "LAB_ROOT", tmp_path)  # so relative_to() resolves
    events = tmp_path / "events.jsonl"
    events.write_text("\n".join(json.dumps({
        "ts": "2026-05-28T00:00:00Z", "event_class": "finding", "lab": "x",
        "cycle": 1, "summary": "s", "id": f"f{i}",
    }) for i in range(5)) + "\n")
    monkeypatch.setattr(m, "EVENTS_PATH", events)
    monkeypatch.setattr(m, "SEASONING_PATH", tmp_path / "seasoning.jsonl")
    monkeypatch.setattr(m, "PRIVATE_MD", tmp_path / "PRIVATE.md")
    monkeypatch.setattr(m, "WEB_DATA", tmp_path / "web_data")
    fdir = tmp_path / "findings"; fdir.mkdir()
    (fdir / "strategist_x.md").write_text("# strat\nbody\n")
    (fdir / "architect_y.md").write_text("# arch\nbody\n")
    monkeypatch.setattr(m, "FINDINGS_DIR", fdir)
    monkeypatch.setattr(m, "CATHEDRAL_DIR", tmp_path / "cathedral")
    rc = _run(m.main)
    assert (rc in (0, 1)) or rc is None


def test_install_nightly(monkeypatch, tmp_path):
    m = importlib.import_module("install_nightly")
    # pure builders
    plist = m.build_plist(hour=22, minute=30)
    assert "<plist" in plist and "22" in plist
    cron = m.build_crontab_line(hour=22, minute=30)
    assert "30 22" in cron
    # mock launchctl + redirect the plist so nothing touches real LaunchAgents
    monkeypatch.setattr(m.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="ok", stderr=""))
    monkeypatch.setattr(m, "_plist_path", lambda: tmp_path / "nightly.plist")
    monkeypatch.setattr(m, "_is_macos", lambda: True)
    assert isinstance(m.install_macos(23, 0), int)
    assert isinstance(m.install_macos(23, 0, print_only=True), int)
    assert isinstance(m.uninstall_macos(), int)
    assert isinstance(m.status_macos(), int)
    # drive main() across each subcommand via argv
    for argv in (["x", "--status"], ["x", "--print-only"], ["x", "--install"],
                 ["x", "--uninstall"], ["x", "--hour", "99"], ["x", "--minute", "77"]):
        monkeypatch.setattr(sys, "argv", argv)
        rc = None
        with contextlib.suppress(SystemExit):
            rc = _run(m.main)
        assert rc is None or isinstance(rc, int)


def test_migrate_to_pace_layers_on_temp_tree(monkeypatch, tmp_path):
    m = importlib.import_module("migrate_to_pace_layers")
    # build a throwaway tree so the migration moves TEMP files, not the repo
    for d in ("core", "prompts", "tests", "tools"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
        (tmp_path / d / "sample.py").write_text("# sample\n")
    monkeypatch.setattr(m, "LAB_ROOT", tmp_path)
    rc = None
    with contextlib.suppress(SystemExit):
        rc = _run(m.main)
    assert rc is None or isinstance(rc, int)


def test_diverse_traffic_helpers():
    import random
    m = importlib.import_module("generate_diverse_traffic")
    templates = m.flatten()
    assert isinstance(templates, list) and len(templates) > 10
    rng = random.Random(0)
    q = m.zipf_sample(rng, templates, alpha=1.1)
    assert isinstance(q, str) and q in templates


def test_observability_traffic_helpers():
    import random
    m = importlib.import_module("generate_observability_traffic")
    rng = random.Random(0)
    assert isinstance(m.pick_role(rng), str)
    assert isinstance(m.pick_query(rng), str)


def test_bm25_traffic_main_bounded():
    # embedder-free (patches core.memory.search → []); max_hours caps runtime
    m = importlib.import_module("generate_bm25_traffic")
    rc = _run(lambda: m.main(target=2, alpha=1.1, max_hours=0.002, seed=0))
    assert rc == 0


def test_diverse_traffic_main_stubbed(monkeypatch):
    # stub hybrid_retrieve → [] so main runs without loading the torch embedder
    from core import retrieval as _ret
    monkeypatch.setattr(_ret, "hybrid_retrieve", lambda *a, **k: [])
    m = importlib.import_module("generate_diverse_traffic")
    rc = _run(lambda: m.main(target=3, alpha=1.1, max_hours=0.01, seed=0))
    assert rc == 0


def main() -> int:
    import inspect
    tests = [
        test_generate_now_page,
        test_export_for_web,
        test_install_nightly,
        test_migrate_to_pace_layers_on_temp_tree,
        test_diverse_traffic_helpers,
        test_observability_traffic_helpers,
        test_bm25_traffic_main_bounded,
        test_diverse_traffic_main_stubbed,
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
