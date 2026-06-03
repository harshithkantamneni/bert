"""Smoke test for L.4: BERT_LAB_PATH routing + /api/labs discovery."""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["BERT_DISABLE_IDLE_COMPUTE"] = "1"

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


def _reload_api_main():
    """Re-import api.main so module-level BERT_LAB_PATH re-evaluates."""
    for mod in list(sys.modules.keys()):
        if mod.startswith("api.main") or mod == "api":
            del sys.modules[mod]
    import api.main as m
    return m


def test_default_lab_path_is_bert_lab() -> None:
    """No BERT_LAB_PATH → LAB_PATH defaults to bert-lab/lab/."""
    os.environ.pop("BERT_LAB_PATH", None)
    m = _reload_api_main()
    assert m.LAB_PATH == m.LAB_ROOT / "lab"


def test_env_override_changes_lab_path() -> None:
    """BERT_LAB_PATH env var redirects LAB_PATH."""
    tmp = Path(tempfile.mkdtemp(prefix="bert_l4_"))
    os.environ["BERT_LAB_PATH"] = str(tmp)
    try:
        m = _reload_api_main()
        assert m.LAB_PATH.resolve() == tmp.resolve()
        assert m.EVENTS_PATH == m.LAB_PATH / "sor" / "events.jsonl"
        assert m.STATE_DIR == m.LAB_PATH / "state"
    finally:
        os.environ.pop("BERT_LAB_PATH", None)
        shutil.rmtree(tmp)


def test_labs_endpoint_shape() -> None:
    """/api/labs returns active + scaffolded list."""
    os.environ.pop("BERT_LAB_PATH", None)
    m = _reload_api_main()
    from fastapi.testclient import TestClient
    r = TestClient(m.app).get("/api/labs")
    assert r.status_code == 200
    body = r.json()
    assert "active" in body
    assert "scaffolded" in body
    assert isinstance(body["scaffolded"], list)
    assert "path" in body["active"]
    assert "is_bert_lab_default" in body["active"]
    assert body["active"]["is_bert_lab_default"] is True


def test_labs_endpoint_discovers_scaffolded() -> None:
    """A scaffolded lab at ~/.bert/labs/X/ with lab.yaml is discovered."""
    fake_home = Path(tempfile.mkdtemp(prefix="bert_l4_home_"))
    labs_dir = fake_home / ".bert" / "labs"
    (labs_dir / "test-lab-a").mkdir(parents=True)
    (labs_dir / "test-lab-a" / "lab.yaml").write_text(
        "name: 'test-lab-a'\n"
        "archetype: product\n"
        "template_origin: demo_note_cli\n"
    )
    (labs_dir / "test-lab-a" / "cycles").mkdir()
    (labs_dir / "test-lab-a" / "cycles" / "001").mkdir()

    saved_home = os.environ.get("HOME")
    os.environ["HOME"] = str(fake_home)
    os.environ.pop("BERT_LAB_PATH", None)
    try:
        m = _reload_api_main()
        from fastapi.testclient import TestClient
        r = TestClient(m.app).get("/api/labs")
        body = r.json()
        names = [s["name"] for s in body["scaffolded"]]
        assert "test-lab-a" in names
        entry = next(s for s in body["scaffolded"] if s["name"] == "test-lab-a")
        assert entry["archetype"] == "product"
        assert entry["template_origin"] == "demo_note_cli"
        assert entry["cycle_count"] == 1
    finally:
        if saved_home is not None:
            os.environ["HOME"] = saved_home
        else:
            os.environ.pop("HOME", None)
        shutil.rmtree(fake_home)


def test_routed_api_serves_routed_events() -> None:
    """When BERT_LAB_PATH is set, /api/events reads from the routed
    lab's events.jsonl, not bert-lab's."""
    import json
    tmp = Path(tempfile.mkdtemp(prefix="bert_l4_routed_"))
    (tmp / "sor").mkdir()
    (tmp / "state").mkdir()
    (tmp / "sor" / "events.jsonl").write_text(
        json.dumps({
            "id": "routed-1",
            "ts": "2026-05-13T20:00:00+00:00",
            "event_class": "finding",
            "agent": "router-test",
            "cycle": 1,
            "content": "routed lab event",
        }) + "\n"
    )
    os.environ["BERT_LAB_PATH"] = str(tmp)
    try:
        m = _reload_api_main()
        from fastapi.testclient import TestClient
        r = TestClient(m.app).get("/api/events?limit=5")
        body = r.json()
        assert body["count"] == 1
        assert body["events"][0]["agent"] == "router-test"
    finally:
        os.environ.pop("BERT_LAB_PATH", None)
        shutil.rmtree(tmp)


def main() -> int:
    tests = [
        test_default_lab_path_is_bert_lab,
        test_env_override_changes_lab_path,
        test_labs_endpoint_shape,
        test_labs_endpoint_discovers_scaffolded,
        test_routed_api_serves_routed_events,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
