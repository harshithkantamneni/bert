"""Smoke test for N.2: per-request ?lab=<name> routing."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["BERT_DISABLE_IDLE_COMPUTE"] = "1"

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


def _reload_api_main():
    for mod in list(sys.modules.keys()):
        if mod.startswith("api.main") or mod == "api":
            del sys.modules[mod]
    import api.main as m
    return m


def _make_user_lab(home: Path, name: str, events: list[dict]) -> None:
    """Materialize a bert-lab-shape lab at home/.bert/labs/name/."""
    lab = home / ".bert" / "labs" / name
    (lab / "sor").mkdir(parents=True)
    (lab / "state").mkdir()
    (lab / "sor" / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + ("\n" if events else "")
    )


def test_default_lab_unchanged() -> None:
    """No ?lab= → behaves exactly like before."""
    os.environ.pop("BERT_LAB_PATH", None)
    m = _reload_api_main()
    from fastapi.testclient import TestClient
    r = TestClient(m.app).get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["lab"] == "(default)"


def test_lab_param_routes_to_user_lab() -> None:
    """?lab=foo reads from ~/.bert/labs/foo/."""
    fake_home = Path(tempfile.mkdtemp(prefix="bert_n2_"))
    _make_user_lab(fake_home, "demo-lab", [
        {"id": "e1", "ts": "2026-05-13T20:00:00+00:00",
         "event_class": "finding", "agent": "researcher", "cycle": 1,
         "content": "user-lab event 1"},
    ])
    saved_home = os.environ.get("HOME")
    os.environ["HOME"] = str(fake_home)
    os.environ.pop("BERT_LAB_PATH", None)
    try:
        m = _reload_api_main()
        from fastapi.testclient import TestClient
        client = TestClient(m.app)
        # Status should report events_total = 1
        r = client.get("/api/status?lab=demo-lab")
        body = r.json()
        assert body["ok"] is True
        assert body["lab"] == "demo-lab"
        assert body["events_total"] == 1
        # Events should return the lab's event, not bert-lab's
        r = client.get("/api/events?lab=demo-lab&limit=5")
        body = r.json()
        assert body["count"] == 1
        assert body["events"][0]["agent"] == "researcher"
        assert body["lab"] == "demo-lab"
    finally:
        if saved_home is not None:
            os.environ["HOME"] = saved_home
        else:
            os.environ.pop("HOME", None)
        shutil.rmtree(fake_home)


def test_lab_param_404_for_missing_lab() -> None:
    """?lab=nonexistent → 404."""
    fake_home = Path(tempfile.mkdtemp(prefix="bert_n2_404_"))
    saved_home = os.environ.get("HOME")
    os.environ["HOME"] = str(fake_home)
    os.environ.pop("BERT_LAB_PATH", None)
    try:
        m = _reload_api_main()
        from fastapi.testclient import TestClient
        r = TestClient(m.app).get("/api/status?lab=ghost-lab")
        assert r.status_code == 404
        assert "not found" in r.json()["detail"].lower()
    finally:
        if saved_home is not None:
            os.environ["HOME"] = saved_home
        else:
            os.environ.pop("HOME", None)
        shutil.rmtree(fake_home)


def test_agents_endpoint_routes_too() -> None:
    """/api/agents respects ?lab=."""
    fake_home = Path(tempfile.mkdtemp(prefix="bert_n2_agents_"))
    _make_user_lab(fake_home, "lab-a", [
        {"id": f"e{i}", "ts": "2026-05-13T20:00:00+00:00",
         "event_class": "finding", "agent": "lab-a-researcher",
         "cycle": 1, "content": "x"}
        for i in range(3)
    ])
    saved_home = os.environ.get("HOME")
    os.environ["HOME"] = str(fake_home)
    os.environ.pop("BERT_LAB_PATH", None)
    try:
        m = _reload_api_main()
        from fastapi.testclient import TestClient
        r = TestClient(m.app).get("/api/agents?lab=lab-a")
        body = r.json()
        assert body["lab"] == "lab-a"
        agent_names = [a["agent"] for a in body["agents"]]
        assert "lab-a-researcher" in agent_names
    finally:
        if saved_home is not None:
            os.environ["HOME"] = saved_home
        else:
            os.environ.pop("HOME", None)
        shutil.rmtree(fake_home)


def test_findings_endpoint_routes_too() -> None:
    """/api/findings respects ?lab=."""
    fake_home = Path(tempfile.mkdtemp(prefix="bert_n2_findings_"))
    _make_user_lab(fake_home, "lab-b", [
        {"id": "f1", "ts": "2026-05-13T20:00:00+00:00",
         "event_class": "finding", "agent": "x", "cycle": 1,
         "content": "This is a real editorial finding sentence."},
    ])
    saved_home = os.environ.get("HOME")
    os.environ["HOME"] = str(fake_home)
    os.environ.pop("BERT_LAB_PATH", None)
    try:
        m = _reload_api_main()
        from fastapi.testclient import TestClient
        r = TestClient(m.app).get("/api/findings?lab=lab-b")
        body = r.json()
        assert body["lab"] == "lab-b"
    finally:
        if saved_home is not None:
            os.environ["HOME"] = saved_home
        else:
            os.environ.pop("HOME", None)
        shutil.rmtree(fake_home)


def test_bert_init_creates_sor_state_dirs() -> None:
    """bert init scaffolds sor/events.jsonl + state/ so labs are routable."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bert_init", LAB_ROOT / "tools" / "bert_init.py",
    )
    bert_init = importlib.util.module_from_spec(spec)
    sys.modules["bert_init"] = bert_init
    spec.loader.exec_module(bert_init)

    fake_home = Path(tempfile.mkdtemp(prefix="bert_n2_init_"))
    bert_init.HOME_BERT = fake_home
    bert_init.LABS_DIR = fake_home / "labs"
    try:
        answers = {
            "archetype": "Product", "name": "scaffolded-test",
            "provider": "Groq", "autonomy": "Collaborator",
            "seed": "verify scaffolded sor/state dirs",
        }
        lab_dir = bert_init._scaffold_lab(answers, from_template="demo_note_cli")
        assert (lab_dir / "sor" / "events.jsonl").exists()
        assert (lab_dir / "state").is_dir()
    finally:
        shutil.rmtree(fake_home)


def main() -> int:
    tests = [
        test_default_lab_unchanged,
        test_lab_param_routes_to_user_lab,
        test_lab_param_404_for_missing_lab,
        test_agents_endpoint_routes_too,
        test_findings_endpoint_routes_too,
        test_bert_init_creates_sor_state_dirs,
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
