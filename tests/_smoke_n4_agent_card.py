"""Smoke test for N.4: agent card reflects active lab."""

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


def test_default_agent_card_unchanged() -> None:
    os.environ.pop("BERT_LAB_PATH", None)
    m = _reload_api_main()
    from fastapi.testclient import TestClient
    r = TestClient(m.app).get("/.well-known/agent.json")
    assert r.status_code == 200
    card = r.json()
    assert card["name"] == "bert-lab"
    # No 'lab' key when no ?lab= override
    assert "lab" not in card


def test_routed_agent_card_reflects_scaffolded_lab() -> None:
    fake_home = Path(tempfile.mkdtemp(prefix="bert_n4_"))
    lab = fake_home / ".bert" / "labs" / "investor-demo"
    lab.mkdir(parents=True)
    (lab / "lab.yaml").write_text(
        "name: 'investor-demo'\n"
        "archetype: product\n"
        "template_origin: demo_note_cli\n"
    )
    (lab / "cycles").mkdir()
    (lab / "cycles" / "001").mkdir()
    (lab / "cycles" / "002").mkdir()
    saved_home = os.environ.get("HOME")
    os.environ["HOME"] = str(fake_home)
    os.environ.pop("BERT_LAB_PATH", None)
    try:
        m = _reload_api_main()
        from fastapi.testclient import TestClient
        r = TestClient(m.app).get("/.well-known/agent.json?lab=investor-demo")
        assert r.status_code == 200
        card = r.json()
        assert "investor-demo" in card["name"]
        assert "product" in card["description"]
        assert "demo_note_cli" in card["description"]
        assert "2 cycles" in card["description"]
        assert card["lab"] == "investor-demo"
    finally:
        if saved_home is not None:
            os.environ["HOME"] = saved_home
        else:
            os.environ.pop("HOME", None)
        shutil.rmtree(fake_home)


def test_routed_agent_card_404_for_missing_lab() -> None:
    fake_home = Path(tempfile.mkdtemp(prefix="bert_n4_404_"))
    saved_home = os.environ.get("HOME")
    os.environ["HOME"] = str(fake_home)
    os.environ.pop("BERT_LAB_PATH", None)
    try:
        m = _reload_api_main()
        from fastapi.testclient import TestClient
        r = TestClient(m.app).get("/.well-known/agent.json?lab=ghost")
        assert r.status_code == 404
    finally:
        if saved_home is not None:
            os.environ["HOME"] = saved_home
        else:
            os.environ.pop("HOME", None)
        shutil.rmtree(fake_home)


def main() -> int:
    tests = [
        test_default_agent_card_unchanged,
        test_routed_agent_card_reflects_scaffolded_lab,
        test_routed_agent_card_404_for_missing_lab,
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
