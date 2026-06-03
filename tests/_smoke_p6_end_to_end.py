"""End-to-end integration smoke (P.6): the full demo flow as one test.

Asserts the chain bert init → /api/labs → /api/status?lab=<name> →
proof packet build → bert verify all hang together. Catches coordination
bugs between L.4/N.2/N.3/N.4/I.2/I.4 that unit tests miss.

This test ONLY uses the existing bert-lab proof packets (no new model
calls). It's a structural integration check.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["BERT_DISABLE_IDLE_COMPUTE"] = "1"

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


def _load_bert_init():
    spec = importlib.util.spec_from_file_location(
        "bert_init", LAB_ROOT / "tools" / "bert_init.py",
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["bert_init"] = m
    spec.loader.exec_module(m)
    return m


def _reload_api_main():
    for mod in list(sys.modules.keys()):
        if mod.startswith("api.main") or mod == "api":
            del sys.modules[mod]
    import api.main as m
    return m


def test_full_demo_flow_integration() -> None:
    """The hero scenario:
    1. `bert init --from-template demo_note_cli` scaffolds ~/.bert/labs/X/
    2. Lab has sor/events.jsonl + state/ subdirs (from L.1+N.2 fix)
    3. /api/labs discovers the scaffolded lab
    4. /api/status?lab=X returns 200 with the lab name echoed
    5. /api/events?lab=X reads the scaffolded lab's events (empty initially)
    6. /.well-known/agent.json?lab=X returns per-lab metadata
    7. bert verify on a known-good packet PASSes (round-trip works)
    """
    fake_home = Path(tempfile.mkdtemp(prefix="bert_p6_"))
    saved_home = os.environ.get("HOME")
    os.environ["HOME"] = str(fake_home)
    os.environ.pop("BERT_LAB_PATH", None)

    try:
        # 1+2. Scaffold a lab via the real bert init helper
        bert_init = _load_bert_init()
        bert_init.HOME_BERT = fake_home / ".bert"
        bert_init.LABS_DIR = fake_home / ".bert" / "labs"
        answers = {
            "archetype": "Product",
            "name": "e2e-lab",
            "provider": "Groq",
            "autonomy": "Collaborator",
            "seed": "p6 end-to-end integration smoke",
        }
        lab_dir = bert_init._scaffold_lab(answers, from_template="demo_note_cli")
        assert lab_dir.exists()
        # 2. Required subdirs created (N.2 fix)
        assert (lab_dir / "sor" / "events.jsonl").exists()
        assert (lab_dir / "state").is_dir()
        # cycle 1 content present
        assert (lab_dir / "cycles" / "001" / "code" / "note.py").exists()
        # No __pycache__ leaked (L.1 fix)
        for p in lab_dir.rglob("*"):
            assert "__pycache__" not in p.parts, f"pycache leak at {p}"

        # 3-6. API endpoints route to the scaffolded lab
        from fastapi.testclient import TestClient
        m = _reload_api_main()
        client = TestClient(m.app)

        # 3. /api/labs discovers it
        labs = client.get("/api/labs").json()
        names = [s["name"] for s in labs["scaffolded"]]
        assert "e2e-lab" in names, (
            f"scaffolded lab not discovered; got {names}"
        )
        e2e_entry = next(s for s in labs["scaffolded"] if s["name"] == "e2e-lab")
        assert e2e_entry["archetype"] == "product"
        assert e2e_entry["template_origin"] == "demo_note_cli"

        # 4. /api/status?lab=e2e-lab routes correctly
        r = client.get("/api/status?lab=e2e-lab")
        assert r.status_code == 200
        body = r.json()
        assert body["lab"] == "e2e-lab"
        # Empty events.jsonl → events_total=0, last_event_ts=None
        assert body["events_total"] == 0

        # 5. /api/events?lab=e2e-lab returns empty list (no events yet)
        r = client.get("/api/events?lab=e2e-lab")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 0
        assert body["lab"] == "e2e-lab"

        # 6. Agent card reflects the scaffolded lab
        r = client.get("/.well-known/agent.json?lab=e2e-lab")
        assert r.status_code == 200
        card = r.json()
        assert "e2e-lab" in card["name"]
        assert card["lab"] == "e2e-lab"
        assert "product" in card["description"]

        # 7. bert verify round-trips on an existing packet
        # Use the canonical cycle-0400 packet from the bert-lab repo
        result = subprocess.run(
            [".venv/bin/python", "tools/bert_verify.py",
             "findings/proof_packets/cycle-0400.tar.gz", "--no-color"],
            cwd=LAB_ROOT, capture_output=True, text=True, timeout=30,
        )
        # PASS or PASS-WITH-WARNINGS = exit 0 or 1, never 2 (FAIL)
        assert result.returncode in (0, 1), (
            f"bert verify shouldn't FAIL on canonical packet; "
            f"got exit={result.returncode}\n{result.stdout}"
        )
        assert "cycle-0400" in result.stdout

    finally:
        if saved_home is not None:
            os.environ["HOME"] = saved_home
        else:
            os.environ.pop("HOME", None)
        shutil.rmtree(fake_home)


def test_404_on_missing_lab_across_endpoints() -> None:
    """Verify the integration is consistent: a non-existent lab returns
    404 (not silently fall through to default) on all routed endpoints."""
    fake_home = Path(tempfile.mkdtemp(prefix="bert_p6_404_"))
    saved_home = os.environ.get("HOME")
    os.environ["HOME"] = str(fake_home)
    os.environ.pop("BERT_LAB_PATH", None)

    try:
        m = _reload_api_main()
        from fastapi.testclient import TestClient
        client = TestClient(m.app)

        for path in (
            "/api/status?lab=phantom",
            "/api/events?lab=phantom",
            "/api/agents?lab=phantom",
            "/api/findings?lab=phantom",
            "/.well-known/agent.json?lab=phantom",
        ):
            r = client.get(path)
            assert r.status_code == 404, (
                f"{path} should return 404 for missing lab; got {r.status_code}"
            )
    finally:
        if saved_home is not None:
            os.environ["HOME"] = saved_home
        else:
            os.environ.pop("HOME", None)
        shutil.rmtree(fake_home)


def test_default_lab_unchanged_under_routing_pressure() -> None:
    """Sanity: even with multiple scaffolded labs around, hitting the
    endpoints WITHOUT ?lab= still reads from bert-lab's own data."""
    fake_home = Path(tempfile.mkdtemp(prefix="bert_p6_def_"))
    saved_home = os.environ.get("HOME")
    os.environ["HOME"] = str(fake_home)
    os.environ.pop("BERT_LAB_PATH", None)

    try:
        bert_init = _load_bert_init()
        bert_init.HOME_BERT = fake_home / ".bert"
        bert_init.LABS_DIR = fake_home / ".bert" / "labs"
        for n in ("lab-a", "lab-b", "lab-c"):
            bert_init._scaffold_lab({
                "archetype": "Product", "name": n, "provider": "Groq",
                "autonomy": "Collaborator", "seed": f"seed for {n}",
            }, from_template="demo_note_cli")

        m = _reload_api_main()
        from fastapi.testclient import TestClient
        client = TestClient(m.app)
        # No ?lab= → default
        r = client.get("/api/status")
        body = r.json()
        assert body["lab"] == "(default)"
        # bert-lab's own events.jsonl is non-empty; this should reflect that
        assert body.get("events_total", 0) > 0, (
            "default lab should still report bert-lab's own events"
        )
        # All 3 scaffolded labs discoverable
        labs = client.get("/api/labs").json()
        names = {s["name"] for s in labs["scaffolded"]}
        assert {"lab-a", "lab-b", "lab-c"}.issubset(names)
    finally:
        if saved_home is not None:
            os.environ["HOME"] = saved_home
        else:
            os.environ.pop("HOME", None)
        shutil.rmtree(fake_home)


def main() -> int:
    tests = [
        test_full_demo_flow_integration,
        test_404_on_missing_lab_across_endpoints,
        test_default_lab_unchanged_under_routing_pressure,
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
