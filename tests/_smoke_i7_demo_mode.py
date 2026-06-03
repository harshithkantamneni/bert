"""Smoke test for I.7: BERT_DEMO_MODE toggle."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["BERT_DISABLE_IDLE_COMPUTE"] = "1"

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_demo_mode_endpoint_default_off() -> None:
    saved = os.environ.pop("BERT_DEMO_MODE", None)
    flag = LAB_ROOT / "lab" / "state" / "demo_mode.on"
    flag_existed = flag.exists()
    if flag_existed:
        flag.unlink()
    try:
        r = client.get("/api/demo-mode")
        assert r.status_code == 200
        body = r.json()
        assert body["enabled"] is False
        assert "policy" in body
        assert "hide_surfaces" in body["policy"]
    finally:
        if saved is not None:
            os.environ["BERT_DEMO_MODE"] = saved
        if flag_existed:
            flag.touch()


def test_demo_mode_env_var_enables() -> None:
    saved = os.environ.get("BERT_DEMO_MODE")
    try:
        for val in ("on", "1", "true", "yes"):
            os.environ["BERT_DEMO_MODE"] = val
            r = client.get("/api/demo-mode")
            assert r.json()["enabled"] is True, f"BERT_DEMO_MODE={val} should enable"
    finally:
        if saved is not None:
            os.environ["BERT_DEMO_MODE"] = saved
        else:
            os.environ.pop("BERT_DEMO_MODE", None)


def test_demo_mode_flag_file_enables() -> None:
    saved = os.environ.pop("BERT_DEMO_MODE", None)
    flag = LAB_ROOT / "lab" / "state" / "demo_mode.on"
    flag.parent.mkdir(parents=True, exist_ok=True)
    try:
        flag.touch()
        r = client.get("/api/demo-mode")
        assert r.json()["enabled"] is True
    finally:
        if flag.exists():
            flag.unlink()
        if saved is not None:
            os.environ["BERT_DEMO_MODE"] = saved


def test_demo_mode_policy_completeness() -> None:
    """The policy dict must include all 6 documented keys."""
    r = client.get("/api/demo-mode")
    pol = r.json()["policy"]
    for k in ("hide_surfaces", "hide_routes", "default_surface",
              "provider_chain", "auto_load_lab",
              "toast_suppress_patterns"):
        assert k in pol, f"policy missing {k!r}"


def test_demo_mode_hide_surfaces_matches_memory() -> None:
    """Per project_bert_demo_mode_and_polish.md, hidden surfaces are
    DevGestures, KeyboardHelp, Choreography, Infrastructure."""
    r = client.get("/api/demo-mode")
    hide = r.json()["policy"]["hide_surfaces"]
    assert set(hide) >= {
        "DevGestures", "KeyboardHelp", "Choreography", "Infrastructure",
    }


def test_demo_mode_provider_chain_groq_first() -> None:
    """Failover chain must start with Groq (fastest free-tier)."""
    r = client.get("/api/demo-mode")
    chain = r.json()["policy"]["provider_chain"]
    assert chain[0] == "groq", f"expected groq first, got {chain[0]}"
    assert "ollama" in chain, "ollama must be in chain as ultimate fallback"


def test_demo_mode_invalid_env_values_remain_off() -> None:
    """BERT_DEMO_MODE=off / 0 / random / empty should NOT enable."""
    saved = os.environ.get("BERT_DEMO_MODE")
    try:
        for val in ("off", "0", "false", "no", "", "random"):
            os.environ["BERT_DEMO_MODE"] = val
            r = client.get("/api/demo-mode")
            assert r.json()["enabled"] is False, (
                f"BERT_DEMO_MODE={val!r} should NOT enable"
            )
    finally:
        if saved is not None:
            os.environ["BERT_DEMO_MODE"] = saved
        else:
            os.environ.pop("BERT_DEMO_MODE", None)


def main() -> int:
    tests = [
        test_demo_mode_endpoint_default_off,
        test_demo_mode_env_var_enables,
        test_demo_mode_flag_file_enables,
        test_demo_mode_policy_completeness,
        test_demo_mode_hide_surfaces_matches_memory,
        test_demo_mode_provider_chain_groq_first,
        test_demo_mode_invalid_env_values_remain_off,
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
