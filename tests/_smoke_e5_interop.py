"""Smoke test for Playwright MCP + A2A Agent Card.

Tests:
  1. state/mcp_servers.json contains playwright + fetch entries
  2. mcp_installer.load_spec returns expected playwright config
  3. /.well-known/agent.json (A2A Agent Card) returns valid shape
     when bert is running (FastAPI test client)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import mcp_installer  # noqa: E402

REGISTRY_PATH = LAB_ROOT / "state" / "mcp_servers.json"


def _require(*paths: Path) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip(
            "requires lab runtime artifact(s) not shipped in the public repo: "
            + ", ".join(str(m) for m in missing)
        )


def test_playwright_registered_in_default_registry() -> None:
    """state/mcp_servers.json should contain a playwright entry."""
    _require(REGISTRY_PATH)
    configured = mcp_installer.list_configured()
    assert "playwright" in configured, f"missing playwright; got {configured}"
    spec = mcp_installer.load_spec("playwright")
    assert spec is not None
    assert spec.command == "npx"
    assert spec.args == ["-y", "@playwright/mcp@latest"]
    assert "browser automation" in spec.description.lower() or "playwright" in spec.description.lower()


def test_fetch_registered_in_default_registry() -> None:
    """state/mcp_servers.json should contain a fetch entry."""
    _require(REGISTRY_PATH)
    configured = mcp_installer.list_configured()
    assert "fetch" in configured
    spec = mcp_installer.load_spec("fetch")
    assert spec is not None
    assert spec.command == "uvx"


def test_agent_card_endpoint_returns_valid_shape() -> None:
    """A2A Agent Card endpoint exists and returns the required fields."""
    pytest.importorskip(
        "api.main",
        reason="requires the FastAPI api/ surface not shipped in the public repo",
    )
    from api.main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    r = client.get("/.well-known/agent.json")
    assert r.status_code == 200
    card = r.json()
    # A2A required fields per a2a-protocol.org spec
    assert "name" in card
    assert "url" in card
    assert "version" in card
    assert "capabilities" in card
    assert "skills" in card
    assert isinstance(card["skills"], list)
    # bert-specific assertion
    assert card["name"] == "bert-lab"
    assert card["capabilities"]["streaming"] is True


def test_agent_card_exposes_registered_mcp_servers_as_skills() -> None:
    """Skills array should mirror what's in state/mcp_servers.json."""
    pytest.importorskip(
        "api.main",
        reason="requires the FastAPI api/ surface not shipped in the public repo",
    )
    _require(REGISTRY_PATH)
    from api.main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    card = client.get("/.well-known/agent.json").json()
    skill_ids = {s["id"] for s in card["skills"]}
    configured = set(mcp_installer.list_configured())
    assert skill_ids == configured


def main() -> int:
    tests = [
        test_playwright_registered_in_default_registry,
        test_fetch_registered_in_default_registry,
        test_agent_card_endpoint_returns_valid_shape,
        test_agent_card_exposes_registered_mcp_servers_as_skills,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
