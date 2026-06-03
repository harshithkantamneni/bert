"""Smoke + TDD: host-first routing (the MCP-first pivot's core contract).

When a host (Claude Code / Cursor with Opus) is present, the host IS the
product's intelligence — it should run EVERY reasoning role, mapped by cost
tier (A->opus, B->sonnet, C->haiku). BYO keys and free-tier only engage when
there is no host.

The bug: resolve_model_for_dispatch only reached Tier-1 (host) when a role's
curated model-card list happened to name a claude model. writer (cards=[opus])
got opus; literature_hunter (cards=[gemini]) silently dropped to BYO gemini,
strategist (cards=[o3]) dropped to free-tier. So most roles never touched the
host — exactly the "why is it using nvidia/gemini" the user flagged.
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import router  # noqa: E402

HOST_MODELS = ["claude-haiku-4-5", "claude-opus-4-6", "claude-opus-4-7",
               "claude-sonnet-4-5", "claude-sonnet-4-6"]


class _Ctx:
    def __init__(self, models):
        self.tier1_models_available = list(models)


def test_host_model_for_tier_maps_by_cost():
    f = router._host_model_for_tier
    s = set(HOST_MODELS)
    assert f("A", s) == "claude-opus-4-7"     # heavy -> best opus
    assert f("B", s) == "claude-sonnet-4-6"   # medium -> best sonnet
    assert f("C", s) == "claude-haiku-4-5"    # light -> haiku
    # Unknown tier defaults to the B (sonnet) lane.
    assert f("Z", s) == "claude-sonnet-4-6"
    # No host models -> None (caller falls through to BYO/free).
    assert f("A", set()) is None


def test_byo_only_role_routes_to_host_when_present():
    # literature_hunter's top card is gemini (BYO), NOT a host model.
    # With a host present it must STILL run on the host (tier B -> sonnet).
    prov, model = router.resolve_model_for_dispatch(
        "literature_hunter",
        task_text="investigate the literature",
        host_ctx=_Ctx(HOST_MODELS),
        byo_keys={"GOOGLE_AI_API_KEY"},
    )
    assert prov == "anthropic-cli", (prov, model)
    assert "sonnet" in model, (prov, model)


def test_tier_a_role_routes_to_host_opus():
    prov, model = router.resolve_model_for_dispatch(
        "writer", task_text="write the report",
        host_ctx=_Ctx(HOST_MODELS), byo_keys=set(),
    )
    assert prov == "anthropic-cli" and "opus" in model, (prov, model)


def test_no_host_falls_through_to_byo():
    # Host absent (headless / cron) -> BYO gemini should win for lit_hunter.
    prov, model = router.resolve_model_for_dispatch(
        "literature_hunter",
        task_text="investigate the literature",
        host_ctx=_Ctx([]),
        byo_keys={"GOOGLE_AI_API_KEY"},
    )
    assert prov == "gemini", (prov, model)


def test_no_host_no_byo_falls_through_to_free():
    prov, model = router.resolve_model_for_dispatch(
        "strategist", task_text="plan the work",
        host_ctx=_Ctx([]), byo_keys=set(),
    )
    assert prov not in ("anthropic-cli",), (prov, model)  # free-tier matrix


def main() -> int:
    tests = [
        test_host_model_for_tier_maps_by_cost,
        test_byo_only_role_routes_to_host_when_present,
        test_tier_a_role_routes_to_host_opus,
        test_no_host_falls_through_to_byo,
        test_no_host_no_byo_falls_through_to_free,
    ]
    for t in tests:
        try:
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
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
