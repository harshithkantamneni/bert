"""Smoke + TDD: model deprecation aliases — graceful transition (launch #32).

#39 (deprecation warnings 7+ days out) was already met; #32 (graceful remap via
aliases) was unmet — when a model passed its deprecation_date the router fell
back to next-best instead of the declared successor. This adds a `deprecated_to`
successor on ModelCard and an alias resolver the router consults so a deprecated
model transparently remaps to its successor (chained, cycle-guarded).
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import model_cards as mc  # noqa: E402

_ON = dt.date(2026, 5, 29)


def _card(cid, *, deprecation_date=None, deprecated_to=None, access=None):
    return mc.ModelCard(
        id=cid, provider="groq", family="llama", generation="3",
        context_window=8000, output_token_max=2000, pricing_per_million_usd={},
        strengths=(), weaknesses=(), best_for_roles=(), best_for_skills=(),
        avoid_for_roles=(), last_validated="2026-01-01",
        access=access or {}, deprecation_date=deprecation_date,
        deprecated_to=deprecated_to)


def test_is_deprecated():
    assert mc.is_deprecated(_card("a", deprecation_date="2026-01-01"), on=_ON) is True
    assert mc.is_deprecated(_card("b", deprecation_date="2027-01-01"), on=_ON) is False
    assert mc.is_deprecated(_card("c"), on=_ON) is False


def test_resolve_active_model_follows_alias(monkeypatch):
    cards = {
        "old": _card("old", deprecation_date="2026-01-01", deprecated_to="mid"),
        "mid": _card("mid", deprecation_date="2026-02-01", deprecated_to="new"),
        "new": _card("new", deprecation_date="2027-01-01"),
    }
    monkeypatch.setattr(mc, "find_by_id", lambda i: cards.get(i))
    # chained: old -> mid -> new (both old and mid are past their dates)
    assert mc.resolve_active_model("old", on=_ON) == "new"
    assert mc.resolve_active_model("new", on=_ON) == "new"      # not deprecated
    assert mc.resolve_active_model("unknown", on=_ON) == "unknown"


def test_resolve_active_model_cycle_guard(monkeypatch):
    cards = {
        "x": _card("x", deprecation_date="2026-01-01", deprecated_to="y"),
        "y": _card("y", deprecation_date="2026-01-01", deprecated_to="x"),
    }
    monkeypatch.setattr(mc, "find_by_id", lambda i: cards.get(i))
    # must terminate (cycle guard), not recurse forever
    out = mc.resolve_active_model("x", on=_ON)
    assert out in ("x", "y")


def test_resolve_deprecated_without_successor_keeps_id(monkeypatch):
    cards = {"d": _card("d", deprecation_date="2026-01-01")}  # no deprecated_to
    monkeypatch.setattr(mc, "find_by_id", lambda i: cards.get(i))
    assert mc.resolve_active_model("d", on=_ON) == "d"


def test_remap_deprecated_replaces_in_candidate_list(monkeypatch):
    old = _card("old", deprecation_date="2026-01-01", deprecated_to="new")
    new = _card("new", deprecation_date="2027-01-01")
    keep = _card("keep")
    monkeypatch.setattr(mc, "find_by_id", lambda i: {"old": old, "new": new, "keep": keep}.get(i))
    out = mc.remap_deprecated([old, keep], on=_ON)
    ids = [c.id for c in out]
    assert "old" not in ids and "new" in ids and "keep" in ids


def test_remap_dedupes_when_successor_already_present(monkeypatch):
    old = _card("old", deprecation_date="2026-01-01", deprecated_to="new")
    new = _card("new", deprecation_date="2027-01-01")
    monkeypatch.setattr(mc, "find_by_id", lambda i: {"old": old, "new": new}.get(i))
    out = mc.remap_deprecated([old, new], on=_ON)
    assert [c.id for c in out] == ["new"]  # not duplicated


def test_yaml_parses_deprecated_to_field():
    # The loader must accept the new field without breaking the registry.
    cards = mc.load_all(force_reload=True)
    assert len(cards) > 0
    # every card has the attribute (default None)
    assert all(hasattr(c, "deprecated_to") for c in cards)


# ── router consults the remap ────────────────────────────────────────


def test_router_remaps_deprecated_to_successor(monkeypatch):
    from core import router
    old = _card("old-model", deprecation_date="2026-01-01", deprecated_to="new-model",
                access={"via_host": ["claude_code"]})
    new = _card("new-model", deprecation_date="2027-01-01",
                access={"via_host": ["claude_code"]})
    monkeypatch.setattr(mc, "cards_for_role", lambda role: [old])
    monkeypatch.setattr(mc, "cards_for_skill", lambda s: [])
    monkeypatch.setattr(mc, "find_by_id", lambda i: {"old-model": old, "new-model": new}.get(i))

    class _Host:
        tier1_models_available = ["new-model", "old-model"]

    prov, model = router.resolve_model_for_dispatch(
        "writer", task_text="write a brief", host_ctx=_Host(), byo_keys=set())
    assert model == "new-model"  # deprecated old-model remapped to successor


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
        test_is_deprecated,
        test_resolve_active_model_follows_alias,
        test_resolve_active_model_cycle_guard,
        test_resolve_deprecated_without_successor_keeps_id,
        test_remap_deprecated_replaces_in_candidate_list,
        test_remap_dedupes_when_successor_already_present,
        test_yaml_parses_deprecated_to_field,
        test_router_remaps_deprecated_to_successor,
    ]
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
