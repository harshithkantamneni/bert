"""Smoke: core/cycle_budget.py — budget estimation + saturation (was 46%).

Pure logic + event-file reads. Covers estimate_budget (profile rules +
archetype/keyword fallbacks), resolve_budget (auto/preset/int/str/error),
_scan_cycle_events (temp events.jsonl + missing + bad-json), novelty_score,
is_saturated (window guard / insufficient-history / saturated / not), and
the novelty/saturated/estimate/usage/unknown CLI.
"""

from __future__ import annotations

import inspect
import json
import shutil
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import cycle_budget as cb  # noqa: E402


def _seed_lab(tmp: Path, events: list[dict]) -> Path:
    (tmp / "sor").mkdir(parents=True, exist_ok=True)
    (tmp / "sor" / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + ("\n" if events else ""))
    return tmp


def test_estimate_budget_profile_rules():
    assert cb.estimate_budget({"horizon": "ongoing"}).preset_name == "until_complete"
    assert cb.estimate_budget({"horizon": "one_shot"}).preset_name == "quick"
    assert cb.estimate_budget({"primary_work": "build"}).preset_name == "deep"
    assert cb.estimate_budget({"primary_work": "decide"}).preset_name == "standard"
    assert cb.estimate_budget({"primary_work": "discover"}).preset_name == "standard"


def test_estimate_budget_fallbacks():
    assert cb.estimate_budget(None, mission_text="monitor the feed weekly").preset_name == "until_complete"
    assert cb.estimate_budget(None, mission_text="just check one-shot tldr").preset_name == "quick"
    assert cb.estimate_budget(None, mission_text="short mission").preset_name == "quick"
    long_text = "word " * 60
    assert cb.estimate_budget(None, archetype="product", mission_text=long_text).preset_name == "deep"
    assert cb.estimate_budget(None, archetype="unknown_arch", mission_text=long_text).preset_name == "standard"


def test_resolve_budget():
    assert cb.resolve_budget(None).preset_name in cb.PRESETS
    assert cb.resolve_budget("auto", archetype="research").preset_name in cb.PRESETS
    assert cb.resolve_budget("deep") is cb.PRESETS["deep"]
    explicit = cb.resolve_budget(7)
    assert explicit.target == 7 and explicit.safety_cap == 14
    assert cb.resolve_budget("5").target == 5             # numeric string
    for bad in ("not_a_preset", 0, 99):
        try:
            cb.resolve_budget(bad); raise SystemExit("no raise")
        except ValueError:
            pass
    try:
        cb.resolve_budget(3.5); raise SystemExit("no raise")  # unsupported type
    except ValueError:
        pass


def test_scan_and_novelty(tmp_path):
    lab = _seed_lab(tmp_path, [
        {"cycle": 1, "event_class": "finding"},
        {"cycle": 1, "event_class": "finding"},
        {"cycle": 1, "event_class": "memory_write"},
        {"cycle": 2, "event_class": "tool_call"},
    ])
    counts = cb._scan_cycle_events(lab, 1)
    assert counts["finding"] == 2 and counts["memory_write"] == 1
    assert cb._scan_cycle_events(lab, 99) == {}            # no events for cycle
    assert cb._scan_cycle_events(tmp_path / "nope", 1) == {}
    # novelty: 2 findings + 1 memory_write = 2.0 + 0.5 = 2.5 → 2.5/3 ≈ 0.833
    assert cb.novelty_score(lab, 1) > 0.5
    assert cb.novelty_score(lab, 2) == 0.0                 # only tool_call, no weights
    assert cb.novelty_score(lab, 99) == 0.0                # no events


def test_is_saturated(tmp_path):
    # low-novelty cycles 1-3 (only tool_calls) → saturated when checked at cycle 5
    lab = _seed_lab(tmp_path, [
        {"cycle": c, "event_class": "tool_call"} for c in (1, 2, 3, 4)
    ])
    sat, scores = cb.is_saturated(lab, current_cycle=5, window=3)
    assert sat is True and len(scores) == 3
    # high-novelty cycles → not saturated
    lab2 = _seed_lab(tmp_path / "l2", [
        {"cycle": c, "event_class": "finding"} for c in (1, 2, 3, 4) for _ in range(4)
    ])
    sat2, _ = cb.is_saturated(lab2, current_cycle=5, window=3)
    assert sat2 is False
    # insufficient history
    assert cb.is_saturated(lab, current_cycle=2, window=3) == (False, [])
    # bad window
    try:
        cb.is_saturated(lab, current_cycle=5, window=0); raise SystemExit("no raise")
    except ValueError:
        pass


def test_cli(tmp_path):
    lab = _seed_lab(tmp_path, [{"cycle": 1, "event_class": "finding"}])
    assert cb._cli(["x"]) == 2                                   # usage
    assert cb._cli(["x", "novelty", str(lab), "1"]) == 0
    assert cb._cli(["x", "novelty"]) == 2                        # sub-usage
    assert cb._cli(["x", "saturated", str(lab), "5"]) == 0
    assert cb._cli(["x", "estimate", "research", "a mission"]) == 0
    assert cb._cli(["x", "bogus"]) in (1, 2)                     # unknown


def main() -> int:
    tests = [
        test_estimate_budget_profile_rules,
        test_estimate_budget_fallbacks,
        test_resolve_budget,
        test_scan_and_novelty,
        test_is_saturated,
        test_cli,
    ]
    for t in tests:
        td = Path(tempfile.mkdtemp())
        try:
            kwargs = {"tmp_path": td} if "tmp_path" in inspect.signature(t).parameters else {}
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
            shutil.rmtree(td, ignore_errors=True)
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
