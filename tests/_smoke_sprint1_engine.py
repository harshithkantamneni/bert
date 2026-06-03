"""Smoke: Sprint 1 verify_engine + Model Intelligence, end-to-end.

Lives in the _smoke_* namespace so the production gauntlet + the 22-stage
industry eval + the coverage gate EXERCISE these modules (verify_engine,
host_detector, model_cards, role_registry, router multi-source resolver).
Before this they were at 0% gauntlet coverage — driven only by
tests/test_*.py, which the gauntlet/eval do not run.

Drives real behavior: run the Python-native verifier against real
artifacts (pass + each failure path), detect the host context, load the
real model-card registry, parse role templates, and resolve a dispatch
model through the host>BYO>free tiers.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import host_detector, model_cards, role_registry, router, verify_engine  # noqa: E402

# A well-formed research finding: H1 + ≥3 H2 + citations + >1500 chars.
_VALID = (
    "# Vector Databases Q2 2026\n\n"
    "## Summary\n\nA substantive overview of the landscape with real findings "
    "and named systems, well over the minimum length floor. " + ("Detail. " * 240) + "\n\n"
    "## Top signals\n\n1. Signal one, see https://arxiv.org/abs/2026.00001 — concrete.\n"
    "2. Signal two, see https://example-vendor.io/docs — concrete.\n"
    "3. Signal three, paper arXiv:2026.00002 — concrete.\n\n"
    "## Gaps\n\nWhat was not covered and why, honestly stated.\n"
)


def _artifact(text: str) -> Path:
    fd, path = tempfile.mkstemp(suffix=".md")
    Path(path).write_text(text)
    return Path(path)


def test_verify_engine_passes_valid_artifact():
    result = verify_engine.verify_artifact(verify_engine.DEFAULT_SPEC, _artifact(_VALID))
    assert result.ok, f"expected pass; failed: {result.checks_failed}"
    assert result.exit_code == 0
    assert len(result.checks_passed) >= 5


def test_verify_engine_fails_empty_and_missing_h1():
    empty = verify_engine.verify_artifact(verify_engine.DEFAULT_SPEC, _artifact(""))
    assert not empty.ok
    no_h1 = verify_engine.verify_artifact(
        verify_engine.DEFAULT_SPEC, _artifact("body without heading. " * 200))
    assert not no_h1.ok
    assert any("H1" in c for c in no_h1.checks_failed)


def test_verify_engine_fails_missing_citations():
    content = "# Title\n\n## A\n\n## B\n\n## C\n\n" + ("prose with no source. " * 80)
    r = verify_engine.verify_artifact(verify_engine.DEFAULT_SPEC, _artifact(content))
    assert not r.ok
    assert any("citation" in c.lower() for c in r.checks_failed)


def test_host_detector_never_raises_and_summarizes():
    ctx = host_detector.detect()
    assert ctx is not None
    assert ctx.host_name in {"claude-code", "cursor", "codex", "standalone"}
    lines = host_detector.summarize(ctx)
    assert isinstance(lines, list) and lines


def test_model_cards_load_and_integrity():
    cards = model_cards.load_all()
    assert len(cards) >= 16, f"expected ≥16 model cards, got {len(cards)}"
    for c in cards:
        assert c.context_window > 0, f"{c.id}: invalid context_window"
    # role-targeted + free-tier slices are non-empty
    assert model_cards.cards_for_role("writer")
    assert model_cards.cards_via_free_tier()


def test_role_registry_templates_and_tier():
    tmpls = role_registry.all_templates(force_reload=True)
    assert len(tmpls) >= 8, f"expected ≥8 role templates, got {len(tmpls)}"
    writer = role_registry.load("writer")
    assert writer is not None
    assert role_registry.get_tier("writer") == writer.tier_default
    assert role_registry.load("definitely_not_a_role_xyz") is None


def test_router_resolves_a_dispatch_model():
    provider, model = router.resolve_model_for_dispatch(role="researcher")
    assert provider and model, f"resolver returned empty: {(provider, model)}"
    # task-text path also resolves
    p2, m2 = router.resolve_model_for_dispatch(
        role="researcher", task_text="survey papers and compare")
    assert p2 and m2


def main() -> int:
    tests = [
        test_verify_engine_passes_valid_artifact,
        test_verify_engine_fails_empty_and_missing_h1,
        test_verify_engine_fails_missing_citations,
        test_host_detector_never_raises_and_summarizes,
        test_model_cards_load_and_integrity,
        test_role_registry_templates_and_tier,
        test_router_resolves_a_dispatch_model,
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
