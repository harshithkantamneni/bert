"""Smoke test for the router + SKILL.md loading."""

import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import router, skills  # noqa: E402

# ── Router tests ────────────────────────────────────────────────────


def test_route_meta_to_nvidia_for_reasoning() -> None:
    assert router._heuristic_select("META", role="threshing_pass") == "nvidia"


def test_route_vision_to_ollama() -> None:
    assert router._heuristic_select("IMPL", role="vision") == "ollama"


def test_route_impl_to_groq() -> None:
    assert router._heuristic_select("IMPL", role="implementer") == "groq"


def test_route_nit_to_ollama() -> None:
    assert router._heuristic_select("NIT-cleanup", role=None) == "ollama"


def test_anthropic_openai_out_of_scope() -> None:
    assert not router.is_in_scope("anthropic")
    assert not router.is_in_scope("openai")
    assert router.is_in_scope("nvidia")
    assert router.is_in_scope("cerebras")


def test_select_first_attempt_works_without_routellm() -> None:
    """When RouteLLM not installed, falls back to heuristic."""
    chosen = router.select_first_attempt_provider(
        prompt="test prompt", altitude="META", role="threshing_pass",
    )
    assert chosen in router.SMART_ROUTABLE_PROVIDERS


# ── Skills tests ────────────────────────────────────────────────────


SAMPLE_SKILL_MD = """---
name: test_skill
description: A test skill that does some thing for the validation tests.
---

# Test skill body

This is the body content of the skill.
"""


def test_parse_frontmatter() -> None:
    fm, body = skills._parse_frontmatter(SAMPLE_SKILL_MD)
    assert fm["name"] == "test_skill"
    assert "validation tests" in fm["description"]
    assert "Test skill body" in body


def test_load_skill_from_dir() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        skill_dir = Path(tmp) / "test_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(SAMPLE_SKILL_MD)
        loaded = skills.load_skill(skill_dir)
        assert loaded is not None
        assert loaded.name == "test_skill"
        valid, errors = skills.validate_skill(loaded)
        assert valid, f"validation failed: {errors}"


def test_skill_missing_description_invalid() -> None:
    """Skill without description in frontmatter rejected."""
    with tempfile.TemporaryDirectory() as tmp:
        skill_dir = Path(tmp) / "bad_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: bad_skill\n---\n\nbody only"
        )
        loaded = skills.load_skill(skill_dir)
        assert loaded is None  # no description → load_skill returns None


def test_skill_invalid_name_rejected() -> None:
    """Names with capitals/spaces rejected."""
    skill = skills.Skill(
        name="Bad Name",
        description="x" * 30,
        path=Path("/tmp"),
        frontmatter={},
        body="content",
    )
    valid, errors = skills.validate_skill(skill)
    assert not valid


def main() -> int:
    tests = [
        test_route_meta_to_nvidia_for_reasoning,
        test_route_vision_to_ollama,
        test_route_impl_to_groq,
        test_route_nit_to_ollama,
        test_anthropic_openai_out_of_scope,
        test_select_first_attempt_works_without_routellm,
        test_parse_frontmatter,
        test_load_skill_from_dir,
        test_skill_missing_description_invalid,
        test_skill_invalid_name_rejected,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
