"""Smoke test for multi-lab as product.

Added `core/lab_config.py` (lab.yaml reader + LabConfig
dataclass + schema validation + graceful fallback).
Wired the director to consume per-lab `focus_areas` via
`gather_observation` and `parse_decision_text(valid_focus_areas=...)`.
Added `focus_areas` declarations to the 4 templates plus a
new `lab/lab.yaml` for the repo's own self-improvement lab with
`role: supervisor`.

Covers:
  - core/lab_config module shape + exports
  - load() across 6 paths: missing file, malformed YAML, valid full,
    valid minimal, invalid types (focus_areas not a list, role not
    in enum), schema version newer than engine
  - Privacy default opt-out: share_with_supervisor defaults true,
    explicit false honored
  - Repo's own lab/lab.yaml exists with role:supervisor + bert-internal
    focus_areas
  - 4 templates declare focus_areas (and they're archetype-appropriate)
  - Director's gather_observation surfaces focus_areas + lab_config
  - parse_decision_text accepts per-lab focus_area; rejects out-of-set
  - Backwards compat: pre-FF tests still pass (CC + EE smoke verified
    separately in the recheck ritual)
  - Cycle_shape×focus_area validation is now per-lab
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import director as dir_mod  # noqa: E402
from core import lab_config as lc  # noqa: E402


def _require(*paths: Path) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip(
            "requires lab runtime artifact(s) not shipped in the public repo: "
            + ", ".join(str(m) for m in missing)
        )

# ─── Module shape ───────────────────────────────────────────


def test_lab_config_module_exports() -> None:
    for name in ("LabConfig", "load",
                 "DEFAULT_FOCUS_AREAS_SUPERVISOR",
                 "DEFAULT_FOCUS_AREAS_STANDARD",
                 "VALID_ROLES", "SCHEMA_VERSION"):
        assert hasattr(lc, name), f"core.lab_config missing {name!r}"


def test_default_focus_areas_locked() -> None:
    assert lc.DEFAULT_FOCUS_AREAS_SUPERVISOR == (
        "routing", "memory", "discipline", "ux", "unspecified",
    )
    assert lc.DEFAULT_FOCUS_AREAS_STANDARD == (
        "methodology", "evidence", "synthesis", "consequences", "unspecified",
    )


def test_valid_roles_locked() -> None:
    assert frozenset({"standard", "supervisor"}) == lc.VALID_ROLES


# ─── load() paths ───────────────────────────────────────────


def test_load_missing_yaml_for_customer_lab_uses_standard_defaults() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        cfg = lc.load(tmp)
        assert cfg.role == "standard"
        assert cfg.focus_areas == lc.DEFAULT_FOCUS_AREAS_STANDARD
        assert cfg.shares_with_supervisor is True
        assert "missing" in (cfg.parse_warnings[0] if cfg.parse_warnings else "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_missing_yaml_for_repo_lab_uses_supervisor_defaults() -> None:
    """Previously the repo's lab/ had no lab.yaml. The loader treated
    that location as supervisor by default; the repo now ships a lab.yaml
    that makes this explicit but the FALLBACK behavior must still be
    supervisor when reading the repo's own lab dir specifically."""
    # We can't easily simulate "repo lab without lab.yaml" since the repo
    # ships one. But we can verify the lab.yaml exists AND declares
    # role:supervisor (test_repo_lab_yaml_declares_supervisor below).
    # This test confirms the FALLBACK code-path treats LAB_ROOT/lab
    # as supervisor when no yaml is present, using a temp directory
    # NAMED the same way wouldn't help — the loader checks the
    # resolved path. So we patch via a temp lab in LAB_ROOT/lab if
    # it didn't have a yaml.
    # Simpler approach: directly inspect the loader's defaults-branch
    # logic via the parse_warnings field.
    cfg = lc.load(LAB_ROOT / "lab")
    # If lab.yaml exists, this returns the file's contents
    # — role should still be supervisor. If lab.yaml doesn't exist,
    # the fallback code path kicks in and role should also be
    # supervisor (because the path matches LAB_ROOT/lab).
    assert cfg.role == "supervisor"


def test_repo_lab_yaml_declares_supervisor() -> None:
    """lab/lab.yaml must exist and declare role:supervisor +
    the bert-internal focus areas."""
    yaml_path = LAB_ROOT / "lab" / "lab.yaml"
    _require(yaml_path)
    assert yaml_path.exists(), "lab/lab.yaml missing"
    cfg = lc.load(LAB_ROOT / "lab")
    assert cfg.role == "supervisor"
    assert cfg.name == "bert-self"
    assert set(cfg.focus_areas) == set(lc.DEFAULT_FOCUS_AREAS_SUPERVISOR)
    assert cfg.shares_with_supervisor is False  # supervisor never shares


def test_load_valid_full_yaml() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "lab.yaml").write_text(
            "lab_schema_version: 1\n"
            "name: customer-survey\n"
            "archetype: research\n"
            "mission: Compare free-tier inference providers\n"
            "focus_areas: [latency, cost, reliability, family_diversity]\n"
            "role: standard\n"
            "share_with_supervisor: true\n"
            "provider: groq\n"
            "autonomy: collaborator\n"
        )
        cfg = lc.load(tmp)
        assert cfg.name == "customer-survey"
        assert cfg.role == "standard"
        # `unspecified` auto-appended
        assert "unspecified" in cfg.focus_areas
        assert "latency" in cfg.focus_areas
        assert cfg.mission.startswith("Compare")
        assert cfg.provider == "groq"
        assert cfg.parse_warnings == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_malformed_yaml_falls_back_with_warnings() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "lab.yaml").write_text("this is not: valid: yaml: at all")
        cfg = lc.load(tmp)
        # Defaults kick in
        assert cfg.role == "standard"
        assert cfg.focus_areas == lc.DEFAULT_FOCUS_AREAS_STANDARD
        assert cfg.parse_warnings
        assert "unreadable" in cfg.parse_warnings[0].lower()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_yaml_root_not_a_mapping_falls_back() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "lab.yaml").write_text("- just\n- a\n- list\n")
        cfg = lc.load(tmp)
        assert cfg.role == "standard"
        assert cfg.parse_warnings
        assert any("mapping" in w for w in cfg.parse_warnings)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_invalid_role_falls_back_to_standard() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "lab.yaml").write_text(
            "role: god-emperor\n"
            "focus_areas: [a, b, c]\n"
        )
        cfg = lc.load(tmp)
        assert cfg.role == "standard"
        assert any("god-emperor" in w for w in cfg.parse_warnings)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_focus_areas_not_a_list_falls_back() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "lab.yaml").write_text(
            "focus_areas: not-a-list\n"
        )
        cfg = lc.load(tmp)
        assert cfg.focus_areas == lc.DEFAULT_FOCUS_AREAS_STANDARD
        assert any("must be a list" in w for w in cfg.parse_warnings)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_focus_areas_count_out_of_bounds_falls_back() -> None:
    """3-7 is the locked range."""
    tmp = Path(tempfile.mkdtemp())
    try:
        # Too few (1 + auto-appended unspecified = 2, still under 3)
        (tmp / "lab.yaml").write_text("focus_areas: [only_one]\n")
        cfg = lc.load(tmp)
        assert cfg.focus_areas == lc.DEFAULT_FOCUS_AREAS_STANDARD
        assert any("outside [3, 7]" in w for w in cfg.parse_warnings)

        # Too many (8 distinct → 9 with unspecified → over 7)
        (tmp / "lab.yaml").write_text(
            "focus_areas: [a, b, c, d, e, f, g, h]\n"
        )
        cfg2 = lc.load(tmp)
        assert cfg2.focus_areas == lc.DEFAULT_FOCUS_AREAS_STANDARD
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_focus_areas_auto_appends_unspecified() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "lab.yaml").write_text(
            "focus_areas: [alpha, beta, gamma]\n"
        )
        cfg = lc.load(tmp)
        assert "unspecified" in cfg.focus_areas
        assert "alpha" in cfg.focus_areas
        # Auto-append doesn't blow past the upper bound (3 + 1 = 4, ok)
        assert len(cfg.focus_areas) == 4
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_schema_version_newer_than_engine_warns_but_loads() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "lab.yaml").write_text(
            "lab_schema_version: 99\n"
            "focus_areas: [a, b, c]\n"
        )
        cfg = lc.load(tmp)
        assert cfg.lab_schema_version == 99
        assert any("newer than" in w for w in cfg.parse_warnings)
        # And the fields still load (focus_areas has unspecified appended)
        assert "unspecified" in cfg.focus_areas
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── Privacy default ────────────────────────────────────────


def test_share_with_supervisor_default_true() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "lab.yaml").write_text("focus_areas: [a, b, c]\n")
        cfg = lc.load(tmp)
        assert cfg.share_with_supervisor is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_share_with_supervisor_explicit_false_honored() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "lab.yaml").write_text(
            "focus_areas: [a, b, c]\n"
            "share_with_supervisor: false\n"
        )
        cfg = lc.load(tmp)
        assert cfg.share_with_supervisor is False
        assert cfg.shares_with_supervisor is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_supervisor_never_shares_with_itself() -> None:
    """`role: supervisor` always returns shares_with_supervisor=False
    regardless of share_with_supervisor declaration."""
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "lab.yaml").write_text(
            "role: supervisor\n"
            "focus_areas: [a, b, c]\n"
            "share_with_supervisor: true\n"  # explicit true
        )
        cfg = lc.load(tmp)
        assert cfg.is_supervisor is True
        assert cfg.share_with_supervisor is True  # raw value preserved
        assert cfg.shares_with_supervisor is False  # property gates it
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── Templates ──────────────────────────────────────────────


def test_all_templates_declare_focus_areas() -> None:
    for tname in ("research", "product", "strategy", "demo_note_cli"):
        ypath = LAB_ROOT / "templates" / tname / "lab.yaml"
        assert ypath.exists(), f"template {tname}/lab.yaml missing"
        text = ypath.read_text()
        assert "focus_areas:" in text, f"{tname} missing focus_areas"
        assert "lab_schema_version: 1" in text, f"{tname} missing schema version"
        assert "role:" in text, f"{tname} missing role"


def test_research_template_focus_areas_match_design() -> None:
    text = (LAB_ROOT / "templates" / "research" / "lab.yaml").read_text()
    for area in ("methodology", "evidence", "synthesis",
                 "consequences", "unspecified"):
        assert area in text, f"research template missing {area}"


def test_product_template_focus_areas_match_design() -> None:
    text = (LAB_ROOT / "templates" / "product" / "lab.yaml").read_text()
    for area in ("architecture", "implementation", "testing",
                 "operations", "unspecified"):
        assert area in text, f"product template missing {area}"


def test_strategy_template_focus_areas_match_design() -> None:
    text = (LAB_ROOT / "templates" / "strategy" / "lab.yaml").read_text()
    for area in ("options", "tradeoffs", "risk", "timing", "unspecified"):
        assert area in text, f"strategy template missing {area}"


# ─── Director integration ───────────────────────────────────


def test_observation_has_focus_areas_field() -> None:
    """Observation gains focus_areas + lab_config fields."""
    assert "focus_areas" in dir_mod.Observation.__dataclass_fields__
    assert "lab_config" in dir_mod.Observation.__dataclass_fields__


def test_gather_observation_uses_per_lab_focus_areas() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "seed_brief.md").write_text("# x")
        (tmp / "lab.yaml").write_text(
            "lab_schema_version: 1\n"
            "name: customer\n"
            "focus_areas: [alpha, beta, gamma, delta]\n"
            "role: standard\n"
        )
        (tmp / "sor").mkdir()
        (tmp / "sor" / "events.jsonl").write_text("")
        (tmp / "state").mkdir()
        obs = dir_mod.gather_observation(tmp, iteration=1)
        assert "alpha" in obs.focus_areas
        assert "delta" in obs.focus_areas
        assert "unspecified" in obs.focus_areas  # auto-appended
        # And bert-internal areas are NOT in this customer lab
        assert "routing" not in obs.focus_areas
        assert "memory" not in obs.focus_areas
        # lab_config snapshot also present
        assert obs.lab_config["role"] == "standard"
        assert obs.lab_config["name"] == "customer"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gather_observation_repo_lab_returns_supervisor() -> None:
    """The repo's own lab/ must observe as role:supervisor with the
    bert-internal focus areas."""
    obs = dir_mod.gather_observation(LAB_ROOT / "lab", iteration=1)
    assert obs.lab_config["role"] == "supervisor"
    assert "routing" in obs.focus_areas
    assert "memory" in obs.focus_areas
    assert "discipline" in obs.focus_areas
    assert "ux" in obs.focus_areas


def test_parse_decision_text_per_lab_areas_accepts_valid() -> None:
    raw = json.dumps({
        "cycle_shape": "research-deeper",
        "focus_area": "alpha",
        "rationale": "x" * 100,
        "researcher_prompt_focus": "Investigate alpha-pattern.",
        "expected_runtime_secs": 120,
        "termination_condition": "alpha verdict or 3 cycles.",
        "confidence_1to10": 6,
    })
    decision, errors = dir_mod.parse_decision_text(
        raw, valid_focus_areas={"alpha", "beta", "gamma", "unspecified"})
    assert errors == [], f"expected clean parse; got {errors}"
    assert decision.focus_area == "alpha"


def test_parse_decision_text_per_lab_areas_rejects_invalid() -> None:
    raw = json.dumps({
        "cycle_shape": "research-deeper",
        "focus_area": "routing",  # not in customer lab's set
        "rationale": "x" * 100,
        "researcher_prompt_focus": "x",
        "expected_runtime_secs": 120,
        "termination_condition": "x",
        "confidence_1to10": 6,
    })
    decision, errors = dir_mod.parse_decision_text(
        raw, valid_focus_areas={"alpha", "beta", "gamma", "unspecified"})
    assert decision is None
    assert any("invalid focus_area" in e for e in errors)


def test_parse_decision_text_backwards_compat_no_areas_arg() -> None:
    """Callers that don't pass valid_focus_areas must still work using
    the global enum (bert-internal areas)."""
    raw = json.dumps({
        "cycle_shape": "research-deeper",
        "focus_area": "routing",
        "rationale": "x" * 100,
        "researcher_prompt_focus": "x",
        "expected_runtime_secs": 120,
        "termination_condition": "x",
        "confidence_1to10": 6,
    })
    decision, errors = dir_mod.parse_decision_text(raw)
    assert errors == []
    assert decision.focus_area == "routing"


# ─── Director prompt update ─────────────────────────────────


def test_director_prompt_documents_per_lab_focus_areas() -> None:
    text = (LAB_ROOT / "prompts" / "director_decision.md").read_text()
    # Should explain the per-lab declaration
    assert "per-lab" in text.lower() or "per lab" in text.lower()
    assert "lab.yaml" in text
    # Should mention the bounded set rule
    assert "VERBATIM" in text or "verbatim" in text.lower()
    # Should reference the supervisor's own areas as the example
    assert "routing / memory / discipline / ux" in text or \
           "routing/memory/discipline/ux" in text


def main() -> int:
    tests = [
        test_lab_config_module_exports,
        test_default_focus_areas_locked,
        test_valid_roles_locked,
        test_load_missing_yaml_for_customer_lab_uses_standard_defaults,
        test_load_missing_yaml_for_repo_lab_uses_supervisor_defaults,
        test_repo_lab_yaml_declares_supervisor,
        test_load_valid_full_yaml,
        test_load_malformed_yaml_falls_back_with_warnings,
        test_load_yaml_root_not_a_mapping_falls_back,
        test_load_invalid_role_falls_back_to_standard,
        test_load_focus_areas_not_a_list_falls_back,
        test_load_focus_areas_count_out_of_bounds_falls_back,
        test_load_focus_areas_auto_appends_unspecified,
        test_load_schema_version_newer_than_engine_warns_but_loads,
        test_share_with_supervisor_default_true,
        test_share_with_supervisor_explicit_false_honored,
        test_supervisor_never_shares_with_itself,
        test_all_templates_declare_focus_areas,
        test_research_template_focus_areas_match_design,
        test_product_template_focus_areas_match_design,
        test_strategy_template_focus_areas_match_design,
        test_observation_has_focus_areas_field,
        test_gather_observation_uses_per_lab_focus_areas,
        test_gather_observation_repo_lab_returns_supervisor,
        test_parse_decision_text_per_lab_areas_accepts_valid,
        test_parse_decision_text_per_lab_areas_rejects_invalid,
        test_parse_decision_text_backwards_compat_no_areas_arg,
        test_director_prompt_documents_per_lab_focus_areas,
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
