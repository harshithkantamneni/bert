"""Smoke test for Track B evals/ scaffolding.

Verifies:
  - evals/ directory exists with README.md
  - 3 sample eval files exist and have valid EVAL_SPEC dicts
  - README documents both Inspect AI + deepeval frameworks
"""

import importlib.util
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent


def test_evals_dir_exists() -> None:
    evals = LAB_ROOT / "evals"
    assert evals.is_dir()
    assert (evals / "README.md").exists()


def test_readme_documents_both_frameworks() -> None:
    readme = (LAB_ROOT / "evals" / "README.md").read_text()
    assert "Inspect AI" in readme
    assert "deepeval" in readme
    assert "A6 §9" in readme
    assert "P-VS-12" in readme  # OTel cross-reference
    assert "reward-hacking" in readme.lower()  # UC Berkeley caveat per L-14


def test_three_sample_evals_load() -> None:
    """Each sample eval module loads + has EVAL_SPEC dict."""
    for fname in ("p_vs_06_threshing", "p_vs_07_clearness_phase1",
                  "cache_token_reduction"):
        path = LAB_ROOT / "evals" / f"{fname}.py"
        assert path.exists()
        spec = importlib.util.spec_from_file_location(fname, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "EVAL_SPEC"), f"{fname} missing EVAL_SPEC"
        eval_spec = mod.EVAL_SPEC
        assert "name" in eval_spec
        assert "description" in eval_spec
        # At least one of falsifiers / falsifier
        assert "falsifiers" in eval_spec or "falsifier" in eval_spec


def test_falsifier_ids_match_a6_pattern() -> None:
    """A6 §9 falsifier IDs follow FALS-A6-9-{N} pattern."""
    import importlib.util
    for fname in ("p_vs_06_threshing", "p_vs_07_clearness_phase1"):
        path = LAB_ROOT / "evals" / f"{fname}.py"
        spec = importlib.util.spec_from_file_location(fname, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for f in mod.EVAL_SPEC.get("falsifiers", []):
            assert f["id"].startswith("FALS-A6-9-"), (
                f"{fname} falsifier id {f['id']} doesn't match pattern"
            )


def main() -> int:
    tests = [
        test_evals_dir_exists,
        test_readme_documents_both_frameworks,
        test_three_sample_evals_load,
        test_falsifier_ids_match_a6_pattern,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
