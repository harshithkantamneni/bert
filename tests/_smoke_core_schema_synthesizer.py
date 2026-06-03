"""Smoke: core/schema_synthesizer.py — profile→schema rule engine (was 59%).

Network-free (heuristic classifier). Covers _match_field (wildcard / list-
alternation / overlap / scalar-in-list / equality), _rule_matches, synthesize
+ _build_schema via real heuristic profiles, scaffold_knowledge_files
(idempotent), list_available_templates (+ roster filter), _parse_frontmatter
(valid / no-fm / unterminated), and the rules/templates/demo/usage/unknown CLI.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import shutil
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import mission_profile as mp  # noqa: E402
from core import schema_synthesizer as ss  # noqa: E402


def test_match_field():
    assert ss._match_field("*", "anything") is True
    assert ss._match_field(["a", "b"], "a") is True          # scalar in list
    assert ss._match_field(["a", "b"], "z") is False
    assert ss._match_field(["a", "b"], ("b", "c")) is True   # list ∩ tuple overlap
    assert ss._match_field("x", ("x", "y")) is True          # rule scalar in profile list
    assert ss._match_field("x", "x") is True and ss._match_field("x", "y") is False


def test_rule_matches():
    assert ss._rule_matches({"data_shape": "document_corpus"},
                            {"data_shape": "document_corpus"}) is True
    assert ss._rule_matches({"data_shape": "code_repo"},
                            {"data_shape": "document_corpus"}) is False


def test_synthesize_and_build():
    for mission in ("Survey vector DB papers and compare recall",
                    "Audit the repo for security issues"):
        profile = mp.classify_mission(mission, use_llm=False)
        schema = ss.synthesize(profile)
        assert schema.rule_id and isinstance(schema.roster_core, tuple)
        d = schema.to_dict()
        assert "rule_id" in d and "roster_core" in d


def test_scaffold_knowledge_files(tmp_path):
    profile = mp.classify_mission("Survey papers comparing methods", use_llm=False)
    schema = ss.synthesize(profile)
    created = ss.scaffold_knowledge_files(tmp_path, schema)
    assert (tmp_path / "knowledge").exists()
    assert isinstance(created, list)
    # idempotent: a second call creates nothing new
    again = ss.scaffold_knowledge_files(tmp_path, schema)
    assert again == []


def test_list_available_templates():
    templates = ss.list_available_templates()
    assert isinstance(templates, list)
    if templates:
        assert "template" in templates[0] and "path" in templates[0]
        name = templates[0]["template"]
        filtered = ss.list_available_templates(roster_filter=(name,))
        assert all(t["template"] == name for t in filtered)


def test_parse_frontmatter():
    assert ss._parse_frontmatter("---\ntemplate: researcher\ntier_default: free\n---\nbody")["template"] == "researcher"
    assert ss._parse_frontmatter("no frontmatter here") is None
    assert ss._parse_frontmatter("---\nunterminated frontmatter") is None


def test_cli():
    assert ss._cli(["x"]) == 2                          # usage
    with contextlib.redirect_stdout(io.StringIO()):
        assert ss._cli(["x", "rules"]) == 0
        assert ss._cli(["x", "templates"]) == 0
        assert ss._cli(["x", "demo", "Survey papers comparing methods"]) == 0
        assert ss._cli(["x", "demo"]) == 2              # demo sub-usage
    assert ss._cli(["x", "bogus"]) == 2                 # unknown


def main() -> int:
    tests = [
        test_match_field,
        test_rule_matches,
        test_synthesize_and_build,
        test_scaffold_knowledge_files,
        test_list_available_templates,
        test_parse_frontmatter,
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
