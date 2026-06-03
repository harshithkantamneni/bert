"""Smoke test for I.6: note-cli demo lab + 3 templates."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

# Load bert_init module
_spec = importlib.util.spec_from_file_location(
    "bert_init", LAB_ROOT / "tools" / "bert_init.py",
)
bert_init = importlib.util.module_from_spec(_spec)
sys.modules["bert_init"] = bert_init
_spec.loader.exec_module(bert_init)


def test_three_templates_exist() -> None:
    templates_root = LAB_ROOT / "templates"
    for name in ("product", "research", "strategy"):
        d = templates_root / name
        assert d.exists(), f"templates/{name}/ missing"
        assert (d / "lab.yaml").exists(), f"templates/{name}/lab.yaml missing"
        assert (d / "README.md").exists()
        assert (d / "EXPECTED_FIRST_CYCLE.md").exists()


def test_demo_note_cli_template_exists() -> None:
    d = LAB_ROOT / "templates" / "demo_note_cli"
    assert d.exists()
    assert (d / "lab.yaml").exists()
    assert (d / "seed_brief.md").exists()
    assert (d / "cycles" / "001" / "plan.md").exists()
    assert (d / "cycles" / "001" / "code" / "note.py").exists()
    assert (d / "cycles" / "001" / "tests" / "test_capture.py").exists()
    assert (d / "cycles" / "001" / "journal.md").exists()


def test_demo_lab_cycle1_tests_pass() -> None:
    """The seeded cycle 1 tests must actually run and pass."""
    test_path = (LAB_ROOT / "templates" / "demo_note_cli" /
                 "cycles" / "001" / "tests" / "test_capture.py")
    result = subprocess.run(
        [sys.executable, str(test_path)],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, (
        f"cycle 1 tests failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "All 5 tests passed" in result.stdout


def test_scaffold_from_template_copies_content() -> None:
    """`bert init --from-template demo_note_cli` copies seed content."""
    tmp = Path(tempfile.mkdtemp(prefix="bert_i6_"))
    saved_labs = bert_init.LABS_DIR
    saved_home = bert_init.HOME_BERT
    try:
        bert_init.HOME_BERT = tmp
        bert_init.LABS_DIR = tmp / "labs"
        answers = {
            "archetype": "Product", "name": "test-demo",
            "provider": "Groq", "autonomy": "Collaborator",
            "seed": "seed text long enough for validation",
        }
        lab_dir = bert_init._scaffold_lab(answers, from_template="demo_note_cli")
        assert (lab_dir / "lab.yaml").exists()
        assert (lab_dir / "seed_brief.md").exists()
        assert (lab_dir / "cycles" / "001" / "code" / "note.py").exists()
        # lab.yaml records template origin
        yaml_text = (lab_dir / "lab.yaml").read_text()
        assert "template_origin: demo_note_cli" in yaml_text
    finally:
        bert_init.LABS_DIR = saved_labs
        bert_init.HOME_BERT = saved_home
        shutil.rmtree(tmp)


def test_scaffold_from_each_archetype_template() -> None:
    """Product / Research / Strategy templates all scaffold cleanly."""
    tmp = Path(tempfile.mkdtemp(prefix="bert_i6_arch_"))
    saved_labs = bert_init.LABS_DIR
    saved_home = bert_init.HOME_BERT
    try:
        bert_init.HOME_BERT = tmp
        bert_init.LABS_DIR = tmp / "labs"
        for arch in ("product", "research", "strategy"):
            answers = {
                "archetype": arch.title(), "name": f"test-{arch}",
                "provider": "Groq", "autonomy": "Collaborator",
                "seed": f"seed for {arch} archetype lab — needs ≥10 chars",
            }
            lab_dir = bert_init._scaffold_lab(answers, from_template=arch)
            assert lab_dir.exists()
            assert (lab_dir / "lab.yaml").exists()
            assert (lab_dir / "README.md").exists()
            assert (lab_dir / "EXPECTED_FIRST_CYCLE.md").exists()
    finally:
        bert_init.LABS_DIR = saved_labs
        bert_init.HOME_BERT = saved_home
        shutil.rmtree(tmp)



def test_scaffold_excludes_pycache_from_template() -> None:
    """L.1 — `bert init --from-template` must not copy __pycache__/."""
    template_pycache = (
        LAB_ROOT / "templates" / "demo_note_cli"
        / "cycles" / "001" / "code" / "__pycache__"
    )
    tmp = Path(tempfile.mkdtemp(prefix="bert_l1_"))
    saved_labs = bert_init.LABS_DIR
    saved_home = bert_init.HOME_BERT
    try:
        bert_init.HOME_BERT = tmp
        bert_init.LABS_DIR = tmp / "labs"
        template_pycache.mkdir(parents=True, exist_ok=True)
        (template_pycache / "test.cpython-313.pyc").write_bytes(b"fake")
        answers = {
            "archetype": "Product", "name": "pycache-test",
            "provider": "Groq", "autonomy": "Collaborator",
            "seed": "verify pycache exclusion — needs >= 10 chars",
        }
        lab_dir = bert_init._scaffold_lab(answers, from_template="demo_note_cli")
        for child in lab_dir.rglob("*"):
            assert "__pycache__" not in child.parts, (
                f"__pycache__ leaked into scaffolded lab at {child}"
            )
            assert not child.name.endswith(".pyc"), (
                f".pyc leaked at {child}"
            )
    finally:
        if template_pycache.exists():
            shutil.rmtree(template_pycache)
        bert_init.LABS_DIR = saved_labs
        bert_init.HOME_BERT = saved_home
        shutil.rmtree(tmp)


def test_unknown_template_raises() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bert_i6_unk_"))
    saved_labs = bert_init.LABS_DIR
    try:
        bert_init.LABS_DIR = tmp / "labs"
        answers = {
            "archetype": "Product", "name": "test",
            "provider": "Groq", "autonomy": "Collaborator",
            "seed": "ten-character seed text minimum",
        }
        try:
            bert_init._scaffold_lab(answers, from_template="bogus_archetype")
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "unknown template" in str(e)
    finally:
        bert_init.LABS_DIR = saved_labs
        shutil.rmtree(tmp)


def test_scaffold_without_template_minimal() -> None:
    """No --from-template → just lab.yaml + .bert/answers.yml."""
    tmp = Path(tempfile.mkdtemp(prefix="bert_i6_min_"))
    saved_labs = bert_init.LABS_DIR
    saved_home = bert_init.HOME_BERT
    try:
        bert_init.HOME_BERT = tmp
        bert_init.LABS_DIR = tmp / "labs"
        answers = {
            "archetype": "Product", "name": "minimal",
            "provider": "Groq", "autonomy": "Collaborator",
            "seed": "minimal seed text — at least ten chars",
        }
        lab_dir = bert_init._scaffold_lab(answers)
        assert (lab_dir / "lab.yaml").exists()
        assert (lab_dir / ".bert" / "answers.yml").exists()
        # No template content copied
        assert not (lab_dir / "cycles").exists()
        # W.3 — scaffold now always writes a minimal seed_brief.md so
        # bert_run has the mission to read. Was previously asserted
        # absent; new behavior intentional.
        assert (lab_dir / "seed_brief.md").exists(), \
            "W.3: scaffold should always produce a seed_brief.md"
        content = (lab_dir / "seed_brief.md").read_text()
        assert "Mission" in content
        assert answers["seed"] in content
    finally:
        bert_init.LABS_DIR = saved_labs
        bert_init.HOME_BERT = saved_home
        shutil.rmtree(tmp)


def main() -> int:
    tests = [
        test_three_templates_exist,
        test_demo_note_cli_template_exists,
        test_demo_lab_cycle1_tests_pass,
        test_scaffold_from_template_copies_content,
        test_scaffold_from_each_archetype_template,
        test_scaffold_excludes_pycache_from_template,
        test_unknown_template_raises,
        test_scaffold_without_template_minimal,
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
