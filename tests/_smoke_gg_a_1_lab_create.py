"""Smoke test for tools/bert_init._scaffold_lab focus-area handling.

`_scaffold_lab` must write a valid lab.yaml: archetype focus-area defaults when
none are given, user-provided focus areas when they are, and the product
archetype's own defaults. (The lab-creation UI/API that originally motivated
this is not part of this repo.)
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


def test_scaffold_lab_writes_focus_areas() -> None:
    from tools import bert_init
    tmp = Path(tempfile.mkdtemp())
    original = bert_init.LABS_DIR
    bert_init.LABS_DIR = tmp
    try:
        ld = bert_init._scaffold_lab(
            {"name": "Test Lab", "archetype": "Research",
             "provider": "groq", "autonomy": "Collaborator",
             "seed": "x" * 50, "focus_areas": None},
            user_provided_seed=True,
        )
        yaml_text = (ld / "lab.yaml").read_text()
        assert "focus_areas:" in yaml_text
        assert "methodology" in yaml_text  # research default
        assert "lab_schema_version: 1" in yaml_text
        assert "role: standard" in yaml_text
        assert "share_with_supervisor: true" in yaml_text
    finally:
        bert_init.LABS_DIR = original
        shutil.rmtree(tmp, ignore_errors=True)


def test_scaffold_lab_honors_user_focus_areas() -> None:
    from tools import bert_init
    tmp = Path(tempfile.mkdtemp())
    original = bert_init.LABS_DIR
    bert_init.LABS_DIR = tmp
    try:
        ld = bert_init._scaffold_lab(
            {"name": "Custom Areas", "archetype": "Research",
             "provider": "groq", "autonomy": "Pilot",
             "seed": "x" * 50,
             "focus_areas": ["latency", "cost", "reliability"]},
            user_provided_seed=True,
        )
        yaml_text = (ld / "lab.yaml").read_text()
        assert "latency" in yaml_text
        assert "cost" in yaml_text
        assert "methodology" not in yaml_text  # archetype default shouldn't win
    finally:
        bert_init.LABS_DIR = original
        shutil.rmtree(tmp, ignore_errors=True)


def test_scaffold_lab_product_archetype_defaults() -> None:
    from tools import bert_init
    tmp = Path(tempfile.mkdtemp())
    original = bert_init.LABS_DIR
    bert_init.LABS_DIR = tmp
    try:
        ld = bert_init._scaffold_lab(
            {"name": "Product Lab", "archetype": "Product",
             "provider": "groq", "autonomy": "Collaborator",
             "seed": "x" * 50, "focus_areas": None},
            user_provided_seed=True,
        )
        yaml_text = (ld / "lab.yaml").read_text()
        assert "architecture" in yaml_text
        assert "implementation" in yaml_text
        assert "testing" in yaml_text
    finally:
        bert_init.LABS_DIR = original
        shutil.rmtree(tmp, ignore_errors=True)
