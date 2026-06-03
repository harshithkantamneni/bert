"""Smoke test for core/sandbox_policy.py + tools/prune_zero_invocation_skills.py (F.8)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import sandbox_policy  # noqa: E402


def _write_skill(d: Path, body: str) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / "SKILL.md"
    p.write_text(body)
    return p


def test_parse_frontmatter_simple() -> None:
    tmp = Path(tempfile.mkdtemp()) / "skill"
    p = _write_skill(tmp, "---\nname: test\nneeds_network: true\ntimeout_secs: 60\n---\n\n# body\n")
    fm = sandbox_policy.parse_frontmatter(p)
    assert fm["name"] == "test"
    assert fm["needs_network"] is True
    assert fm["timeout_secs"] == 60


def test_parse_frontmatter_list_value() -> None:
    tmp = Path(tempfile.mkdtemp()) / "skill"
    p = _write_skill(tmp,
                     "---\nneeds_read_paths:\n  - /etc\n  - /usr/local/var/ollama\nneeds_write_paths:\n  - /tmp/work\n---\n\n# body\n")
    fm = sandbox_policy.parse_frontmatter(p)
    assert fm["needs_read_paths"] == ["/etc", "/usr/local/var/ollama"]
    assert fm["needs_write_paths"] == ["/tmp/work"]


def test_parse_frontmatter_missing_returns_empty() -> None:
    tmp = Path(tempfile.mkdtemp()) / "skill"
    p = _write_skill(tmp, "# no frontmatter\n")
    assert sandbox_policy.parse_frontmatter(p) == {}


def test_build_policy_translates_frontmatter() -> None:
    tmp = Path(tempfile.mkdtemp()) / "skill"
    p = _write_skill(tmp,
                     "---\nneeds_network: true\nneeds_read_paths:\n  - /etc\ntimeout_secs: 45\n---\n")
    pol = sandbox_policy.build_policy(p)
    assert pol["allow_network"] is True
    assert pol["allow_read_paths"] == ["/etc"]
    assert pol["timeout_secs"] == 45


def test_build_policy_ollama_flag_expands_paths() -> None:
    tmp = Path(tempfile.mkdtemp()) / "skill"
    p = _write_skill(tmp, "---\nneeds_ollama: true\n---\n")
    pol = sandbox_policy.build_policy(p)
    paths = pol["allow_read_paths"]
    assert any("/ollama" in path for path in paths)


def test_explain_prose_summary() -> None:
    tmp = Path(tempfile.mkdtemp()) / "skill"
    p = _write_skill(tmp,
                     "---\nneeds_network: false\nneeds_read_paths:\n  - /etc\nneeds_subprocess: true\n---\n")
    text = sandbox_policy.explain(p)
    assert "denies network" in text
    assert "1 explicit path" in text
    assert "subprocess" in text


# ── prune_zero_invocation_skills ────────────────────────────────────


def test_prune_dry_run_doesnt_move_files() -> None:
    import tools.prune_zero_invocation_skills as prune_mod
    base = Path(tempfile.mkdtemp(prefix="bert_prune_"))
    events = base / "lab" / "sor" / "events.jsonl"
    active = base / "skills" / "active"
    archived = base / "skills" / "archived"
    events.parent.mkdir(parents=True, exist_ok=True)
    active.mkdir(parents=True, exist_ok=True)
    # Write a skill that has 0 invocations
    skill_dir = active / "test-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: test-tool\n---\n")
    # Seed events from cycles 1..40 (none reference test-tool)
    events.write_text("\n".join(
        json.dumps({"event_class": "tool_call", "cycle": c,
                    "tool_name": "Read"})
        for c in range(1, 41)
    ))
    prune_mod.EVENTS_PATH = events
    prune_mod.ACTIVE_DIR = active
    prune_mod.ARCHIVED_DIR = archived
    summary = prune_mod.prune(cycles=30, dry_run=True)
    assert summary["pruned_count"] == 1
    assert summary["pruned"][0]["name"] == "test-tool"
    assert skill_dir.exists()  # not moved
    assert not archived.exists()


def test_prune_actually_moves() -> None:
    import tools.prune_zero_invocation_skills as prune_mod
    base = Path(tempfile.mkdtemp(prefix="bert_prune_"))
    events = base / "lab" / "sor" / "events.jsonl"
    active = base / "skills" / "active"
    archived = base / "skills" / "archived"
    events.parent.mkdir(parents=True, exist_ok=True)
    active.mkdir(parents=True, exist_ok=True)
    skill_dir = active / "stale"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: stale-tool\n---\n")
    events.write_text("\n".join(
        json.dumps({"event_class": "tool_call", "cycle": c,
                    "tool_name": "Read"})
        for c in range(1, 41)
    ))
    prune_mod.EVENTS_PATH = events
    prune_mod.ACTIVE_DIR = active
    prune_mod.ARCHIVED_DIR = archived
    summary = prune_mod.prune(cycles=30, dry_run=False)
    assert summary["pruned_count"] == 1
    assert not skill_dir.exists()
    assert (archived / "stale").exists()


def test_prune_keeps_active_skill() -> None:
    import tools.prune_zero_invocation_skills as prune_mod
    base = Path(tempfile.mkdtemp(prefix="bert_prune_"))
    events = base / "lab" / "sor" / "events.jsonl"
    active = base / "skills" / "active"
    events.parent.mkdir(parents=True, exist_ok=True)
    active.mkdir(parents=True, exist_ok=True)
    skill_dir = active / "active-tool"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: active-tool\n---\n")
    # 5 invocations of active-tool
    events.write_text("\n".join(
        json.dumps({"event_class": "tool_call", "cycle": 40,
                    "tool_name": "active-tool"})
        for _ in range(5)
    ))
    prune_mod.EVENTS_PATH = events
    prune_mod.ACTIVE_DIR = active
    prune_mod.ARCHIVED_DIR = base / "skills" / "archived"
    summary = prune_mod.prune(cycles=30, dry_run=False)
    assert summary["pruned_count"] == 0
    assert summary["kept_count"] == 1


def main() -> int:
    tests = [
        test_parse_frontmatter_simple,
        test_parse_frontmatter_list_value,
        test_parse_frontmatter_missing_returns_empty,
        test_build_policy_translates_frontmatter,
        test_build_policy_ollama_flag_expands_paths,
        test_explain_prose_summary,
        test_prune_dry_run_doesnt_move_files,
        test_prune_actually_moves,
        test_prune_keeps_active_skill,
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
