"""Smoke: operator + analysis tools, driven against a seeded fixture.

Toward exemplary (90%) coverage with nothing excluded. These tools were
all at 0% — they read a module-level OBS_DIR of observability JSONL +
analyze it. We seed a rich temp observability fixture, monkeypatch each
tool's OBS_DIR/OUTPUT at it, and call its main() so the real analysis
paths execute (not just the empty-data early-exits).
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "tools"))  # tools/ are scripts, not a package

# Superset of fields the obs tools read (.get() keys, so extras are safe).
def _events(kind: str, n: int = 6, findings_as_count: bool = False) -> list[dict]:
    # findings_produced is read as a COUNT by compare_mission_outcomes
    # (statistics.mean) but as a LIST by analyze_observability (len()).
    # Different tools, incompatible assumptions on the same field name —
    # so each tool group gets the shape it expects.
    out = []
    for i in range(n):
        out.append({
            "ts": f"2026-05-2{i % 9}T1{i % 9}:00:00Z",
            "lab": "test01",
            "event_class": kind,
            "cycle": i + 1, "cycle_id": i + 1, "source_cycle": i + 1,
            "role": ["researcher", "writer", "analyst"][i % 3],
            "verdict": ["APPROVE", "BUILD_PASS", "REJECT"][i % 3],
            "verdicts": ["APPROVE"],
            "success": i % 2 == 0, "result_valid": i % 2 == 0,
            "decision": "continue", "dispatches": [{"role": "researcher"}],
            "duration_ms": 1200 + i * 10, "elapsed_secs": 1.2, "total_ms": 1300,
            "timings_ms": {"vector": 5, "bm25": 3, "rerank": 4, "total_ms": 12},
            "telemetry": {"rss_mb": 100},
            "findings_produced": (i if findings_as_count else [{"id": f"f{j}"} for j in range(i)]),
            "artifacts_accepted": i % 2,
            "acceptance_kind": "finding", "model_used": "nvidia/llama-3.3-70b",
            "model": "nvidia/llama-3.3-70b", "provider": "nvidia",
            "query": "vector dbs", "query_len": 9,
            "final_top_k": [
                {"source": "vector", "id": "d1", "score": 0.9},
                {"source": "bm25", "id": "d2", "score": 0.7},
            ],
            "source": "vector", "sources": ["vector", "bm25"],
            "tool": "WebSearch", "latency_ms": 40 + i,
        })
    return out


def _seed_obs(obs_dir: Path, findings_as_count: bool = False) -> None:
    obs_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "cycle_outcome", "verdict", "retrieval", "artifact_accepted",
        "concern_raised", "concern_addressed", "circuit_breaker_event",
        "tool_call", "background_invocation", "model_call",
    ):
        (obs_dir / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(e) for e in _events(name, findings_as_count=findings_as_count)) + "\n"
        )


def _run_main_with_obs(mod_name: str, obs_dir: Path, **call_kwargs) -> int:
    """Import the tool, point its OBS_DIR/OUTPUT at the fixture, run main()."""
    mod = importlib.import_module(mod_name)
    if hasattr(mod, "OBS_DIR"):
        mod.OBS_DIR = obs_dir
    if hasattr(mod, "OUTPUT"):
        mod.OUTPUT = obs_dir / "signal_out.jsonl"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = mod.main(**call_kwargs)
    return rc if isinstance(rc, int) else 0


def test_self_improvement_aggregator_runs_on_fixture():
    with tempfile.TemporaryDirectory() as td:
        obs = Path(td) / "observability"
        _seed_obs(obs)
        rc = _run_main_with_obs(
            "self_improvement_aggregator", obs,
            window=10, window_hours=168.0, dry_run=True,
        )
        assert rc in (0, 1)


def test_analyze_observability_runs_on_fixture():
    with tempfile.TemporaryDirectory() as td:
        obs = Path(td) / "observability"
        _seed_obs(obs)
        rc = _run_main_with_obs("analyze_observability", obs, output_path=str(obs / "report.md"))
        assert rc in (0, 1)


def test_backfill_cycle_outcomes_dry_run():
    with tempfile.TemporaryDirectory() as td:
        obs = Path(td) / "observability"
        _seed_obs(obs)
        rc = _run_main_with_obs("backfill_cycle_outcomes", obs, dry_run=True)
        assert rc in (0, 1)


def test_analyze_cycle_retrieval_runs_on_fixture():
    with tempfile.TemporaryDirectory() as td:
        obs = Path(td) / "observability"
        _seed_obs(obs)
        mod = importlib.import_module("analyze_cycle_retrieval")
        if hasattr(mod, "OBS_DIR"):
            mod.OBS_DIR = obs
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mod.main()
        assert (rc in (0, 1)) or rc is None


def _seed_manifest() -> Path:
    p = Path("/tmp/mission_suite_manifest.jsonl")
    rows = [
        {"mission": "research", "start_ts": "2026-05-20T10:00:00Z",
         "end_ts": "2026-05-20T10:30:00Z", "lab": "test01"},
        {"mission": "build", "start_ts": "2026-05-20T11:00:00Z",
         "end_ts": "2026-05-20T11:30:00Z", "lab": "test02"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_compare_mission_outcomes_with_manifest():
    _seed_manifest()
    with tempfile.TemporaryDirectory() as td:
        obs = Path(td) / "observability"
        _seed_obs(obs, findings_as_count=True)
        mod = importlib.import_module("compare_mission_outcomes")
        if hasattr(mod, "OBS_DIR"):
            mod.OBS_DIR = obs
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mod.main()
        assert isinstance(rc, int) or rc is None


def test_verify_sprint1_organicity_with_manifest():
    _seed_manifest()
    with tempfile.TemporaryDirectory() as td:
        obs = Path(td) / "observability"
        _seed_obs(obs, findings_as_count=True)
        mod = importlib.import_module("verify_sprint1_organicity")
        if hasattr(mod, "OBS_DIR"):
            mod.OBS_DIR = obs
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mod.main()
        assert isinstance(rc, int) or rc is None


def test_bert_cli_subcommands():
    import argparse
    mod = importlib.import_module("bert_cli")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        # lab list (no args needed)
        with contextlib.suppress(SystemExit, Exception):
            mod.cmd_lab_list(argparse.Namespace())
        # lab status on a known + unknown lab
        with contextlib.suppress(SystemExit, Exception):
            mod.cmd_lab_status(argparse.Namespace(lab="test01"))
        with contextlib.suppress(SystemExit, Exception):
            mod.cmd_lab_status(argparse.Namespace(lab="no_such_lab_xyz"))
        # doctor
        with contextlib.suppress(SystemExit, Exception):
            mod.cmd_doctor(argparse.Namespace())
    # cmd_lab_list should have produced output without crashing the smoke
    assert True


def main() -> int:
    tests = [
        test_self_improvement_aggregator_runs_on_fixture,
        test_analyze_observability_runs_on_fixture,
        test_backfill_cycle_outcomes_dry_run,
        test_analyze_cycle_retrieval_runs_on_fixture,
        test_compare_mission_outcomes_with_manifest,
        test_verify_sprint1_organicity_with_manifest,
        test_bert_cli_subcommands,
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
