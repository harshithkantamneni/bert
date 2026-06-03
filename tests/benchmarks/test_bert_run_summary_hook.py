"""TDD: bert_run.run() must dump its per-cycle/per-dispatch summary (incl.
telemetry) to BERT_RUN_SUMMARY_PATH when that env var is set. The B7 benchmark's
arm C runs entirely through the claude -p bridge, whose real token/latency/cost
telemetry is in the dispatch summary -- the model_call.jsonl rows for bridge
dispatches are all-zero placeholders. This opt-in JSON dump gives the benchmark
clean, per-run, run-isolated telemetry without scraping pretty-printed stdout.
Default (env unset) behavior is unchanged."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import tools.bert_run as br  # noqa: E402


def test_dump_run_summary_writes_when_path_set(tmp_path):
    out = tmp_path / "run_summary.json"
    summaries = [
        {"cycle": 7, "success": True, "elapsed_secs": 250.0,
         "dispatches": [
             {"role": "researcher", "verdict": "APPROVE", "result_valid": True,
              "telemetry": {"tokens_in": 1200, "tokens_out": 800,
                            "latency_secs": 130.0, "provider": "anthropic-cli",
                            "model_used": "claude-opus-4-7", "cost_usd": 0.4,
                            "retry_count": 0}},
             {"role": "strategist", "verdict": "APPROVE", "result_valid": True,
              "telemetry": {"tokens_in": 1500, "tokens_out": 900,
                            "latency_secs": 140.0, "provider": "anthropic-cli",
                            "model_used": "claude-opus-4-7", "cost_usd": 0.5,
                            "retry_count": 1}},
         ]},
    ]
    br._dump_run_summary(summaries, str(out), lab_name="test01", wall_secs=300.0)
    assert out.exists()
    obj = json.loads(out.read_text())
    assert obj["lab"] == "test01"
    assert obj["wall_secs"] == 300.0
    assert len(obj["cycles"]) == 1
    disp = obj["cycles"][0]["dispatches"]
    assert disp[0]["telemetry"]["provider"] == "anthropic-cli"
    assert disp[1]["telemetry"]["retry_count"] == 1


def test_dump_run_summary_noop_when_path_empty(tmp_path):
    # No path -> no write, no raise (default production behavior).
    br._dump_run_summary([{"cycle": 1, "dispatches": []}], "", lab_name="x",
                         wall_secs=1.0)
    br._dump_run_summary([{"cycle": 1, "dispatches": []}], None, lab_name="x",
                         wall_secs=1.0)
    assert list(tmp_path.iterdir()) == []   # nothing written
