"""Smoke test for the per-lab paused-flag polling in tools/bert_run.py.

The runner watches `<lab_path>/state/paused` and idles (5s poll cadence, Ctrl-C
honored) while it exists. (The multi-lab state-isolation API that originally
motivated this is not part of this repo.)
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


def test_bert_run_polls_paused_flag_per_lab() -> None:
    src = (LAB_ROOT / "tools" / "bert_run.py").read_text()
    # The flag lives under <lab_path>/state/paused per the refactor.
    assert 'paused_flag = lab_path / "state" / "paused"' in src
    # Loop polls with a sleep cadence (don't busy-wait).
    assert "while paused_flag.exists():" in src
    # Honors Ctrl-C while paused.
    assert 'interrupted["caught"]' in src
    # 5-second poll cadence.
    assert "time.sleep(5)" in src
