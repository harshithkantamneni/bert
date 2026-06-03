"""Smoke + TDD: tools/refresh_model_cards.py — daily model-registry refresh (#31).

The capability harness + model-card registry existed, but nothing refreshed the
registry on a daily cadence. This is the refresh orchestrator the existing daily
runner (tools/bert_nightly.sh, scheduled by tools/install_nightly.py via
launchd/cron) calls: it reloads + validates the registry, surfaces models within
7 days of deprecation (#39) and already-deprecated ones (#32), and stamps a
last-refreshed marker so staleness is observable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "tools"))

from tools import refresh_model_cards as rmc  # noqa: E402


def test_refresh_returns_summary_and_writes_marker(tmp_path):
    marker = tmp_path / "model_registry_refresh.json"
    summary = rmc.refresh(marker_path=marker)
    assert summary["cards"] > 0
    assert isinstance(summary["pending_deprecation"], list)
    assert isinstance(summary["deprecated"], list)
    assert "refreshed_at" in summary
    # marker persisted with the same shape
    on_disk = json.loads(marker.read_text())
    assert on_disk["cards"] == summary["cards"]


def test_refresh_no_marker_when_disabled(tmp_path):
    marker = tmp_path / "nope.json"
    rmc.refresh(marker_path=marker, write_marker=False)
    assert not marker.exists()


def test_main_returns_zero(tmp_path):
    rc = rmc.main(["--marker", str(tmp_path / "m.json")])
    assert rc == 0


def test_daily_runner_wires_the_refresh():
    # The criterion is a DAILY refresh — assert the nightly runner invokes it
    # (install_nightly.py schedules bert_nightly.sh via launchd/cron).
    sh = (LAB_ROOT / "tools" / "bert_nightly.sh").read_text()
    assert "refresh_model_cards.py" in sh


def main() -> int:
    import inspect
    import tempfile
    tests = [
        test_refresh_returns_summary_and_writes_marker,
        test_refresh_no_marker_when_disabled,
        test_main_returns_zero,
        test_daily_runner_wires_the_refresh,
    ]
    for t in tests:
        try:
            if "tmp_path" in inspect.signature(t).parameters:
                with tempfile.TemporaryDirectory() as d:
                    t(Path(d))
            else:
                t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
