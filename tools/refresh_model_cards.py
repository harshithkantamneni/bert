"""Daily model-registry refresh (launch criterion #31).

The model-card registry (core/library/model_cards.yaml) + the capability harness
(tools/run_capability_harness.py) existed, but nothing refreshed the registry on
a daily cadence. This is the refresh orchestrator that the existing daily runner
(tools/bert_nightly.sh, scheduled by tools/install_nightly.py via launchd on
macOS / cron on Linux) invokes each night. It:

  1. reloads + validates the card registry (a malformed card is dropped by the
     loader, so a successful load is the validation),
  2. surfaces models within 7 days of deprecation (#39 warning) and any already
     past their deprecation_date (#32 — these now remap via deprecated_to),
  3. stamps a last-refreshed marker (state/model_registry_refresh.json) so
     registry staleness is observable (e.g. by `bert doctor`).

Live provider-catalog fetching (querying each provider's model list) is a future
operational upgrade; the daily job's load-bearing work today is validation +
deprecation surfacing + the freshness stamp.

Usage:
  .venv/bin/python tools/refresh_model_cards.py
  .venv/bin/python tools/refresh_model_cards.py --marker /tmp/m.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DEFAULT_MARKER = _REPO / "state" / "model_registry_refresh.json"


def refresh(*, marker_path: Path | None = None, write_marker: bool = True) -> dict:
    """Reload + validate the registry, surface deprecations, stamp freshness.
    Returns a summary dict."""
    from core import model_cards
    cards = model_cards.load_all(force_reload=True)
    pending = [c.id for c in model_cards.cards_with_pending_deprecation(7)]
    deprecated = [c.id for c in cards if model_cards.is_deprecated(c)]
    summary = {
        "refreshed_at": datetime.now(UTC).isoformat(),
        "cards": len(cards),
        "pending_deprecation": pending,
        "deprecated": deprecated,
    }
    if write_marker:
        path = Path(marker_path) if marker_path is not None else DEFAULT_MARKER
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Daily model-registry refresh (#31)")
    p.add_argument("--marker", default=None, help="path for the freshness marker JSON")
    p.add_argument("--no-marker", action="store_true", help="don't write the marker")
    args = p.parse_args(argv)
    summary = refresh(
        marker_path=Path(args.marker) if args.marker else None,
        write_marker=not args.no_marker,
    )
    print(f"model-registry refresh: {summary['cards']} cards · "
          f"{len(summary['pending_deprecation'])} nearing deprecation · "
          f"{len(summary['deprecated'])} deprecated (remapped via aliases)")
    for mid in summary["pending_deprecation"]:
        print(f"  ⚠ {mid} deprecates within 7 days")
    for mid in summary["deprecated"]:
        print(f"  ⓧ {mid} is deprecated — routing remaps to its successor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
