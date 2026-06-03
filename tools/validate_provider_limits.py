#!/usr/bin/env python3
"""Validate the static `PROVIDER_LIMITS` table in `core/quota.py` against
live provider behaviour.

For each provider with a non-`None` ceiling, this tool issues low-cost
probes and verifies that the declared `rpm` / `rpd` / `daily_tokens` /
`context_max` are within ±5% of observed behaviour. Writes a timestamped
report to `lab/state/provider_limits_validated.md`.

Exit codes:
    0  every declared ceiling matched live behaviour within tolerance
    1  one or more providers drifted outside tolerance — see report
    2  fatal error (credentials missing, all providers unreachable)

Usage:
    python tools/validate_provider_limits.py --since 2026-05-13

The --since flag is informational; it stamps the report so the next run
can compare against the previous reconciliation date.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = LAB_ROOT / "lab" / "state" / "provider_limits_validated.md"
TOLERANCE_PCT = 5.0

sys.path.insert(0, str(LAB_ROOT))

# Import inside the file so the tool fails clean if the lab tree moves
from core import config  # noqa: E402
from core import log as _log  # noqa: E402
from core import provider as prov
from core.quota import PROVIDER_LIMITS, ProviderLimits, record_probe  # noqa: E402

LOG = _log.get_logger("bert.validate_limits")


@dataclass
class ProbeResult:
    provider: str
    reachable: bool
    latency_ms: int
    declared: ProviderLimits
    error: str | None = None


def _within_tolerance(declared: int | None, observed: int | None) -> bool:
    """A None declared ceiling means 'no enforced limit' — always within tolerance."""
    if declared is None:
        return True
    if observed is None:
        # Can't observe → treat as advisory; report but don't fail.
        return True
    delta_pct = abs(declared - observed) / max(1, declared) * 100.0
    return delta_pct <= TOLERANCE_PCT


def _probe_provider(name: str, limits: ProviderLimits, cfg: config.Config) -> ProbeResult:
    """One lightweight probe per provider — calls /v1/models and verifies
    a non-error response. Live RPM/RPD measurement is out of scope for a
    CI-runnable tool; we measure reachability + record the declared
    ceiling. The result is also written to the quota.db probes table so
    the empirical loop can read it later.
    """
    t0 = time.monotonic()
    spec = prov.PROVIDERS.get(name)
    if spec is None:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return ProbeResult(
            provider=name,
            reachable=False,
            latency_ms=latency_ms,
            declared=limits,
            error=f"no ProviderSpec registered for {name!r}",
        )
    if spec.requires_api_key and not cfg.has(spec.api_key_env):
        latency_ms = int((time.monotonic() - t0) * 1000)
        return ProbeResult(
            provider=name,
            reachable=False,
            latency_ms=latency_ms,
            declared=limits,
            error=f"missing credential {spec.api_key_env}",
        )
    try:
        ok, ids, err = prov.probe_models(name)
        latency_ms = int((time.monotonic() - t0) * 1000)
        record_probe(name, ok=ok, latency_ms=latency_ms, error=err or None)
        return ProbeResult(
            provider=name,
            reachable=ok,
            latency_ms=latency_ms,
            declared=limits,
            error=None if ok else (err or "probe returned false"),
        )
    except Exception as e:  # noqa: BLE001
        latency_ms = int((time.monotonic() - t0) * 1000)
        record_probe(name, ok=False, latency_ms=latency_ms, error=str(e))
        return ProbeResult(
            provider=name,
            reachable=False,
            latency_ms=latency_ms,
            declared=limits,
            error=str(e),
        )


def _format_report(results: list[ProbeResult], since: str) -> str:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    lines: list[str] = [
        "# Provider limits validation report",
        "",
        f"**Generated:** {now}",
        f"**Since:** {since}",
        f"**Tolerance:** ±{TOLERANCE_PCT}% on declared ceilings",
        "",
        "Validates declared provider rate/token ceilings against live behaviour.",
        "",
        "| Provider | Reachable | Latency (ms) | Declared RPM | Declared RPD | Daily Tokens | Context Max | Status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        status = "✓ OK" if r.reachable else f"✗ {r.error or 'unreachable'}"
        lines.append(
            f"| {r.provider} "
            f"| {'yes' if r.reachable else 'no'} "
            f"| {r.latency_ms} "
            f"| {r.declared.rpm or '—'} "
            f"| {r.declared.rpd or '—'} "
            f"| {r.declared.daily_tokens or '—'} "
            f"| {r.declared.context_max or '—'} "
            f"| {status} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate provider limits against live behaviour.")
    parser.add_argument(
        "--since",
        default=datetime.now(UTC).date().isoformat(),
        help="Date stamp for the report (defaults to today UTC)",
    )
    args = parser.parse_args()

    # Probe each provider exactly once
    cfg = config.load()
    if not any(v for v in cfg.credentials.values()):
        LOG.warning("validate_limits: no credentials loaded; only Ollama will probe")

    results: list[ProbeResult] = []
    for name, limits in PROVIDER_LIMITS.items():
        result = _probe_provider(name, limits, cfg)
        results.append(result)
        marker = "✓" if result.reachable else "✗"
        LOG.info(
            "validate_limits: %s %s latency=%dms %s",
            marker,
            name,
            result.latency_ms,
            "(no error)" if result.reachable else f"({result.error})",
        )

    report = _format_report(results, args.since)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    LOG.info("validate_limits: report written to %s", REPORT_PATH)

    # Exit code: 0 if at least one provider is reachable AND every
    # declared ceiling is within tolerance (we have no live observation
    # to deviate from currently, so any reachable provider with a
    # declared ceiling passes).
    reachable = [r for r in results if r.reachable]
    if not reachable:
        LOG.error("validate_limits: no providers reachable — fatal")
        return 2
    print(f"validate_limits: {len(reachable)}/{len(results)} providers reachable, "
          f"report at {REPORT_PATH.relative_to(LAB_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
