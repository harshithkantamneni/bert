"""Cost estimation with a confidence interval (Sprint 3 c6, recheck Q-2).

A cost preview that shows a single number is false precision — the user
can't tell a calibrated $0.05 from a wild guess. So every estimate
carries a CI:

  - With ≥ MIN_HISTORY observed runs: point = median(history), CI half-
    width derived from the observed coefficient of variation (stddev /
    mean), capped at MAX_CI_PCT.
  - Otherwise: point = the feature's hand-authored estimate, CI = a wide
    DEFAULT_CI_PCT band, and the basis string says so explicitly.

The CI narrows as real runs accumulate — the estimate earns precision
from evidence rather than asserting it up front.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CI_PCT = 0.30   # ±30% when we have no/insufficient history
MAX_CI_PCT = 0.50       # never claim a band wider than ±50%
MIN_HISTORY = 3         # runs needed before history overrides the default


@dataclass
class CostEstimate:
    cost_usd: float          # point estimate
    ci_low: float            # cost_usd * (1 - ci_pct), floored at 0
    ci_high: float           # cost_usd * (1 + ci_pct)
    ci_pct: float            # relative half-width (0.30 == ±30%)
    n_samples: int           # historical runs informing the CI
    basis: str               # human-readable methodology

    def as_label(self) -> str:
        """e.g. '$0.05 ±30% (feature default, no history)'."""
        return f"${self.cost_usd:.3f} ±{int(self.ci_pct * 100)}% ({self.basis})"


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def estimate(point_usd: float, history: list[float] | None = None) -> CostEstimate:
    """Build a CostEstimate. `point_usd` is the fallback point estimate
    (typically a feature's hand-authored estimated_cost_usd); `history`
    is observed per-run costs in USD."""
    samples = [h for h in (history or []) if h is not None and h >= 0]
    n = len(samples)

    if n >= MIN_HISTORY:
        point = _median(samples)
        mean = sum(samples) / n
        if mean > 0:
            var = sum((x - mean) ** 2 for x in samples) / n
            cv = (var ** 0.5) / mean          # coefficient of variation
        else:
            cv = DEFAULT_CI_PCT
        ci_pct = min(MAX_CI_PCT, max(0.01, cv))
        basis = f"median of {n} historical runs (CI ±{int(ci_pct * 100)}% from sample spread)"
    else:
        point = point_usd
        ci_pct = DEFAULT_CI_PCT
        basis = (
            f"feature default estimate, no history "
            f"(need ≥{MIN_HISTORY} runs; have {n})"
        )

    ci_low = max(0.0, point * (1 - ci_pct))
    ci_high = point * (1 + ci_pct)
    return CostEstimate(
        cost_usd=point, ci_low=ci_low, ci_high=ci_high,
        ci_pct=ci_pct, n_samples=n, basis=basis,
    )


def estimate_llm_calls(roster_size: int, cycles: int, calls_per_role: int = 1) -> int:
    """Rough LLM-call count: each role in the roster fires ~calls_per_role
    times per cycle, plus one director decision per cycle. Floored at 1."""
    per_cycle = max(1, roster_size) * max(1, calls_per_role) + 1  # +1 director
    return max(1, per_cycle * max(1, cycles))


def estimate_from_feature(feature, history: list[float] | None = None) -> CostEstimate:
    """Convenience: estimate a feature's cost from its hand-authored
    estimated_cost_usd + any observed history."""
    return estimate(point_usd=float(feature.estimated_cost_usd), history=history)
