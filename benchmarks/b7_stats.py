"""Pure, no-network small-n paired statistics for the B7 infra-value A/B
benchmark. Separated from the runner so it is fully unit-testable without Opus
or any provider. Honest about small n: never emits a misleading p-value, uses
robust effect sizes (Cliff's delta, bootstrap CI) per
benchmarks/B7_INFRA_VALUE_METHODOLOGY.md §8.

All functions are deterministic given their inputs (bootstrap takes an explicit
seed) and degrade to finite sentinels rather than inf/NaN so the results
serialize cleanly to JSON.
"""

from __future__ import annotations

import random
from collections import defaultdict

# ── central tendency ─────────────────────────────────────────────────

def _percentile(sorted_xs: list[float], p: float) -> float:
    """Linear-interpolation percentile (numpy default 'linear' method).
    p in [0,1]. Assumes sorted_xs non-empty and sorted ascending."""
    if len(sorted_xs) == 1:
        return float(sorted_xs[0])
    idx = (len(sorted_xs) - 1) * p
    lo = int(idx)
    frac = idx - lo
    if lo + 1 >= len(sorted_xs):
        return float(sorted_xs[lo])
    return float(sorted_xs[lo] + frac * (sorted_xs[lo + 1] - sorted_xs[lo]))


def median(xs: list[float]) -> float:
    return _percentile(sorted(xs), 0.5)


def median_iqr(xs: list[float]) -> tuple[float, float, float]:
    """Return (median, q1, q3) using linear-interpolation percentiles."""
    s = sorted(xs)
    return (_percentile(s, 0.5), _percentile(s, 0.25), _percentile(s, 0.75))


# ── paired differences + effect sizes ────────────────────────────────

def paired_diffs(c_scores: list[float], b_scores: list[float]) -> list[float]:
    """Per-instance paired differences C-B (or any two matched arms)."""
    return [round(c - b, 10) for c, b in zip(c_scores, b_scores, strict=True)]


def cliffs_delta(xs: list[float], ys: list[float]) -> float:
    """Cliff's delta = (#(x>y) - #(x<y)) / (nx*ny). Robust at small n."""
    if not xs or not ys:
        return 0.0
    gt = lt = 0
    for x in xs:
        for y in ys:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    return (gt - lt) / (len(xs) * len(ys))


def cliffs_band(d: float) -> str:
    """Interpretation bands on |delta| (Romano et al.):
    <0.147 negligible, <0.33 small, <0.474 medium, else large."""
    a = abs(d)
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


def cohens_dz(diffs: list[float]) -> float:
    """Paired Cohen's d_z = mean(diffs)/sd(diffs) (sample sd, ddof=1).
    Returns 0.0 when there is no spread (avoids inf/NaN). Secondary metric,
    caveated at n<10 in the report."""
    n = len(diffs)
    if n < 2:
        return 0.0
    m = sum(diffs) / n
    var = sum((d - m) ** 2 for d in diffs) / (n - 1)
    sd = var ** 0.5
    if sd == 0.0:
        return 0.0
    return m / sd


def length_slope(scores: list[float], tokens_out: list[float]) -> float:
    """OLS slope of weighted_score on tokens_out — the length-bias diagnostic.
    Returns 0.0 if tokens have no variance (avoids div-by-zero)."""
    n = len(scores)
    if n < 2:
        return 0.0
    mx = sum(tokens_out) / n
    my = sum(scores) / n
    sxx = sum((x - mx) ** 2 for x in tokens_out)
    if sxx == 0.0:
        return 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(tokens_out, scores, strict=True))
    return sxy / sxx


def wilcoxon_p(diffs: list[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank p — but ONLY when n is large enough to
    be meaningful. Below n=6 the two-sided p-floor is ~0.25, so no result can
    reach significance; we return None rather than a misleading number. For
    n>=6 we defer to scipy; if every diff is zero (degenerate) we also return
    None."""
    nz = [d for d in diffs if d != 0]
    if len(nz) < 6:
        return None
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    try:
        res = wilcoxon(nz, alternative="two-sided", zero_method="wilcox")
        return float(res.pvalue)
    except ValueError:
        return None


def bootstrap_ci(diffs: list[float], n_boot: int = 10000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI on the MEDIAN of paired differences.
    Reproducible for a fixed seed. At small n the interval is wide — that
    width is the honest signal, not a defect."""
    if not diffs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(diffs)
    meds: list[float] = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        meds.append(median(sample))
    meds.sort()
    lo = _percentile(meds, alpha / 2)
    hi = _percentile(meds, 1 - alpha / 2)
    return (lo, hi)


# ── per-instance derivation (the 3-arm decomposition) ────────────────

def _safe_ratio(num: float, den: float, *, sentinel: float = 0.0) -> float:
    return num / den if den else sentinel


def derive_instance(row: dict) -> dict:
    """Derive the per-instance A/B/C metrics. The decomposition identity
    quality_gain_total (C-A) == quality_gain_orch (C-B) + leakage (B-A) holds
    EXACTLY here (it is the per-instance arithmetic, not the median)."""
    a, b, c = row["A_score"], row["B_score"], row["C_score"]
    orch = c - b
    leakage = b - a
    total = c - a
    extra_tokens = row["C_tokens"] - row["A_tokens"]
    extra_ktokens = extra_tokens / 1000.0
    return {
        "tier": row.get("tier"),
        "instance": row.get("instance"),
        "quality_gain_orch": orch,
        "leakage": leakage,
        "quality_gain_total": total,
        "overhead_token_ratio": _safe_ratio(row["C_tokens"], row["A_tokens"]),
        "overhead_lat_ratio": _safe_ratio(row["C_latency"], row["A_latency"]),
        "gain_per_extra_ktoken": _safe_ratio(orch, extra_ktokens),
        "gain_per_extra_second": _safe_ratio(orch, row["C_latency"] - row["A_latency"]),
        "A_score": a, "B_score": b, "C_score": c,
    }


def per_tier_summary(rows: list[dict], *, tau_ktoken: float = 0.0,
                     boot_seed: int = 0) -> dict:
    """Aggregate per-instance rows into a per-tier summary dict keyed by tier.
    Quality gain is reported as the median of paired differences + a bootstrap
    CI; effect size as Cliff's delta on C vs B scores."""
    by_tier: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_tier[r["tier"]].append(derive_instance(r))

    out: dict[str, dict] = {}
    for tier, dl in by_tier.items():
        orch = [d["quality_gain_orch"] for d in dl]
        leak = [d["leakage"] for d in dl]
        total = [d["quality_gain_total"] for d in dl]
        c_scores = [d["C_score"] for d in dl]
        b_scores = [d["B_score"] for d in dl]
        ci_low, ci_high = bootstrap_ci(orch, n_boot=10000, seed=boot_seed)
        gain_med = median(orch)
        # gain per extra ktoken at the tier level uses median gain over median
        # extra tokens so it is not dominated by one outlier instance.
        med_token_ratio = median([d["overhead_token_ratio"] for d in dl])
        med_lat_ratio = median([d["overhead_lat_ratio"] for d in dl])
        gpk = median([d["gain_per_extra_ktoken"] for d in dl])
        out[tier] = {
            "n_pairs": len(dl),
            "quality_gain_orch_median": gain_med,
            "quality_gain_total_median": median(total),
            "leakage_median": median(leak),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "cliffs_delta": cliffs_delta(c_scores, b_scores),
            "cliffs_band": cliffs_band(cliffs_delta(c_scores, b_scores)),
            "cohens_dz": cohens_dz(orch),
            "wilcoxon_p": wilcoxon_p(orch),
            "overhead_token_ratio": med_token_ratio,
            "overhead_lat_ratio": med_lat_ratio,
            "gain_per_extra_ktoken": gpk,
            "justified": (gpk >= tau_ktoken) and (ci_low > 0),
        }
    return out


def _tier_key(t: str) -> tuple:
    """Order tiers by trailing integer (T0<T1<T2<T3); fall back to string."""
    digits = "".join(ch for ch in t if ch.isdigit())
    return (0, int(digits)) if digits else (1, t)


def find_crossover(per_tier: dict, tau_ktoken: float = 0.0) -> str | None:
    """The smallest tier that is justified AND for which every higher tier is
    also justified. justified := gain_per_extra_ktoken >= tau AND ci_low > 0
    (the noise guard: a positive point estimate whose CI includes 0 does NOT
    count). Returns None ('no crossover in range') when nothing qualifies."""
    tiers = sorted(per_tier.keys(), key=_tier_key)

    def justified(t: str) -> bool:
        s = per_tier[t]
        return (s.get("gain_per_extra_ktoken", 0.0) >= tau_ktoken
                and s.get("ci_low", -1.0) > 0)

    for i, t in enumerate(tiers):
        if justified(t) and all(justified(h) for h in tiers[i:]):
            return t
    return None
