"""TDD for benchmarks/b7_stats.py — the pure, no-network small-n paired stats
for the B7 infra-value A/B benchmark. Every value here is hand-checkable so the
stats can't silently drift. See benchmarks/B7_INFRA_VALUE_METHODOLOGY.md §8."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks import b7_stats as S  # noqa: E402


def test_paired_diffs():
    assert S.paired_diffs([0.9, 0.8, 0.7], [0.6, 0.5, 0.5]) == [
        round(0.9 - 0.6, 10), round(0.8 - 0.5, 10), round(0.7 - 0.5, 10)]


def test_median_iqr_n3_linear_percentile():
    # numpy-linear percentile on [1,2,3]: median=2.0, q1=1.5, q3=2.5
    med, q1, q3 = S.median_iqr([3, 1, 2])
    assert (med, q1, q3) == (2.0, 1.5, 2.5)


def test_cliffs_delta_hand_value_and_band():
    # xs=[2,3,4], ys=[1,2,3]: (#x>y - #x<y)/(9) = (6-1)/9
    d = S.cliffs_delta([2, 3, 4], [1, 2, 3])
    assert abs(d - (5 / 9)) < 1e-9
    assert S.cliffs_band(d) == "large"      # |d|>0.474
    assert S.cliffs_band(0.1) == "negligible"
    assert S.cliffs_band(-0.3) == "small"    # |0.3| in [0.147,0.33) -> small
    assert S.cliffs_band(-0.4) == "medium"   # bands are on |d|, [0.33,0.474)


def test_cohens_dz_hand_value_and_zero_sd():
    # diffs=[2,4,6]: mean=4, sample sd=2 -> dz=2.0
    assert abs(S.cohens_dz([2, 4, 6]) - 2.0) < 1e-9
    # zero spread -> 0.0, never a NaN/inf that poisons JSON
    assert S.cohens_dz([1, 1, 1]) == 0.0


def test_length_slope_hand_value():
    # y=[1,2,3] on x=[10,20,30]: OLS slope = 0.1
    assert abs(S.length_slope([1, 2, 3], [10, 20, 30]) - 0.1) < 1e-9
    # degenerate x (no variance) -> 0.0, not div-by-zero
    assert S.length_slope([1, 2, 3], [5, 5, 5]) == 0.0


def test_wilcoxon_p_none_below_floor():
    # n<6 cannot reach significance (two-sided floor ~0.25) -> None, never a p
    assert S.wilcoxon_p([0.1, 0.2, 0.3]) is None
    # n>=6 returns a float in [0,1] (or None if degenerate all-zero diffs)
    p = S.wilcoxon_p([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert p is None or (isinstance(p, float) and 0.0 <= p <= 1.0)


def test_bootstrap_ci_reproducible_and_brackets_median():
    diffs = [0.05, 0.12, 0.09, 0.20, 0.02]
    lo1, hi1 = S.bootstrap_ci(diffs, n_boot=2000, seed=42)
    lo2, hi2 = S.bootstrap_ci(diffs, n_boot=2000, seed=42)
    assert (lo1, hi1) == (lo2, hi2)            # deterministic under fixed seed
    assert lo1 <= S.median_iqr(diffs)[0] <= hi1
    assert lo1 <= hi1


def test_derive_instance_decomposition_identity():
    # The 3-arm decomposition must hold EXACTLY per instance:
    #   quality_gain_total (C-A) == quality_gain_orch (C-B) + leakage (B-A)
    row = {"tier": "T1", "instance": "i1",
           "A_score": 0.60, "B_score": 0.72, "C_score": 0.80,
           "A_tokens": 1000, "C_tokens": 5000,
           "A_latency": 60.0, "C_latency": 300.0}
    d = S.derive_instance(row)
    assert abs(d["quality_gain_total"] - (d["quality_gain_orch"] + d["leakage"])) < 1e-12
    assert abs(d["quality_gain_orch"] - 0.08) < 1e-9   # C-B
    assert abs(d["leakage"] - 0.12) < 1e-9             # B-A
    assert abs(d["overhead_token_ratio"] - 5.0) < 1e-9
    assert abs(d["overhead_lat_ratio"] - 5.0) < 1e-9
    # gain per extra ktoken = orch / ((C-A)/1000) = 0.08 / 4.0 = 0.02
    assert abs(d["gain_per_extra_ktoken"] - 0.02) < 1e-9


def test_derive_instance_divzero_guard():
    row = {"tier": "T0", "instance": "i1",
           "A_score": 0.5, "B_score": 0.5, "C_score": 0.5,
           "A_tokens": 0, "C_tokens": 0, "A_latency": 0.0, "C_latency": 0.0}
    d = S.derive_instance(row)
    # no extra tokens -> gain_per_extra_ktoken must be a finite sentinel, not inf/NaN
    assert math.isfinite(d["gain_per_extra_ktoken"])
    assert math.isfinite(d["overhead_token_ratio"])


def _tier_rows(tier, gains, *, a=0.6, tokens_extra=4000):
    # build n instances for a tier with controlled C-B gains
    rows = []
    for i, g in enumerate(gains):
        rows.append({"tier": tier, "instance": f"i{i}",
                     "A_score": a, "B_score": a + 0.05, "C_score": a + 0.05 + g,
                     "A_tokens": 1000, "C_tokens": 1000 + tokens_extra,
                     "A_latency": 60.0, "C_latency": 300.0})
    return rows


def test_per_tier_summary_basic():
    rows = _tier_rows("T2", [0.10, 0.12, 0.08])
    summ = S.per_tier_summary(rows, tau_ktoken=0.0, boot_seed=1)["T2"]
    assert summ["n_pairs"] == 3
    assert abs(summ["quality_gain_orch_median"] - 0.10) < 1e-9
    assert abs(summ["leakage_median"] - 0.05) < 1e-9
    assert "ci_low" in summ and "cliffs_delta" in summ and "gain_per_extra_ktoken" in summ


def test_find_crossover_noise_guard_and_monotone_higher():
    # T0: gain over tau but CI includes 0 -> NOT justified (noise guard)
    # T1,T2,T3: justified -> crossover is the smallest justified tier whose
    # higher tiers are ALL justified -> T1.
    per_tier = {
        "T0": {"gain_per_extra_ktoken": 0.50, "ci_low": -0.10},
        "T1": {"gain_per_extra_ktoken": 0.50, "ci_low": 0.02},
        "T2": {"gain_per_extra_ktoken": 0.50, "ci_low": 0.05},
        "T3": {"gain_per_extra_ktoken": 0.50, "ci_low": 0.10},
    }
    assert S.find_crossover(per_tier, tau_ktoken=0.1) == "T1"

    # A gap in the middle: T2 not justified -> only T3 (and all-higher=none) qualifies
    per_tier2 = {
        "T1": {"gain_per_extra_ktoken": 0.50, "ci_low": 0.02},
        "T2": {"gain_per_extra_ktoken": 0.50, "ci_low": -0.01},  # CI includes 0
        "T3": {"gain_per_extra_ktoken": 0.50, "ci_low": 0.10},
    }
    assert S.find_crossover(per_tier2, tau_ktoken=0.1) == "T3"

    # Nothing justified -> honest None
    per_tier3 = {"T0": {"gain_per_extra_ktoken": 0.0, "ci_low": -0.2}}
    assert S.find_crossover(per_tier3, tau_ktoken=0.1) is None
