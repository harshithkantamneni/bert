"""v2 benchmark statistics — paired-design inference done right.

The v1 sweep reported bare proportions at n=20 with no uncertainty, so fine
differences (hybrid vs vector) were indistinguishable from noise but presented
as if real. This module makes the uncertainty explicit and uses the CORRECT
test for the design:

  - Arms are evaluated on the SAME questions -> paired binary outcomes -> the
    right test for "does arm A differ from arm B" is McNemar's (on the
    discordant pairs), NOT a two-sample proportion test.
  - Per-arm accuracy uncertainty -> bootstrap 95% CI by resampling questions.
  - Many arm-pairs compared at once -> Holm-Bonferroni family-wise correction.

Pure functions, scipy/numpy only, fully unit-testable. No network, no LLM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


# ── per-arm accuracy with bootstrap CI ───────────────────────────────
@dataclass
class ArmStat:
    arm: str
    n: int
    accuracy: float
    ci_low: float
    ci_high: float
    # if k>1 repeats per item were collapsed, std across repeats of the mean
    repeat_std: float | None = None


def bootstrap_ci(correct: list[int] | np.ndarray, *, iters: int = 10000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of a 0/1 vector (resample items)."""
    x = np.asarray(correct, dtype=float)
    n = len(x)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iters, n))
    means = x[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (lo, hi)


def arm_stat(arm: str, correct: list[int], *, seed: int = 0,
             repeat_std: float | None = None) -> ArmStat:
    x = np.asarray(correct, dtype=float)
    n = len(x)
    acc = float(x.mean()) if n else float("nan")
    lo, hi = bootstrap_ci(x, seed=seed)
    return ArmStat(arm=arm, n=n, accuracy=acc, ci_low=lo, ci_high=hi,
                   repeat_std=repeat_std)


# ── paired comparison: McNemar exact ─────────────────────────────────
@dataclass
class PairTest:
    arm_a: str
    arm_b: str
    b: int          # a correct, b wrong (a wins)
    c: int          # a wrong, b correct (b wins)
    acc_a: float
    acc_b: float
    diff: float            # acc_a - acc_b
    diff_ci: tuple[float, float]
    p_value: float         # exact McNemar (two-sided)
    p_holm: float | None = None
    significant: bool | None = None


def mcnemar_exact(a_correct: list[int], b_correct: list[int]) -> tuple[int, int, float]:
    """Exact McNemar on paired 0/1 vectors. Returns (b, c, two-sided p).
    b = #(a=1,b=0), c = #(a=0,b=1). p from the binomial sign test on the
    n=b+c discordant pairs under H0 p=0.5 (exact; correct for small n)."""
    a = np.asarray(a_correct, dtype=int)
    bb = np.asarray(b_correct, dtype=int)
    if len(a) != len(bb):
        raise ValueError("paired vectors must be equal length")
    b = int(np.sum((a == 1) & (bb == 0)))
    c = int(np.sum((a == 0) & (bb == 1)))
    n = b + c
    if n == 0:
        return (b, c, 1.0)
    p = float(stats.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
    return (b, c, p)


def paired_diff_ci(a_correct: list[int], b_correct: list[int], *,
                   iters: int = 10000, alpha: float = 0.05,
                   seed: int = 0) -> tuple[float, float]:
    """Bootstrap CI for the PAIRED accuracy difference (resample items jointly,
    preserving pairing) — the honest interval for acc_a - acc_b."""
    a = np.asarray(a_correct, dtype=float)
    b = np.asarray(b_correct, dtype=float)
    n = len(a)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iters, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return (float(np.quantile(diffs, alpha / 2)),
            float(np.quantile(diffs, 1 - alpha / 2)))


def pair_test(arm_a: str, a_correct: list[int],
              arm_b: str, b_correct: list[int], *, seed: int = 0) -> PairTest:
    b, c, p = mcnemar_exact(a_correct, b_correct)
    acc_a = float(np.mean(a_correct)) if a_correct else float("nan")
    acc_b = float(np.mean(b_correct)) if b_correct else float("nan")
    dci = paired_diff_ci(a_correct, b_correct, seed=seed)
    return PairTest(arm_a=arm_a, arm_b=arm_b, b=b, c=c, acc_a=acc_a, acc_b=acc_b,
                    diff=acc_a - acc_b, diff_ci=dci, p_value=p)


# ── multiple-comparison correction ───────────────────────────────────
def holm_bonferroni(tests: list[PairTest], alpha: float = 0.05) -> list[PairTest]:
    """Annotate each PairTest with Holm-adjusted p + significance flag.
    Controls family-wise error across all pairwise comparisons."""
    m = len(tests)
    order = sorted(range(m), key=lambda i: tests[i].p_value)
    max_adj = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * tests[i].p_value)
        max_adj = max(max_adj, adj)   # enforce monotonicity
        tests[i].p_holm = max_adj
        tests[i].significant = max_adj < alpha
    return tests


# ── inter-judge agreement (Cohen's kappa, 2 raters) ──────────────────
def cohens_kappa(r1: list[int], r2: list[int]) -> float:
    """Cohen's kappa for two binary raters (judge agreement)."""
    a = np.asarray(r1, dtype=int)
    b = np.asarray(r2, dtype=int)
    n = len(a)
    if n == 0:
        return float("nan")
    po = float(np.mean(a == b))
    p1 = (np.mean(a == 1) * np.mean(b == 1)) + (np.mean(a == 0) * np.mean(b == 0))
    return float((po - p1) / (1 - p1)) if p1 < 1 else 1.0


if __name__ == "__main__":
    # self-test: sanity on known cases
    import sys
    # 1. bootstrap CI brackets the mean
    x = [1] * 15 + [0] * 5
    s = arm_stat("t", x)
    assert s.ci_low < s.accuracy < s.ci_high, s
    assert abs(s.accuracy - 0.75) < 1e-9
    # 2. identical arms -> McNemar p=1, diff 0
    b, c, p = mcnemar_exact(x, x)
    assert (b, c) == (0, 0) and p == 1.0
    # 3. strongly different arms -> small p
    a1 = [1] * 20
    a2 = [0] * 20
    _, _, p2 = mcnemar_exact(a1, a2)
    assert p2 < 1e-4, p2
    # 4. the v1 hybrid(0.70) vs vector(0.80) case: 2 discordant -> NOT significant
    #    A3: 14/20 correct, A4: 16/20, discordant on 6 (b=2, c=4)
    A3 = [1]*12 + [1, 1, 0, 0, 0, 0, 0, 0]   # 14 ones
    A4 = [1]*12 + [0, 0, 1, 1, 1, 1, 0, 0]   # 16 ones, net +2 to A4
    bb, cc, pv = mcnemar_exact(A3, A4)
    pt = pair_test("A3", A3, "A4", A4)
    print(f"hybrid-vs-vector replica: b={bb} c={cc} p={pv:.3f} diff={pt.diff:+.2f} "
          f"diff_ci=({pt.diff_ci[0]:+.2f},{pt.diff_ci[1]:+.2f}) -> "
          f"{'SIGNIFICANT' if pv < 0.05 else 'noise (cannot distinguish)'}")
    # 5. holm
    tests = [pair_test("A1", [0]*18+[1,1], "A4", [1]*16+[0]*4),
             pair_test("A3", A3, "A4", A4)]
    holm_bonferroni(tests)
    for t in tests:
        print(f"  {t.arm_a} vs {t.arm_b}: p={t.p_value:.4f} holm={t.p_holm:.4f} sig={t.significant}")
    # 6. kappa identical -> 1
    assert abs(cohens_kappa([1,0,1,1,0],[1,0,1,1,0]) - 1.0) < 1e-9
    print("v2_stats self-test: OK")
    sys.exit(0)
