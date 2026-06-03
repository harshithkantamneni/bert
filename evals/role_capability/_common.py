"""Shared scaffolding for L-24 Phase 2 role-capability batteries.

Each role battery imports from here so the battery files stay short
and focused on their specific task corpus + scoring.

Two execution modes:

  - offline (default): synthesize per-task scores from deterministic
    seed (provider, model, task_id) hashes. This keeps the matrix
    structure warm + monotonic when network is down. Caller passes
    `live=False`. The harness's first deployment runs offline.

  - live: actually dispatch the task through core.provider.call,
    apply the role-specific scorer to the response. Caller passes
    `live=True`. Live runs consume real quota and need credentials.

Both modes return a CapabilityRow that the harness appends to
lab/state/capability_matrix.jsonl.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

LAB_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class Task:
    id: str
    prompt: str
    reference: str | None = None  # optional held-out reference for scoring
    rubric_hint: str = ""


@dataclass
class BatteryRunResult:
    role: str
    provider: str
    model: str
    score: float
    task_count: int
    latency_p50_ms: int
    latency_p95_ms: int
    cost_per_task_usd: float
    notes: str = ""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seeded_score(role: str, provider: str, model: str, task_id: str) -> float:
    """Deterministic 0..1 score from (role, provider, model, task_id).

    Used in offline mode to populate matrix with reproducible values.
    Bias: model size proxy (param count in name) nudges score up; this
    keeps the L-24 router sensible during offline warm-up.
    """
    seed = f"{role}|{provider}|{model}|{task_id}"
    h = hashlib.sha256(seed.encode()).digest()
    # Map first 4 bytes to 0..1
    base = int.from_bytes(h[:4], "big") / 0xFFFFFFFF
    # Bias: 70B+ models +0.1; 8B/small -0.1; cross-family Qwen evaluator +0.05
    bias = 0.0
    name = (provider + "/" + model).lower()
    if "70b" in name or "next-80b" in name or "120b" in name:
        bias += 0.10
    elif "8b" in name or "small" in name or "mini" in name:
        bias -= 0.08
    if "qwen" in name and role == "evaluator":
        bias += 0.05
    if "deepseek-r1" in name and role in ("evaluator", "threshing"):
        bias += 0.04
    return max(0.05, min(0.98, base * 0.7 + 0.2 + bias))


def run_battery(
    role: str,
    tasks: list[Task],
    scorer: Callable[[Task, str], float],
    *,
    provider: str,
    model: str,
    live: bool = False,
    sample: int | None = None,
) -> BatteryRunResult:
    """Run `tasks` against (provider, model) and aggregate scores.

    offline mode synthesizes per-task scores; live mode dispatches and
    invokes `scorer(task, response_text)`. Either way we return one
    CapabilityRow-shaped summary.
    """
    if sample and sample < len(tasks):
        tasks = tasks[:sample]
    if not tasks:
        return BatteryRunResult(
            role=role, provider=provider, model=model,
            score=0.0, task_count=0,
            latency_p50_ms=0, latency_p95_ms=0,
            cost_per_task_usd=0.0,
            notes="empty task list",
        )

    if not live:
        scores = [_seeded_score(role, provider, model, t.id) for t in tasks]
        return BatteryRunResult(
            role=role, provider=provider, model=model,
            score=sum(scores) / len(scores),
            task_count=len(tasks),
            latency_p50_ms=0, latency_p95_ms=0,
            cost_per_task_usd=0.0,
            notes="offline (deterministic seeded scores)",
        )

    # Live mode — dispatch through core/provider.py
    from core import provider as prov
    latencies: list[int] = []
    scores: list[float] = []
    for task in tasks:
        t0 = time.monotonic()
        try:
            resp = prov.call(
                provider,
                messages=[{"role": "user", "content": task.prompt}],
                model=model,
                max_tokens=400,
                temperature=0.3,
            )
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            latencies.append(elapsed_ms)
            scores.append(scorer(task, resp.text or ""))
        except Exception:  # noqa: BLE001
            scores.append(0.0)
            latencies.append(0)
    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0
    return BatteryRunResult(
        role=role, provider=provider, model=model,
        score=sum(scores) / len(scores),
        task_count=len(tasks),
        latency_p50_ms=p50, latency_p95_ms=p95,
        cost_per_task_usd=0.0,  # free-tier providers
        notes="live dispatch",
    )


def write_matrix_row(result: BatteryRunResult, *, reference_set: str) -> None:
    """Append the run result to lab/state/capability_matrix.jsonl."""
    from core import capability_matrix
    row = capability_matrix.CapabilityRow(
        ts=now_iso(),
        role=result.role,
        provider=result.provider,
        model=result.model,
        score=round(result.score, 4),
        cost_per_task_usd=result.cost_per_task_usd,
        latency_p50_ms=result.latency_p50_ms,
        latency_p95_ms=result.latency_p95_ms,
        quota_headroom_pct=100,  # filled by harness when it queries quota.stats
        task_count=result.task_count,
        reference_set=reference_set,
        notes=result.notes,
    )
    capability_matrix.append_row(row)
