"""Empirical model-capability-per-role matrix.

Replaces the static EVAL_PROVIDER_PREFERENCE / MODEL_FAMILIES heuristic in
core/subagent.py with measured per-role scores from
`lab/state/capability_matrix.jsonl`.

The matrix is append-only JSONL with one row per
`{model, role, ts}` triple:

  {
    "ts": "2026-05-13T08:00:00Z",
    "role": "researcher",
    "provider": "nvidia",
    "model": "meta/llama-3.3-70b-instruct",
    "score": 0.78,
    "cost_per_task_usd": 0.0,
    "latency_p50_ms": 2340,
    "latency_p95_ms": 4880,
    "quota_headroom_pct": 42,
    "task_count": 20,
    "reference_set": "researcher_battery_v1",
    "notes": ""
  }

`tools/run_capability_harness.py` writes new rows weekly via cron;
the router reads the file each cycle into an in-memory dict keyed by
`{role: [rows]}` and picks the max-score-within-headroom for the
first attempt.

`pick_evaluator_model` must reference `capability_matrix` in its
body — this module is the integration point for that constraint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parent.parent
MATRIX_PATH = LAB_ROOT / "lab" / "state" / "capability_matrix.jsonl"


@dataclass
class CapabilityRow:
    ts: str
    role: str
    provider: str
    model: str
    score: float
    cost_per_task_usd: float
    latency_p50_ms: int
    latency_p95_ms: int
    quota_headroom_pct: int
    task_count: int
    reference_set: str
    notes: str = ""


def load_rows(path: Path | None = None) -> list[CapabilityRow]:
    """Read the matrix as a list of CapabilityRow objects.

    Empty file or missing file returns []. Malformed lines are skipped
    (best-effort; one corrupt row shouldn't kill the cascade).
    """
    p = path or MATRIX_PATH
    if not p.exists():
        return []
    rows: list[CapabilityRow] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            rows.append(CapabilityRow(
                ts=d["ts"],
                role=d["role"],
                provider=d["provider"],
                model=d["model"],
                score=float(d["score"]),
                cost_per_task_usd=float(d.get("cost_per_task_usd", 0.0)),
                latency_p50_ms=int(d.get("latency_p50_ms", 0)),
                latency_p95_ms=int(d.get("latency_p95_ms", 0)),
                quota_headroom_pct=int(d.get("quota_headroom_pct", 100)),
                task_count=int(d.get("task_count", 0)),
                reference_set=str(d.get("reference_set", "")),
                notes=str(d.get("notes", "")),
            ))
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    return rows


def rows_by_role(rows: list[CapabilityRow]) -> dict[str, list[CapabilityRow]]:
    """Group rows by role, keeping the latest row per (role, provider, model)."""
    latest: dict[tuple[str, str, str], CapabilityRow] = {}
    for r in rows:
        key = (r.role, r.provider, r.model)
        prior = latest.get(key)
        if prior is None or r.ts > prior.ts:
            latest[key] = r
    out: dict[str, list[CapabilityRow]] = {}
    for r in latest.values():
        out.setdefault(r.role, []).append(r)
    for role in out:
        out[role].sort(key=lambda x: x.score, reverse=True)
    return out


def best_for_role(
    role: str,
    *,
    exclude_provider: str | None = None,
    exclude_family: str | None = None,
    family_of_fn: Any = None,
    min_headroom_pct: int = 20,
) -> CapabilityRow | None:
    """Best capability row for `role`, subject to filters.

    Returns the highest-score row for the given role that:
    - has quota_headroom_pct >= min_headroom_pct (can be dispatched without
      hitting a ceiling)
    - if exclude_provider given: skip rows from that provider
    - if exclude_family + family_of_fn given: skip rows whose
      family_of_fn(row.provider, row.model) == exclude_family
      (family_of_fn may accept (provider) or (provider, model) — we
      prefer the slot-aware (provider, model) form to honor explicit
      Qwen-via-NVIDIA / etc. routing.)

    Returns None if no row matches — caller should fall back to the
    static cascade.
    """
    rows = rows_by_role(load_rows())
    candidates = rows.get(role, [])
    for r in candidates:
        if r.quota_headroom_pct < min_headroom_pct:
            continue
        if exclude_provider and r.provider == exclude_provider:
            continue
        if exclude_family and family_of_fn is not None:
            try:
                fam = family_of_fn(r.provider, r.model)
            except TypeError:
                fam = family_of_fn(r.provider)
            if fam == exclude_family:
                continue
        return r
    return None


def append_row(row: CapabilityRow, path: Path | None = None) -> None:
    """Append a new capability measurement. Used by the harness."""
    p = path or MATRIX_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps({
            "ts": row.ts,
            "role": row.role,
            "provider": row.provider,
            "model": row.model,
            "score": row.score,
            "cost_per_task_usd": row.cost_per_task_usd,
            "latency_p50_ms": row.latency_p50_ms,
            "latency_p95_ms": row.latency_p95_ms,
            "quota_headroom_pct": row.quota_headroom_pct,
            "task_count": row.task_count,
            "reference_set": row.reference_set,
            "notes": row.notes,
        }) + "\n")
