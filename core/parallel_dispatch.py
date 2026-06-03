"""Async parallel dispatch helper for the cycle runner.

Phase C3 of the v3 plan. Groups dispatch_specs into parallelizable
batches + serial batches, then runs each in turn.

The runner (tools/bert_run.py or wherever the cycle loop lives) feeds
this module a list of dispatch_specs; this module returns the list of
results in the same order.

Concurrency model: asyncio.gather for parallelizable groups; sequential
for anything that touches shared write targets. ThreadPoolExecutor
under the hood since subagent.run_subagent is sync — we wrap it in
asyncio.to_thread() so the awaitable surface composes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

LOG = logging.getLogger("bert.parallel_dispatch")


@dataclass
class DispatchResult:
    """One sub-agent's result + ordering metadata."""
    index: int                  # position in original spec list
    spec_role: str
    spec_label: str
    summary: dict               # whatever the runner returns
    elapsed_secs: float
    errors: tuple[str, ...] = ()


def group_by_parallelizable(specs: list[dict]) -> list[list[dict]]:
    """Return list of groups. Each group is either:
      - len > 1 and all `parallelizable: True` → fire concurrently
      - len == 1 → fire sequentially

    Greedy grouping: a span of consecutive parallelizable specs becomes
    one group; a non-parallelizable spec starts a new singleton group.

    Also breaks parallel groups when two specs would write the same
    output_path (which would race).
    """
    groups: list[list[dict]] = []
    current: list[dict] = []
    seen_outputs: set[str] = set()

    for spec in specs:
        is_par = bool(spec.get("parallelizable", False))
        output_path = spec.get("output_path", "")
        # Break group on conflict OR on non-parallelizable
        if not is_par:
            if current:
                groups.append(current)
                current = []
                seen_outputs.clear()
            groups.append([spec])  # singleton
            continue
        # Parallelizable — check for write conflict
        if output_path and output_path in seen_outputs:
            # Conflict — flush current group, start new
            if current:
                groups.append(current)
            current = [spec]
            seen_outputs = {output_path}
        else:
            current.append(spec)
            if output_path:
                seen_outputs.add(output_path)

    if current:
        groups.append(current)
    return groups


async def _run_one_async(
    spec: dict,
    index: int,
    runner_fn: Callable[[dict], dict],
) -> DispatchResult:
    """Wrap a sync runner_fn(spec) → dict in asyncio.to_thread."""
    t0 = time.monotonic()
    try:
        summary = await asyncio.to_thread(runner_fn, spec)
        errors: tuple[str, ...] = ()
    except Exception as exc:  # noqa: BLE001 — every dispatch is opt-fail
        summary = {
            "verdict": "OTHER",
            "result_valid": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
        errors = (f"{type(exc).__name__}: {exc}",)
    return DispatchResult(
        index=index,
        spec_role=spec.get("role", "unknown"),
        spec_label=spec.get("label", spec.get("role", "")),
        summary=summary,
        elapsed_secs=round(time.monotonic() - t0, 2),
        errors=errors,
    )


async def _dispatch_all_async(
    specs: list[dict],
    runner_fn: Callable[[dict], dict],
) -> list[DispatchResult]:
    """The real async work — groups, then runs each group concurrently."""
    groups = group_by_parallelizable(specs)
    flat_index = 0
    results: list[DispatchResult] = []
    for group in groups:
        if len(group) > 1:
            LOG.info("dispatch_parallel: firing %d specs in parallel",
                     len(group))
            tasks = [
                _run_one_async(s, flat_index + i, runner_fn)
                for i, s in enumerate(group)
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            flat_index += len(group)
        else:
            r = await _run_one_async(group[0], flat_index, runner_fn)
            results.append(r)
            flat_index += 1
    return results


def dispatch_all(
    specs: list[dict],
    runner_fn: Callable[[dict], dict],
) -> list[DispatchResult]:
    """Synchronous entry point — the cycle loop calls this. Internally
    uses asyncio for parallel groups.

    `runner_fn` is the sync function that takes a spec dict and returns
    a result summary dict (e.g., subagent.run_subagent).

    Returns DispatchResult list in the SAME ORDER as input specs.
    """
    if not specs:
        return []
    # Single-spec fast path — no asyncio overhead
    if len(specs) == 1:
        return asyncio.run(_dispatch_all_async(specs, runner_fn))
    # Multi-spec — go async
    return asyncio.run(_dispatch_all_async(specs, runner_fn))
