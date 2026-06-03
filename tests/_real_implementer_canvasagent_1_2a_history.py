"""Phase 1.2a — run history (backend store + frontend sidebar).

Adds the foundation for Phase 1.2 replay+branch: persistent (in-memory
for now) run store on the backend, sidebar UI for listing and inspecting
past runs. Canvas-replay scrubbing is deferred to 1.2b — keeps this
dispatch focused.

Architecture target:

  Backend (phase1/canvasagent/backend/):
    backend/store.py        — RunStore class with FIFO cap (50 runs);
                              records each run's metadata + event list;
                              thread-safe for FastAPI's async loop
    backend/main.py         — POST /run now returns run_id in the first
                              SSE event ({event: "started", data:
                              {run_id, started_at}}); GET /runs lists
                              recent runs (id, started_at, ended_at,
                              node_count, verdict); GET /runs/{id}
                              returns full event list
    backend/types.py        — RunSummary, RunDetail pydantic models
    tests/test_store.py     — append/list/get/cap behavior

  Frontend (phase1/canvasagent/src/):
    src/RunHistory.tsx      — fixed-position sidebar (right side); shows
                              recent runs as cards (verdict pill, time
                              ago, node count, model, click to expand);
                              expanded view shows event timeline as text
                              with [hh:mm:ss] node_id state lines
    src/backendClient.ts    — extended with listRuns() + getRun(id)
                              (GET endpoints; null on backend offline so
                              UI shows graceful empty state)
    src/Toolbar.tsx         — new "History" toggle button (between
                              Run All and Stats)
    src/App.tsx             — captures run_id from the first SSE event;
                              passes onHistoryToggle prop; holds
                              showHistory state; on Run All success,
                              triggers a refetch of the runs list

Run: PYTHONPATH=. uv run python tests/_real_implementer_canvasagent_1_2a_history.py
"""
from __future__ import annotations

import json
import sys

from core import subagent

DISPATCH = {
    "dispatch_altitude": "IMPL",
    "role": "implementer",
    "cycle": 14,
    "task": (
        "PHASE 1.2a — run history. Backend run store + frontend sidebar.\n\n"
        "READ FIRST:\n"
        "- phase1/canvasagent/backend/backend/main.py (existing /run + SSE)\n"
        "- phase1/canvasagent/backend/backend/types.py (DAGRequest +\n"
        "  NodeStateUpdate)\n"
        "- phase1/canvasagent/backend/backend/dag.py (execute_dag emits\n"
        "  NodeStateUpdate via callback; we'll capture those events)\n"
        "- phase1/canvasagent/src/backendClient.ts (existing fetch + SSE)\n"
        "- phase1/canvasagent/src/App.tsx (handleRunAll's SSE consumer)\n"
        "- phase1/canvasagent/src/Toolbar.tsx (existing button layout)\n\n"
        "GOAL: every Run All invocation produces a stored run; users can\n"
        "see a sidebar listing recent runs with verdict + timing; click\n"
        "to expand and see the event timeline. No canvas replay yet —\n"
        "that's 1.2b.\n\n"
        "BACKEND CHANGES:\n\n"
        "1. NEW backend/store.py (~80 LoC):\n"
        "   class RunStore:\n"
        "       def __init__(self, cap: int = 50) -> None\n"
        "       def start_run(self, request: DAGRequest) -> str  # returns run_id\n"
        "       def append_event(self, run_id: str, event: NodeStateUpdate) -> None\n"
        "       def end_run(self, run_id: str, verdict: str) -> None\n"
        "       def list_runs(self) -> list[RunSummary]\n"
        "       def get_run(self, run_id: str) -> RunDetail | None\n"
        "   Threadsafe via asyncio.Lock around mutations. FIFO cap means\n"
        "   inserting run #51 evicts run #1. run_id format: timestamp-\n"
        "   prefixed UUIDv4 short ('20260506T024500-a3f2c1') so listings\n"
        "   sort naturally.\n\n"
        "2. ADD to backend/types.py:\n"
        "   class RunSummary(BaseModel):\n"
        "       run_id: str; started_at: float; ended_at: float | None = None\n"
        "       node_count: int; model: str; verdict: str | None = None\n"
        "   class RunDetail(BaseModel):\n"
        "       summary: RunSummary; events: list[NodeStateUpdate]\n"
        "       request: DAGRequest  # the original input\n\n"
        "3. UPDATE backend/main.py:\n"
        "   - Module-level singleton: STORE = RunStore(cap=50)\n"
        "   - POST /run now: calls STORE.start_run(request) → run_id;\n"
        "     yields a 'started' SSE event with {run_id, started_at}\n"
        "     BEFORE invoking execute_dag; in execute_dag's emit\n"
        "     callback, also calls STORE.append_event; on completion,\n"
        "     STORE.end_run(run_id, verdict='ok' | 'error') and yields\n"
        "     terminal 'complete' event with {run_id, summary}\n"
        "   - NEW GET /runs returns list[RunSummary] (newest first)\n"
        "   - NEW GET /runs/{run_id} returns RunDetail or 404\n"
        "   CORS already set for localhost:5173 — no change needed.\n\n"
        "4. NEW tests/test_store.py (5+ tests):\n"
        "   - test_start_returns_unique_ids\n"
        "   - test_append_and_get_roundtrip\n"
        "   - test_list_returns_newest_first\n"
        "   - test_cap_evicts_oldest\n"
        "   - test_get_unknown_returns_none\n"
        "   - test_concurrent_appends (asyncio gather of 100 appends\n"
        "     to one run; assert count == 100)\n\n"
        "FRONTEND CHANGES:\n\n"
        "5. EXTEND src/backendClient.ts (~50 LoC added):\n"
        "   export interface RunSummary { ... mirror backend ... }\n"
        "   export interface RunDetail { ... mirror backend ... }\n"
        "   export async function listRuns(): Promise<RunSummary[] | null>;\n"
        "     (null when backend offline; empty array when no runs)\n"
        "   export async function getRun(runId: string):\n"
        "     Promise<RunDetail | null>;\n"
        "   Both use 1s-2s timeouts. Convert backend snake_case to TS\n"
        "   camelCase or keep snake_case — pick one and be consistent\n"
        "   (recommend keeping snake_case so types match backend exactly,\n"
        "   matches existing NodeStateUpdate convention).\n\n"
        "6. NEW src/RunHistory.tsx (~140 LoC):\n"
        "   Props: { open: boolean; onClose: () => void; refetchTrigger: number }\n"
        "   Fixed-position sidebar on the RIGHT (300px wide, full height).\n"
        "   On mount + when refetchTrigger changes: fetch listRuns().\n"
        "   Render:\n"
        "     - Header: 'Run history' + close (×) button\n"
        "     - List of run cards (newest top), each showing:\n"
        "       * verdict pill (green=ok, red=error, yellow=running)\n"
        "       * 'X minutes ago' (use Intl.RelativeTimeFormat)\n"
        "       * '{node_count} nodes · {model}'\n"
        "       * click handler that fetches getRun(id) and expands\n"
        "         inline to show the event timeline\n"
        "     - Empty state when listRuns()=== [] : 'No runs yet —\n"
        "       click Run All to record one'\n"
        "     - Backend-offline state when listRuns()===null:\n"
        "       'Backend offline — history requires the Python server'\n"
        "   Event timeline (in expanded card): each event as one line:\n"
        "     [03:24:51] node_3 done (3.2s, 142 tokens)\n"
        "     [03:24:54] node_5 error: timed out\n"
        "   Scroll the list (overflow: auto). Style consistent with\n"
        "   MetricsPanel.\n\n"
        "7. UPDATE src/Toolbar.tsx:\n"
        "   - New 'History' button between Run All and Stats\n"
        "   - Pass onHistoryToggle callback prop from parent\n"
        "   - Disabled while isRunning (same as other action buttons)\n\n"
        "8. UPDATE src/App.tsx:\n"
        "   - showHistory state + setShowHistory\n"
        "   - historyRefetchTrigger state (incremented after every\n"
        "     successful Run All to nudge RunHistory to refetch)\n"
        "   - When SSE yields a 'started' event with run_id (via the\n"
        "     extended runDAGViaBackend signature, see #5b below),\n"
        "     stash run_id locally for analytics; when the run finishes,\n"
        "     bump historyRefetchTrigger\n"
        "   - Render <RunHistory open={showHistory} onClose=...\n"
        "     refetchTrigger={historyRefetchTrigger} /> alongside the\n"
        "     Toolbar / ReactFlow / MetricsPanel\n"
        "   - Pass onHistoryToggle to Toolbar\n\n"
        "5b. EXTEND runDAGViaBackend's yielded events to include the\n"
        "    'started' SSE event so the frontend can capture run_id.\n"
        "    Easiest: change return type to AsyncGenerator yielding a\n"
        "    discriminated union { type: 'started'; run_id: string;\n"
        "    started_at: number } | { type: 'state'; ...NodeStateUpdate\n"
        "    fields }; consumers in App.tsx switch on type.\n\n"
        "BUDGET: total LoC across frontend src + backend (excl tests +\n"
        "lock files) ≤2100. Was 1447 + ~290 backend = ~1740 after 1.1b.2;\n"
        "+360 budget for this dispatch (most goes to RunHistory.tsx).\n\n"
        "DO NOT touch tests/test_dag.py — that suite stays.\n"
        "DO NOT add storage persistence to disk yet — in-memory FIFO is\n"
        "the MVP. SQLite can come in a later dispatch if PMF demands it.\n\n"
        "DO write build report to drafts/canvasagent_1_2a_history_report.md\n"
        "with file diff summary, new endpoints + their curl examples, and\n"
        "a 'Quick Verify' that walks the user through:\n"
        "  - Start backend, frontend, click Run All, History sidebar\n"
        "    appears with 1 run\n"
        "  - Click run → expanded timeline\n"
        "  - Run All again → list grows to 2"
    ),
    "success_criterion": (
        "Backend: store.py + types.py + main.py all compile, pytest "
        "passes (test_dag.py + test_store.py, all green). Frontend: "
        "RunHistory.tsx + extended backendClient.ts + updated Toolbar/App, "
        "build passes. Total LoC ≤2100. ResultPacket valid."
    ),
    "output_path": "drafts/canvasagent_1_2a_history_report.md",
    "model": "mistral/mistral-small-latest",
    "process_hygiene": (
        "BEHAVIOR PARITY: existing POST /run callers (1.1b.2 frontend) "
        "must keep working unchanged. The new 'started' SSE event is "
        "additive; the existing NodeStateUpdate stream behaves exactly "
        "as before. PRIVACY: no remote calls; backend is still localhost-"
        "only. THREAD SAFETY: RunStore mutations under asyncio.Lock since "
        "FastAPI handles concurrent /run requests. NO PERSISTENCE: in-"
        "memory only; restarting the backend wipes runs. That's the MVP "
        "trade. PYDANTIC v2 syntax."
    ),
    "confidence_required": True,
    "falsifier_text": (
        "Failure if any of: (a) test_dag.py regression (any of the 6 "
        "existing tests fails); (b) test_store.py has <5 tests; (c) "
        "backend build/import breaks; (d) frontend build breaks; (e) "
        "any HTTP host other than 127.0.0.1; (f) total LoC >2100; (g) "
        "ResultPacket schema-invalid."
    ),
    "verification_command": (
        "set -eo pipefail && "
        "cd phase1/canvasagent/backend && "
        "uv sync --extra dev --quiet 2>&1 | tail -5 && "
        "uv run python -m pytest -q 2>&1 | tail -5 && "
        "cd ../.. && cd phase1/canvasagent && "
        "npm run build 2>&1 | tail -5 && "
        "echo '--- privacy hard-check ---' && "
        "BAD=$(grep -lE 'fetch\\(|XMLHttpRequest|sendBeacon|new WebSocket' "
        "  src/*.ts src/*.tsx 2>/dev/null | grep -vE 'src/(agent|backendClient)\\.tsx?$' | wc -l) && "
        "if [ \"$BAD\" -eq 0 ]; then echo 'privacy: OK'; "
        "else echo \"privacy: FAIL ($BAD)\" && exit 1; fi && "
        "test -f src/RunHistory.tsx && echo 'RunHistory: present' && "
        "grep -q 'listRuns' src/backendClient.ts && echo 'API: listRuns' && "
        "grep -q 'getRun' src/backendClient.ts && echo 'API: getRun'"
    ),
    "verification_timeout_secs": 480,
}


def main() -> int:
    print("=" * 72)
    print("PHASE 1.2a — Implementer adds run history (backend + frontend)")
    print("=" * 72)
    print()

    summary = subagent.run_subagent(DISPATCH)

    print()
    print("=" * 72)
    print("Implementer return")
    print("=" * 72)
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "calibration_reasoning"},
                     indent=2, default=str))
    print()
    print("--- calibration_reasoning ---")
    print(summary.get("calibration_reasoning", "")[:1500])
    print()
    print("--- verification ---")
    tel = summary.get("telemetry", {})
    verify = tel.get("verification") if isinstance(tel, dict) else None
    if verify:
        print(f"  ok={verify['ok']} exit={verify['exit_code']} elapsed={verify['elapsed_ms']}ms")
        print(f"  stdout tail:\n{verify.get('stdout', '')[-1000:]}")
        if verify.get("stderr"):
            print(f"  stderr tail:\n{verify['stderr'][-400:]}")
    else:
        print("  (no verification block)")
    return 0 if summary["spec_valid"] and summary["result_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
