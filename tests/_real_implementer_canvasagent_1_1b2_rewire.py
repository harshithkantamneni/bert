"""Phase 1.1b.2 — frontend↔backend rewire.

Connects the existing Vite-bundled React frontend (1.0/1.1a/1.1c) to the
new FastAPI backend (1.1b). Strategy: try-backend-first with graceful
fallback to the existing browser-side DAG executor.

User experience:
  Backend running   →  Run All POSTs to /run, streams SSE node updates,
                       same UI behavior but real Python orchestration
  Backend not running →  fall back to browser-side topoSort+runAgent loop
                         (current 1.1a behavior); user sees a gray
                         "backend offline" dot in the toolbar

Per-node Run button stays browser-side regardless — it's a one-shot
single-prompt call that doesn't need backend orchestration.

Run: PYTHONPATH=. uv run python tests/_real_implementer_canvasagent_1_1b2_rewire.py
"""
from __future__ import annotations

import json
import sys

from core import subagent

DISPATCH = {
    "dispatch_altitude": "IMPL",
    "role": "implementer",
    "cycle": 13,
    "task": (
        "PHASE 1.1b.2 — frontend↔backend rewire.\n\n"
        "READ FIRST:\n"
        "- phase1/canvasagent/src/App.tsx (handleRunAll currently does\n"
        "  browser-side topoSort + runAgent loop)\n"
        "- phase1/canvasagent/src/agent.ts (single-prompt Ollama call)\n"
        "- phase1/canvasagent/src/dag.ts (browser topoSort)\n"
        "- phase1/canvasagent/src/Toolbar.tsx (existing toolbar layout)\n"
        "- phase1/canvasagent/backend/backend/main.py (POST /run endpoint,\n"
        "  GET /healthz, SSE stream format)\n"
        "- phase1/canvasagent/backend/backend/types.py (DAGRequest /\n"
        "  NodeStateUpdate pydantic shape; mirror in TS)\n\n"
        "GOAL: Run All button uses the backend when it's running, falls\n"
        "back to the existing browser-side loop when it's not. Per-node\n"
        "Run button stays browser-side. No breaking changes to existing\n"
        "UI; users see no diff except: (a) Run All is now real Python\n"
        "orchestration when backend is up, (b) a small status dot in\n"
        "the toolbar shows backend connection.\n\n"
        "REQUIREMENTS:\n\n"
        "1. NEW src/backendClient.ts (~80 LoC) — backend HTTP wrapper.\n"
        "   Module API:\n"
        "     export const BACKEND_URL = 'http://127.0.0.1:8765';\n"
        "       (matches main.py's run() default port)\n"
        "     export async function checkBackendHealth(): Promise<boolean>;\n"
        "       (GET /healthz with 1s timeout; true on 200, false otherwise)\n"
        "     export interface NodeStateUpdate {\n"
        "       node_id: string;\n"
        "       state: 'running' | 'done' | 'error';\n"
        "       result?: { node_id: string; text: string; tokens: number;\n"
        "                  latency_ms: number };\n"
        "       error?: string;\n"
        "     }\n"
        "     export async function* runDAGViaBackend(\n"
        "       nodes: Node<AgentNodeData>[], edges: Edge[], model?: string,\n"
        "     ): AsyncGenerator<NodeStateUpdate, void, void>;\n"
        "       (POSTs to /run with {nodes: ReactFlow→DAGNode, edges: ReactFlow→\n"
        "        DAGEdge, model}, parses SSE stream, yields each\n"
        "        NodeStateUpdate. Throws on HTTP error so the caller can\n"
        "        catch + fall back. Each SSE event line is `data: {json}\\n`;\n"
        "        the terminal `event: complete` indicates clean finish.)\n"
        "   Use the native EventSource API for SSE if it works for POST\n"
        "   bodies; if not (EventSource is GET-only), use fetch() + a\n"
        "   manual ReadableStream reader splitting on `\\n\\n`.\n"
        "   (Hint: native EventSource is GET-only; use fetch+ReadableStream.)\n\n"
        "2. UPDATE src/App.tsx — `handleRunAll` becomes:\n"
        "     1. Check backend health once at function start (cached if\n"
        "        recently checked within 5s).\n"
        "     2. If backend up: try runDAGViaBackend; on success, route\n"
        "        each yielded NodeStateUpdate to the same setNodes(...)\n"
        "        update path the existing browser-side loop uses\n"
        "     3. On any backend error (non-200, parse failure, network):\n"
        "        log the error, set a brief 'falling back to browser mode'\n"
        "        flag, then continue with the existing browser-side topoSort\n"
        "        loop unchanged\n"
        "     4. If backend not up: skip the try, go straight to browser\n"
        "        loop\n"
        "   The existing browser-side code stays exactly as 1.1a left it.\n"
        "   Do NOT delete it — it's the fallback.\n\n"
        "3. UPDATE src/Toolbar.tsx — add a small backend-status dot.\n"
        "   Right of the Stats button, render a 8px circle + label:\n"
        "     - Green dot + 'backend' when /healthz returned 200 in last 30s\n"
        "     - Gray dot + 'browser' when /healthz failed\n"
        "     - Animated yellow when health-check is in flight\n"
        "   Health-check polls every 30s when the toolbar is mounted.\n"
        "   Single useEffect with setInterval. Don't poll if window\n"
        "   isn't focused (use document.hidden).\n\n"
        "4. UPDATE src/types.ts — re-export Node<AgentNodeData>-shaped\n"
        "   types if needed for backendClient. Don't duplicate; import\n"
        "   what already exists.\n\n"
        "BUDGET: total LoC across phase1/canvasagent/src/* + config files\n"
        "≤1500 (was 1214 after 1.1c; +286 budget for this dispatch).\n\n"
        "DO NOT touch backend/ at all — this dispatch is frontend-only.\n\n"
        "DO NOT touch agent.ts — per-node Run button keeps using direct\n"
        "Ollama for simplicity. backendClient.ts is the new path; it\n"
        "doesn't replace agent.ts.\n\n"
        "DO write build report to drafts/canvasagent_1_1b2_rewire_report.md\n"
        "with file diff summary, new module API, and a 'Quick verify'\n"
        "section showing how to test:\n"
        "  - With backend off: open localhost:5173, click Run All →\n"
        "    works in browser mode (gray dot)\n"
        "  - With backend on (uvicorn ... --port 8765): refresh,\n"
        "    click Run All → works via backend (green dot, server logs\n"
        "    show /run requests)\n"
        "  - Mid-run, kill backend → next Run All falls back to browser\n"
        "    cleanly with a console warning"
    ),
    "success_criterion": (
        "src/backendClient.ts exists with the listed API. App.tsx "
        "handleRunAll has a try-backend-first / fallback path. Toolbar "
        "renders a status dot. drafts/canvasagent_1_1b2_rewire_report.md "
        "exists. Total LoC ≤1500. Build passes. ResultPacket valid."
    ),
    "output_path": "drafts/canvasagent_1_1b2_rewire_report.md",
    "model": "mistral/mistral-small-latest",
    "process_hygiene": (
        "FALLBACK FIRST. The browser-side path must continue to work "
        "exactly as before when backend is unreachable — that's the "
        "user-visible 'don't break my app' contract. SSE PARSING: split "
        "on `\\n\\n` (event boundary), then look for `data: ` prefix on "
        "any line of each event. Buffer partial events until the \\n\\n. "
        "TYPE PARITY: NodeStateUpdate fields must match backend/types.py "
        "exactly (snake_case, optional result/error). NO TYPE MIXING: "
        "don't widen any to silence TypeScript. PRIVACY: no remote pings "
        "outside agent.ts (existing) and backendClient.ts (new, "
        "localhost-only). Reuse Node/Edge from reactflow."
    ),
    "confidence_required": True,
    "falsifier_text": (
        "Failure if any of: (a) build breaks; (b) backendClient.ts "
        "missing or doesn't export required API; (c) handleRunAll doesn't "
        "fall back to browser when backend errors; (d) status dot stuck "
        "on one state forever; (e) any HTTP host other than 127.0.0.1; "
        "(f) ResultPacket schema-invalid."
    ),
    "verification_command": (
        "set -eo pipefail && cd phase1/canvasagent && "
        "npm run build 2>&1 | tail -5 && "
        "echo '--- privacy hard-check ---' && "
        "BAD=$(grep -lE 'fetch\\(|XMLHttpRequest|sendBeacon|new WebSocket' "
        "  src/*.ts src/*.tsx 2>/dev/null | grep -vE 'src/(agent|backendClient)\\.tsx?$' | wc -l) && "
        "if [ \"$BAD\" -eq 0 ]; then echo 'privacy: OK (network calls only in agent.ts + backendClient.ts)'; "
        "else echo \"privacy: FAIL ($BAD files have remote calls)\" && exit 1; fi && "
        "test -f src/backendClient.ts && echo 'backendClient.ts: present' && "
        "grep -q 'runDAGViaBackend' src/backendClient.ts && echo 'API: runDAGViaBackend exported' && "
        "grep -q 'checkBackendHealth' src/backendClient.ts && echo 'API: checkBackendHealth exported' && "
        "grep -q '127.0.0.1' src/backendClient.ts && echo 'localhost: enforced'"
    ),
    "verification_timeout_secs": 180,
}


def main() -> int:
    print("=" * 72)
    print("PHASE 1.1b.2 — Implementer rewires frontend to backend")
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
        print(f"  stdout tail:\n{verify.get('stdout', '')[-800:]}")
        if verify.get("stderr"):
            print(f"  stderr tail:\n{verify['stderr'][-400:]}")
    else:
        print("  (no verification block)")
    return 0 if summary["spec_valid"] and summary["result_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
