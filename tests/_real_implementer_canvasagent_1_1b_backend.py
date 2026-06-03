"""Phase 1.1b — LangGraph backend (Python server).

Stands up a self-contained Python server that orchestrates DAG execution
via LangGraph. Frontend will rewire in a follow-on dispatch (1.1b.2);
this dispatch focuses on a working backend with a curlable HTTP API.

Architecture target:
  phase1/canvasagent/backend/
    pyproject.toml       # uv-managed; deps: fastapi, uvicorn, langgraph, httpx, pydantic
    backend/
      __init__.py
      main.py            # FastAPI app, GET /healthz, POST /run (SSE), GET /v1/models
      dag.py             # ReactFlow nodes+edges → LangGraph StateGraph; topo execution
      ollama_client.py   # thin httpx wrapper around http://127.0.0.1:11434
      types.py           # pydantic models for DAGRequest / NodeUpdate / RunResult
    tests/
      test_dag.py        # in-process tests: DAG topology, single-node run, multi-node propagation
    README.md            # install + run + curl test
    .gitignore           # .venv, __pycache__, .pytest_cache

The backend is the future replacement for browser-side DAG execution but
ships side-by-side with it for now. Verification: uv-builds, pytest passes,
GET /healthz returns 200.

Run: PYTHONPATH=. uv run python tests/_real_implementer_canvasagent_1_1b_backend.py
"""
from __future__ import annotations

import json
import sys

from core import subagent

DISPATCH = {
    "dispatch_altitude": "IMPL",
    "role": "implementer",
    "cycle": 11,
    "task": (
        "PHASE 1.1b BACKEND — scaffold a self-contained Python server\n"
        "for CanvasAgent DAG execution. Self-contained means: own\n"
        "pyproject.toml under phase1/canvasagent/backend/, NOT shared\n"
        "with the bert-lab parent pyproject. Future bert-lab deploys\n"
        "this backend as part of the CanvasAgent product, not as part\n"
        "of the bert-lab orchestrator itself.\n\n"
        "READ FIRST:\n"
        "- memories/mission.md (full mission spec)\n"
        "- phase1/canvasagent/src/dag.ts (current browser-side topo sort —\n"
        "  match its semantics so frontend can swap with no behavior change)\n"
        "- phase1/canvasagent/src/agent.ts (current Ollama wrapper —\n"
        "  match request/response shape)\n"
        "- phase1/canvasagent/src/types.ts (RunResult shape)\n\n"
        "FILES TO CREATE under phase1/canvasagent/backend/ via the Write\n"
        "tool. Use Bash + uv only for venv/install verification, NOT for\n"
        "creating files (Write tool is the source of truth).\n\n"
        "1. pyproject.toml — uv-managed Python ≥3.12 project. Deps:\n"
        "     fastapi >= 0.115\n"
        "     uvicorn[standard] >= 0.30\n"
        "     httpx >= 0.27\n"
        "     pydantic >= 2.7\n"
        "     langgraph >= 0.2\n"
        "   Dev-deps: pytest >= 8.3, pytest-asyncio >= 0.23, httpx (already\n"
        "   in main; test client uses it). Add a `[project.scripts]` entry:\n"
        "     canvasagent-backend = \"backend.main:run\"\n"
        "   Add `[tool.hatch.build.targets.wheel] packages = [\"backend\"]`.\n\n"
        "2. backend/__init__.py — empty (just makes it a package)\n\n"
        "3. backend/types.py — pydantic models:\n"
        "     class NodePosition(BaseModel): x: float; y: float\n"
        "     class NodeData(BaseModel):\n"
        "       prompt: str; state: str = 'idle'; output: dict | None = None\n"
        "     class DAGNode(BaseModel):\n"
        "       id: str; type: str = 'agentNode'\n"
        "       position: NodePosition; data: NodeData\n"
        "     class DAGEdge(BaseModel):\n"
        "       id: str; source: str; target: str\n"
        "       sourceHandle: str | None = None\n"
        "       targetHandle: str | None = None\n"
        "     class DAGRequest(BaseModel):\n"
        "       nodes: list[DAGNode]; edges: list[DAGEdge]\n"
        "       model: str = 'llama3.2:3b'\n"
        "     class NodeRunResult(BaseModel):\n"
        "       node_id: str; text: str; tokens: int; latency_ms: float\n"
        "     class NodeStateUpdate(BaseModel):\n"
        "       node_id: str\n"
        "       state: str  # 'running' | 'done' | 'error'\n"
        "       result: NodeRunResult | None = None\n"
        "       error: str | None = None\n\n"
        "4. backend/ollama_client.py — async httpx client. Single function:\n"
        "     async def generate(prompt: str, model: str = 'llama3.2:3b',\n"
        "                        host: str = 'http://127.0.0.1:11434') -> dict\n"
        "   POSTs to {host}/api/generate with stream=False, returns\n"
        "   {'text': str, 'tokens': int, 'latency_ms': float}. Raises\n"
        "   RuntimeError on non-2xx with status code in message.\n\n"
        "5. backend/dag.py — DAG execution engine. Public API:\n"
        "     def topo_sort(nodes: list[DAGNode],\n"
        "                   edges: list[DAGEdge]) -> list[str]\n"
        "       (Kahn's algo; raises ValueError on cycle. Same semantics as\n"
        "        phase1/canvasagent/src/dag.ts so frontend swap is\n"
        "        behavior-preserving.)\n"
        "     async def execute_dag(\n"
        "         request: DAGRequest,\n"
        "         emit: Callable[[NodeStateUpdate], Awaitable[None]],\n"
        "     ) -> dict[str, NodeRunResult]\n"
        "       (walks topo order; for each node, computes effective_prompt\n"
        "        as `{upstream_outputs}\\n\\n{node.data.prompt}` (joined by\n"
        "        \\n\\n if multiple upstream); emits 'running' update,\n"
        "        calls ollama_client.generate, emits 'done' or 'error'\n"
        "        update; returns dict mapping node_id → NodeRunResult.\n"
        "        On error, marks node failed but continues with non-\n"
        "        downstream nodes — same partial-DAG semantics as 1.1a.)\n\n"
        "6. backend/main.py — FastAPI app. Endpoints:\n"
        "     GET /healthz → {'ok': True, 'version': '0.1.0'}\n"
        "     GET /v1/models → proxies http://127.0.0.1:11434/api/tags\n"
        "       and returns the model list. Returns 503 if Ollama unreachable.\n"
        "     POST /run → accepts DAGRequest as JSON. Returns Server-Sent\n"
        "       Events stream where each event is a NodeStateUpdate JSON.\n"
        "       Final event is `event: complete\\ndata: {summary}\\n\\n`.\n"
        "       Use FastAPI's StreamingResponse with media_type='text/event-stream'.\n"
        "     CORS: allow http://localhost:5173 and 127.0.0.1:5173 origins\n"
        "       so the Vite dev server can call us.\n"
        "   Define `def run() -> None` that calls\n"
        "     uvicorn.run('backend.main:app', host='127.0.0.1', port=8765,\n"
        "                  reload=False, log_level='info')\n"
        "   so `canvasagent-backend` console script works.\n\n"
        "7. tests/test_dag.py — pytest tests:\n"
        "     - test_topo_sort_linear: 3-node chain a→b→c returns [a, b, c]\n"
        "     - test_topo_sort_branch: a→b, a→c returns [a, b, c] in some\n"
        "       order with a first\n"
        "     - test_topo_sort_cycle_raises: a→b, b→a raises ValueError\n"
        "     - test_topo_sort_disconnected: 2 isolated nodes both appear\n"
        "     - test_execute_dag_single_node_happy: mock ollama_client.generate,\n"
        "       run a 1-node DAG, assert one 'running' + one 'done' update\n"
        "       and result text matches mock\n"
        "     - test_execute_dag_propagation: 2-node chain, mock ollama;\n"
        "       assert downstream node's effective_prompt contains upstream\n"
        "       output\n"
        "   Use pytest-asyncio with `@pytest.mark.asyncio` on async tests.\n"
        "   Mock ollama via monkeypatching backend.ollama_client.generate.\n\n"
        "8. README.md — title, what it is, install (`uv sync`), run\n"
        "   (`uv run uvicorn backend.main:app --port 8765`), curl test,\n"
        "   pytest run, architecture note (one paragraph), pointer to\n"
        "   `memories/mission.md` for falsifiers.\n\n"
        "9. .gitignore — .venv, __pycache__, .pytest_cache, .ruff_cache,\n"
        "   *.pyc, dist, build, *.egg-info, uv.lock (debatable; keep in\n"
        "   for now since it's a standalone subproject)\n\n"
        "BUDGET: ≤600 LoC across all .py files (excluding generated/lock).\n\n"
        "DO NOT touch any phase1/canvasagent/src/* file — frontend rewire\n"
        "is a separate dispatch (1.1b.2). This is backend-only.\n\n"
        "DO write build report to drafts/canvasagent_1_1b_backend_report.md\n"
        "with file list, LoC, install command, dev-server command, curl\n"
        "examples for /healthz + /v1/models + /run."
    ),
    "success_criterion": (
        "All 9 files exist under phase1/canvasagent/backend/. uv sync "
        "succeeds. pytest passes (4+ tests green). uvicorn starts, "
        "GET /healthz returns 200. Total LoC ≤600 across .py files."
    ),
    "output_path": "drafts/canvasagent_1_1b_backend_report.md",
    "model": "mistral/mistral-small-latest",
    "process_hygiene": (
        "BEHAVIOR PARITY: topo_sort + execute_dag must match the JS in "
        "src/dag.ts and src/App.tsx so the frontend swap is invisible to "
        "the user. Read those files first, then mirror semantics. "
        "PRIVACY: backend ONLY talks to localhost:11434 (Ollama). NO "
        "external HTTP. ASYNC: ollama_client and execute_dag must be "
        "async; the SSE stream depends on it. ERROR HANDLING: any node "
        "that errors marks itself failed and the DAG continues for "
        "non-downstream nodes (partial DAG semantics, NOT halt-on-first-"
        "error). PYDANTIC v2 syntax (model_dump not dict; field_validator "
        "not validator)."
    ),
    "confidence_required": True,
    "falsifier_text": (
        "Failure if any of: (a) any of the 9 files missing; (b) uv sync "
        "fails; (c) pytest fails or runs <4 tests; (d) GET /healthz "
        "returns non-200 after server start; (e) total .py LoC >600; "
        "(f) any HTTP call to a host other than 127.0.0.1; (g) ResultPacket "
        "schema-invalid."
    ),
    # set -o pipefail is critical: without it, `pytest | tail` swallows pytest's
    # nonzero exit and verification reports ok=true even with failing tests.
    # `python -m pytest` (rather than bare `pytest`) ensures we use the
    # in-venv pytest, not whatever happens to be on $PATH.
    "verification_command": (
        "set -eo pipefail && cd phase1/canvasagent/backend && "
        "uv sync --extra dev --quiet 2>&1 | tail -5 && "
        "uv run python -m pytest -q 2>&1 | tail -10 && "
        "uv run python -m uvicorn backend.main:app --host 127.0.0.1 --port 8766 "
        "  --log-level error > /tmp/canvasagent_backend.log 2>&1 & "
        "PID=$! && sleep 3 && "
        "STATUS=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8766/healthz) && "
        "kill $PID 2>/dev/null; wait $PID 2>/dev/null; "
        "echo \"healthz_status=$STATUS\" && [ \"$STATUS\" = \"200\" ]"
    ),
    "verification_timeout_secs": 480,
}


def main() -> int:
    print("=" * 72)
    print("PHASE 1.1b — Implementer scaffolds CanvasAgent LangGraph backend")
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
        print(f"  stdout tail: {verify.get('stdout', '')[-600:]}")
        if verify.get("stderr"):
            print(f"  stderr tail: {verify['stderr'][-400:]}")
    else:
        print("  (no verification block)")
    return 0 if summary["spec_valid"] and summary["result_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
