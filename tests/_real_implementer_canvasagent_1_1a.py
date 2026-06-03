"""Phase 1.1a — multi-agent wiring (frontend-only DAG execution).

Builds on the v1.0 hello-canvas scaffold (now committed). Adds:
  - "+ Add Node" toolbar button (spawn a new AgentNode at offset position)
  - Per-node state badge: idle / running / done / error
  - Edge propagation: when a node completes, its output flows into the
    downstream node's prompt context (input/output handles + state machine)
  - "Run All" button: topological DAG execution, root nodes first
  - Persist canvas state to localStorage so reloads don't lose work

Still no backend — frontend orchestrates the DAG by calling Ollama once
per node in topological order. LangGraph backend lands in separate 1.1b.

Constraint: total LoC across phase1/canvasagent/src/* + config ≤700
(was 320 after 1.0; +380 budget for this dispatch).

Run: PYTHONPATH=. uv run python tests/_real_implementer_canvasagent_1_1a.py
"""
from __future__ import annotations

import json
import sys

from core import subagent

DISPATCH = {
    "dispatch_altitude": "IMPL",
    "role": "implementer",
    "cycle": 9,
    "task": (
        "PHASE 1.1a — multi-agent wiring on top of the v1.0 scaffold.\n\n"
        "READ FIRST: phase1/canvasagent/src/App.tsx, AgentNode.tsx, "
        "agent.ts, types.ts. The scaffold renders one starter node with a "
        "prompt + Run button; output is shown inline. We're now extending "
        "to a real multi-agent DAG.\n\n"
        "REQUIREMENTS:\n\n"
        "1. ADD-NODE TOOLBAR — A small toolbar fixed to the top-left of the "
        "viewport (above the canvas, z-index above ReactFlow) with:\n"
        "   - A '+ Add Node' button that spawns a new AgentNode at an offset "
        "     position (last-spawned-position + (60, 60) so they don't pile up)\n"
        "   - A 'Run All' button (described in #4)\n"
        "   - A 'Clear' button that resets to {one starter node, no edges}\n\n"
        "2. PER-NODE STATE BADGE — Each AgentNode shows a small pill in "
        "the top-right showing its current state:\n"
        "   - 'idle' (gray) — never run since last reset\n"
        "   - 'running' (yellow) — Ollama call in flight\n"
        "   - 'done' (green) — last run succeeded; pill shows token + latency\n"
        "   - 'error' (red) — last run threw; pill shows error preview\n\n"
        "3. INPUT HANDLE + EDGE PROPAGATION — Add a target Handle on the "
        "left of each AgentNode (currently only has a source Handle on right). "
        "When a node runs:\n"
        "   - If it has incoming edges: its effective prompt is "
        "     `{upstream output text}\\n\\n{node prompt}` (concatenated;\n"
        "     this is the simplest useful semantic for v1.1a — refine later)\n"
        "   - On completion, its output is exposed via the source Handle for\n"
        "     downstream nodes to pull when they run\n"
        "   - Store output in node.data.output as before; also expose via\n"
        "     the global ReactFlow nodes state so children can read it\n\n"
        "4. RUN-ALL DAG EXECUTION — The 'Run All' button:\n"
        "   - Computes topological order of the current graph (Kahn's algo "
        "     or simple DFS). Edges define dependency direction.\n"
        "   - Walks topologically, calling runAgent(effectivePrompt) per node\n"
        "   - Updates per-node state as each step transitions\n"
        "   - On any node error, marks it 'error' but continues for nodes\n"
        "     not downstream of the failure (graceful partial DAG)\n"
        "   - Disabled while any node is 'running' to avoid concurrent runs\n\n"
        "5. PERSIST TO LOCALSTORAGE — Save {nodes, edges} to "
        "localStorage('canvasagent.dag.v1') on every change. On app mount, "
        "restore if present, else fall back to the v1.0 single-starter-node "
        "default. Add a 'Reset' option to the toolbar that clears storage.\n\n"
        "FILES TO TOUCH:\n"
        "- src/App.tsx — add toolbar (probably as a separate Toolbar.tsx),\n"
        "  wire onConnect / onNodesChange / onEdgesChange to localStorage,\n"
        "  add Run All handler, add new node spawning logic\n"
        "- src/AgentNode.tsx — add target Handle on left, add state badge,\n"
        "  surface effective prompt logic (read upstream output if any),\n"
        "  expose state transitions through node.data updates\n"
        "- src/types.ts — add NodeState enum/union, expand AgentNodeData\n"
        "- src/agent.ts — minor: surface error vs success cleanly\n"
        "- NEW src/Toolbar.tsx — the fixed-position toolbar component\n"
        "- NEW src/dag.ts — pure topological-sort helper (Kahn's algo)\n"
        "  with one exported function: topoSort(nodes, edges): NodeId[]\n"
        "- NEW src/storage.ts — localStorage save/load wrapper with\n"
        "  versioning ('canvasagent.dag.v1' key, ignore on schema mismatch)\n\n"
        "BUDGET: total non-blank LoC across phase1/canvasagent/src/*.{ts,tsx,css} "
        "+ config files ≤ 700. v1.0 was 320, leaves +380 for this dispatch.\n\n"
        "DO: write the build report to drafts/canvasagent_1_1a_build_report.md\n"
        "with file diff summary, total LoC, and a Quick Start showing the\n"
        "user how to test multi-agent wiring (steps to add 2 nodes, wire\n"
        "them, click Run All, observe propagation).\n\n"
        "DO NOT: run npm install or npm run build. Just edit code. The\n"
        "harness will verify build separately."
    ),
    "success_criterion": (
        "All 6 file changes/additions present (App.tsx, AgentNode.tsx, "
        "types.ts, agent.ts, Toolbar.tsx, dag.ts, storage.ts). Total LoC "
        "≤ 700. drafts/canvasagent_1_1a_build_report.md exists with file "
        "list, LoC, Quick Start. ResultPacket schema-validates."
    ),
    "output_path": "drafts/canvasagent_1_1a_build_report.md",
    "model": "mistral/mistral-small-latest",
    "process_hygiene": (
        "Atomic edits — every file should compile after this dispatch. No "
        "half-implemented features. The dag.ts topoSort must handle: empty "
        "graph, single node, disconnected nodes (each becomes its own root), "
        "and reject cyclic graphs by returning an error rather than infinite "
        "loop. localStorage save must be debounced or onChange (avoid "
        "storage-quota issues with rapid drag events). Use React Flow's "
        "existing Connection / Edge / Node types — don't re-type them."
    ),
    "confidence_required": True,
    "falsifier_text": (
        "Failure if any of: (a) any of the 7 src files missing or unparseable; "
        "(b) total LoC > 700; (c) topoSort rejects valid DAGs or accepts "
        "cyclic graphs; (d) Run All doesn't propagate upstream output to "
        "downstream prompts; (e) localStorage breaks on schema-version "
        "mismatch instead of falling back; (f) ResultPacket schema-invalid."
    ),
    "verification_command": "set -eo pipefail && cd phase1/canvasagent && npm run build 2>&1 | tail -8",
    "verification_timeout_secs": 180,
}


def main() -> int:
    print("=" * 72)
    print("PHASE 1.1a — Implementer extends CanvasAgent: multi-agent wiring")
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
    return 0 if summary["spec_valid"] and summary["result_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
