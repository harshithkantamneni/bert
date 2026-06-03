"""Phase 1.0 — Implementer dispatch: scaffold CanvasAgent v1.0 hello-canvas.

Goal: produce a buildable React Flow + Vite + TypeScript scaffold under
`phase1/canvasagent/` with a single demo node that calls Ollama and
renders the structured response as a child node. ≤500 LoC across all
src files. README explains npm install && npm run dev workflow.

Output_path is in drafts/ (the dispatch schema doesn't allow phase1/),
but the Implementer will use Write tool freely to create the actual
project files under phase1/canvasagent/. The drafts/ file is a build
report listing what got scaffolded.

Run: PYTHONPATH=. uv run python tests/_real_implementer_canvasagent_1_0.py
"""
from __future__ import annotations

import json
import sys

from core import subagent

DISPATCH = {
    "dispatch_altitude": "IMPL",
    "role": "implementer",
    "cycle": 8,
    "task": (
        "PHASE 1.0 SCAFFOLD — CanvasAgent v1.0 hello-canvas prototype.\n\n"
        "Read mission spec at memories/mission.md (CanvasAgent v1, locked\n"
        "stack: TypeScript + React Flow + Python + LangGraph + Ollama +\n"
        "Electron). Phase 1.0 target is ≤500 LoC, builds in a browser, no\n"
        "Electron yet — that ships in 1.3.\n\n"
        "Scaffold these files under phase1/canvasagent/ via the Write tool:\n"
        "1. package.json — Vite + React 18 + TypeScript + reactflow.\n"
        "   Scripts: 'dev' (vite), 'build' (tsc + vite build), 'preview'.\n"
        "2. tsconfig.json — standard React+Vite TypeScript config.\n"
        "3. vite.config.ts — minimal Vite config with React plugin.\n"
        "4. index.html — single-page entry mounting #root.\n"
        "5. src/main.tsx — React entry, mounts App.\n"
        "6. src/App.tsx — top-level layout: full-screen ReactFlow canvas\n"
        "   with one starter 'AgentNode' (custom node type) prewired.\n"
        "7. src/AgentNode.tsx — custom React Flow node component:\n"
        "     - Input: a prompt textarea inside the node\n"
        "     - 'Run' button that POSTs to http://127.0.0.1:11434/api/generate\n"
        "       with model='llama3.2:3b' (configurable), prompt=user text,\n"
        "       stream=false\n"
        "     - On response: spawns a child node containing the structured\n"
        "       output (raw text + token count + latency_ms)\n"
        "     - Connects parent → child via a React Flow edge automatically\n"
        "8. src/agent.ts — Ollama client wrapper. Single function\n"
        "   `runAgent(prompt: string, model?: string): Promise<{text, tokens, latency_ms}>`.\n"
        "   No frameworks; just fetch().\n"
        "9. src/types.ts — AgentNodeData / RunResult interfaces.\n"
        "10. src/index.css — minimal styling so the canvas fills the viewport\n"
        "    and nodes look clean (background, borders, font).\n"
        "11. README.md — title, screenshot placeholder, prereqs (Node 20+,\n"
        "    Ollama running locally with llama3.2:3b pulled), install\n"
        "    + dev commands, brief architecture note. End with 'Mission:\n"
        "    `memories/mission.md`. Falsifiers: FALS-CANVASAGENT-{01,02,03}.'\n"
        "12. .gitignore — node_modules, dist, .DS_Store.\n\n"
        "Constraints:\n"
        "- Total LoC across src/ + config files ≤500. Counts: rough wc -l.\n"
        "- All TypeScript strict-mode-clean. No `any` unless commented why.\n"
        "- No backend yet (Phase 1.1+) — agent runs entirely in browser\n"
        "  via fetch to user's local Ollama. Document the CORS workaround\n"
        "  in README (OLLAMA_ORIGINS=* or running browser with --disable-\n"
        "  web-security; pick the README-correct option).\n"
        "- Use React Flow's <Background> + <Controls> for polish out of box.\n"
        "- Initial canvas should have ONE AgentNode pre-placed at (200, 200)\n"
        "  with a default prompt 'Summarize React Flow in one sentence.'\n"
        "  so demo works on first load.\n\n"
        "Do NOT npm install. Just scaffold. Do not run dev server.\n"
        "Final task: write a build-report markdown to the output_path\n"
        "summarizing what files you created, total LoC, any decisions\n"
        "you made, and a verbatim 'Quick Start' section the user can\n"
        "copy-paste."
    ),
    "success_criterion": (
        "All 12 files exist under phase1/canvasagent/ (verified by Read or "
        "Bash ls). Total non-blank LoC across .ts, .tsx, .css, .html, .json "
        "files ≤500. drafts/canvasagent_1_0_build_report.md exists with file "
        "list + LoC count + Quick Start. ResultPacket schema-validates."
    ),
    "output_path": "drafts/canvasagent_1_0_build_report.md",
    "model": "mistral/mistral-small-latest",
    "process_hygiene": (
        "TDD-when-feasible doesn't apply yet (no backend). Do apply: atomic "
        "files, no half-finished components, no `any` without justification, "
        "every file builds the moment npm install completes. Use Bash 'ls' "
        "or wc -l to verify LoC budget at the end. If you exceed 500 LoC, "
        "trim AgentNode.tsx (the most likely offender) before reporting."
    ),
    "confidence_required": True,
    "falsifier_text": (
        "Failure if any of: (a) any of the 12 files missing; (b) total LoC "
        ">500 across the listed file types; (c) AgentNode.tsx doesn't fetch "
        "Ollama; (d) README missing CORS guidance; (e) ResultPacket schema-"
        "invalid."
    ),
    "verification_command": "set -eo pipefail && cd phase1/canvasagent && npm install --silent --no-audit --no-fund 2>&1 | tail -3 && npm run build 2>&1 | tail -8",
    "verification_timeout_secs": 240,
}


def main() -> int:
    print("=" * 72)
    print("PHASE 1.0 — Implementer scaffolds CanvasAgent v1.0 hello-canvas")
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
