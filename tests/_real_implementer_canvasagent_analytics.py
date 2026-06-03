"""Phase 1.1c — wire FALS-CANVASAGENT-* analytics observable from Day 1.

The three falsifiers locked at Phase 0 → Phase 1 transition:
  FALS-01: onboarding completion <60% on fresh-install funnel (≥50 users)
  FALS-02: canvas interaction latency >300ms on RTX 4070
           (drag / wire / click-to-run, 100-rep median)
  FALS-03: daily active users <200 within 60 days of public launch

Privacy-first: all metrics stay in localStorage, no remote pings, no
SaaS dependency. Users can later opt in to share via export. The point
is to make the falsifiers observable on the user's machine from the
moment v1.0 ships, so we can extract numbers via dogfooding before
public launch.

Run: PYTHONPATH=. uv run python tests/_real_implementer_canvasagent_analytics.py
"""
from __future__ import annotations

import json
import sys

from core import subagent

DISPATCH = {
    "dispatch_altitude": "IMPL",
    "role": "implementer",
    "cycle": 10,
    "task": (
        "PHASE 1.1c — instrument FALS-CANVASAGENT-* analytics in CanvasAgent.\n\n"
        "READ FIRST:\n"
        "- memories/mission.md (locked falsifiers at the bottom)\n"
        "- memories/governance/pi_notes.md (Phase 0 → Phase 1 Approval section)\n"
        "- phase1/canvasagent/src/{App,AgentNode,Toolbar,types}.tsx + dag.ts + storage.ts\n\n"
        "GOAL: add a privacy-first analytics layer that makes all three\n"
        "falsifiers observable from Day 1 of v1.0. No remote pings, all\n"
        "data in localStorage under key `canvasagent.analytics.v1`.\n\n"
        "REQUIREMENTS:\n\n"
        "1. NEW src/analytics.ts — localStorage-only event log + helpers.\n"
        "   Module API:\n"
        "     export type EventName = 'app_load' | 'node_run_click'\n"
        "       | 'node_added' | 'edge_wired' | 'run_all_click'\n"
        "       | 'onboarding_stage';\n"
        "     export interface AnalyticsEvent {\n"
        "       name: EventName;\n"
        "       ts: number;       // Date.now()\n"
        "       props: Record<string, number | string | boolean>;\n"
        "     }\n"
        "     export function recordEvent(name, props={}): void;\n"
        "       (appends to localStorage; caps log at 1000 events FIFO)\n"
        "     export function recordLatency(action, ms): void;\n"
        "       (specialized — keeps a rolling 100-rep window per action;\n"
        "        action ∈ 'drag_node' | 'wire_edge' | 'click_to_run')\n"
        "     export function getOnboardingFunnel(): {\n"
        "       loaded: boolean; ran_starter: boolean; added_node: boolean;\n"
        "       wired_edge: boolean; ran_dag: boolean; complete: boolean;\n"
        "     };\n"
        "     export function getLatencyP50(action): number | null;\n"
        "     export function getDau(windowDays=60): number;\n"
        "       (count distinct YYYY-MM-DD calendar days where any event\n"
        "        was recorded within window)\n"
        "     export function exportAnalyticsJson(): string;\n"
        "       (returns full state as JSON for manual share/dogfood export)\n"
        "     export function clearAnalytics(): void;\n"
        "   ALL writes wrapped in try/catch with console.warn on quota fail.\n\n"
        "2. INSTRUMENT in App.tsx and AgentNode.tsx:\n"
        "   - On App mount: recordEvent('app_load')\n"
        "   - On Toolbar's + Add Node click: recordEvent('node_added',\n"
        "     {total_nodes_after}); also measure ms from click to next\n"
        "     render via a useLayoutEffect or performance.now() pair, log\n"
        "     via recordLatency('drag_node', ms)  (it's actually \"add a\n"
        "     node\" but the FALS-02 grouping is fine for now)\n"
        "   - On onConnect (edge wired): recordEvent('edge_wired') +\n"
        "     recordLatency('wire_edge', ms-since-callback-start)\n"
        "   - On Run All click: recordEvent('run_all_click') +\n"
        "     recordLatency('click_to_run', ms-from-click-to-first-node-\n"
        "     state-update)\n"
        "   - On per-node Run button click in AgentNode: recordEvent(\n"
        "     'node_run_click', {node_id})\n"
        "   - Onboarding stages — record once per fresh-install:\n"
        "     stage 1 = app_load (always), 2 = node_run_click, 3 =\n"
        "     node_added, 4 = edge_wired, 5 = run_all_click. Use\n"
        "     recordEvent('onboarding_stage', {stage: N}) on the FIRST\n"
        "     time each stage fires per browser. Persist 'seen stages'\n"
        "     bitmask in localStorage.\n\n"
        "3. NEW src/MetricsPanel.tsx — small toggleable panel showing\n"
        "   current state. Triggered by a small 'Stats' button at the\n"
        "   right of the Toolbar (next to Clear). Panel content:\n"
        "     - Onboarding funnel checklist (5 stages, ✓ or empty)\n"
        "     - Latency p50 for each of 3 actions (or 'no data' if <5 reps)\n"
        "     - DAU (last 60 days)\n"
        "     - Total events recorded\n"
        "     - Two buttons: 'Copy JSON' (clipboard via navigator.clipboard)\n"
        "       and 'Reset analytics' (clearAnalytics + close)\n"
        "   Style consistent with existing Toolbar (rounded corners, light\n"
        "   shadow, fixed position top-right of viewport). Closes on\n"
        "   backdrop click or X button.\n\n"
        "4. UPDATE src/Toolbar.tsx — add Stats button alongside the existing\n"
        "   3 (+ Add Node / Run All / Clear). Pass an onStats callback\n"
        "   prop. Don't bake MetricsPanel directly inside Toolbar — keep\n"
        "   them sibling components in App.tsx.\n\n"
        "BUDGET: total non-blank LoC across phase1/canvasagent/src/*.{ts,tsx,css}\n"
        "+ config files ≤900. Was 744 after 1.1a; +156 budget for this dispatch.\n\n"
        "DO NOT: run npm install or npm run build. The harness will verify.\n\n"
        "DO: write build report to drafts/canvasagent_analytics_build_report.md\n"
        "with file diff summary, total LoC, and a 'Quick Verify' section\n"
        "showing how to:\n"
        "  - Open localhost:5173 (after npm run dev)\n"
        "  - Open the Stats panel\n"
        "  - Click through the 5 onboarding stages and watch ✓s appear\n"
        "  - Inspect localStorage in devtools — key 'canvasagent.analytics.v1'"
    ),
    "success_criterion": (
        "src/analytics.ts and src/MetricsPanel.tsx exist; App.tsx, "
        "Toolbar.tsx, AgentNode.tsx instrumented at all 6 touchpoints. "
        "Total LoC ≤900. drafts/canvasagent_analytics_build_report.md "
        "exists. ResultPacket schema-validates."
    ),
    "output_path": "drafts/canvasagent_analytics_build_report.md",
    "model": "mistral/mistral-small-latest",
    "process_hygiene": (
        "PRIVACY HARD RULE: NO REMOTE PINGS. No fetch() / XMLHttpRequest / "
        "navigator.sendBeacon / WebSocket targeting any external host. The "
        "ONLY network call in CanvasAgent is the Ollama localhost POST in "
        "agent.ts; do not add others. All analytics writes go to "
        "localStorage only. If you find yourself reaching for a network "
        "call to ship metrics, STOP — the design is intentional, the user "
        "exports manually if at all. "
        "Build correctness: useNodesState/useEdgesState return 3-tuples, "
        "not 2; React Flow's Connection type for onConnect; AgentNodeData "
        "should NOT have a position field (Node has it natively). All "
        "additions must compile under noEmit + strict mode."
    ),
    "confidence_required": True,
    "falsifier_text": (
        "Failure if any of: (a) analytics.ts missing or doesn't export "
        "the listed API; (b) any remote network call introduced; (c) "
        "MetricsPanel doesn't render onboarding funnel + latency p50 + "
        "DAU; (d) total LoC > 900; (e) build breaks."
    ),
    # Verification = build passes AND no new fetch/XHR/sendBeacon outside agent.ts
    # (the privacy hard rule). egrep -L returns 0 if no match, 1 if match — we
    # invert via 'test ! -s' on a temp file. Simpler: count matches in non-agent
    # files and require zero.
    "verification_command": (
        "set -eo pipefail && cd phase1/canvasagent && npm run build 2>&1 | tail -8 && "
        "echo '--- privacy check ---' && "
        "BAD=$(grep -lE 'fetch\\(|XMLHttpRequest|sendBeacon|new WebSocket' "
        "  src/*.ts src/*.tsx 2>/dev/null | grep -v 'src/agent.ts$' | wc -l) && "
        "if [ \"$BAD\" -eq 0 ]; then echo 'privacy: OK (no remote pings outside agent.ts)'; "
        "else echo \"privacy: FAIL ($BAD files have remote calls)\" && exit 1; fi"
    ),
    "verification_timeout_secs": 180,
}


def main() -> int:
    print("=" * 72)
    print("PHASE 1.1c — Implementer wires FALS-CANVASAGENT-* analytics")
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
