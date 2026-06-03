"""LIVE-TEST — requires real model provider credentials. Smoke test for the new verification_command Spawn contract.

Two scenarios tested end-to-end:

1. AGENT CLAIMS X, EXTERNAL CHECK PASSES → verdict overridden to BUILD_PASS.
   Run a tiny no-op task (agent says "I did nothing meaningful") with a
   verification_command that runs `cd phase1/canvasagent && npm run build`.
   That build is currently green. Expected: agent's verdict (whatever it
   self-reports) gets overridden to BUILD_PASS with calibration_reasoning
   noting the override; telemetry.verification populated with exit=0.

2. AGENT CRASHES BEFORE PACKET, EXTERNAL CHECK PASSES → verdict synthesized
   to BUILD_PASS. Same dispatch but with a max_iterations=1 that's likely
   too small to write the ResultPacket; verification still passes.
   (Skipped here because it's hard to force a crash deterministically;
   the recovery path is unit-tested via _synthesize_packet_from_verification.)

Run: PYTHONPATH=. uv run python tests/_smoke_verification_command.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import subagent  # noqa: E402

SPEC = {
    "dispatch_altitude": "INFRA",
    "role": "implementer",
    "cycle": 99,
    "task": (
        "SMOKE TEST FOR verification_command. Do NOT modify any code in "
        "phase1/canvasagent — just write a tiny note to drafts/verify_smoke_note.md "
        "saying 'agent ran the smoke test' and then write the ResultPacket "
        "with verdict=CHANGES_REQUESTED (deliberate — verifies override path). "
        "The verification_command will independently run `npm run build` to "
        "confirm the canvasagent project still builds; that's the source of "
        "truth, not your verdict."
    ),
    "success_criterion": (
        "drafts/verify_smoke_note.md exists; ResultPacket schema-valid; "
        "Spawn override flips verdict from CHANGES_REQUESTED to BUILD_PASS "
        "based on verification_command exit code 0."
    ),
    "output_path": "drafts/verify_smoke_note.md",
    "model": "cerebras/qwen-3-235b-a22b-instruct-2507",
    "process_hygiene": (
        "This is a deliberate test of verdict override. Do NOT alter any "
        "canvasagent source files — verification depends on the existing "
        "build being green."
    ),
    "confidence_required": True,
    "verification_command": "cd phase1/canvasagent && npm run build 2>&1 | tail -3",
    "verification_timeout_secs": 180,
}


def main() -> int:
    print("=" * 72)
    print("Smoke test: verification_command override path")
    print("=" * 72)
    print()

    summary = subagent.run_subagent(SPEC)

    print()
    print("=" * 72)
    print("Summary returned to Director")
    print("=" * 72)
    safe = {k: v for k, v in summary.items() if k != "calibration_reasoning"}
    print(json.dumps(safe, indent=2, default=str))
    print()
    print("--- calibration_reasoning ---")
    print(summary.get("calibration_reasoning", "")[:1500])
    print()
    print("--- verification block from telemetry ---")
    tel = summary.get("telemetry", {})
    verify = tel.get("verification") if isinstance(tel, dict) else None
    print(json.dumps(verify, indent=2)[:600] if verify else "  (no verification block)")
    print()
    print("=" * 72)
    print("EXPECTED:")
    print("  - verdict == BUILD_PASS  (overridden from agent's CHANGES_REQUESTED)")
    print("  - verification.ok == True, verification.exit_code == 0")
    print("  - calibration_reasoning mentions 'overridden' or 'external check'")
    print("=" * 72)

    verdict_ok = summary.get("verdict") == "BUILD_PASS"
    verify_ok = verify and verify.get("ok") is True
    return 0 if (verdict_ok and verify_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
