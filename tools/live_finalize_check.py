"""Live-LLM finalize harness (Sprint 7 caveat — harness-ready).

finalize_project has been verified end-to-end against the real registry but only
with STUBBED providers. This harness runs it with a REAL provider cascade to
validate synthesis + 4-judge grading quality on an actual lab. It is guarded: if
no provider key is configured (env or ~/.bert-lab/credentials.json), it prints a
clear SKIP and exits 0 — it is a validation aid, not a CI gate, so a missing key
is not a failure. Re-runnable the instant a key is exported:

    export GROQ_API_KEY=...        # free tier, fastest
    python tools/live_finalize_check.py --lab test01

It writes the artifact into the lab and prints the grade, ready flag, and the
signed-hash prefix so you can eyeball real output quality.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `core` importable when run directly as `python tools/live_finalize_check.py`.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# env var -> provider lane label
PROVIDER_KEYS = {
    "GROQ_API_KEY": "groq",
    "NVIDIA_API_KEY": "nvidia",
    "CEREBRAS_API_KEY": "cerebras",
    "GOOGLE_AI_API_KEY": "google",
    "MISTRAL_API_KEY": "mistral",
    "OPENROUTER_API_KEY": "openrouter",
}


def available_lanes() -> list[str]:
    """Which provider lanes have a usable key (env or credentials.json)."""
    from core import config
    cfg = config.load(reload=True)
    return [lane for key, lane in PROVIDER_KEYS.items() if cfg.has(key)]


def run(lab: str = "test01", objective: str | None = None,
        output: str | None = None) -> int:
    lanes = available_lanes()
    if not lanes:
        print("[live-finalize] SKIP: no provider key configured. Export one and "
              "re-run, e.g.  export GROQ_API_KEY=...  (free tier).")
        return 0
    from tools.mcp import bert_lab
    lab_path = bert_lab._resolve_lab(lab)
    if lab_path is None:
        print(f"[live-finalize] lab not found: {lab!r}")
        return 1
    objective = objective or "Synthesize the lab's findings into an honest, cited brief."
    output = output or "live_finalize_out.md"
    print(f"[live-finalize] lanes={lanes} lab={lab} — running finalize_project "
          f"with a REAL provider cascade (synthesis + 4-judge grade)...")
    from core import skill_runner
    res = skill_runner.run_skill(
        "finalize_project",
        {"objective": objective, "output_path": output},
        lab_path=lab_path,
    )
    if res.get("ok"):
        out = res.get("outputs", {})
        print(f"[live-finalize] OK  grade={out.get('grade')}  ready={out.get('ready')}  "
              f"hash={str(out.get('signed_hash', ''))[:16]}  artifact={out.get('artifact_path')}")
        print("[live-finalize] Inspect the artifact + gaps.md to judge real output quality.")
        return 0
    print(f"[live-finalize] FAILED: errors={res.get('errors')} "
          f"steps={res.get('steps_executed')}")
    return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Live-LLM finalize quality check")
    p.add_argument("--lab", default="test01", help="lab name or path (default: test01)")
    p.add_argument("--objective", default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)
    return run(lab=args.lab, objective=args.objective, output=args.output)


if __name__ == "__main__":
    sys.exit(main())
