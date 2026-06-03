"""bert-lab Python entry point.

Usage:
    python lab.py --role director --cycle <N>     # run a Director cycle
    python lab.py --role researcher --task <...>  # spawn a sub-agent (usually invoked by core, not directly)
    python lab.py probe                           # check provider readiness
    python lab.py mcp <server-name>               # run a custom bert MCP server
    python lab.py memory <op> [args]              # invoke memory tools from CLI

This is the thin entry point. Real logic lives in core/.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Prefix-cache discipline + Ollama keep_alive.
# Ollama auto-reuses prefix KV cache when a byte-identical prefix arrives;
# keep_alive=24h keeps the model warm across cycles so the cache survives
# the gap between dispatches. Without this, Ollama unloads the model after
# ~5 min idle and the next cycle pays full TTFT (model load + cold KV)
# instead of cached TTFT (5-10× faster on repeated reads). Per-cycle cost:
# ~no extra RAM (model already loaded once per cycle); benefit: 5-10×
# TTFT speedup on local-Ollama dispatches. setdefault preserves user
# override (export OLLAMA_KEEP_ALIVE=0 to force unload-after-each-call
# for memory-constrained testing).
os.environ.setdefault("OLLAMA_KEEP_ALIVE", "24h")

# F.6 follow-up: default OTel endpoint to local Jaeger when set up via
# tools/setup_jaeger.sh. setdefault means a user/operator who pointed
# the lab at Honeycomb or another collector wins; absent that env var,
# this falls through to localhost:4318 where bert's local Jaeger
# container (started by setup_jaeger.sh) listens.
os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
os.environ.setdefault("OTEL_SERVICE_NAME", "bert-lab")


LAB_ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(prog="bert", description="bert-lab entry point")
    sub = parser.add_subparsers(dest="cmd")

    # `--role <role> --cycle <N>` is the primary path used by run.sh.
    # Implemented as a positional command-or-flag union for simplicity.
    parser.add_argument("--role", help="Role to run (director / researcher / implementer / evaluator / ...)")
    parser.add_argument("--cycle", type=int, default=1, help="Cycle number (default 1)")
    parser.add_argument("--task", help="Inline task description (for sub-agent invocations)")
    parser.add_argument("--provider", default="nvidia",
                        help="Inference provider (groq, nvidia, cerebras, gemini, mistral, openrouter, hf_router, ollama)")
    parser.add_argument("--model", help="Model id; uses provider default if omitted")
    parser.add_argument("--max-iter", type=int, default=20, help="Max agent-loop iterations per cycle")

    # Subcommands
    sub.add_parser("probe", help="Probe provider readiness")
    mcp_p = sub.add_parser("mcp", help="Run a custom bert MCP server")
    mcp_p.add_argument("server", help="Which MCP server to run (e.g., bert-orchestrator, bert-memory)")
    mem_p = sub.add_parser("memory", help="Invoke a memory tool from CLI")
    mem_p.add_argument("op", choices=["view", "create", "str_replace", "insert", "delete",
                                       "rename", "search", "graph", "index", "stats", "extract"])
    mem_p.add_argument("args", nargs="*", help="Op-specific args")
    verify_p = sub.add_parser("verify", help="Verify a proof packet (.tar.gz)")
    verify_p.add_argument("packets", nargs="+", help="proof packet .tar.gz file(s)")
    verify_p.add_argument("--chain", action="store_true",
                          help="verify packets as a lineage chain")
    verify_p.add_argument("--fetch-rekor", action="store_true",
                          help="fetch Rekor inclusion proof (requires network)")
    verify_p.add_argument("--no-color", action="store_true",
                          help="disable ANSI color output")

    # `bert project {list|status|finalize|verify|approve}` — Sprint 7 #31.
    from tools import project_cli
    project_cli.build_subparser(sub)

    args = parser.parse_args()

    # Lazy imports so `--help` is fast and missing optional deps don't block CLI
    if args.cmd == "probe":
        from core import probe
        return probe.run()

    if args.cmd == "mcp":
        from core import mcp_server
        return mcp_server.run(args.server)

    if args.cmd == "memory":
        from core import memory
        # `memory` subparser uses choices including some not yet implemented
        # in MVP — memory.cli() returns 1 with a "not yet implemented" notice
        # for those ops.
        return memory.cli(args.op, args.args)

    if args.cmd == "verify":
        from tools import bert_verify
        return bert_verify.run(args.packets, chain=args.chain,
                               fetch_rekor=args.fetch_rekor, no_color=args.no_color)

    if args.cmd == "project":
        from tools import project_cli
        return project_cli.dispatch(args)

    if args.role:
        from core import agent
        return agent.run_role(
            args.role,
            cycle=args.cycle,
            task=args.task,
            provider_name=args.provider,
            model=args.model,
            max_iterations=args.max_iter,
        )

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
