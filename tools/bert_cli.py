"""bert — top-level CLI dispatcher.

Phase D4 of the v3 plan. The CLI surface complementing the MCP plugin
(the MCP plugin is primary for adoption; CLI is for power users +
scripts + CI).

Same core/ wiring as the MCP server — no duplication. Commands:

  bert lab list
  bert lab start <name> "<mission>"
  bert lab status <name>
  bert lab cycle <name> [--budget auto|quick|standard|deep|until_complete]
  bert lab reshape <name> [field=value ...]
  bert lab cost <name> [--since 7d]
  bert memory search <name> "<query>" [--method hybrid|vector|grep]
  bert memory ingest <name> <source_path_or_url>
  bert packet export <name> [--cycle N]
  bert packet verify <path/to/packet.tar.gz>
  bert doctor                          (wraps tools/bert_doctor.py)
  bert dashboard                       (spawns optional React UI; default OFF)

Exit codes:
  0  success
  1  partial / soft failure
  2  bad arguments
  3  lab not found / missing precondition
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

LABS_DIR = Path.home() / ".bert" / "labs"


def _resolve_lab(arg: str) -> Path | None:
    if not arg:
        return None
    p = Path(arg).expanduser()
    if p.is_absolute() and p.exists():
        return p
    cand = LABS_DIR / arg
    return cand if cand.exists() else None


# ── lab subcommands ──────────────────────────────────────────────


def cmd_lab_list(args) -> int:
    from tools.mcp.bert_lab import _t_lab_list
    out = _t_lab_list({"prefix": args.prefix or ""})
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"{len(out['labs'])} labs at {out['labs_dir']}:")
        for lab in out["labs"]:
            print(f"  {lab['lab']:25s} cycle={lab['last_cycle']:4d} "
                  f"findings={lab['findings_count']:3d} "
                  f"events={lab['events_total']:6d}")
    return 0


def cmd_lab_start(args) -> int:
    from tools.mcp.bert_lab import _t_lab_start
    out = _t_lab_start({
        "name": args.name,
        "mission": args.mission,
        "archetype": args.archetype or "research",
        "use_llm_classifier": not args.no_llm,
    })
    if not out.get("ok"):
        print(f"error: {out.get('error', '?')}", file=sys.stderr)
        return 3
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"created lab: {out['lab']}")
        print(f"  path: {out['path']}")
        print(f"  profile: {out['profile']['domain']} / "
              f"{out['profile']['primary_work']} / "
              f"{out['profile']['data_shape']}")
        print(f"  schema: {out['schema']['rule_id']}")
        print(f"  scaffolded: {out['scaffolded_knowledge_files']}")
        print(f"  next: {out['next']}")
    return 0


def cmd_lab_status(args) -> int:
    from tools.mcp.bert_lab import _t_lab_status
    out = _t_lab_status({"lab": args.name})
    if not out.get("ok"):
        print(f"error: {out.get('error', '?')}", file=sys.stderr)
        return 3
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"lab: {out['lab']}")
        print(f"  path:           {out['path']}")
        print(f"  mission:        {out['mission'][:120]}")
        print(f"  last cycle:     {out['last_cycle']}")
        print(f"  findings count: {out['findings_count']}")
        print(f"  events total:   {out['events_total']}")
    return 0


def cmd_lab_cycle(args) -> int:
    from tools.mcp.bert_lab import _t_lab_cycle
    out = _t_lab_cycle({
        "lab": args.name,
        "budget": args.budget or "auto",
        "via_claude": not args.no_claude,
    })
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        if not out.get("ok"):
            print(f"cycle failed: {out.get('error', '?')}", file=sys.stderr)
            return 1
        delta = out.get("cycles_after", 0) - out.get("cycles_before", 0)
        print(f"cycle complete: +{delta} cycles, "
              f"+{out.get('findings_delta', 0)} findings, "
              f"+{out.get('events_delta', 0)} events "
              f"in {out['elapsed_secs']}s")
        if out.get("saturation", {}).get("saturated"):
            print("  ⚠ saturation detected — consider reshaping or stopping")
    return 0


def cmd_lab_reshape(args) -> int:
    from tools.mcp.bert_lab import _t_lab_reshape
    updates: dict = {}
    for kv in args.updates or []:
        if "=" not in kv:
            print(f"bad update arg: {kv} (expected key=value)", file=sys.stderr)
            return 2
        k, v = kv.split("=", 1)
        try:
            updates[k] = json.loads(v)
        except json.JSONDecodeError:
            updates[k] = v
    out = _t_lab_reshape({
        "lab": args.name,
        "updates": updates or None,
    })
    print(json.dumps(out, indent=2))
    return 0 if out.get("ok") else 1


# ── memory subcommands ───────────────────────────────────────────


def cmd_memory_search(args) -> int:
    from tools.mcp.bert_lab import _t_memory_search
    out = _t_memory_search({
        "lab": args.name,
        "query": args.query,
        "k": args.k or 5,
        "use_vector": (args.method == "vector"),
    })
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        if not out.get("ok"):
            print(f"error: {out.get('error', '?')}", file=sys.stderr)
            return 1
        print(f"{len(out['results'])} hits (method={out.get('method', '?')})")
        for h in out["results"][:args.k or 5]:
            path = h.get("path", "?")
            snip = h.get("snippet", "")[:120].replace("\n", " ")
            print(f"  {path}: {snip}")
    return 0


def cmd_memory_ingest(args) -> int:
    lab = _resolve_lab(args.name)
    if lab is None:
        print(f"lab not found: {args.name!r}", file=sys.stderr)
        return 3
    from core.memory_adapters import find_adapter_for_shape
    # Detect data_shape from lab.yaml
    try:
        import yaml
        cfg = yaml.safe_load((lab / "lab.yaml").read_text())
        shape = (cfg or {}).get("mission_profile", {}).get(
            "data_shape", "document_corpus")
    except Exception:  # noqa: BLE001
        shape = "document_corpus"
    cls = find_adapter_for_shape(shape)
    ad = cls(lab)
    result = ad.ingest(args.source)
    print(json.dumps({
        "ok": result.items_added > 0,
        "items_added": result.items_added,
        "bytes_in": result.bytes_in,
        "duration_ms": result.duration_ms,
        "warnings": list(result.warnings),
        "method": shape,
    }, indent=2))
    return 0 if result.items_added > 0 else 1


# ── packet subcommands ───────────────────────────────────────────


def cmd_packet_export(args) -> int:
    from tools.mcp.bert_lab import _t_packet_export
    out = _t_packet_export({"lab": args.name, "cycle_id": args.cycle})
    print(json.dumps(out, indent=2))
    return 0 if out.get("ok") else 1


def cmd_packet_verify(args) -> int:
    """Wrap existing tools/bert_verify.py."""
    cmd = [sys.executable, str(LAB_ROOT / "tools" / "bert_verify.py"), args.path]
    if args.fetch_rekor:
        cmd.append("--fetch-rekor")
    result = subprocess.run(cmd)
    return result.returncode


# ── doctor / dashboard ───────────────────────────────────────────


def cmd_doctor(args) -> int:
    """Wrap existing tools/bert_doctor.py."""
    cmd = [sys.executable, str(LAB_ROOT / "tools" / "bert_doctor.py")]
    result = subprocess.run(cmd)
    return result.returncode


def cmd_dashboard(args) -> int:
    """Spin up the optional React dashboard. Default OFF; opt-in."""
    print("Starting vite dev server (Ctrl-C to stop)...")
    print("Dashboard will be at http://127.0.0.1:5173 once vite reports ready.")
    cmd = ["npm", "run", "dev"]
    cwd = str(LAB_ROOT / "bert" / "v4")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


# ── argparse wiring ──────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="bert",
        description="bert — adaptive autonomous lab framework",
    )
    ap.add_argument("--json", action="store_true",
                    help="JSON output instead of pretty text")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # lab
    p_lab = sub.add_parser("lab", help="lab commands")
    p_lab_sub = p_lab.add_subparsers(dest="lab_cmd", required=True)

    p_ll = p_lab_sub.add_parser("list", help="list labs")
    p_ll.add_argument("--prefix", help="filter by name prefix")
    p_ll.set_defaults(func=cmd_lab_list)

    p_ls = p_lab_sub.add_parser("start", help="create a new lab")
    p_ls.add_argument("name")
    p_ls.add_argument("mission")
    p_ls.add_argument("--archetype", choices=["research", "product", "strategy"])
    p_ls.add_argument("--no-llm", action="store_true",
                      help="skip Haiku classifier (use heuristic fallback)")
    p_ls.set_defaults(func=cmd_lab_start)

    p_lst = p_lab_sub.add_parser("status", help="show lab state")
    p_lst.add_argument("name")
    p_lst.set_defaults(func=cmd_lab_status)

    p_lc = p_lab_sub.add_parser("cycle", help="run autonomous cycle(s)")
    p_lc.add_argument("name")
    p_lc.add_argument(
        "--budget",
        choices=["auto", "quick", "standard", "deep", "until_complete"],
        help="budget preset (default: auto)",
    )
    p_lc.add_argument("--no-claude", action="store_true",
                      help="skip Opus-via-Max-plan researcher routing")
    p_lc.set_defaults(func=cmd_lab_cycle)

    p_lr = p_lab_sub.add_parser("reshape", help="reshape lab profile (within-shape)")
    p_lr.add_argument("name")
    p_lr.add_argument("updates", nargs="*",
                      help="key=value pairs of profile fields to update")
    p_lr.set_defaults(func=cmd_lab_reshape)

    # memory
    p_mem = sub.add_parser("memory", help="memory commands")
    p_mem_sub = p_mem.add_subparsers(dest="mem_cmd", required=True)

    p_ms = p_mem_sub.add_parser("search", help="search lab memory")
    p_ms.add_argument("name")
    p_ms.add_argument("query")
    p_ms.add_argument("-k", type=int, help="max hits (default 5)")
    p_ms.add_argument("--method", choices=["hybrid", "vector", "grep"],
                      help="search method (default: hybrid)")
    p_ms.set_defaults(func=cmd_memory_search)

    p_mi = p_mem_sub.add_parser("ingest", help="ingest a source into lab memory")
    p_mi.add_argument("name")
    p_mi.add_argument("source", help="file path, directory, or URL")
    p_mi.set_defaults(func=cmd_memory_ingest)

    # packet
    p_pk = sub.add_parser("packet", help="proof packet commands")
    p_pk_sub = p_pk.add_subparsers(dest="pk_cmd", required=True)

    p_pe = p_pk_sub.add_parser("export", help="export proof packet")
    p_pe.add_argument("name")
    p_pe.add_argument("--cycle", type=int)
    p_pe.set_defaults(func=cmd_packet_export)

    p_pv = p_pk_sub.add_parser("verify", help="verify a proof packet")
    p_pv.add_argument("path")
    p_pv.add_argument("--fetch-rekor", action="store_true")
    p_pv.set_defaults(func=cmd_packet_verify)

    # doctor + dashboard
    p_d = sub.add_parser("doctor", help="health check")
    p_d.set_defaults(func=cmd_doctor)

    p_dash = sub.add_parser("dashboard", help="optional React dashboard")
    p_dash.set_defaults(func=cmd_dashboard)

    return ap


def main(argv: list[str]) -> int:
    ap = build_parser()
    args = ap.parse_args(argv[1:])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
