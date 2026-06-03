"""`bert project` CLI family (Sprint 7 #31).

A thin command surface over production backends — no new logic:
  list      -> bert_lab._t_lab_list
  status    -> bert_lab._t_lab_status (+ readiness line)
  finalize  -> skill_runner.run_skill("finalize_project", ...) (mission contract
               auto-injected by skill_runner from the lab schema)
  verify    -> bert_verify.run (proof-packet verification)
  approve   -> proposal_activate.activate (install tool / promote skill)

"project" == "lab" (the ~/.bert/labs/<name> directory). Wired into lab.py via
build_subparser(sub) + dispatch(args).
"""

from __future__ import annotations

import json


def build_subparser(sub) -> None:
    """Register the `project` subcommand family on an argparse subparsers obj."""
    p = sub.add_parser("project", help="Project (lab) lifecycle: list/status/finalize/verify/approve")
    psub = p.add_subparsers(dest="project_cmd", required=True)

    pl = psub.add_parser("list", help="List all bert projects")
    pl.add_argument("--json", action="store_true", help="emit raw JSON")

    ps = psub.add_parser("status", help="Show a project's state + readiness")
    ps.add_argument("lab", help="Project/lab name or path")
    ps.add_argument("--json", action="store_true")

    pf = psub.add_parser("finalize", help="Finalize a project: synthesize + grade + sign")
    pf.add_argument("lab", help="Project/lab name or path")
    pf.add_argument("--objective", required=True, help="What the artifact should answer/deliver")
    pf.add_argument("--output", required=True, help="Where to write the artifact (e.g. final.md)")
    pf.add_argument("--json", action="store_true")

    pv = psub.add_parser("verify", help="Verify a project's proof packet(s)")
    pv.add_argument("packets", nargs="+", help="proof packet .tar.gz file(s)")
    pv.add_argument("--chain", action="store_true", help="verify as a lineage chain")
    pv.add_argument("--no-color", action="store_true")

    pa = psub.add_parser("approve", help="Approve a pending tool/skill proposal (the PI gate)")
    pa.add_argument("proposal_id", help="Proposal id (tool-* or prop-*)")
    pa.add_argument("--json", action="store_true")


def _emit(payload: dict, args) -> None:
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, default=str))
    else:
        for k, v in payload.items():
            print(f"  {k}: {v}")


# ── command handlers ─────────────────────────────────────────────────


def cmd_list(args) -> int:
    from tools.mcp import bert_lab
    res = bert_lab._t_lab_list({})
    _emit(res, args)
    return 0


def cmd_status(args) -> int:
    from tools.mcp import bert_lab
    res = bert_lab._t_lab_status({"lab": args.lab})
    # Cheap rolled-up snapshot (Q-6) for a readiness line — delta-folds only
    # events appended since the last snapshot. Best-effort: never break status.
    if not res.get("error"):
        try:
            from core import project_snapshot
            lab_path = bert_lab._resolve_lab(args.lab)
            if lab_path is not None:
                snap = project_snapshot.delta(lab_path)
                res["snapshot"] = {k: snap.get(k) for k in
                                   ("cycle", "findings", "artifacts_accepted", "new_events")}
        except Exception:  # noqa: BLE001
            pass
    _emit(res, args)
    return 0 if not res.get("error") else 1


def cmd_finalize(args) -> int:
    from core import skill_runner
    from tools.mcp import bert_lab
    lab_path = bert_lab._resolve_lab(args.lab)
    if lab_path is None:
        print(f"project not found: {args.lab!r}")
        return 1
    # The artifact must land INSIDE the lab — reject a traversing/absolute path
    # (it flows to the Write tool, which would otherwise escape the lab root).
    from pathlib import Path as _P
    if _P(args.output).is_absolute() or ".." in _P(args.output).parts:
        print(f"unsafe --output (must be a relative path within the project): {args.output!r}")
        return 1
    res = skill_runner.run_skill(
        "finalize_project",
        {"objective": args.objective, "output_path": args.output},
        lab_path=lab_path,
    )
    if res.get("ok"):
        out = res.get("outputs", {})
        _emit({"ok": True, "grade": out.get("grade"),
               "signed_hash": out.get("signed_hash"),
               "ready": out.get("ready")}, args)
        return 0
    _emit({"ok": False, "errors": res.get("errors", []),
           "steps_executed": res.get("steps_executed", [])}, args)
    return 1


def cmd_verify(args) -> int:
    from tools import bert_verify
    return bert_verify.run(args.packets, chain=args.chain, no_color=args.no_color)


def cmd_approve(args) -> int:
    from core import proposal_activate
    res = proposal_activate.activate(args.proposal_id)
    _emit(res, args)
    return 0 if res.get("ok") else 1


_DISPATCH = {
    "list": "cmd_list", "status": "cmd_status", "finalize": "cmd_finalize",
    "verify": "cmd_verify", "approve": "cmd_approve",
}


def dispatch(args) -> int:
    """Route a parsed `project` args namespace to its handler. Looks handlers up
    by name on this module so tests can monkeypatch them."""
    import sys as _sys
    handler_name = _DISPATCH.get(getattr(args, "project_cmd", None))
    if handler_name is None:
        print("usage: bert project {list|status|finalize|verify|approve}")
        return 1
    return getattr(_sys.modules[__name__], handler_name)(args)
