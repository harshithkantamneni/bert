"""bert-lab MCP server — public-facing surface for Claude Code / Codex / Cursor.

The plug-in entry point for bert's "lab mode" pivot. Exposes 11 tools
that turn any MCP-compatible AI agent (Claude Desktop, Claude Code,
Cursor with MCP, Continue, etc.) into a persistent lab orchestrator:

  - lab_list                  enumerate known labs
  - lab_status(lab)           inspect a lab's current state
  - lab_start(name, mission)  create a new lab
  - lab_cycle(lab, n)         run N autonomous cycles
  - lab_reshape(lab)          re-derive roles from the mission
  - lab_resume(lab)           resume a paused lab
  - lab_finalize(lab)         run the finalization ceremony
  - lab_approve(lab, id)      approve a pending proposal
  - lab_synthesize_tool(lab)  synthesize + sandbox a new tool
  - memory_search(lab, q)     vector + keyword search across a lab's
                              memories + findings
  - packet_export(lab, cycle) build a signed proof-packet tar.gz

Labs live under ~/.bert/labs/<name>/ by default. Each lab has its own
seed_brief.md, memories/, findings/, sor/events.jsonl, state/graph.db.

Install in Claude Desktop (~/Library/Application Support/Claude/
claude_desktop_config.json):

  {
    "mcpServers": {
      "bert": {
        "command": "/abs/path/to/bert-lab/.venv/bin/python",
        "args": ["-m", "tools.mcp.bert_lab"]
      }
    }
  }

Or via the existing entry point `python lab.py mcp bert-lab`.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core.mcp_server import MCPServer  # noqa: E402

LABS_DIR = Path.home() / ".bert" / "labs"


def _resolve_lab(arg: str | None) -> Path | None:
    """Accepts a lab name (`test01`) or absolute path. Returns the
    resolved directory or None if not found."""
    if not arg:
        return None
    p = Path(arg).expanduser()
    if p.is_absolute() and p.exists():
        return p
    candidate = LABS_DIR / arg
    if candidate.exists():
        return candidate
    return None


def _lab_summary(lab_path: Path) -> dict:
    """Cheap inspection: cycle count, last cycle, finding count, mission."""
    seed = lab_path / "seed_brief.md"
    events = lab_path / "sor" / "events.jsonl"
    findings = lab_path / "findings"

    mission = ""
    if seed.exists():
        text = seed.read_text(errors="replace")
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                mission = line[:280]
                break

    events_total = 0
    last_cycle = 0
    last_ts = ""
    if events.exists():
        try:
            with events.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    events_total += 1
                    try:
                        ev = json.loads(line)
                        c = ev.get("cycle")
                        if isinstance(c, int) and c > last_cycle:
                            last_cycle = c
                        ts = ev.get("ts")
                        if ts and ts > last_ts:
                            last_ts = ts
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass

    findings_count = 0
    if findings.exists():
        findings_count = sum(
            1 for p in findings.iterdir() if p.is_file() and p.suffix == ".md"
        )

    return {
        "lab": lab_path.name,
        "path": str(lab_path),
        "mission": mission,
        "events_total": events_total,
        "last_cycle": last_cycle,
        "last_event_ts": last_ts,
        "findings_count": findings_count,
    }


# ── Tool: lab_list ──────────────────────────────────────────────────


def _t_lab_list(args: dict) -> dict:
    prefix = (args.get("prefix") or "").strip().lower()
    LABS_DIR.mkdir(parents=True, exist_ok=True)
    labs = []
    for child in sorted(LABS_DIR.iterdir()):
        if not child.is_dir():
            continue
        if prefix and not child.name.lower().startswith(prefix):
            continue
        labs.append(_lab_summary(child))
    return {"labs": labs, "count": len(labs), "labs_dir": str(LABS_DIR)}


# ── Tool: lab_status ────────────────────────────────────────────────


def _t_lab_status(args: dict) -> dict:
    lab_arg = args.get("lab", "")
    lab_path = _resolve_lab(lab_arg)
    if lab_path is None:
        return {"ok": False, "error": f"lab not found: {lab_arg!r}"}
    summary = _lab_summary(lab_path)
    summary["ok"] = True
    return summary


# ── Tool: lab_finalize ──────────────────────────────────────────────


def _t_lab_finalize(args: dict) -> dict:
    """Run the finalize_project skill end-to-end (task #69 trigger): gather
    evidence -> synthesize a polished, cited artifact -> disclose honest gaps ->
    grade (4-judge median+variance) + sign -> record in the ledger. Returns the
    grade, the proof-packet signed hash, the artifact + gaps paths, and ready."""
    objective = (args.get("objective") or "").strip()
    output_path = (args.get("output_path") or "").strip()
    if not objective or not output_path:
        return {"ok": False, "error": "objective and output_path are required"}
    # The artifact must land inside the lab — reject traversing/absolute paths.
    if Path(output_path).is_absolute() or ".." in Path(output_path).parts:
        return {"ok": False, "error": "output_path must be a relative path within the lab"}
    lab_arg = args.get("lab")
    lab_path = _resolve_lab(lab_arg) if lab_arg else None
    if lab_arg and lab_path is None:
        return {"ok": False, "error": f"lab not found: {lab_arg!r}"}
    from core import skill_runner
    res = skill_runner.run_skill(
        "finalize_project",
        {"objective": objective, "output_path": output_path},
        lab_path=lab_path,
    )
    if not res.get("ok"):
        return {"ok": False, "error": "finalize_project did not complete",
                "errors": res.get("errors", []),
                "steps_executed": res.get("steps_executed", [])}
    out = res.get("outputs", {})
    return {"ok": True, "grade": out.get("grade"),
            "signed_hash": out.get("signed_hash"),
            "artifact_path": out.get("artifact_path"),
            "gaps_path": out.get("gaps_path"), "ready": out.get("ready")}


# ── Tool: lab_synthesize_tool ───────────────────────────────────────


def _t_lab_synthesize_tool(args: dict) -> dict:
    """Synthesize a new tool from a spec (Sprint 6 #30): a cross-family LLM
    writes the source + smoke test, an AST scan flags foot-guns, the candidate
    runs in core.sandbox, and the result is written to state/tools_pending_pi.md
    for PI review. The tool is NOT registered or callable until a PI /approve —
    `active` is always False here. Returns the proposal id, scan safety, and the
    sandbox exit code so the reviewer can judge."""
    from core import tool_synthesizer as ts
    name = (args.get("name") or "").strip()
    description = (args.get("description") or "").strip()
    if not name or not description:
        return {"ok": False, "error": "name and description are required"}
    if not ts.is_valid_tool_name(name):
        return {"ok": False, "error": (
            "name must be a snake_case Python identifier "
            "(^[a-z_][a-z0-9_]{0,63}$) — it becomes a filename + function name")}
    spec = ts.ToolSpec(
        name=name, description=description,
        params_schema=args.get("params_schema") or {"type": "object"},
        returns=args.get("returns") or "",
        implementation_hint=args.get("implementation_hint") or "",
    )
    cand = ts.synthesize(spec)
    if cand.method == "unavailable":
        return {"ok": False, "error": cand.error, "active": False}
    res = ts.sandbox_validate(cand.source, cand.smoke_test, name=name)
    cand.sandbox = {"exit_code": res.exit_code, "tier_used": str(res.tier_used),
                    "timed_out": res.timed_out}
    proposal_id = ts.propose(cand)
    return {"ok": True, "proposal_id": proposal_id, "active": False,
            "scan_safe": cand.scan.safe, "scan_violations": cand.scan.violations,
            "sandbox_exit": res.exit_code,
            "note": "Pending PI review — run /approve to install + activate."}


# ── Tool: lab_approve ───────────────────────────────────────────────


def _t_lab_approve(args: dict) -> dict:
    """Approve a pending proposal so it activates (Sprint 7): a `tool-*` id
    installs + registers the synthesized tool; a `prop-*` id promotes the drafted
    skill to active. Calling this IS the PI blessing. Idempotent. Returns the
    activation result (ok, kind, name/skill_id, already?)."""
    proposal_id = (args.get("proposal_id") or "").strip()
    if not proposal_id:
        return {"ok": False, "error": "proposal_id is required"}
    from core import proposal_activate
    return proposal_activate.activate(proposal_id)


# ── Tool: lab_start ─────────────────────────────────────────────────


def _t_lab_start(args: dict) -> dict:
    name = (args.get("name") or "").strip()
    mission = (args.get("mission") or "").strip()
    # archetype kept for back-compat; mission_profile.data_shape now
    # carries the real type information.
    archetype = (args.get("archetype") or "research").strip()
    # `use_llm_classifier=false` lets callers (tests, offline scenarios)
    # opt out of the Haiku classifier and use stage-0 heuristics only.
    use_llm = bool(args.get("use_llm_classifier", True))

    if not name or "/" in name or ".." in name:
        return {"ok": False, "error": "name must be a simple slug"}
    if not mission or len(mission) < 20:
        return {"ok": False, "error": "mission must be ≥20 chars"}
    if archetype not in ("research", "product", "strategy"):
        return {"ok": False, "error": "archetype must be research|product|strategy"}

    lab_path = LABS_DIR / name
    if lab_path.exists():
        return {"ok": False, "error": f"lab already exists at {lab_path}"}

    # A4: Classify mission + synthesize lab schema
    from core import mission_profile as mp_mod
    from core import schema_synthesizer as ss_mod
    try:
        profile = mp_mod.classify_mission(mission, use_llm=use_llm)
        schema = ss_mod.synthesize(profile)
    except Exception as e:  # noqa: BLE001
        # Never block lab creation on classifier failure — fall back
        # to a safe default profile.
        import core.log as _log
        _log.get_logger("bert.lab_start").warning(
            "classifier/synthesizer failed; using safe default: %s", e
        )
        profile = mp_mod.default_profile(mission)
        schema = ss_mod.synthesize(profile)

    # Scaffold directory structure (mirrors bert_init.py's layout)
    for sub in ("memories", "findings", "drafts", "sor", "state",
                "state/results", "knowledge", "agents"):
        (lab_path / sub).mkdir(parents=True, exist_ok=True)
    (lab_path / "sor" / "events.jsonl").touch()
    (lab_path / "seed_brief.md").write_text(
        f"# Mission\n\n{mission}\n"
    )

    # A4: lab.yaml now includes the full mission_profile + schema
    lab_yaml = (
        f"name: {name}\n"
        f"archetype: {archetype}\n"
        f"mission: |\n  {mission}\n"
        f"role: standard\n"
        f"share_with_supervisor: false\n"
        f"\n"
        f"mission_profile:\n"
        f"{profile.to_yaml_block()}\n"
        f"\n"
        f"lab_schema:\n"
        f"  rule_id: {schema.rule_id!r}\n"
        f"  profile_id: {schema.profile_id!r}\n"
        f"  roster_core: {list(schema.roster_core)}\n"
        f"  roster_initial: {list(schema.roster_initial)}\n"
        f"  memory_adapters: {list(schema.memory_adapters)}\n"
        f"  knowledge_files: {list(schema.knowledge_files)}\n"
        f"  graph_schema: {schema.graph_schema!r}\n"
        f"  workflow: {schema.workflow!r}\n"
        f"  output_format: {schema.output_format!r}\n"
    )
    (lab_path / "lab.yaml").write_text(lab_yaml)

    # A4: scaffold profile-appropriate knowledge files from library
    scaffolded = ss_mod.scaffold_knowledge_files(lab_path, schema)

    # Determine if classifier was ambiguous; surface to user if so
    ambiguous = mp_mod.is_ambiguous(profile)

    return {
        "ok": True,
        "lab": name,
        "path": str(lab_path),
        "archetype": archetype,
        "mission": mission,
        "profile": {
            "domain": profile.domain,
            "primary_work": profile.primary_work,
            "horizon": profile.horizon,
            "data_shape": profile.data_shape,
            "output_kind": profile.output_kind,
            "rigor": profile.rigor,
            "classifier_confidence": profile.classifier_confidence,
            "classifier_stage": profile.stage_used,
            "ambiguous": ambiguous,
        },
        "schema": {
            "rule_id": schema.rule_id,
            "roster_initial": list(schema.roster_initial),
            "knowledge_files": list(schema.knowledge_files),
            "graph_schema": schema.graph_schema,
            "workflow": schema.workflow,
        },
        "scaffolded_knowledge_files": [str(p.name) for p in scaffolded],
        "next": (
            f"Call lab_cycle(lab='{name}', budget='auto') to start "
            f"autonomous cycles. Budget will auto-derive from the mission."
            + (" Profile classification is ambiguous — you may want to "
               "review lab.yaml's mission_profile and adjust before "
               "running cycles."
               if ambiguous else "")
        ),
    }


# ── Tool: lab_cycle ─────────────────────────────────────────────────


def _t_lab_cycle(args: dict) -> dict:
    lab_arg = args.get("lab", "")
    # A1 — accept `budget` enum OR legacy `max_cycles` int.
    # Quality-first: default is "auto" (derive from profile/mission)
    # rather than forcing the user to guess a number.
    budget_arg = args.get("budget", args.get("max_cycles", "auto"))
    via_claude = bool(args.get("via_claude", True))  # default: route researcher to Opus

    lab_path = _resolve_lab(lab_arg)
    if lab_path is None:
        return {"ok": False, "error": f"lab not found: {lab_arg!r}"}

    # Read lab.yaml (mission + archetype) to inform budget estimation
    from core import cycle_budget as cb
    try:
        from core import lab_config as lc_mod
        cfg = lc_mod.load(lab_path)
        archetype = cfg.archetype
        mission_text = cfg.mission or _lab_summary(lab_path).get("mission", "")
    except Exception:  # noqa: BLE001
        archetype = "research"
        mission_text = _lab_summary(lab_path).get("mission", "")

    try:
        budget = cb.resolve_budget(
            budget_arg, profile=None, archetype=archetype,
            mission_text=mission_text,
        )
    except ValueError as e:
        return {"ok": False, "error": str(e)}

    # Snapshot pre-state so we can report deltas
    before = _lab_summary(lab_path)

    cmd = [
        sys.executable,
        str(LAB_ROOT / "tools" / "bert_run.py"),
        "--lab", str(lab_path),
        "--max-cycles", str(budget.target),
        "--autonomous",
    ]
    env = None
    if via_claude:
        import os
        env = {**os.environ, "BERT_RESEARCHER_VIA_CLAUDE": "1"}

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=60 * budget.safety_cap * 15,  # 15 min/cycle generous
            env=env, cwd=str(LAB_ROOT),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "lab_cycle timed out",
                "elapsed_secs": round(time.monotonic() - t0, 1)}

    after = _lab_summary(lab_path)
    new_findings = []
    findings_dir = lab_path / "findings"
    if findings_dir.exists():
        for p in sorted(findings_dir.iterdir(), key=lambda x: x.stat().st_mtime,
                        reverse=True):
            if p.stat().st_mtime > t0 - 5:  # 5s slack
                new_findings.append({
                    "name": p.name,
                    "bytes": p.stat().st_size,
                    "preview": p.read_text(errors="replace")[:400],
                })
            if len(new_findings) >= 6:
                break

    # A1 — report saturation status after the run so the host can show
    # the user whether the lab is "done" or just paused at the budget cap.
    sat, novelty_scores = cb.is_saturated(
        lab_path, current_cycle=after["last_cycle"], window=3, threshold=0.3
    )
    return {
        "ok": result.returncode == 0,
        "lab": lab_path.name,
        "exit_code": result.returncode,
        "elapsed_secs": round(time.monotonic() - t0, 1),
        "cycles_before": before["last_cycle"],
        "cycles_after": after["last_cycle"],
        "events_delta": after["events_total"] - before["events_total"],
        "findings_delta": after["findings_count"] - before["findings_count"],
        "new_findings": new_findings,
        "via_claude_researcher": via_claude,
        "budget": {
            "preset": budget.preset_name,
            "target": budget.target,
            "safety_cap": budget.safety_cap,
        },
        "saturation": {
            "saturated": sat,
            "recent_novelty_scores": novelty_scores,
        },
        "stderr_tail": result.stderr[-2000:] if result.stderr else "",
    }


# ── Tool: lab_resume (C1) ───────────────────────────────────────────


def _t_lab_resume(args: dict) -> dict:
    """Resume a cycle that paused at a needs_user_input fork.

    Args:
      token  — the resume_token from the prior NeedsUserInput envelope
      answer — the user's chosen Option value (or free-form string)

    The runner consumes the saved_state from the verified token and
    continues the cycle from the recorded step_id. Implementation
    is intentionally minimal in C1 — agent loop integration lands in
    C3 (async dispatch) and C4 (organic spawn).
    """
    from core import pause_resume as _pr
    token = (args.get("token") or "").strip()
    answer = (args.get("answer") or "").strip()
    if not token:
        return {"ok": False, "error": "token required"}
    if not answer:
        return {"ok": False, "error": "answer required"}
    state = _pr.verify_resume_token(token)
    if state is None:
        return {"ok": False, "error": "token invalid or expired"}
    lab_path = _resolve_lab(state.lab)
    if lab_path is None:
        return {"ok": False,
                "error": f"lab from token not found: {state.lab!r}"}
    # Clear the paused-state file (the answer is being applied)
    _pr.clear_paused(lab_path, state.step_id)
    return {
        "ok": True,
        "lab": state.lab,
        "cycle": state.cycle,
        "step_id": state.step_id,
        "answer": answer,
        "saved_state": state.saved_state,
        "next": (
            "Call lab_cycle to continue; the runner will pick up the "
            "saved_state and apply the user's answer. (Full async "
            "pause/resume integration lands in Phase C3.)"
        ),
    }


# ── Tool: lab_reshape (C6) ──────────────────────────────────────────


def _t_lab_reshape(args: dict) -> dict:
    """Reshape a lab's mission_profile (within-shape only in v1).

    Two modes:
      auto:    args has no `updates` — runs propose_reshape() based on
               drift_score() of recent cycles; returns the proposal.
               Caller (host) shows it to user; user confirms by calling
               again with `updates` populated.
      apply:   args has `updates` dict — applies the within-shape
               reshape immediately.

    Cross-shape reshape is DEFERRED to v1.1; users with shape changes
    archive + create new lab.
    """
    from core import profile_drift as pd
    lab_arg = args.get("lab", "")
    updates = args.get("updates")
    lab_path = _resolve_lab(lab_arg)
    if lab_path is None:
        return {"ok": False, "error": f"lab not found: {lab_arg!r}"}

    if not updates:
        # Auto-propose based on drift
        report = pd.drift_score(lab_path)
        return {
            "ok": True,
            "mode": "propose",
            "lab": lab_path.name,
            "drift_score": report.score,
            "cycles_inspected": report.cycles_inspected,
            "recommendation": report.recommendation,
            "proposed_changes": report.proposed_changes,
            "next": (
                "If you agree with the proposal, call lab_reshape again "
                "with updates={...} populated from proposed_changes. "
                "Only within-shape reshapes are supported in v1."
            ),
        }

    # Apply mode
    result = pd.within_shape_reshape(lab_path, updates)
    result["mode"] = "apply"
    result["lab"] = lab_path.name
    return result


# ── Tool: memory_search ─────────────────────────────────────────────


def _grep_memory(lab_path, query: str, k: int) -> list[dict]:
    """Literal substring scan over a lab's memories/ + findings/ markdown.
    The fast escape hatch and the graceful fallback when retrieval fails."""
    hits: list[dict] = []
    for root in (lab_path / "memories", lab_path / "findings"):
        if not root.exists():
            continue
        for p in root.rglob("*.md"):
            try:
                text = p.read_text(errors="replace")
            except OSError:
                continue
            idx = text.lower().find(query.lower())
            if idx < 0:
                continue
            start = max(0, idx - 100)
            end = min(len(text), idx + len(query) + 300)
            hits.append({
                "path": str(p.relative_to(lab_path)),
                "snippet": text[start:end],
            })
            if len(hits) >= k:
                break
        if len(hits) >= k:
            break
    return hits[:k]


def _t_memory_search(args: dict) -> dict:
    lab_arg = args.get("lab", "")
    query = (args.get("query") or "").strip()
    k = max(1, min(int(args.get("k", 5)), 20))
    # mode: "hybrid" (default) | "vector" | "grep".
    #   hybrid = the full retrieval engine (dense bge + BM25, fused by RRF,
    #            then a bge-reranker-v2-m3 cross-encoder) — what makes this a
    #            semantic memory rather than a grep box.
    #   vector = dense-only (no BM25/rerank); grep = literal substring.
    # The prewarm daemon (serve() → prewarm.prewarm()) pins the embedder +
    # reranker resident at server start, so hybrid-by-default does not pay a
    # cold-start load on the first call. Back-compat: a legacy `use_vector`
    # flag maps true→hybrid, false→grep when `mode` is unset.
    mode = (args.get("mode") or "").strip().lower()
    if not mode:
        if "use_vector" in args:
            mode = "hybrid" if bool(args.get("use_vector")) else "grep"
        else:
            mode = "hybrid"

    lab_path = _resolve_lab(lab_arg)
    if lab_path is None:
        return {"ok": False, "error": f"lab not found: {lab_arg!r}"}
    if not query:
        return {"ok": False, "error": "query required"}

    # Keep any ingested external sources fresh before searching (best-effort,
    # TTL-gated). This is what lets a re-ingested project stay current with no
    # agent effort. A resync failure must never break search.
    try:
        from core import lab_context as _lc
        from core import memory as _mem
        _tok = _lc.set_active_lab_path(lab_path)
        try:
            _mem.resync_sources(force=False, eager_index=False)
        finally:
            _lc.reset_active_lab_path(_tok)
    except Exception:  # noqa: BLE001 — freshness is best-effort
        pass

    fallback_reason = ""
    if mode in ("hybrid", "vector"):
        try:
            from core import lab_context, memory
            token = lab_context.set_active_lab_path(lab_path)
            try:
                if mode == "hybrid":
                    from core import retrieval as _ret
                    results = _ret.hybrid_retrieve(
                        query, k_per_source=max(k * 4, 20), top_n=k,
                    )
                    hits = [{
                        "path": (r.metadata or {}).get("path", ""),
                        "chunk_idx": (r.metadata or {}).get("chunk_idx", 0),
                        "snippet": r.text,
                        "score": r.final_score,
                        "sources": r.sources,  # which signal(s) surfaced this
                    } for r in results]
                else:
                    hits = memory.search(query, k=k)
            finally:
                lab_context.reset_active_lab_path(token)
            return {
                "ok": True,
                "lab": lab_path.name,
                "query": query,
                "method": mode,
                "results": hits,
            }
        except Exception as e:  # noqa: BLE001
            fallback_reason = (f"{mode} retrieval failed, fell back to grep: "
                               f"{type(e).__name__}: {e}")
        # fall through to grep on any retrieval failure
    elif mode != "grep":
        fallback_reason = f"unknown mode {mode!r}; used grep"

    hits = _grep_memory(lab_path, query, k)
    out = {
        "ok": True,
        "lab": lab_path.name,
        "query": query,
        "method": "grep",
        "results": hits,
    }
    if fallback_reason:
        out["fallback_reason"] = fallback_reason
    return out


# ── Tool: memory_ingest ─────────────────────────────────────────────


def _t_memory_ingest(args: dict) -> dict:
    lab_arg = args.get("lab", "")
    source = (args.get("source") or "").strip()

    lab_path = _resolve_lab(lab_arg)
    if lab_path is None:
        return {"ok": False, "error": f"lab not found: {lab_arg!r}"}
    if not source:
        return {"ok": False, "error": "source required"}
    src = Path(source).expanduser()
    if not src.exists() or not src.is_dir():
        return {"ok": False,
                "error": f"source is not an existing directory: {source!r}"}

    exts_t = None
    raw_exts = args.get("exts")
    if isinstance(raw_exts, list) and raw_exts:
        exts_t = tuple(e if e.startswith(".") else f".{e}" for e in raw_exts)

    from core import lab_context, memory
    token = lab_context.set_active_lab_path(lab_path)
    try:
        kw: dict = {"eager_index": True}
        if exts_t:
            kw["exts"] = exts_t
        rep = memory.ingest_corpus_report(src, **kw)
        memory.register_ingest_source(src, exts=exts_t, dest="findings/corpus")
        result = {
            "ok": True,
            "lab": lab_path.name,
            "source": str(src.resolve()),
            "files_ingested": rep["written"],
            "files_skipped": rep["skipped"],
            "files_removed": rep["removed"],
            "truncated": rep["truncated"],
        }
        try:
            st = memory.stats()
            result["chunks_indexed"] = st.get("chunks_indexed")
            result["db_bytes"] = st.get("db_bytes")
        except Exception:  # noqa: BLE001 — stats needs sqlite-vec; degrade
            pass
        return result
    finally:
        lab_context.reset_active_lab_path(token)


# ── Tool: packet_export ─────────────────────────────────────────────


def _t_packet_export(args: dict) -> dict:
    lab_arg = args.get("lab", "")
    cycle_id = args.get("cycle_id")

    lab_path = _resolve_lab(lab_arg)
    if lab_path is None:
        return {"ok": False, "error": f"lab not found: {lab_arg!r}"}

    if cycle_id is None:
        summary = _lab_summary(lab_path)
        cycle_id = summary["last_cycle"]
    cycle_id = int(cycle_id)
    if cycle_id <= 0:
        return {"ok": False, "error": "no cycles in this lab yet — run lab_cycle first"}

    try:
        from core import lab_context, proof_packet
        token = lab_context.set_active_lab_path(lab_path)
        try:
            output_dir = lab_path / "findings" / "proof_packets"
            packet_path = proof_packet.build_packet(
                cycle_id=cycle_id,
                output_dir=output_dir,
            )
        finally:
            lab_context.reset_active_lab_path(token)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "cycle_id": cycle_id}

    return {
        "ok": True,
        "lab": lab_path.name,
        "cycle_id": cycle_id,
        "packet_path": str(packet_path),
        "packet_bytes": packet_path.stat().st_size,
        "verify_with": f"bert verify {packet_path}",
    }


# ── Server factory ──────────────────────────────────────────────────


def make_server() -> MCPServer:
    srv = MCPServer(name="bert", version="0.1.0", namespace="bert")

    srv.register_tool(
        "lab_list",
        description=(
            "List bert labs available on this machine. Returns name, "
            "path, mission, last cycle, event count, findings count "
            "for each. Use this to discover existing labs before "
            "calling lab_status or lab_cycle."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "prefix": {
                    "type": "string",
                    "description": "Optional name prefix filter",
                },
            },
        },
        handler=_t_lab_list,
    )

    srv.register_tool(
        "lab_status",
        description=(
            "Inspect a single lab's current state: cycle count, "
            "findings count, last activity. Lab can be referenced "
            "by name (e.g. 'test01') or absolute path."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "lab": {
                    "type": "string",
                    "description": "Lab name or absolute path",
                },
            },
            "required": ["lab"],
        },
        handler=_t_lab_status,
    )

    srv.register_tool(
        "lab_finalize",
        description=(
            "Finalize a project end-to-end: gather the lab's evidence, "
            "synthesize a polished + cited artifact, disclose honest gaps, then "
            "grade it (4 judges, median+variance across 8 quality dimensions) and "
            "sign it for the proof packet. Returns grade (A-F), the signed hash, "
            "the artifact + gaps.md paths, and `ready` (true iff grade A/B with "
            "gaps disclosed and a ledger row written). This is the deliverable-"
            "producing step — run it when the lab's investigation is complete."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "objective": {"type": "string",
                              "description": "What the artifact should answer/deliver"},
                "output_path": {"type": "string",
                                "description": "Where to write the artifact (e.g. final.md)"},
                "lab": {"type": "string",
                        "description": "Lab name or absolute path (default: the bert-lab repo)"},
            },
            "required": ["objective", "output_path"],
        },
        handler=_t_lab_finalize,
    )

    srv.register_tool(
        "lab_synthesize_tool",
        description=(
            "Synthesize a NEW tool when a mission needs a capability the registry "
            "lacks. A cross-family LLM writes the source + a smoke test, an AST "
            "scan flags foot-guns, and the candidate runs sandboxed. The result is "
            "queued in state/tools_pending_pi.md for PI review — the tool is NOT "
            "active until a human /approves it. Returns the proposal id, scan "
            "safety, and the sandbox exit code. Use for organic capability growth."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string",
                         "description": "Function name (snake_case, valid Python identifier)"},
                "description": {"type": "string",
                                "description": "What the tool does"},
                "params_schema": {"type": "object",
                                  "description": "JSON Schema for the tool's params (optional)"},
                "returns": {"type": "string",
                            "description": "Description of the return shape (optional)"},
                "implementation_hint": {"type": "string",
                                        "description": "A hint for how to implement it (optional)"},
            },
            "required": ["name", "description"],
        },
        handler=_t_lab_synthesize_tool,
    )

    srv.register_tool(
        "lab_approve",
        description=(
            "Approve a pending proposal so it activates. A `tool-*` proposal id "
            "installs + registers the sandboxed synthesized tool; a `prop-*` id "
            "promotes the drafted skill to active. Calling this is the PI blessing "
            "— it's the human gate between propose and use. Idempotent (re-approve "
            "is a no-op). Returns {ok, kind, ...}."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "proposal_id": {"type": "string",
                                "description": "Proposal id from lab_synthesize_tool (tool-*) or skill mining (prop-*)"},
            },
            "required": ["proposal_id"],
        },
        handler=_t_lab_approve,
    )

    srv.register_tool(
        "lab_start",
        description=(
            "Create a new bert lab with a mission. Scaffolds the "
            "directory structure (memories, findings, sor, state) "
            "and a seed_brief.md. After this, call lab_cycle to "
            "start autonomous research."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Slug-safe lab name (no slashes)",
                },
                "mission": {
                    "type": "string",
                    "description": "What the lab should investigate (≥20 chars)",
                },
                "archetype": {
                    "type": "string",
                    "enum": ["research", "product", "strategy"],
                    "description": "Lab type (default: research)",
                },
            },
            "required": ["name", "mission"],
        },
        handler=_t_lab_start,
    )

    srv.register_tool(
        "lab_cycle",
        description=(
            "Run autonomous cycles on a lab. Each cycle: director "
            "decides focus → researcher gathers (via Opus 4.7 if "
            "via_claude=true, default) → strategist proposes next "
            "action → both write findings to disk.\n\n"
            "Quality-first defaults: budget='auto' derives cycle count "
            "from the lab's mission (one-shot questions get 'quick' 3 "
            "cycles; multi-week investigations get 'deep' 15-30; "
            "monitor missions get 'until_complete' which runs until "
            "director signals mission-complete, capped at 50). "
            "Director also signals mission-complete when saturation "
            "is detected (3 consecutive cycles producing no new "
            "findings).\n\n"
            "Returns: delta summary (events, findings, cycles), "
            "previews of new findings, budget actually used, and "
            "saturation status."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "lab": {
                    "type": "string",
                    "description": "Lab name or absolute path",
                },
                "budget": {
                    "oneOf": [
                        {
                            "type": "string",
                            "enum": [
                                "auto", "quick", "standard",
                                "deep", "until_complete",
                            ],
                        },
                        {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50,
                        },
                    ],
                    "description": (
                        "Budget preset or explicit cycle count. "
                        "Defaults to 'auto' (derive from mission). "
                        "'quick' = 1-3 cycles; 'standard' = 5-10; "
                        "'deep' = 15-30; 'until_complete' = run until "
                        "director signals mission-complete (cap 50). "
                        "Integer = explicit target with 2× safety cap."
                    ),
                },
                "max_cycles": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": (
                        "Legacy alias for `budget` when passed an int. "
                        "Prefer `budget` for new code."
                    ),
                },
                "via_claude": {
                    "type": "boolean",
                    "description": (
                        "Route the researcher dispatch through Claude "
                        "Code (Opus 4.7) instead of the free-tier "
                        "model. Default true. Set false to use bert's "
                        "default provider chain."
                    ),
                },
            },
            "required": ["lab"],
        },
        handler=_t_lab_cycle,
    )

    srv.register_tool(
        "lab_reshape",
        description=(
            "Reshape a lab's mission_profile based on observed drift "
            "or explicit updates (within-shape only in v1; "
            "cross-shape reshapes archive + recreate). Call with no "
            "updates to get an auto-proposal from drift_score; call "
            "with `updates={...}` to apply specific field changes."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "lab": {"type": "string"},
                "updates": {
                    "type": "object",
                    "description": (
                        "Field-level updates to mission_profile. "
                        "Omit for auto-proposal mode. Only same-shape "
                        "changes accepted in v1."
                    ),
                },
            },
            "required": ["lab"],
        },
        handler=_t_lab_reshape,
    )

    srv.register_tool(
        "lab_resume",
        description=(
            "Resume a paused lab cycle that asked the user a question "
            "via a needs_user_input envelope. Pass the resume_token "
            "from the prior envelope along with the user's chosen "
            "answer. The runner reads the saved_state from the "
            "verified token + continues from the recorded step."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "token": {
                    "type": "string",
                    "description": "resume_token from prior needs_user_input envelope",
                },
                "answer": {
                    "type": "string",
                    "description": "User's chosen Option value (or free-form answer)",
                },
            },
            "required": ["token", "answer"],
        },
        handler=_t_lab_resume,
    )

    srv.register_tool(
        "memory_search",
        description=(
            "Semantic search over a lab's memories + findings. Defaults to the "
            "full hybrid retrieval engine: dense (bge-base-en-v1.5) + BM25, fused "
            "by reciprocal rank fusion, then re-scored by a bge-reranker-v2-m3 "
            "cross-encoder. Set mode='grep' for a fast literal substring scan, or "
            "mode='vector' for dense-only. Returns the top k matches with path + "
            "snippet (and, for hybrid, the fusion score + which signals hit). "
            "Falls back to grep if the retrieval models are unavailable."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "lab": {"type": "string"},
                "query": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["hybrid", "vector", "grep"],
                    "description": "Retrieval mode (default hybrid).",
                },
                "k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Max hits to return (default 5)",
                },
            },
            "required": ["lab", "query"],
        },
        handler=_t_memory_search,
    )

    srv.register_tool(
        "memory_ingest",
        description=(
            "Index an EXISTING external project/codebase into a lab so the "
            "agent can retrieve over it. Walks `source` (a directory), shards "
            "the supported text/code files into the lab corpus, and embeds "
            "them. Incremental on re-call (only changed files re-embed); the "
            "source is remembered so later memory_search calls auto-refresh "
            "it. Returns file counts. Never modifies the source tree; this is "
            "files-only (not a live database connector)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "lab": {
                    "type": "string",
                    "description": "Lab name or absolute path",
                },
                "source": {
                    "type": "string",
                    "description": "Absolute path to the project directory to index",
                },
                "exts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional file extensions to include "
                                   "(default: common code + docs types)",
                },
            },
            "required": ["lab", "source"],
        },
        handler=_t_memory_ingest,
    )

    srv.register_tool(
        "packet_export",
        description=(
            "Build a signed proof packet (.tar.gz) for a specific "
            "cycle. The packet contains: cycle.json (events + "
            "claims), failures.md (separately signed), in-toto "
            "attestation, Sigstore bundle. Verifiable with "
            "`bert verify <packet>`. If cycle_id is omitted, uses "
            "the latest cycle in the lab."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "lab": {"type": "string"},
                "cycle_id": {
                    "type": "integer",
                    "description": "Specific cycle to attest (default: latest)",
                },
            },
            "required": ["lab"],
        },
        handler=_t_packet_export,
    )

    _wire_resources_and_prompts(srv)
    return srv


def _wire_resources_and_prompts(srv: MCPServer) -> None:
    """Populate the MCP resource + prompt primitives (Sprint 4 A1).

    Resources: each lab's read-only artifacts (seed_brief.md, lab.yaml).
    Prompts: each seed feature exposed as a reusable, topic-templated prompt
    a host can fetch via prompts/get(name, {topic}).
    """
    lab_dirs: list[Path] = []
    if LABS_DIR.exists():
        lab_dirs += [d for d in sorted(LABS_DIR.iterdir()) if d.is_dir()]
    supervisor = LAB_ROOT / "lab"
    if supervisor.exists():
        lab_dirs.append(supervisor)
    for lab in lab_dirs:
        for artifact, mime in (("seed_brief.md", "text/markdown"),
                               ("lab.yaml", "application/yaml")):
            f = lab / artifact
            if not f.exists():
                continue
            srv.register_resource(
                uri=f"bert://lab/{lab.name}/{artifact}",
                name=f"{lab.name}/{artifact}",
                description=f"{artifact} for lab {lab.name!r}",
                mime_type=mime,
                reader=lambda p=f: p.read_text(encoding="utf-8", errors="replace"),
            )

    features_dir = LAB_ROOT / "core" / "library" / "features"
    if features_dir.exists():
        for fp in sorted(features_dir.glob("*.md")):
            srv.register_prompt(
                fp.stem,
                description=f"bert `{fp.stem}` feature — run its pipeline on a topic",
                arguments=[{"name": "topic",
                            "description": "the subject / target to run the feature on",
                            "required": True}],
                builder=lambda a, body=fp.read_text(encoding="utf-8", errors="replace"),
                fn=fp.stem: [{
                    "role": "user",
                    "content": {"type": "text",
                                "text": f"Run the bert `{fn}` feature on: "
                                        f"{a.get('topic', '')}\n\n---\n{body}"},
                }],
            )


def serve() -> int:
    """Real server entry point: pin the retrieval models resident (so the first
    search doesn't pay the cold-start tail), then serve over stdio. Kept separate
    from make_server() — building a server (which tests do constantly) must NOT
    load models."""
    from core import prewarm
    prewarm.prewarm()  # background daemon: embedder + reranker
    return make_server().serve_stdio()


if __name__ == "__main__":
    sys.exit(serve())
