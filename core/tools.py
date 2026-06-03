"""Built-in tool implementations.

Read, Write, Edit, Bash, Grep, Glob, WebFetch, WebSearch, Spawn-subagent.
Each tool is a function with JSON Schema input + handler. Registers itself
with tool_registry on import.

P-019: Bash respects sandbox tier. Trusted = subprocess+timeout. Docker =
spawn isolated container. sandbox-exec = macOS profile (Phase 3+).
"""

from __future__ import annotations

import re
import subprocess
import urllib.parse
from datetime import UTC
from pathlib import Path

import httpx

from core import tool_registry
from core.lab_context import get_active_lab_path
from core.types import PermissionMode, SandboxTier

LAB_ROOT = Path(__file__).resolve().parent.parent


def _resolve_relative_path(file_path: str) -> Path:
    """Resolve a relative file path against the active lab's directory
    (if set via core.lab_context) else against the bert-lab repo root.

    Without this, every Write tool call from a subagent dispatched on
    a user lab landed in bert-lab/drafts/ instead of <lab>/drafts/, so
    canvas_watcher (which watches LAB_ROOT/findings/) never saw the
    user-lab's artifacts and Atlas/Manuscript/Loom stayed empty for
    user labs even after the F1–F4 fixes routed events correctly.
    """
    p = Path(file_path).expanduser()
    if p.is_absolute():
        # Absolute paths are a documented escape hatch (the schema allows them;
        # tests write to tmp dirs; an agent with Bash can already write anywhere).
        return p
    root = get_active_lab_path() or LAB_ROOT
    candidate = root / p
    # Containment: a lab-relative path must NOT escape the lab root via `..`.
    # This is the surprising, dangerous case (a "lab-relative" path landing
    # outside the lab). Raise so each file tool returns its native error shape.
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"path escapes the lab root: {file_path!r}")
    return candidate

# Identify ourselves transparently — some servers will still block, in which
# case the agent can use Bash + curl with custom headers as an escape hatch.
_BERT_USER_AGENT = (
    "bert-lab/0.1 (educational research; +https://github.com/harshithkantamneni/bert)"
)
# DuckDuckGo HTML endpoint sniffs UAs; use a real-browser UA for search to
# avoid empty-result pages.
_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15"
)


# ── Read ────────────────────────────────────────────────────────────


def _read(file_path: str, offset: int | None = None, limit: int | None = None) -> str:
    """Read a file. Empty-state guarantee: missing file → empty string."""
    try:
        p = _resolve_relative_path(file_path)
    except ValueError as e:
        return f"[Read error: {e}]"
    if not p.exists():
        return ""
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except (PermissionError, IsADirectoryError) as e:
        return f"[Read error: {type(e).__name__}: {e}]"
    if offset is not None or limit is not None:
        lines = text.splitlines(keepends=True)
        start = max(0, (offset or 1) - 1)
        end = start + (limit if limit is not None else len(lines))
        text = "".join(lines[start:end])
    return text


tool_registry.register_function(
    name="Read",
    description=(
        "Read a file from the bert-lab filesystem. Returns the file contents. "
        "If the file is missing, returns an empty string (empty-state guarantee). "
        "Optional `offset` (1-indexed line number) and `limit` (line count) for "
        "large files."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute or lab-relative path"},
            "offset": {"type": "integer", "minimum": 1, "description": "Starting line (1-indexed)"},
            "limit":  {"type": "integer", "minimum": 1, "description": "Number of lines"},
        },
        "required": ["file_path"],
    },
    handler=_read,
)


# ── Write ───────────────────────────────────────────────────────────


def _write(file_path: str, content: str) -> str:
    """Write content to a file (atomic: tmp + rename). Creates parent dirs.

    Relative paths route into the active lab's directory (see
    _resolve_relative_path). For the bert-lab supervisor default that
    resolves to LAB_ROOT — same behavior as before. For a user lab
    (set via core.lab_context.set_active_lab_path) the path lands
    under ~/.bert/labs/<lab>/.

    Side effect: writes whose resolved path is `findings/<file>.md`
    (relative to the active lab root) emit a `finding` event so the
    Manuscript surface and the Atlas strata get populated without
    waiting on canvas_watcher to scan the directory. Without this
    auto-emit, the file would only become visible via canvas_watcher
    (which today watches the bert-lab default repo only, not user
    labs) — so the artifact would sit on disk unseen by the UI.
    """
    try:
        p = _resolve_relative_path(file_path)
    except ValueError as e:
        return f"[Write error: {e}]"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(p)
    _maybe_emit_finding_event(p, content)
    return f"wrote {p} ({len(content)} bytes)"


def _maybe_emit_finding_event(p: Path, content: str, *, lineage: list | None = None) -> None:
    """If the write is into the active lab's findings/ dir, emit a
    `finding` event so Manuscript + Atlas pick it up immediately.

    `lineage` (prior findings this one cites) populates the event's lineage
    field — record_finding passes it; the plain Write tool leaves it empty.

    Detection is by directory match (must contain a `findings`
    component) + filename ending in `.md`. Anything under
    `findings/archive/` is skipped — matches canvas_watcher's policy
    of treating archived findings as historical, not live.
    """
    try:
        if p.suffix != ".md":
            return
        if "archive" in p.parts:
            return
        # Find the "findings" segment in the path
        try:
            idx = p.parts.index("findings")
        except ValueError:
            return
        # Reject if findings/ has any deeper subdir other than .md
        # (e.g. findings/2026-05/foo.md) — keep flat for the v1.
        if idx + 1 != len(p.parts) - 1:
            return

        base = get_active_lab_path() or LAB_ROOT
        try:
            rel = str(p.relative_to(base))
        except ValueError:
            rel = str(p)

        # Build the canvas event shape directly + append to the
        # active lab's events.jsonl. We bypass observability.emit
        # because its canvas mapper builds `content` from role/model
        # /provider/verdict/tool keys — none of which a finding has;
        # the mapper would produce an empty content string and the
        # /api/findings endpoint's _is_editorial filter would reject
        # the event as non-prose.
        import hashlib
        import json
        from datetime import datetime

        summary = _first_paragraph(content)[:500]
        cycle = _infer_cycle(p.name)
        agent = _infer_agent(p.name) or "researcher"
        # SHA1 as deterministic id (matches collate's _hash_id); not security
        eid = "find_" + hashlib.sha1(rel.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]

        canvas_event = {
            "id": eid,
            "ts": datetime.now(UTC).isoformat(),
            "event_class": "finding",
            "source_path": rel,
            "agent": agent,
            "content": summary,
            "tags": ["#finding"],
            "lineage": list(lineage or []),
            "cycle": cycle,
            "significance": None,
            "phase": None,
            "system": None,
            "severity_grade": None,
            "memory_tier": None,
            "judge_provider": None,
            "position_swap_delta": None,
            "revival_conditions": None,
            "confidence_1to10": None,
            "verdict": None,
            "enrichment_provenance": "write_tool_emit",
        }

        # Resolve target events.jsonl — default lab uses LAB_ROOT/lab/
        # (the conventional supervisor lab dir), user labs use
        # <lab>/sor/events.jsonl directly.
        if base == LAB_ROOT:
            events_path = LAB_ROOT / "lab" / "sor" / "events.jsonl"
        else:
            events_path = base / "sor" / "events.jsonl"
        events_path.parent.mkdir(parents=True, exist_ok=True)
        with events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(canvas_event, separators=(",", ":")) + "\n")

        # F11 — Heuristic graph extraction from the finding content.
        # Best-effort; never blocks the write.
        try:
            from core import graph_extract
            graph_extract.extract_and_persist(
                finding_id=eid,
                content=content,
                cycle=cycle,
                role=agent,
                source_path=rel,
            )
        except Exception:
            pass
    except Exception:
        # Never let event emission break the underlying write.
        pass


def _first_paragraph(text: str) -> str:
    """First non-heading paragraph. Matches canvas_watcher._take_summary."""
    for chunk in text.split("\n\n"):
        s = chunk.strip()
        if not s:
            continue
        if all(line.startswith("#") for line in s.splitlines() if line.strip()):
            continue
        return s
    return text.strip()


def _infer_cycle(filename: str) -> int | None:
    """Pull cycle number out of files like 'bert_run_C5_researcher.md'."""
    m = re.search(r"_C(\d+)", filename)
    return int(m.group(1)) if m else None


def _infer_agent(filename: str) -> str | None:
    """Pull role hint out of filenames like 'bert_run_C5_researcher.md'."""
    for role in ("researcher", "strategist", "evaluator", "implementer",
                 "reflector", "consolidator", "director"):
        if role in filename:
            return role
    return None


tool_registry.register_function(
    name="Write",
    description=(
        "Atomic write to a file. Overwrites existing content. Creates parent "
        "directories. Returns a confirmation with byte count. Per P-011, "
        "writes to system paths and credentials are blocked at the permission gate."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "content":   {"type": "string"},
        },
        "required": ["file_path", "content"],
    },
    handler=_write,
    permission_mode=PermissionMode.DEFAULT,  # writes ask in default mode
)


# ── Edit ────────────────────────────────────────────────────────────


def _edit(file_path: str, old_string: str, new_string: str,
          replace_all: bool = False) -> dict:
    """Atomic exact-string replacement in a file.

    Returns dict with: ok, file_path, replacements, error.

    Errors (return ok=False with descriptive error):
    - file does not exist
    - old_string equals new_string (no-op)
    - old_string not found in file
    - old_string appears multiple times and replace_all=False
    """
    try:
        p = _resolve_relative_path(file_path)
    except ValueError as e:
        return {"ok": False, "file_path": file_path, "replacements": 0, "error": str(e)}
    base = {"ok": False, "file_path": str(p), "replacements": 0, "error": ""}

    if old_string == new_string:
        base["error"] = "old_string and new_string are identical (no-op)"
        return base
    if not p.exists():
        base["error"] = f"file does not exist: {p}"
        return base

    try:
        text = p.read_text(encoding="utf-8")
    except (PermissionError, IsADirectoryError, UnicodeDecodeError) as e:
        base["error"] = f"{type(e).__name__}: {e}"
        return base

    count = text.count(old_string)
    if count == 0:
        base["error"] = (
            f"old_string not found in {p.name}. Verify exact match including "
            f"whitespace and indentation (file has {len(text)} chars, "
            f"{text.count(chr(10))} lines)."
        )
        return base
    if count > 1 and not replace_all:
        base["error"] = (
            f"old_string appears {count} times in {p.name}. Either provide a "
            f"longer old_string with surrounding context to make it unique, or "
            f"pass replace_all=true."
        )
        return base

    if replace_all:
        new_text = text.replace(old_string, new_string)
        replaced = count
    else:
        new_text = text.replace(old_string, new_string, 1)
        replaced = 1

    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(new_text, encoding="utf-8")
    tmp.replace(p)
    base.update({"ok": True, "replacements": replaced})
    return base


tool_registry.register_function(
    name="Edit",
    description=(
        "Atomic exact-string replacement in a file. Cheaper than re-Write for "
        "small changes. old_string must match the file content exactly (including "
        "whitespace and indentation). If old_string appears multiple times, you "
        "must either include more surrounding context to make it unique, or pass "
        "replace_all=true. Returns dict with ok, file_path, replacements, error. "
        "Atomic via tmp+rename. P-011 destructive paths (credentials, /etc/, "
        "/.ssh/) blocked at the permission gate same as Write."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "file_path":   {"type": "string"},
            "old_string":  {"type": "string", "description": "Exact text to find (whitespace-sensitive)"},
            "new_string":  {"type": "string", "description": "Replacement text (must differ from old_string)"},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence; default false (require uniqueness)"},
        },
        "required": ["file_path", "old_string", "new_string"],
    },
    handler=_edit,
    permission_mode=PermissionMode.DEFAULT,
)


# ── Bash ────────────────────────────────────────────────────────────


def _bash(command: str, timeout: int = 120, sandbox: str = "trusted") -> dict:
    """Run a shell command. Returns dict with stdout, stderr, exit_code, elapsed_ms.

    sandbox = "trusted": subprocess + timeout (lab tools only)
    sandbox = "docker":  Phase 3 — implementation pending in core.sandbox
    sandbox = "sandbox-exec": Phase 3 — macOS sandbox-exec profile
    """
    if sandbox != "trusted":
        return {
            "stdout": "",
            "stderr": (
                f"[bert] sandbox tier '{sandbox}' not yet implemented in MVP. "
                f"Use sandbox='trusted' for lab tools, or wait for core.sandbox."
            ),
            "exit_code": 2,
            "elapsed_ms": 0,
        }
    import time as _t
    start = _t.monotonic()
    try:
        r = subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(LAB_ROOT),
        )
        return {
            "stdout": r.stdout[-8000:],   # cap output to avoid context blowup
            "stderr": r.stderr[-2000:],
            "exit_code": r.returncode,
            "elapsed_ms": int((_t.monotonic() - start) * 1000),
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"[bert] command timed out after {timeout}s",
            "exit_code": 124,
            "elapsed_ms": int((_t.monotonic() - start) * 1000),
        }


tool_registry.register_function(
    name="Bash",
    description=(
        "Execute a bash command in bert-lab's working directory. Returns stdout, "
        "stderr, exit_code, elapsed_ms. Default 120s timeout. `sandbox` tier: "
        "'trusted' (subprocess, lab-tools only), 'docker' (isolated container — "
        "Phase 3), 'sandbox-exec' (macOS profile — Phase 3). Per P-011, "
        "destructive operations (rm -rf, drop, force-push) hard-route to human "
        "approval regardless of sandbox tier."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer", "minimum": 1, "maximum": 600},
            "sandbox": {"type": "string", "enum": ["trusted", "docker", "sandbox-exec"]},
        },
        "required": ["command"],
    },
    handler=_bash,
    permission_mode=PermissionMode.DEFAULT,
    sandbox_tier=SandboxTier.TRUSTED,
)


# ── WebFetch ────────────────────────────────────────────────────────


def _clean_html(html: str) -> tuple[str, str]:
    """Extract (title, body_text) from HTML. Strips script/style/nav/footer.
    Prefers <main> or <article> for body; falls back to <body>."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    title_el = soup.find("title")
    title = title_el.get_text(strip=True) if title_el else ""
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header",
                     "aside", "iframe", "svg"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator="\n", strip=True)
    # Collapse runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text


def _webfetch(url: str, prompt: str | None = None, timeout: int = 15,
              max_chars: int = 30000) -> dict:
    """Fetch a URL and return cleaned text content.

    Returns a dict with: ok, url, status_code, content_type, title, content,
    truncated, elapsed_ms, error (if any).

    `prompt` is a context hint from the caller; not used for filtering, but
    surfaced so the caller can record what they were looking for.
    """
    import time as _t
    start = _t.monotonic()
    headers = {
        "User-Agent": _BERT_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml,application/json,text/plain;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    base = {
        "ok": False, "url": url, "status_code": 0, "content_type": "",
        "title": "", "content": "", "truncated": False,
        "elapsed_ms": 0, "error": "",
        "prompt": prompt or "",
    }
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            r = client.get(url, headers=headers)
        elapsed_ms = int((_t.monotonic() - start) * 1000)
        ctype = r.headers.get("content-type", "").lower()
        base.update({
            "status_code": r.status_code,
            "content_type": ctype,
            "elapsed_ms": elapsed_ms,
            "url": str(r.url),  # may have changed via redirect
        })
        if r.status_code >= 400:
            base["error"] = f"HTTP {r.status_code}: {r.text[:200]}"
            return base

        # Choose extraction strategy by content-type
        if "html" in ctype:
            title, content = _clean_html(r.text)
        elif "json" in ctype:
            title, content = "", r.text
        else:
            # text/plain, markdown, csv, etc.
            title, content = "", r.text

        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n... (truncated {len(content) - max_chars} chars) ..."
            base["truncated"] = True
        base.update({"ok": True, "title": title, "content": content})
        return base
    except (httpx.TimeoutException, httpx.NetworkError, httpx.InvalidURL) as e:
        base.update({
            "elapsed_ms": int((_t.monotonic() - start) * 1000),
            "error": f"{type(e).__name__}: {e}",
        })
        return base


tool_registry.register_function(
    name="WebFetch",
    description=(
        "Fetch a URL and return the cleaned text content. HTML is stripped "
        "of script/style/nav/footer; <main> or <article> is preferred for body. "
        "JSON / text / markdown returned as-is. Returns dict with ok, url, "
        "status_code, title, content (capped at 30 KB), truncated, elapsed_ms, "
        "error. Use this for reading articles, papers, GitHub READMEs, blog posts, "
        "API endpoints. For sites that block our UA, fall back to Bash + curl."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "url":     {"type": "string", "description": "Absolute URL (http/https)"},
            "prompt":  {"type": "string", "description": "Optional context hint about why you're fetching this URL"},
            "timeout": {"type": "integer", "minimum": 1, "maximum": 60, "description": "Seconds (default 15)"},
        },
        "required": ["url"],
    },
    handler=_webfetch,
    permission_mode=PermissionMode.AUTO,  # read-only network call
)


# ── WebSearch ───────────────────────────────────────────────────────


def _websearch(query: str, max_results: int = 5) -> dict:
    """Search DuckDuckGo (HTML endpoint) and return ranked results.

    Returns dict with ok, query, results (list of {title, url, snippet}),
    elapsed_ms, error. Free-tier-only: no API key required, but DDG can rate-
    limit aggressive scraping. Cap max_results at 10.
    """
    import time as _t
    start = _t.monotonic()
    max_results = max(1, min(max_results, 10))
    base = {"ok": False, "query": query, "results": [],
            "elapsed_ms": 0, "error": ""}
    headers = {
        "User-Agent": _BROWSER_USER_AGENT,
        "Accept": "text/html",
        "Accept-Language": "en-US,en;q=0.9",
    }
    # POST form-encoded is more reliable than GET; DDG redirects GET to the
    # JS-heavy main site, but the /html/ POST endpoint serves static results.
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            r = client.post(
                "https://html.duckduckgo.com/html/",
                headers=headers,
                data={"q": query, "kl": "us-en"},
            )
        elapsed_ms = int((_t.monotonic() - start) * 1000)
        base["elapsed_ms"] = elapsed_ms
        if r.status_code >= 400:
            base["error"] = f"HTTP {r.status_code}: {r.text[:200]}"
            return base
        results = _parse_ddg_results(r.text, max_results)
        if not results:
            base["error"] = (
                "DuckDuckGo returned no parsable results "
                "(possibly rate-limited or HTML format changed)."
            )
            return base
        base.update({"ok": True, "results": results})
        return base
    except (httpx.TimeoutException, httpx.NetworkError) as e:
        base["error"] = f"{type(e).__name__}: {e}"
        base["elapsed_ms"] = int((_t.monotonic() - start) * 1000)
        return base


def _parse_ddg_results(html: str, max_results: int) -> list[dict]:
    """Extract result rows from DDG html endpoint."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    results: list[dict] = []
    for div in soup.select("div.result")[:max_results * 2]:
        a = div.select_one("a.result__a")
        snippet_el = div.select_one(".result__snippet")
        if not a or not a.get("href"):
            continue
        # DDG wraps real URLs in a redirect — extract uddg= param
        raw_href = a.get("href", "")
        url = _unwrap_ddg_url(raw_href)
        title = a.get_text(" ", strip=True)
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
        if not url or not title:
            continue
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= max_results:
            break
    return results


def _unwrap_ddg_url(href: str) -> str:
    """DDG hrefs look like //duckduckgo.com/l/?uddg=<encoded_real_url>&...
    Return the decoded real URL."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    qs = urllib.parse.parse_qs(parsed.query)
    if "uddg" in qs:
        return urllib.parse.unquote(qs["uddg"][0])
    return href


tool_registry.register_function(
    name="WebSearch",
    description=(
        "Search the web (DuckDuckGo HTML endpoint, free-tier, no API key). "
        "Returns dict with ok, query, results (list of {title, url, snippet}), "
        "elapsed_ms, error. max_results capped at 10. Use this to discover "
        "URLs to feed into WebFetch. For ArXiv-specific search, query "
        "'site:arxiv.org <terms>' to constrain results. If DDG rate-limits, "
        "wait + retry or fall back to direct ArXiv/GitHub API queries via "
        "Bash + curl."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "query":       {"type": "string", "minLength": 2},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Default 5"},
        },
        "required": ["query"],
    },
    handler=_websearch,
    permission_mode=PermissionMode.AUTO,
)


# ── memory_search ───────────────────────────────────────────────────


def _memory_search(query: str, k: int = 5) -> dict:
    """Hybrid retrieval across indexed memories/ + findings/ corpus.

    A5 — upgraded from vector-only to hybrid (vector + graph + semantic
    cache via RRF fusion). Returns the same dict shape as before for
    back-compat (hits = list of {path, chunk_idx, content, distance})
    so existing callers don't break.

    On any hybrid_retrieve failure, falls back to direct vector search
    via `core.memory.search` (this preserves bert's behavior at the
    old call site if the new retrieval stack has a transient issue).
    """
    import sqlite3 as _sql
    import time as _t
    start = _t.monotonic()
    # LLMs sometimes pass k as a string ("5") via the tool call; coerce to int.
    try:
        k = int(k)
    except (TypeError, ValueError):
        k = 5
    # First try the hybrid retrieval (vector + graph + cache + RRF)
    try:
        from core import retrieval as _ret
        results = _ret.hybrid_retrieve(
            query, k_per_source=max(k * 4, 20), top_n=k,
        )
        # Convert RetrievalResult → legacy hit dict shape
        hits = []
        for r in results:
            meta = r.metadata or {}
            hits.append({
                "path": meta.get("path", ""),
                "chunk_idx": meta.get("chunk_idx", 0),
                "content": r.text,
                "distance": 1.0 - r.final_score if r.final_score else 1.0,
                "score": r.final_score,
                "sources": r.sources,  # which signal(s) surfaced this
            })
        return {
            "ok": True, "query": query, "hits": hits,
            "method": "hybrid",
            "elapsed_ms": int((_t.monotonic() - start) * 1000),
            "error": "",
        }
    except Exception as hybrid_err:  # noqa: BLE001
        import logging
        import traceback
        logging.getLogger("bert.retrieval").warning(
            "memory_search hybrid path failed → fallback to vector. "
            "query=%r exc=%s: %s\nTRACE:\n%s",
            query[:80], type(hybrid_err).__name__, hybrid_err,
            traceback.format_exc(),
        )
        # Fall back to vector-only on any hybrid failure
        try:
            from core import memory as _mem
            hits = _mem.search(query, k=k)
            return {
                "ok": True, "query": query, "hits": hits,
                "method": "vector_fallback",
                "fallback_reason": f"{type(hybrid_err).__name__}: {hybrid_err}",
                "elapsed_ms": int((_t.monotonic() - start) * 1000),
                "error": "",
            }
        except (_sql.OperationalError, RuntimeError, ImportError, OSError) as e:
            return {
                "ok": False, "query": query, "hits": [],
                "method": "failed",
                "elapsed_ms": int((_t.monotonic() - start) * 1000),
                "error": f"{type(e).__name__}: {e}",
            }


tool_registry.register_function(
    name="memory_search",
    description=(
        "Vector search across bert's markdown memory corpus (memories/ + findings/). "
        "Returns top-k chunks ranked by cosine distance (lower = more similar). "
        "Auto-indexes new/changed files on each call. Use this to recall prior "
        "decisions, prior research findings, killed ideas, or governance directives "
        "before re-doing work that's already been done. k capped at 20."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 2},
            "k":     {"type": "integer", "minimum": 1, "maximum": 20, "description": "Default 5"},
        },
        "required": ["query"],
    },
    handler=_memory_search,
    permission_mode=PermissionMode.AUTO,  # read-only
)


# ── memory_create ───────────────────────────────────────────────────


def _memory_create(path: str, content: str) -> dict:
    """Atomic write of a memory file. Path must be under memories/ or findings/.

    Returns dict with ok, path, bytes, error. Indexing is lazy — happens on the
    next memory_search call (mtime-driven).
    """
    from core import memory as _mem
    return _mem.create(path, content)


tool_registry.register_function(
    name="memory_create",
    description=(
        "Atomic write of a memory file (markdown). Path is scoped to memories/ or "
        "findings/ — rejects everything else. Use this for findings reports, "
        "decision-log entries, killed-idea entries, semantic notes. For arbitrary "
        "paths (state/, agents/, code), use Write instead. Indexing is lazy: "
        "memory_search will pick up the new content on its next call."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "path":    {"type": "string", "description": "Path under memories/ or findings/"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
    handler=_memory_create,
    permission_mode=PermissionMode.DEFAULT,  # writes ask in default mode
)


# ── record_finding (macro-op fusion: Write + memory_create + lineage) ─


def _record_finding(content: str, name: str, summary: str = "",
                    lineage: list | None = None) -> dict:
    """Atomically record a finding — the fused form of the common
    Write(findings/…) + memory_create(log) pair (25% of agent tool calls).

    Writes `content` into the active lab's findings/<name>.md (with the lineage
    recorded in the file), emits the finding event carrying that lineage, and
    appends a one-line `summary` to memories/log.md. Returns {ok, finding_path,
    log_path, lineage, error}."""
    from datetime import datetime
    lineage = list(lineage or [])
    rel = name if name.startswith("findings/") else f"findings/{name}.md"
    if not rel.endswith(".md"):
        rel += ".md"
    body = content
    if lineage:
        body = f"**Lineage:** {', '.join(str(x) for x in lineage)}\n\n{content}"

    try:
        fp = _resolve_relative_path(rel)
    except ValueError as e:
        return {"ok": False, "finding_path": None, "log_path": None,
                "lineage": lineage, "error": str(e)}
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = fp.with_suffix(fp.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(fp)
    _maybe_emit_finding_event(fp, body, lineage=lineage)

    out = {"ok": True, "finding_path": rel, "log_path": None,
           "lineage": lineage, "error": ""}

    if summary.strip():
        try:
            lp = _resolve_relative_path("memories/log.md")
        except ValueError:
            return out
        lp.parent.mkdir(parents=True, exist_ok=True)
        existing = lp.read_text(encoding="utf-8") if lp.exists() else ""
        entry = f"- {datetime.now(UTC).isoformat()} {summary.strip()} (finding: {name}"
        if lineage:
            entry += f"; lineage: {lineage}"
        entry += ")\n"
        ltmp = lp.with_suffix(lp.suffix + ".tmp")
        ltmp.write_text(existing + entry, encoding="utf-8")
        ltmp.replace(lp)
        out["log_path"] = "memories/log.md"
    return out


tool_registry.register_function(
    name="record_finding",
    description=(
        "Record a finding in ONE atomic call — the fused form of writing a "
        "findings/ artifact and logging it (the common Write+memory_create pair). "
        "Writes `content` to findings/<name>.md with `lineage` (prior findings "
        "this cites) recorded in the file + the finding event, and appends a "
        "one-line `summary` to memories/log.md. Prefer this over separate Write + "
        "memory_create when recording a finding."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The finding body (markdown)"},
            "name":    {"type": "string", "description": "Finding file stem -> findings/<name>.md"},
            "summary": {"type": "string", "description": "One-line log summary (optional)"},
            "lineage": {"type": "array", "items": {"type": "string"},
                        "description": "Prior finding names/paths this cites (optional)"},
        },
        "required": ["content", "name"],
    },
    handler=_record_finding,
    permission_mode=PermissionMode.DEFAULT,
)


# ── Spawn (sub-agent dispatch) ──────────────────────────────────────


def _spawn(spec: dict) -> dict:
    """Dispatch a sub-agent with a scoped DispatchSpec.

    Validates the spec against schemas/dispatch_spec.json, runs the named
    role's agent loop, reads the ResultPacket the sub-agent wrote, and
    returns a Director-friendly summary.
    """
    # Lazy import to avoid circular: tools loads at agent boot, subagent imports agent
    from core import subagent
    return subagent.run_subagent(spec)


tool_registry.register_function(
    name="Spawn",
    description=(
        "Dispatch a sub-agent (researcher / strategist / implementer / evaluator / "
        "reflector / consolidator) with a scoped DispatchSpec. The sub-agent runs "
        "its own agent loop and returns a structured ResultPacket summary "
        "(verdict, findings_count, confidence_1to10, calibration_reasoning, "
        "telemetry). Use this for delegation; do not run sub-agent work inline. "
        "DispatchSpec must validate against schemas/dispatch_spec.json."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "spec": {
                "type": "object",
                "description": (
                    "Full DispatchSpec. Required fields: dispatch_altitude "
                    "(META|SPEC|IMPL|INFRA|NIT-cleanup), role, cycle (int), "
                    "task (>50 chars), success_criterion (>20 chars), "
                    "output_path (must match agents/.../output_cycleN.md, "
                    "findings/...md, or drafts/...), model (provider/model), "
                    "process_hygiene (>20 chars), confidence_required (bool)."
                ),
            },
        },
        "required": ["spec"],
    },
    handler=_spawn,
    permission_mode=PermissionMode.DEFAULT,  # spawning runs code → ask in default mode
)


# ── evaluate_artifact_rubric (Sprint 5 item 24 — backs the finalize grader) ──


def _grade_letter(weighted: float) -> str:
    """Letter grade from a normalized 0-1 weighted score."""
    if weighted >= 0.90:
        return "A"
    if weighted >= 0.80:
        return "B"
    if weighted >= 0.70:
        return "C"
    if weighted >= 0.60:
        return "D"
    return "F"


def _evaluate_artifact_rubric(artifact: str, gaps: str = "",
                              evidence_count: int = 0,
                              rubric_path: str | None = None,
                              contract: dict | None = None,
                              cascade=None) -> dict:
    """4-judge median+variance grade of a finalized artifact (core.grader).

    `contract` is the mission's QualityContract as a dict; when omitted a
    balanced contract (all weights 3, threshold 0.7) is used as a transparent
    default — the finalize flow should pass the mission-declared contract once
    contract-plumbing lands (see SPRINT5_DESIGN finalize-wiring follow-up).
    """
    from core import grader as _grader
    from core import quality as _quality
    qc = (_quality.QualityContract.from_dict(contract) if contract
          else _quality.QualityContract(3, 3, 3, 3, 3, 3, 3, 3))
    kwargs: dict = {"contract": qc, "evidence_count": evidence_count}
    if rubric_path:
        kwargs["rubric_path"] = Path(rubric_path)
    if cascade is not None:
        kwargs["cascade"] = cascade
    res = _grader.grade_artifact(artifact, gaps, **kwargs)
    return {
        "grade": _grade_letter(res.weighted_score),
        "components": res.medians,
        **res.to_dict(),
    }


tool_registry.register_function(
    name="evaluate_artifact_rubric",
    description=(
        "Grade a finalized artifact on the 8 quality dimensions using 4 judge "
        "personas (correctness, gap_finder, honesty, reproducibility+efficiency). "
        "Each judge scores 0-5 per dimension against the rubric; the grader takes "
        "the MEDIAN per dimension (a single outlier judge can't swing the grade), "
        "reports per-dimension VARIANCE (judge disagreement), and collapses the "
        "medians through the mission's quality contract into a weighted 0-1 score "
        "plus a letter grade. Resilient: a judge that fails every provider lane is "
        "dropped and the grade is computed over survivors."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "artifact":       {"type": "string", "description": "The finalized deliverable text"},
            "gaps":           {"type": "string", "description": "Declared gaps/limitations"},
            "evidence_count": {"type": "integer", "minimum": 0},
            "rubric_path":    {"type": "string", "description": "Override rubric YAML path"},
            "contract":       {"type": "object", "description": "Mission QualityContract (8 weights + pass_threshold); omitted -> balanced default"},
        },
        "required": ["artifact"],
    },
    handler=_evaluate_artifact_rubric,
    permission_mode=PermissionMode.AUTO,  # read-only grading (LLM calls, no state mutation)
)


# Register the finalize_project tool suite (task #68). Importing here means
# `import core.tools` registers the full tool set the finalize skills need.
from core import finalize_tools  # noqa: E402,F401

__all__ = ["_read", "_write", "_edit", "_bash",
           "_webfetch", "_websearch",
           "_memory_search", "_memory_create",
           "_spawn", "_evaluate_artifact_rubric"]
