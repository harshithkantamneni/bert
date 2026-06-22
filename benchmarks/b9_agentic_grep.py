"""B9 arm A7 — agentic grep: the baseline a senior engineer actually points to.

This is what Claude Code / Cursor do instead of a vector index: give the model
read-only filesystem tools (grep, read_file, list_files) and let it iteratively
search the codebase to answer the question. No embeddings, no chunk store — the
model adapts its search based on what it finds. Uses native function-calling on
the same free reader (llama-3.3-70b) bert runs on, so the only thing that varies
vs the RAG arms is HOW context is gathered.

The honest hypothesis: agentic grep should do WELL on needles (it can grep the
exact symbol then read around it), which is exactly why coding agents use it.
This arm tests whether bert's hybrid retrieval actually beats just grepping.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

MAX_GREP_MATCHES = 40
MAX_READ_LINES = 160
DEFAULT_MAX_STEPS = 8

_TOOLS = [
    {"type": "function", "function": {
        "name": "grep",
        "description": "Regex-search every .py file in the codebase. Returns matching path:line: text lines.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Python regular expression"}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a line range of a file (relative path). Returns numbered lines.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "path relative to repo root, e.g. httpx/_decoders.py"},
            "start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List all .py files in the codebase (relative paths).",
        "parameters": {"type": "object", "properties": {}}}},
]


def _rg(corpus_dir: Path, pattern: str) -> str:
    try:
        re.compile(pattern)
    except re.error as e:
        return f"(invalid regex: {e})"
    rx = re.compile(pattern)
    out: list[str] = []
    for f in sorted(corpus_dir.rglob("*.py")):
        rel = str(f.relative_to(corpus_dir))
        try:
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    out.append(f"{rel}:{i}: {line.strip()[:200]}")
                    if len(out) >= MAX_GREP_MATCHES:
                        out.append(f"... (truncated at {MAX_GREP_MATCHES} matches)")
                        return "\n".join(out)
        except OSError:
            continue
    return "\n".join(out) if out else "(no matches)"


def _read(corpus_dir: Path, path: str, start: int | None, end: int | None) -> str:
    fp = (corpus_dir / path)
    if not fp.is_file():
        # tolerate basename-only or wrong prefix
        cands = list(corpus_dir.rglob(Path(path).name))
        if not cands:
            return f"(file not found: {path})"
        fp = cands[0]
    lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
    s = max(1, start or 1)
    e = min(len(lines), end or (s + MAX_READ_LINES - 1))
    if e - s + 1 > MAX_READ_LINES:
        e = s + MAX_READ_LINES - 1
    return "\n".join(f"{i}: {lines[i-1]}" for i in range(s, e + 1)) or "(empty range)"


def _list(corpus_dir: Path) -> str:
    return "\n".join(str(p.relative_to(corpus_dir)) for p in sorted(corpus_dir.rglob("*.py")))


def _exec_tool(corpus_dir: Path, name: str, args: dict) -> str:
    try:
        if name == "grep":
            return _rg(corpus_dir, str(args.get("pattern", "")))
        if name == "read_file":
            return _read(corpus_dir, str(args.get("path", "")),
                         args.get("start_line"), args.get("end_line"))
        if name == "list_files":
            return _list(corpus_dir)
    except Exception as e:  # noqa: BLE001
        return f"(tool error: {e})"
    return f"(unknown tool: {name})"


def agentic_grep_answer(question: str, corpus_dir: Path, cascade,
                        *, max_steps: int = DEFAULT_MAX_STEPS, verbose: bool = False) -> dict:
    """Run the grep/read loop with native tool-calling on the free reader.
    Returns {answer, steps, tool_log}."""
    from core import provider as prov

    sysp = (
        "You are answering a factual question about a Python codebase by searching it. "
        "Always START by calling the grep tool to locate the relevant symbol or constant, "
        "then call read_file to read the exact lines around it. Quote the exact value from "
        "the source in your answer. Once you have found the answer, state it directly. "
        f"You have at most {max_steps} tool calls."
    )
    messages: list = [
        {"role": "system", "content": sysp},
        {"role": "user", "content": f"Question: {question}"},
    ]
    tool_log: list = []
    steps = 0
    usage = {"in": 0, "out": 0}  # total tokens burned across the whole loop
    for _ in range(max_steps):
        resp = None
        for prov_name, model in cascade:
            try:
                r = prov.call(prov_name, messages, model=model, tools=_TOOLS,
                              max_tokens=700, temperature=0.0, timeout=70.0)
                if r.finish_reason != "error" and not (r.text or "").startswith("[bert]"):
                    usage["in"] += getattr(r, "usage_prompt_tokens", 0) or 0
                    usage["out"] += getattr(r, "usage_completion_tokens", 0) or 0
                    resp = r
                    break
            except Exception:  # noqa: BLE001
                continue
        if resp is None:
            return {"answer": "[agentic-grep: all reader lanes errored]", "steps": steps, "tool_log": tool_log, "tokens_in": usage["in"], "tokens_out": usage["out"]}

        if resp.tool_calls:
            # record assistant turn with tool calls, then execute each
            messages.append({"role": "assistant", "content": resp.text or "",
                             "tool_calls": [{"id": tc.id, "type": "function",
                                             "function": {"name": tc.name,
                                                          "arguments": json.dumps(tc.arguments)}}
                                            for tc in resp.tool_calls]})
            for tc in resp.tool_calls:
                obs = _exec_tool(corpus_dir, tc.name, tc.arguments or {})
                tool_log.append({"tool": tc.name, "args": tc.arguments, "obs_len": len(obs)})
                if verbose:
                    print(f"    [{tc.name}] {tc.arguments} -> {len(obs)} chars")
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": obs[:6000]})
            steps += 1
            continue

        # no tool call -> final answer
        return {"answer": (resp.text or "").strip(), "steps": steps, "tool_log": tool_log, "tokens_in": usage["in"], "tokens_out": usage["out"]}

    # budget exhausted: force a final answer without tools
    messages.append({"role": "user",
                     "content": "Budget reached. Give your single best final answer now, no tools."})
    for prov_name, model in cascade:
        try:
            r = prov.call(prov_name, messages, model=model, max_tokens=400,
                          temperature=0.0, timeout=60.0)
            if r.finish_reason != "error" and not (r.text or "").startswith("[bert]"):
                usage["in"] += getattr(r, "usage_prompt_tokens", 0) or 0
                usage["out"] += getattr(r, "usage_completion_tokens", 0) or 0
                return {"answer": (r.text or "").strip(), "steps": steps, "tool_log": tool_log, "tokens_in": usage["in"], "tokens_out": usage["out"]}
        except Exception:  # noqa: BLE001
            continue
    return {"answer": "[agentic-grep: no final answer]", "steps": steps, "tool_log": tool_log, "tokens_in": usage["in"], "tokens_out": usage["out"]}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    CASCADE = [("nvidia", "meta/llama-3.3-70b-instruct"), ("groq", "llama-3.3-70b-versatile")]
    corpus = Path("/tmp/b9_corpus")
    q = sys.argv[1] if len(sys.argv) > 1 else \
        "In httpx's DeflateDecoder, what does the decoder do on the first chunk of data?"
    print(f"Q: {q}\n")
    out = agentic_grep_answer(q, corpus, CASCADE, verbose=True)
    print(f"\nsteps={out['steps']}  tools={[t['tool'] for t in out['tool_log']]}")
    print(f"ANSWER: {out['answer']}")
