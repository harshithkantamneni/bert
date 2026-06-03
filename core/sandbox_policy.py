"""Tailored sandbox-exec profile generator from SKILL.md frontmatter.

Per E.3 deferred scope. The base sandbox.py builds a generic
deny-by-default profile; this module reads a skill's `needs_*`
frontmatter and emits a profile that allows only what the skill
declares it needs.

Frontmatter keys understood (all optional; default deny):
  needs_network: true|false
  needs_read_paths: [list of paths]
  needs_write_paths: [list of paths]
  needs_subprocess: true|false
  needs_ollama: true|false                  # convenience flag
  timeout_secs: int                          # default 30

The frontmatter is YAML-style between the leading and trailing
`---` lines of a SKILL.md.
"""

from __future__ import annotations

import contextlib
import logging
import re
from pathlib import Path

LOG = logging.getLogger("bert.sandbox_policy")


def parse_frontmatter(skill_md_path: Path) -> dict:
    """Extract YAML-style frontmatter from a SKILL.md.

    No yaml dependency: we hand-parse the small subset we use
    (key: scalar / list of strings).
    """
    if not skill_md_path.exists():
        return {}
    text = skill_md_path.read_text()
    m = re.match(r"---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not m:
        return {}
    body = m.group(1)
    out: dict = {}
    current_key: str | None = None
    current_list: list[str] | None = None
    for line in body.splitlines():
        if not line.strip():
            continue
        # List continuation
        if current_list is not None and line.startswith("  -"):
            current_list.append(line.split("-", 1)[1].strip().strip("\"'"))
            continue
        if current_list is not None:
            out[current_key] = current_list
            current_list = None
            current_key = None
        m2 = re.match(r"([\w_]+):\s*(.*)$", line)
        if not m2:
            continue
        key, val = m2.group(1), m2.group(2).strip()
        if not val:
            # Start of a list
            current_key = key
            current_list = []
            continue
        if val.lower() in ("true", "false"):
            out[key] = val.lower() == "true"
        elif val.lstrip("-").isdigit():
            out[key] = int(val)
        else:
            out[key] = val.strip("\"'")
    if current_list is not None and current_key:
        out[current_key] = current_list
    return out


def build_policy(skill_md_path: Path) -> dict:
    """Return a kwarg dict that core.sandbox.run_restricted() accepts."""
    fm = parse_frontmatter(skill_md_path)
    allow_read = list(fm.get("needs_read_paths", []) or [])
    allow_write = list(fm.get("needs_write_paths", []) or [])
    if fm.get("needs_ollama"):
        # Local Ollama needs the unix socket + temp dir
        allow_read.extend(["/usr/local/var/ollama", "/tmp"])
        allow_write.append("/tmp/ollama")
    policy = {
        "allow_read_paths": allow_read,
        "allow_write_paths": allow_write,
        "allow_network": bool(fm.get("needs_network", False)),
    }
    # Optional timeout
    if "timeout_secs" in fm:
        with contextlib.suppress(TypeError, ValueError):
            policy["timeout_secs"] = int(fm["timeout_secs"])
    return policy


def explain(skill_md_path: Path) -> str:
    """Render a one-paragraph plain-English summary of what the policy
    allows. Useful for PI review during P-005 permission gate."""
    fm = parse_frontmatter(skill_md_path)
    parts = []
    if fm.get("needs_network"):
        parts.append("permits outbound network")
    else:
        parts.append("denies network")
    read = fm.get("needs_read_paths") or []
    write = fm.get("needs_write_paths") or []
    parts.append(
        f"read access to {len(read)} explicit path(s)"
        if read else "no read access beyond stdlib"
    )
    parts.append(
        f"write access to {len(write)} explicit path(s)"
        if write else "no write access"
    )
    if fm.get("needs_subprocess"):
        parts.append("permits subprocess spawning")
    if fm.get("needs_ollama"):
        parts.append("permits Ollama IPC")
    timeout = fm.get("timeout_secs", 30)
    parts.append(f"timeout {timeout}s")
    return "; ".join(parts) + "."
