"""note — sub-100ms Markdown capture CLI.

Stdlib only. No deps. Single file.

Usage:
  note "thought"                    # capture to ~/.notes/YYYY-MM-DD.md
  note "thought" --tag deep --tag idea
  note --where                      # print today's note path
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


def _today_path() -> Path:
    """Resolve ~/.notes/YYYY-MM-DD.md, creating ~/.notes/ if absent."""
    notes_dir = Path(os.path.expanduser("~/.notes"))
    notes_dir.mkdir(parents=True, exist_ok=True)
    return notes_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"


def capture(text: str, tags: list[str] | None = None,
             path: Path | None = None) -> Path:
    """Append a frontmatter+text block to today's notes file.

    Returns the path written to. Raises ValueError on empty text.
    """
    if not text or not text.strip():
        raise ValueError("note text cannot be empty")
    path = path or _today_path()
    tags = tags or []
    ts = datetime.now().isoformat(timespec="seconds")
    block = (
        f"\n---\n"
        f"ts: {ts}\n"
        f"tags: [{', '.join(tags)}]\n"
        f"---\n\n"
        f"{text.strip()}\n"
    )
    with path.open("a", encoding="utf-8") as f:
        f.write(block)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="note — sub-100ms Markdown capture")
    ap.add_argument("text", nargs="?", help="the thought to capture")
    ap.add_argument("--tag", action="append", default=[],
                    help="add a tag (repeatable)")
    ap.add_argument("--where", action="store_true",
                    help="print today's note path and exit")
    args = ap.parse_args(argv)

    if args.where:
        print(_today_path())
        return 0
    if not args.text:
        print("error: text required (or --where)", file=sys.stderr)
        return 2
    try:
        path = capture(args.text, args.tag)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"captured → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
