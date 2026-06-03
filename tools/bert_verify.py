"""bert verify <packet>.tar.gz CLI (I.4 entry point).

Usage:
  bert verify cycle-0042.tar.gz
  bert verify cycle-0042.tar.gz --fetch-rekor
  bert verify --chain cycle-0001.tar.gz cycle-0002.tar.gz ...

Exit codes:
  0: PASS
  1: PASS-WITH-WARNINGS
  2: FAIL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import verify_packet


def run(packets: list[str], *, chain: bool = False, fetch_rekor: bool = False,
        no_color: bool = False) -> int:
    """Verify proof packet(s) and return an exit code (0 PASS / 1 WARN / 2 FAIL).
    Shared by this CLI's main() and lab.py's `bert verify` subcommand."""
    if chain:
        chain_result = verify_packet.verify_chain([Path(p) for p in packets])
        for pkt in chain_result["packets"]:
            print(f"  {pkt['cycle_id']:20s} {pkt['overall']:20s} fails={pkt['fail_count']}")
        print()
        print("Chain links:")
        for link in chain_result["chain_links"]:
            state = link.get("state", "linked" if link["linked"] else "unlinked")
            mark = {"linked": "✓", "unlinked": "·", "mismatch": "✗"}.get(state, "?")
            label = {
                "linked": "lineage verified",
                "unlinked": "no parent declared",
                "mismatch": "parent mismatch — integrity failure",
            }.get(state, "")
            print(f"  {mark} {link['from']} → {link['to']}  ({label})")
        print()
        # L.3 — soften wording. Only call the chain "BROKEN" when a
        # declared parent actually mismatches (integrity failure).
        # Unrelated cycles with no declared parent are "unlinked",
        # which is a softer state, not an alarm.
        if chain_result.get("has_mismatch"):
            print("Chain BROKEN — at least one declared parent doesn't match")
            return 2
        elif chain_result["chain_ok"]:
            print("Chain OK — every link's declared parent matches")
            return 0
        else:
            print("Chain unlinked — cycles have no declared lineage between them")
            return 1

    # Single-packet verification
    exit_code = 0
    for p in packets:
        result = verify_packet.verify_packet(Path(p), fetch_rekor=fetch_rekor)
        print(verify_packet.format_result(result, color=not no_color))
        print()
        if result.overall == "FAIL":
            exit_code = max(exit_code, 2)
        elif result.overall == "PASS-WITH-WARNINGS":
            exit_code = max(exit_code, 1)
    return exit_code


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify a bert proof packet (I.4).")
    ap.add_argument("packets", nargs="+", help="proof packet .tar.gz file(s)")
    ap.add_argument("--chain", action="store_true",
                    help="verify packets as a lineage chain")
    ap.add_argument("--fetch-rekor", action="store_true",
                    help="fetch Rekor v2 inclusion proof (requires network)")
    ap.add_argument("--no-color", action="store_true",
                    help="disable ANSI color output")
    args = ap.parse_args()
    return run(args.packets, chain=args.chain, fetch_rekor=args.fetch_rekor,
               no_color=args.no_color)


if __name__ == "__main__":
    sys.exit(main())
