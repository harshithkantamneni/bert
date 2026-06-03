"""Install / uninstall bert nightly automation.

On macOS, writes a launchd plist to ~/Library/LaunchAgents/ that runs
tools/bert_nightly.sh every night at the configured hour. On Linux,
prints the crontab line you'd add manually (launchd isn't available).

Usage:
  .venv/bin/python tools/install_nightly.py --install              # at 23:00 local
  .venv/bin/python tools/install_nightly.py --install --hour 6     # custom hour
  .venv/bin/python tools/install_nightly.py --uninstall
  .venv/bin/python tools/install_nightly.py --status
  .venv/bin/python tools/install_nightly.py --print-only           # print plist, don't write
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
NIGHTLY_SH = LAB_ROOT / "tools" / "bert_nightly.sh"
LOG_PATH = LAB_ROOT / "lab" / "state" / "nightly.log"

PLIST_LABEL = "dev.bert.bert.nightly"
PLIST_NAME = f"{PLIST_LABEL}.plist"
LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _plist_path() -> Path:
    return LAUNCH_AGENTS_DIR / PLIST_NAME


def build_plist(hour: int = 23, minute: int = 0) -> str:
    """Return the launchd plist contents for the configured schedule."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>{PLIST_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{NIGHTLY_SH}</string>
  </array>
  <key>WorkingDirectory</key>
  <string>{LAB_ROOT}</string>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>{hour}</integer>
    <key>Minute</key>
    <integer>{minute}</integer>
  </dict>
  <key>StandardOutPath</key>
  <string>{LOG_PATH}</string>
  <key>StandardErrorPath</key>
  <string>{LOG_PATH}</string>
  <key>RunAtLoad</key>
  <false/>
</dict>
</plist>
"""


def build_crontab_line(hour: int = 23, minute: int = 0) -> str:
    """Crontab line for Linux installs."""
    return f"{minute} {hour} * * *  cd {LAB_ROOT} && {NIGHTLY_SH} >> {LOG_PATH} 2>&1"


def install_macos(hour: int, minute: int, *, print_only: bool = False) -> int:
    if not NIGHTLY_SH.exists():
        print(f"[ERROR] {NIGHTLY_SH} not found", file=sys.stderr)
        return 2
    if not os.access(NIGHTLY_SH, os.X_OK):
        print(f"[hint] making {NIGHTLY_SH} executable")
        NIGHTLY_SH.chmod(0o755)

    plist_contents = build_plist(hour, minute)
    if print_only:
        print(plist_contents)
        return 0

    LAUNCH_AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    plist_path = _plist_path()

    # If already loaded, unload first (idempotent reinstall)
    if plist_path.exists():
        subprocess.run(["launchctl", "unload", str(plist_path)],
                       capture_output=True)

    plist_path.write_text(plist_contents)
    result = subprocess.run(["launchctl", "load", str(plist_path)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] launchctl load failed: {result.stderr.strip()}",
              file=sys.stderr)
        return 2

    print(f"installed {plist_path}")
    print(f"  scheduled: daily at {hour:02d}:{minute:02d} local time")
    print(f"  log: {LOG_PATH}")
    print("  uninstall: .venv/bin/python tools/install_nightly.py --uninstall")
    return 0


def uninstall_macos() -> int:
    plist_path = _plist_path()
    if not plist_path.exists():
        print("[info] no installed plist found at "
              f"{plist_path} — already uninstalled?")
        return 0
    subprocess.run(["launchctl", "unload", str(plist_path)],
                   capture_output=True)
    plist_path.unlink()
    print(f"uninstalled {plist_path}")
    return 0


def status_macos() -> int:
    plist_path = _plist_path()
    if not plist_path.exists():
        print(f"[status] not installed (no plist at {plist_path})")
        return 1
    print(f"[status] installed at {plist_path}")
    # launchctl list shows whether it's loaded
    result = subprocess.run(
        ["launchctl", "list"], capture_output=True, text=True
    )
    loaded = any(PLIST_LABEL in line for line in result.stdout.splitlines())
    print(f"[status] loaded into launchd: {loaded}")
    if LOG_PATH.exists():
        size_kb = LOG_PATH.stat().st_size // 1024
        print(f"[status] log: {LOG_PATH} ({size_kb} KB)")
    else:
        print(f"[status] log: {LOG_PATH} (not yet created — runs at scheduled time)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--install", action="store_true")
    g.add_argument("--uninstall", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--print-only", action="store_true",
                   help="Print the plist (macOS) or crontab line (Linux) to stdout.")
    ap.add_argument("--hour", type=int, default=23,
                    help="Hour 0-23 in local time (default: 23).")
    ap.add_argument("--minute", type=int, default=0,
                    help="Minute 0-59 (default: 0).")
    args = ap.parse_args()

    if not 0 <= args.hour <= 23:
        print(f"[ERROR] --hour must be 0..23 (got {args.hour})", file=sys.stderr)
        return 2
    if not 0 <= args.minute <= 59:
        print("[ERROR] --minute must be 0..59", file=sys.stderr)
        return 2

    if _is_macos():
        if args.install:
            return install_macos(args.hour, args.minute)
        if args.uninstall:
            return uninstall_macos()
        if args.status:
            return status_macos()
        if args.print_only:
            return install_macos(args.hour, args.minute, print_only=True)
    else:
        # Linux: print the crontab line, no auto-install
        if args.print_only or args.install:
            print(build_crontab_line(args.hour, args.minute))
            if args.install:
                print()
                print(
                    "[info] On Linux we don't auto-install — copy the line above\n"
                    "[info] into your crontab:\n"
                    "[info]    crontab -e\n"
                    "[info] then paste, save, exit."
                )
            return 0
        if args.uninstall:
            print("[info] On Linux, remove the bert_nightly.sh line from `crontab -e`.")
            return 0
        if args.status:
            print("[info] On Linux, `crontab -l` to inspect your installed entries.")
            return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
