#!/bin/bash
# Install / verify the launchd plist that drives nightly bert backups.
#
# On macOS we use launchd (the native scheduler) rather than cron.
# This script writes ~/Library/LaunchAgents/com.bert-lab.backup.plist
# and loads it. Runs nightly at 04:30 local time.
#
# Usage:
#   bash tools/setup_backup_cron.sh install
#   bash tools/setup_backup_cron.sh status
#   bash tools/setup_backup_cron.sh uninstall

set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLIST_NAME="com.bert-lab.backup"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"
SCRIPT="$LAB_ROOT/tools/nightly_backup.sh"
LOG_DIR="$LAB_ROOT/state/launchd_logs"

cmd="${1:-status}"

if [ ! -x "$SCRIPT" ]; then
    if [ -f "$SCRIPT" ]; then
        chmod +x "$SCRIPT"
    else
        echo "error: $SCRIPT not found" >&2
        exit 2
    fi
fi

mkdir -p "$LOG_DIR"

case "$cmd" in
install)
    mkdir -p "$(dirname "$PLIST_PATH")"
    cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>${PLIST_NAME}</string>
    <key>ProgramArguments</key>
    <array>
      <string>/bin/bash</string>
      <string>${SCRIPT}</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
      <key>Hour</key><integer>4</integer>
      <key>Minute</key><integer>30</integer>
    </dict>
    <key>WorkingDirectory</key>
    <string>${LAB_ROOT}</string>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/backup.stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/backup.stderr.log</string>
    <key>RunAtLoad</key>
    <false/>
  </dict>
</plist>
EOF
    echo "wrote $PLIST_PATH"
    launchctl unload "$PLIST_PATH" 2>/dev/null || true
    launchctl load "$PLIST_PATH"
    echo "loaded ${PLIST_NAME} — fires nightly at 04:30 local"
    ;;
status)
    if [ -f "$PLIST_PATH" ]; then
        echo "plist installed: $PLIST_PATH"
        if launchctl list | grep -q "${PLIST_NAME}"; then
            echo "loaded: yes"
        else
            echo "loaded: no (run install to load)"
        fi
    else
        echo "plist not installed (run: bash tools/setup_backup_cron.sh install)"
    fi
    if [ -d "${LAB_ROOT}/backup/state" ]; then
        echo "recent backups:"
        ls -lh "${LAB_ROOT}/backup/state/" 2>/dev/null | head -6
    fi
    ;;
uninstall)
    if [ -f "$PLIST_PATH" ]; then
        launchctl unload "$PLIST_PATH" 2>/dev/null || true
        rm -f "$PLIST_PATH"
        echo "removed $PLIST_PATH"
    else
        echo "not installed; nothing to do"
    fi
    ;;
run-once)
    echo "running $SCRIPT manually…"
    bash "$SCRIPT"
    ;;
*)
    echo "usage: $0 {install|status|uninstall|run-once}" >&2
    exit 2
    ;;
esac
