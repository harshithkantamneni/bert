#!/bin/bash
# Local Jaeger setup for bert's telemetry (F.6 follow-up).
#
# Runs Jaeger all-in-one in Docker, exposing:
#   :4318  OTLP HTTP receiver (where bert sends spans)
#   :16686 web UI (where you read them)
#
# Bert's lab.py now defaults OTEL_EXPORTER_OTLP_ENDPOINT to
# http://localhost:4318 — once this container is up, every cycle's
# spans land here automatically.
#
# Usage:
#   bash tools/setup_jaeger.sh install        # run container + launchd plist
#   bash tools/setup_jaeger.sh status         # report state
#   bash tools/setup_jaeger.sh restart        # restart the container
#   bash tools/setup_jaeger.sh uninstall      # stop + remove container + plist
#   bash tools/setup_jaeger.sh logs           # tail container logs
#   bash tools/setup_jaeger.sh open           # open the UI in default browser

set -euo pipefail

CONTAINER="bert-jaeger"
IMAGE="jaegertracing/all-in-one:latest"
LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLIST_NAME="com.bert-lab.jaeger"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"
LOG_DIR="$LAB_ROOT/state/launchd_logs"

cmd="${1:-status}"
mkdir -p "$LOG_DIR"

case "$cmd" in
install)
    # Container — uses badger storage on a docker volume so traces
    # survive restart. The Jaeger image runs as uid=10001; we chown
    # the volume root before first start so badger can mkdir its
    # subdirectories.
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}\$"; then
        echo "container ${CONTAINER} exists; restarting"
        docker start "$CONTAINER" >/dev/null
    else
        echo "starting ${CONTAINER} (badger storage, persisted to docker volume)"
        docker volume create bert-jaeger-data >/dev/null
        # One-shot chown so Jaeger's uid=10001 can write to /badger
        docker run --rm -v bert-jaeger-data:/badger alpine \
            chown -R 10001:10001 /badger >/dev/null
        docker run -d --name "$CONTAINER" \
            -p 4318:4318 -p 16686:16686 \
            -e COLLECTOR_OTLP_ENABLED=true \
            -e SPAN_STORAGE_TYPE=badger \
            -e BADGER_EPHEMERAL=false \
            -e BADGER_DIRECTORY_VALUE=/badger/data \
            -e BADGER_DIRECTORY_KEY=/badger/key \
            -v bert-jaeger-data:/badger \
            --restart unless-stopped \
            "$IMAGE" >/dev/null
    fi

    # launchd plist — ensures Jaeger comes back after reboot by running
    # `docker start bert-jaeger` at user login.
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
      <string>/usr/local/bin/docker</string>
      <string>start</string>
      <string>${CONTAINER}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/jaeger.stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/jaeger.stderr.log</string>
  </dict>
</plist>
EOF
    launchctl unload "$PLIST_PATH" 2>/dev/null || true
    launchctl load "$PLIST_PATH"
    echo
    echo "✓ Jaeger up on:"
    echo "    OTLP receiver:  http://localhost:4318"
    echo "    Web UI:         http://localhost:16686"
    echo
    echo "Verify with: .venv/bin/python tools/check_otel_setup.py"
    ;;
status)
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}\$"; then
        echo "container: running"
        docker ps --filter "name=${CONTAINER}" --format "  {{.Status}}    ports: {{.Ports}}"
    elif docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}\$"; then
        echo "container: stopped (run: bash tools/setup_jaeger.sh restart)"
    else
        echo "container: not installed"
    fi
    if [ -f "$PLIST_PATH" ]; then
        if launchctl list | grep -q "${PLIST_NAME}"; then
            echo "launchd:   loaded (auto-starts at login)"
        else
            echo "launchd:   plist present but not loaded"
        fi
    else
        echo "launchd:   not installed"
    fi
    echo "OTLP env:  ${OTEL_EXPORTER_OTLP_ENDPOINT:-(unset; lab.py defaults to http://localhost:4318)}"
    ;;
restart)
    docker restart "$CONTAINER"
    echo "✓ restarted"
    ;;
uninstall)
    docker stop "$CONTAINER" 2>/dev/null || true
    docker rm "$CONTAINER" 2>/dev/null || true
    if [ -f "$PLIST_PATH" ]; then
        launchctl unload "$PLIST_PATH" 2>/dev/null || true
        rm -f "$PLIST_PATH"
    fi
    if [ "${2:-}" = "--purge" ]; then
        docker volume rm bert-jaeger-data 2>/dev/null || true
        echo "✓ removed (including trace history)"
    else
        echo "✓ removed (trace history preserved in bert-jaeger-data docker volume)"
        echo "  rerun with --purge to delete trace history too"
    fi
    ;;
logs)
    docker logs --tail 80 -f "$CONTAINER"
    ;;
open)
    open "http://localhost:16686/search?service=bert-lab"
    ;;
*)
    echo "usage: $0 {install|status|restart|uninstall|logs|open}" >&2
    exit 2
    ;;
esac
