#!/bin/bash
# bert-lab autonomous runner
#
# Launches the Director in a while-true loop with exit-reason dispatch.
# Designed to run inside tmux:
#   tmux new-session -d -s bert-lab './run.sh'
#   tmux attach -t bert-lab
#
# Adapted from a prior lab/AGI patterns — six exit reasons, three-layer rate-limit
# handling, holding-loop detector, watchdog, signature verifier.

set -uo pipefail

LAB_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$LAB_DIR"

# Configuration
MAX_CYCLES="${BERT_MAX_CYCLES:-1000}"
RESTART_DELAY="${BERT_RESTART_DELAY:-5}"
RATE_LIMIT_WAIT="${BERT_RATE_LIMIT_WAIT:-300}"
UNKNOWN_WAIT="${BERT_UNKNOWN_WAIT:-120}"
MAX_CONSECUTIVE_FAILS="${BERT_MAX_CONSECUTIVE_FAILS:-5}"
HEARTBEAT_INTERVAL_MIN="${BERT_HEARTBEAT_INTERVAL_MIN:-240}"  # 4h default
STABLE_WINDOW_MIN="${BERT_STABLE_WINDOW_MIN:-30}"

# L-08 Phase A: keep Ollama models warm across cycles for prefix-cache
# reuse on local dispatches. 24h survives a tmux session of typical work;
# Ollama unloads after this idle. Per FINAL plan §5.1 H1 day 1.
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-24h}"

SESSION_EXIT="$LAB_DIR/state/session_exit.md"
SESSION_START="$LAB_DIR/state/session_start.md"
LOG_DIR="$LAB_DIR/logs"
RATE_LIMIT_RESETS_FILE="$LAB_DIR/state/rate_limit_resets_at"
PI_NOTES_FILE="$LAB_DIR/memories/governance/pi_notes.md"
PI_POLL_INTERVAL="${BERT_PI_POLL_INTERVAL:-30}"  # P-021: fast-poll pi_notes mtime

mkdir -p "$LOG_DIR" "$LAB_DIR/state"

# P-021: sleep with fast pi_notes.md mtime check.
# Aborts sleep early if PI sent /inject (file mtime changed) → next cycle starts within ≤30s.
wait_with_pi_check() {
    local total_secs=$1
    local elapsed=0
    local initial_mtime=0
    if [ -f "$PI_NOTES_FILE" ]; then
        initial_mtime=$(stat -f %m "$PI_NOTES_FILE" 2>/dev/null || stat -c %Y "$PI_NOTES_FILE" 2>/dev/null || echo 0)
    fi
    while [ $elapsed -lt $total_secs ]; do
        sleep "$PI_POLL_INTERVAL"
        elapsed=$((elapsed + PI_POLL_INTERVAL))
        if [ -f "$PI_NOTES_FILE" ]; then
            local current_mtime=$(stat -f %m "$PI_NOTES_FILE" 2>/dev/null || stat -c %Y "$PI_NOTES_FILE" 2>/dev/null || echo 0)
            if [ "$current_mtime" -gt "$initial_mtime" ] 2>/dev/null; then
                echo "[$(date)] pi_notes.md changed (PI nudge detected) — aborting sleep, starting next cycle"
                notify_pi "PI nudge received. Bert is starting next cycle."
                return
            fi
        fi
    done
}

# macOS notification helper
notify_pi() {
    osascript -e "display notification \"$1\" with title \"bert-lab\"" 2>/dev/null || true
}

# Probe: cheap "are providers reachable" check
probe_providers() {
    # TODO: implement via lab.py probe (queries each provider's /models)
    # Placeholder: always return ready until implemented
    return 0
}

# Wait for rate limit reset (precise if resetsAt timestamp written by stream formatter,
# fallback to probe loop)
wait_for_rate_limit_reset() {
    if [ -f "$RATE_LIMIT_RESETS_FILE" ]; then
        local resets_at=$(cat "$RATE_LIMIT_RESETS_FILE" | tr -d '[:space:]')
        local now=$(date +%s)
        local wait_secs=$((resets_at - now + 30))
        if [ $wait_secs -gt 0 ] && [ $wait_secs -lt 86400 ]; then
            local reset_human=$(date -r "$resets_at" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "$resets_at")
            echo "[$(date)] Rate limit resets at $reset_human. Sleeping ${wait_secs}s (+30s buffer)."
            notify_pi "Rate limit hit. Resets at $reset_human."
            sleep "$wait_secs"
            rm -f "$RATE_LIMIT_RESETS_FILE"
            return
        fi
    fi
    echo "[$(date)] No reset timestamp. Probing every ${RATE_LIMIT_WAIT}s..."
    notify_pi "Rate limit hit. Waiting for reset."
    while ! probe_providers; do
        sleep "$RATE_LIMIT_WAIT"
    done
}

echo "================================================"
echo "  bert-lab — Autonomous Operation"
echo "  Lab: $LAB_DIR"
echo "  Started: $(date)"
echo "================================================"

cycle=0
consecutive_fails=0

while [ $cycle -lt $MAX_CYCLES ]; do
    cycle=$((cycle + 1))
    start_time=$(date +%s)
    timestamp=$(date +%Y%m%d_%H%M%S)

    echo "--- Cycle $cycle starting at $(date) ---"

    # Clean up previous exit file
    rm -f "$SESSION_EXIT"

    # Heartbeat
    echo "PID: $$ | Start: $(date) | Cycle: $cycle" > "$SESSION_START"

    # Cycle log
    LOGFILE="$LOG_DIR/cycle_${cycle}_${timestamp}.log"

    # Launch Director via lab.py
    # TODO: implement core/agent.py + lab.py to dispatch the Director
    # For now this is a placeholder; once core is built it'll be:
    #   python -u lab.py --role director --cycle "$cycle" 2>&1 | tee "$LOGFILE"
    if python -u lab.py --role director --cycle "$cycle" 2>&1 | tee "$LOGFILE"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "--- Cycle $cycle finished in ${duration}s ---"

        # Check exit reason
        if [ -f "$SESSION_EXIT" ]; then
            EXIT_REASON=$(head -1 "$SESSION_EXIT")
            case "$EXIT_REASON" in
                *GRACEFUL_CHECKPOINT*)
                    echo "Graceful checkpoint. Restarting in ${RESTART_DELAY}s (or sooner if PI nudge)..."
                    consecutive_fails=0
                    wait_with_pi_check "$RESTART_DELAY"
                    ;;
                *CONTEXT_FULL*)
                    echo "Context full. Restarting in ${RESTART_DELAY}s..."
                    consecutive_fails=0
                    sleep "$RESTART_DELAY"
                    ;;
                *RATE_LIMIT*)
                    echo "Rate limit hit."
                    consecutive_fails=0
                    wait_for_rate_limit_reset
                    ;;
                *VICTORY*)
                    echo "=== VICTORY ==="
                    notify_pi "bert-lab: phase complete!"
                    exit 0
                    ;;
                *CATASTROPHIC*)
                    echo "=== CATASTROPHIC FAILURE ==="
                    notify_pi "bert-lab: CATASTROPHIC. Check logs."
                    exit 1
                    ;;
                *PIVOT*)
                    echo "Pivot detected. Restarting in ${RESTART_DELAY}s..."
                    notify_pi "bert-lab: direction pivot. Check pi_notes.md"
                    consecutive_fails=0
                    sleep "$RESTART_DELAY"
                    ;;
                *)
                    echo "Unknown exit reason: $EXIT_REASON"
                    consecutive_fails=0
                    sleep "$RESTART_DELAY"
                    ;;
            esac
        else
            # No exit file — duration-based fallback
            if [ $duration -lt 10 ]; then
                consecutive_fails=$((consecutive_fails + 1))
                echo "Short session (${duration}s) — possible crash (fail #$consecutive_fails)"
                if [ $consecutive_fails -ge $MAX_CONSECUTIVE_FAILS ]; then
                    echo "=== Too many consecutive failures. Stopping. ==="
                    notify_pi "bert-lab: too many failures. Stopped."
                    exit 1
                fi
                sleep "$UNKNOWN_WAIT"
            else
                echo "Session ran ${duration}s without exit file. Restarting in ${RESTART_DELAY}s..."
                consecutive_fails=0
                sleep "$RESTART_DELAY"
            fi
        fi
    else
        # CLI returned error
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        consecutive_fails=$((consecutive_fails + 1))
        echo "--- Cycle $cycle FAILED (${duration}s, fail #$consecutive_fails) ---"

        if grep -qiE "rate limit|hit your limit|429" "$LOGFILE" 2>/dev/null; then
            echo "Rate limit detected in output."
            consecutive_fails=0
            wait_for_rate_limit_reset
            continue
        fi

        if [ $consecutive_fails -ge $MAX_CONSECUTIVE_FAILS ]; then
            echo "=== Too many consecutive failures. Stopping. ==="
            notify_pi "bert-lab: too many failures. Stopped."
            exit 1
        fi
        sleep "$UNKNOWN_WAIT"
    fi
done

echo "=== Max cycles ($MAX_CYCLES) reached ==="
notify_pi "bert-lab: max cycles reached."
exit 0
