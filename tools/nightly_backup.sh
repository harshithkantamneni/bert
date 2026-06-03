#!/bin/bash
# Nightly backup of bert state per P-015 (Disk-growth + state-backup discipline).
#
# Per FINAL_implementation_plan_2026-05-07.md §5.1 H1 day 5 (Gap 2 closure).
# Per docs/ARCHITECTURE.md key-files inventory:
#   "backup/state/state_YYYY-MM-DD.tar.gz nightly state backup"
#
# What gets backed up:
#   - state/ (cycle queue, results, session state, lab-org queue)
#   - lab/sor/ (canonical events, log, findings, results — append-only;
#     Merkle-hashed in Phase H1 day 4)
#   - lab/sod/ (procedures, mission, governance — ratification-edit-only)
#   - memories/ (current, log, heuristics, killed, mission, procedures,
#     governance, programs, glossary, INDEX, shared)
#
# What does NOT get backed up:
#   - lab/soi/ (freely mutable; redundant with memories/ during build phase)
#   - findings/ (currently top-level; will be in lab/sor/ after migration)
#   - drafts/ (mutable; recreate-from-source if lost)
#   - memory.db (rebuildable from markdown sources via core/memory._index_corpus)
#   - graph.kuzu/ (rebuildable from KG-extractor; not yet built per L-22)
#   - logs/ (operational, not state)
#   - .venv/, node_modules/, .git/ (vendor / VCS)
#
# Retention: ≤30 days via find -mtime +30 -delete after backup creates.
#
# Install as cron entry (PI runs once):
#   crontab -e
#   # Add: 0 3 * * *  /Users/harshithkantamneni/Desktop/bert-lab/tools/nightly_backup.sh >> /Users/harshithkantamneni/Desktop/bert-lab/logs/nightly_backup.log 2>&1
#
# Manual run:
#   ./tools/nightly_backup.sh

set -euo pipefail

LAB_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BACKUP_DIR="${LAB_DIR}/backup/state"
DATE_STAMP=$(date -u +%Y-%m-%d)
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
ARCHIVE="${BACKUP_DIR}/state_${DATE_STAMP}.tar.gz"

mkdir -p "${BACKUP_DIR}"

echo "[${TIMESTAMP}] starting backup → ${ARCHIVE}"

cd "${LAB_DIR}"

# What to include — each path relative to LAB_DIR. Skip if path doesn't
# exist (e.g., lab/sod/ may be empty during early build).
INCLUDE_PATHS=()
for path in state lab/sor lab/sod memories; do
  if [ -d "${LAB_DIR}/${path}" ]; then
    INCLUDE_PATHS+=("${path}")
  fi
done

if [ ${#INCLUDE_PATHS[@]} -eq 0 ]; then
  echo "[${TIMESTAMP}] WARN: no paths to back up; bailing"
  exit 1
fi

tar -czf "${ARCHIVE}" "${INCLUDE_PATHS[@]}"

ARCHIVE_SIZE=$(du -h "${ARCHIVE}" | cut -f1)
echo "[${TIMESTAMP}] created ${ARCHIVE} (${ARCHIVE_SIZE})"

# Retention: delete backups older than 30 days
PRUNED=$(find "${BACKUP_DIR}" -name 'state_*.tar.gz' -type f -mtime +30 -print -delete 2>&1 | wc -l | tr -d ' ')
if [ "${PRUNED}" -gt 0 ]; then
  echo "[${TIMESTAMP}] pruned ${PRUNED} old backup(s) (>30 days)"
fi

# Compute Merkle roots over append-only files for verifiability
# (per L-01 + Phase H1 day 4 core/merkle.py).
if [ -f "${LAB_DIR}/lab/sor/events.jsonl" ]; then
  ROOT=$(cd "${LAB_DIR}" && .venv/bin/python -m core.merkle lab/sor/events.jsonl 2>/dev/null || echo "<merkle-unavailable>")
  echo "[${TIMESTAMP}] lab/sor/events.jsonl Merkle root: ${ROOT}"
fi

echo "[${TIMESTAMP}] backup complete"
