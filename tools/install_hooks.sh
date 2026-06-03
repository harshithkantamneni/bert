#!/usr/bin/env bash
# Install bert's git hooks into .git/hooks/.
#
# Idempotent — running twice does no harm. If the hooks already exist
# from another source, this script backs them up rather than overwriting.

set -e

LAB_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOOKS_SRC="$LAB_ROOT/tools/git-hooks"
HOOKS_DST="$LAB_ROOT/.git/hooks"

if [ ! -d "$LAB_ROOT/.git" ]; then
  echo "[ERROR] not a git repository (or .git/ missing at $LAB_ROOT/.git)" >&2
  exit 2
fi

mkdir -p "$HOOKS_DST"
INSTALLED=0
for src in "$HOOKS_SRC"/*; do
  [ -f "$src" ] || continue
  name=$(basename "$src")
  dst="$HOOKS_DST/$name"

  # Back up an existing hook that isn't ours
  if [ -f "$dst" ] && ! grep -q "bert · git" "$dst"; then
    backup="$dst.pre-bert.$(date +%s)"
    echo "[install_hooks] backing up existing $name → $backup"
    mv "$dst" "$backup"
  fi

  cp "$src" "$dst"
  chmod +x "$dst"
  echo "[install_hooks] ✓ installed $name"
  INSTALLED=$((INSTALLED+1))
done

if [ "$INSTALLED" -eq 0 ]; then
  echo "[install_hooks] no hooks found in $HOOKS_SRC" >&2
  exit 1
fi

echo
echo "[install_hooks] $INSTALLED hook(s) installed."
echo "[install_hooks] to bypass on a specific push: git push --no-verify"
