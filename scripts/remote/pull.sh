#!/usr/bin/env bash
# Copy results back from the home PC into this repo, via ssh+tar (no
# rsync available locally). Each path is relative to the remote project
# dir and is recreated at the same relative path here.
#
# Usage: scripts/remote/pull.sh [path ...]
#   (default: outputs checkpoints)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "$SCRIPT_DIR/_common.sh"

check_ssh

if [ "$#" -gt 0 ]; then
  PATHS=("$@")
else
  PATHS=("outputs" "checkpoints")
fi

for p in "${PATHS[@]}"; do
  echo "== $p =="
  if ! ssh "$SSH_HOST" "test -e '$REMOTE_DIR/$p'"; then
    echo "  skip: $SSH_HOST:$REMOTE_DIR/$p does not exist"
    continue
  fi
  ssh "$SSH_HOST" "tar -czf - -C '$REMOTE_DIR' '$p'" | tar -xzf - -C "$REPO_ROOT"
  echo "  pulled into $REPO_ROOT/$p"
done
