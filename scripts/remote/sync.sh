#!/usr/bin/env bash
# Push this repo's code to the home PC. No rsync available on this
# laptop, so this streams a tar archive straight over ssh instead.
#
# Usage: scripts/remote/sync.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "$SCRIPT_DIR/_common.sh"

check_ssh

echo "syncing $REPO_ROOT -> $SSH_HOST:$REMOTE_DIR ..."

# Never sync: git internals, datasets (they already live permanently on
# the PC under ~/data/), the HF/Ultralytics model caches (several GB,
# re-fetchable on the PC if ever needed), run outputs/checkpoints/logs
# (those flow the other way, via pull.sh), venvs, caches, and large
# binary weight files that might be lying around locally.
tar -czf - \
  --exclude='.git' \
  --exclude='data' \
  --exclude='datasets' \
  --exclude='models' \
  --exclude='checkpoints' \
  --exclude='outputs' \
  --exclude='runs' \
  --exclude='logs' \
  --exclude='wandb' \
  --exclude='venv' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.ckpt' \
  --exclude='*.zip' \
  -C "$REPO_ROOT" . \
  | ssh "$SSH_HOST" "mkdir -p '$REMOTE_DIR' && tar -xzf - -C '$REMOTE_DIR'"

echo "done: $SSH_HOST:$REMOTE_DIR"
