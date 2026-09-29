#!/usr/bin/env bash
# Kill a run's tmux session on the home PC. Does not touch its log file
# or anything under outputs/checkpoints/ - use pull.sh first if you want
# results off the PC.
#
# Usage: scripts/remote/stop.sh <run-name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "$SCRIPT_DIR/_common.sh"

if [ "$#" -lt 1 ]; then
  echo "usage: stop.sh <run-name>" >&2
  exit 1
fi
RUN_NAME="$1"
validate_run_name "$RUN_NAME"

check_ssh

if ssh "$SSH_HOST" "tmux has-session -t '$RUN_NAME'" 2>/dev/null; then
  ssh "$SSH_HOST" "tmux kill-session -t '$RUN_NAME'"
  echo "stopped: $RUN_NAME"
else
  echo "error: no tmux session named '$RUN_NAME' on $SSH_HOST" >&2
  echo "       (scripts/remote/status.sh to see what's running)" >&2
  exit 1
fi
