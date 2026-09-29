#!/usr/bin/env bash
# Show what's running on the home PC: tmux sessions, the tail of a run's
# log (the most recently modified one if no run name is given), and GPU
# memory/utilization.
#
# Usage: scripts/remote/status.sh [run-name]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "$SCRIPT_DIR/_common.sh"

RUN_NAME="${1:-}"
if [ -n "$RUN_NAME" ]; then
  validate_run_name "$RUN_NAME"
fi

check_ssh

# $RUN_NAME and $REMOTE_DIR are substituted here, locally, right now (this
# script's own variables); everything prefixed \$ below is left literal so
# the REMOTE shell evaluates it when the command actually runs there.
REMOTE_CMD=$(cat <<EOF
echo "== tmux sessions on \$(hostname) =="
tmux ls 2>/dev/null || echo "(none running)"
echo

LOG_DIR="$REMOTE_DIR/logs"
if [ -n "$RUN_NAME" ]; then
  LOG_FILE="\$LOG_DIR/$RUN_NAME.log"
else
  LOG_FILE="\$(ls -t "\$LOG_DIR"/*.log 2>/dev/null | head -n1 || true)"
fi

if [ -n "\$LOG_FILE" ] && [ -f "\$LOG_FILE" ]; then
  echo "== last 40 lines: \$LOG_FILE =="
  tail -n 40 "\$LOG_FILE"
else
  echo "== no matching log file under $REMOTE_DIR/logs =="
fi
echo

echo "== nvidia-smi =="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \\
  --format=csv || echo "(nvidia-smi not available)"
EOF
)

ssh "$SSH_HOST" "$REMOTE_CMD"
