#!/usr/bin/env bash
# Shared config and helpers for scripts/remote/*.sh. Sourced, never run
# directly - each caller already set `set -euo pipefail` for itself.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

_REMOTE_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Machine-specific bits (SSH host alias, remote username, ...) live in
# config.sh, which is gitignored: nothing personal to this setup belongs
# in the tracked scripts themselves.
CONFIG_FILE="$_REMOTE_SCRIPTS_DIR/config.sh"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "error: $CONFIG_FILE not found." >&2
  echo "       cp scripts/remote/config.sh.example scripts/remote/config.sh" >&2
  echo "       then edit it for your setup." >&2
  exit 1
fi
# shellcheck source=./config.sh
source "$CONFIG_FILE"

: "${SSH_HOST:?config.sh must set SSH_HOST}"
: "${REMOTE_USER:?config.sh must set REMOTE_USER}"
REMOTE_HOME="${REMOTE_HOME:-/home/$REMOTE_USER}"

REMOTE_CONDA_SH="$REMOTE_HOME/miniconda3/etc/profile.d/conda.sh"
REMOTE_CONDA_ACTIVATE="source $REMOTE_CONDA_SH && conda activate ml"

# Absolute paths only, everywhere below - no "~" - so nothing here depends
# on whether a variable ends up inside single or double quotes by the time
# it reaches the remote shell.
REPO_ROOT="$(cd "$_REMOTE_SCRIPTS_DIR/../.." && pwd)"
PROJECT_NAME="$(basename "$REPO_ROOT")"
REMOTE_DIR="$REMOTE_HOME/projects/$PROJECT_NAME"

# Fails fast and clearly instead of hanging or prompting for a password
# when the home PC is asleep, offline, or Tailscale is down.
check_ssh() {
  if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "$SSH_HOST" true \
      >/dev/null 2>&1; then
    echo "error: can't reach '$SSH_HOST' over ssh." >&2
    echo "       Is the home PC awake, online, and Tailscale up?" >&2
    echo "       (tried: ssh -o ConnectTimeout=8 $SSH_HOST true)" >&2
    exit 1
  fi
}

check_tmux() {
  if ! ssh "$SSH_HOST" "command -v tmux >/dev/null 2>&1"; then
    echo "error: tmux is not installed on $SSH_HOST." >&2
    echo "       Install it with: ssh $SSH_HOST 'sudo apt install -y tmux'" >&2
    exit 1
  fi
}

# Keeps run names safe to drop into tmux session names, file paths and
# shell command strings without ever needing to think about escaping them.
validate_run_name() {
  case "$1" in
    ''|*[!A-Za-z0-9_.-]*)
      echo "error: run name must match [A-Za-z0-9_.-]+" \
           "(got: '${1:-<empty>}')" >&2
      exit 1
      ;;
  esac
}
