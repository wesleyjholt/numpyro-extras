#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKLIST_FILE="${1:-$ROOT_DIR/checklist.md}"
MAX_ITERS="${MAX_ITERS:-50}"
PROMPT_TEXT="${PROMPT_TEXT:-do next step}"
CODEX_FLAGS="${CODEX_FLAGS:---full-auto --sandbox workspace-write}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/.codex-step-runs}"

if ! command -v codex >/dev/null 2>&1; then
  echo "ERROR: codex CLI not found in PATH."
  exit 1
fi

if [[ ! -f "$CHECKLIST_FILE" ]]; then
  echo "ERROR: checklist file not found: $CHECKLIST_FILE"
  echo "Create checklist.md first, then rerun."
  exit 1
fi

mkdir -p "$LOG_DIR"

count_open_steps() {
  rg -c '^\s*[-*]\s+\[ \]\s+' "$CHECKLIST_FILE" || true
}

first_open_step() {
  rg -n '^\s*[-*]\s+\[ \]\s+' "$CHECKLIST_FILE" | head -n 1 | sed -E 's/^[0-9]+://'
}

open_before="$(count_open_steps)"
if [[ "$open_before" -eq 0 ]]; then
  echo "No unchecked steps found in $CHECKLIST_FILE."
  exit 0
fi

echo "Starting step runner."
echo "Checklist: $CHECKLIST_FILE"
echo "Unchecked steps: $open_before"
echo "Max iterations: $MAX_ITERS"
echo

for ((i=1; i<=MAX_ITERS; i++)); do
  open_before="$(count_open_steps)"
  if [[ "$open_before" -eq 0 ]]; then
    echo "All checklist steps are complete."
    exit 0
  fi

  next_step="$(first_open_step)"
  run_log="$LOG_DIR/run_${i}.last_message.txt"

  echo "[$i/$MAX_ITERS] Next unchecked step: $next_step"
  echo "[$i/$MAX_ITERS] Launching fresh Codex session..."

  run_prompt=$(
    cat <<EOF
$PROMPT_TEXT

Read AGENTS.md and $CHECKLIST_FILE first.
Execute exactly one assigned step and stop.
When done, update $CHECKLIST_FILE by marking the completed step.
EOF
  )

  # shellcheck disable=SC2086
  if ! codex exec \
    --cd "$ROOT_DIR" \
    --output-last-message "$run_log" \
    $CODEX_FLAGS \
    "$run_prompt"; then
    echo "Codex run failed on iteration $i. See logs in $LOG_DIR."
    exit 2
  fi

  open_after="$(count_open_steps)"
  if [[ "$open_after" -ge "$open_before" ]]; then
    echo "No checklist progress detected after iteration $i."
    echo "Stopping to avoid infinite loop."
    echo "Last message log: $run_log"
    exit 3
  fi

  echo "[$i/$MAX_ITERS] Progress made: $open_before -> $open_after unchecked."
  echo
done

echo "Reached MAX_ITERS=$MAX_ITERS before checklist completion."
exit 4

