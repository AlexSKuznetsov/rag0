#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./dev.sh

Launches a local Temporal dev server, the RAG0 worker, and starts an interactive workflow session.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: missing required command '$1'." >&2
    exit 1
  fi
}

require_cmd temporal
require_cmd make

if [[ -n "${PYTHON_BIN:-}" ]]; then
  require_cmd "$PYTHON_BIN"
else
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Error: missing required command 'python3' (or 'python')." >&2
    exit 1
  fi
fi

ADDRESS="${TEMPORAL_ADDRESS:-127.0.0.1:7233}"
NAMESPACE="${TEMPORAL_NAMESPACE:-default}"
TASK_QUEUE="${TEMPORAL_TASK_QUEUE:-rag0}"

HOST="${ADDRESS%:*}"
PORT="${ADDRESS##*:}"
if [[ -z "$PORT" ]]; then PORT="7233"; fi

cleanup() {
  local status=$?
  if [[ -n "${WORKER_PID:-}" ]]; then
    kill "$WORKER_PID" >/dev/null 2>&1 || true
    wait "$WORKER_PID" 2>/dev/null || true
  fi
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

echo "Starting Temporal dev server on ${ADDRESS} (namespace=${NAMESPACE})..."
temporal server start-dev \
  --ip "$HOST" \
  --port "$PORT" \
  --namespace "$NAMESPACE" \
  > /dev/null 2>&1 &
SERVER_PID=$!
wait_for_port() {
  "$PYTHON_BIN" - <<PY
import socket
import sys
import time

host = "$HOST"
port = int("$PORT")
deadline = time.time() + 60
while time.time() < deadline:
    with socket.socket() as sock:
        sock.settimeout(1)
        try:
            sock.connect((host, port))
            sys.exit(0)
        except OSError:
            time.sleep(0.5)
print(f"Timed out waiting for {host}:{port}")
sys.exit(1)
PY
}
wait_for_port
echo "Temporal server is ready."

sleep 2
echo "Starting Temporal worker (task queue=${TASK_QUEUE})..."
"$PYTHON_BIN" -m src.temporal.worker --address "$ADDRESS" --namespace "$NAMESPACE" --task-queue "$TASK_QUEUE" &
WORKER_PID=$!

if ! kill -0 "$WORKER_PID" 2>/dev/null; then
  echo "Error: Worker failed to start."
  exit 1
fi
echo "Worker is running."

cmd_env=(
  "TEMPORAL_ADDRESS=$ADDRESS"
  "TEMPORAL_NAMESPACE=$NAMESPACE"
  "TEMPORAL_TASK_QUEUE=$TASK_QUEUE"
)

echo "Launching interactive workflow session..."
env "${cmd_env[@]}" make interactive

echo "Interactive session complete. Cleaning up background processes."
