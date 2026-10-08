#!/bin/bash
set -Eeuo pipefail

app_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/App"
cd "$app_path"
umask 0002

export TASK_RESULTS_DB="${TASK_RESULTS_DB:-$app_path/task-results/task_results.db}"
redis_path="${REDIS_DATA_DIR:-$app_path/task-results/redis}"
redis_port="${REDIS_PORT:-6379}"
export REDIS_URL="${REDIS_URL:-redis://127.0.0.1:$redis_port/0}"
export RQ_WORKER_NAME="${RQ_WORKER_NAME:-bioautoml-container}"
shutdown_timeout="${SHUTDOWN_TIMEOUT:-25}"
redis_pid=""
worker_pid=""
web_pid=""

if [[ ! "$shutdown_timeout" =~ ^[0-9]+$ ]]; then
    echo "SHUTDOWN_TIMEOUT must be an integer number of seconds." >&2
    exit 1
fi

stop_group() {
    local pid="$1" signal="$2"
    if [[ -n "$pid" ]]; then
        kill -"$signal" -- "-$pid" 2>/dev/null || true
    fi
}

group_running() {
    [[ -n "$1" ]] && kill -0 -- "-$1" 2>/dev/null
}

cleanup() {
    local status=$?
    trap - EXIT
    trap '' TERM INT
    # Keep Redis available while the RQ worker records its shutdown.
    if [[ -n "$worker_pid" ]]; then
        kill -TERM "$worker_pid" 2>/dev/null || true
    fi
    stop_group "$web_pid" TERM
    local deadline=$((SECONDS + shutdown_timeout))
    while (( SECONDS < deadline )); do
        if ! group_running "$worker_pid" && ! group_running "$web_pid"; then
            break
        fi
        sleep 0.2
    done
    stop_group "$worker_pid" KILL
    stop_group "$web_pid" KILL
    stop_group "$redis_pid" TERM
    if [[ -n "$redis_pid" ]]; then
        wait "$redis_pid" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -p "$redis_path" "$(dirname -- "$TASK_RESULTS_DB")" jobs

# External Redis is optional; only manage a local server when requested.
if [[ "${START_REDIS:-1}" == "1" ]]; then
    # Refuse to attach a worker to an unrelated Redis already on this port.
    python - "$redis_port" <<'PY'
import socket
import sys

with socket.socket() as probe:
    try:
        probe.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError as error:
        raise SystemExit("Redis port is already in use; choose REDIS_PORT or explicitly use START_REDIS=0.") from error
PY
    setsid redis-server --bind 127.0.0.1 --protected-mode yes \
        --port "$redis_port" --daemonize no --loglevel warning \
        --dir "$redis_path" --appendonly yes --appendfsync everysec &
    redis_pid=$!
fi

echo "Waiting for Redis..."
redis_ready=0
for ((attempt=0; attempt<60; attempt++)); do
    if [[ -n "$redis_pid" ]] && ! kill -0 "$redis_pid" 2>/dev/null; then
        echo "Redis exited before startup completed." >&2
        exit 1
    fi
    if python -c 'import os; from redis import Redis; Redis.from_url(os.environ["REDIS_URL"], socket_connect_timeout=1, socket_timeout=1).ping()' 2>/dev/null; then
        redis_ready=1
        break
    fi
    sleep 0.5
done
if [[ "$redis_ready" != "1" ]]; then
    echo "Redis did not become ready." >&2
    exit 1
fi
echo "Redis ready."

setsid python worker.py &
worker_pid=$!
setsid streamlit run app.py --server.address=0.0.0.0 \
    --server.port="${PORT:-8501}" --server.headless=true \
    --server.runOnSave="${STREAMLIT_SERVER_RUN_ON_SAVE:-false}" &
web_pid=$!

# If any service exits, stop the others and fail the container so a restart
# policy can recover it. An exec of Streamlit alone would hide worker failures.
service_pids=("$worker_pid" "$web_pid")
if [[ -n "$redis_pid" ]]; then
    service_pids+=("$redis_pid")
fi
status=0
wait -n "${service_pids[@]}" || status=$?
echo "A BioAutoML-FAST service exited (status $status); stopping services." >&2
if [[ "$status" == "0" ]]; then
    status=1
fi
exit "$status"
