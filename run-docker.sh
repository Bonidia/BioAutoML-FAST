#!/usr/bin/env bash
# Rebuild and run the local application with live UI reload, preserving jobs and state.
set -Eeuo pipefail

project_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
image_name="${IMAGE_NAME:-bioautoml-fast:local}"
container_name="${CONTAINER_NAME:-bioautoml-fast}"
host_port="${HOST_PORT:-8501}"
bind_address="${BIND_ADDRESS:-127.0.0.1}"
jobs_path="${JOBS_DIR:-$project_path/App/jobs}"
state_path="${STATE_DIR:-$project_path/App/task-results}"
datasets_path="${DATASETS_DIR:-$project_path/App/datasets}"
owner_label="io.bioautoml-fast.launcher"
# Accept the previously documented flag; rebuilding is now unconditional.
if [[ "${1:-}" == "--build" ]]; then
    shift
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: $0 [Docker build options, e.g. --no-cache-filter mathfeature]"
    echo "Always rebuilds the image and recreates the container with live UI reload."
    echo "Replaces this script's container and image tag. Jobs/state are NEVER deleted."
    echo "Any running training jobs in that container will be interrupted."
    echo "Settings: IMAGE_NAME, CONTAINER_NAME, HOST_PORT, BIND_ADDRESS, JOBS_DIR, STATE_DIR, DATASETS_DIR."
    echo "Defaults: bioautoml-fast:local, bioautoml-fast, 8501, 127.0.0.1, App/jobs, App/task-results, App/datasets."
    echo "DATASETS_DIR is mounted read-only and must contain references.bib and your model repository."
    echo "Run as a non-root user with Docker access. Paths are resolved on the Docker host."
    exit 0
fi

fail() { echo "Error: $*" >&2; exit 1; }
command -v docker >/dev/null || fail "Docker is not installed."
docker info >/dev/null 2>&1 || fail "Cannot access Docker. Check the daemon and your Docker permissions."
[[ "$(id -u)" != "0" ]] || fail "Run this script as a non-root user with Docker access (without sudo)."
[[ "$container_name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || fail "Invalid CONTAINER_NAME."
[[ "$image_name" != -* && "$image_name" != *@* && "$image_name" != sha256:* ]] || fail "IMAGE_NAME must be a local image name/tag, not an ID or digest."
[[ "$host_port" =~ ^[0-9]{1,5}$ ]] || fail "HOST_PORT must be between 1 and 65535."
host_port=$((10#$host_port))
(( host_port >= 1 && host_port <= 65535 )) || fail "HOST_PORT must be between 1 and 65535."

development_options=()
[[ "$project_path" != *,* ]] || fail "Bind-mount paths cannot contain commas."
for path in App/app.py App/modules start.sh; do
    [[ -r "$project_path/$path" ]] || fail "Missing or unreadable development source: $path"
    development_options+=(--mount "type=bind,source=$project_path/$path,target=/app/$path,readonly")
done
# Mount the startup script too so this works with previously built images.
development_options+=(--env STREAMLIT_SERVER_RUN_ON_SAVE=true
                      --env STREAMLIT_SERVER_FILE_WATCHER_TYPE=poll)

# Do not accidentally replace an unrelated application using the same name.
if docker container inspect "$container_name" >/dev/null 2>&1; then
    owner="$(docker container inspect --format "{{index .Config.Labels \"$owner_label\"}}" "$container_name")"
    [[ "$owner" == "run-docker.sh" ]] || fail "Container '$container_name' was not created by this script. Choose another CONTAINER_NAME or remove it manually after checking it."
fi

# Validate repository data before stopping the current application. Do not
# silently create an empty datasets directory that would break the Repository tab.
[[ -d "$datasets_path" && -r "$datasets_path" && -x "$datasets_path" ]] || fail "DATASETS_DIR must be an existing readable directory: $datasets_path"
datasets_path="$(cd -- "$datasets_path" && pwd -P)"
[[ "$datasets_path" != *,* ]] || fail "Bind-mount paths cannot contain commas."
[[ -f "$datasets_path/references.bib" && -r "$datasets_path/references.bib" ]] || fail "Missing or unreadable bibliography: $datasets_path/references.bib"

mkdir -p -- "$jobs_path" "$state_path"
jobs_path="$(cd -- "$jobs_path" && pwd -P)"
state_path="$(cd -- "$state_path" && pwd -P)"
for path in "$jobs_path" "$state_path"; do
    [[ "$path" != *,* ]] || fail "Bind-mount paths cannot contain commas."
    [[ -w "$path" && -x "$path" ]] || fail "Directory is not writable by your user: $path"
done
[[ "$jobs_path" != "$state_path" ]] || fail "JOBS_DIR and STATE_DIR must be separate directories."

if docker container inspect "$container_name" >/dev/null 2>&1; then
    echo "Stopping and removing container $container_name (running jobs will be interrupted)..."
    docker stop --time 40 "$container_name" >/dev/null
    docker container rm "$container_name" >/dev/null
fi
if docker image inspect "$image_name" >/dev/null 2>&1; then
    echo "Removing existing image tag $image_name..."
    docker image rm "$image_name" || fail "Image is still in use. No forced image removal was attempted."
fi

echo "Building $image_name..."
docker build "$@" --tag "$image_name" --file "$project_path/Dockerfile" "$project_path"

echo "Live UI reload enabled with read-only source mounts."
echo "Avoid editing job-processing code while jobs are running; worker changes require a restart."
echo "Starting $container_name with persistent local directories..."
docker run --detach --name "$container_name" \
    --label "$owner_label=run-docker.sh" \
    --restart unless-stopped --stop-timeout 40 \
    --user "$(id -u):$(id -g)" \
    --publish "$bind_address:$host_port:8501" \
    --mount "type=bind,source=$jobs_path,target=/app/App/jobs" \
    --mount "type=bind,source=$state_path,target=/app/App/task-results" \
    --mount "type=bind,source=$datasets_path,target=/app/App/datasets,readonly" \
    --env TASK_RESULTS_DB=/app/App/task-results/task_results.db \
    --env REDIS_DATA_DIR=/app/App/task-results/redis \
    "${development_options[@]}" \
    "$image_name"

echo "Started. Web: http://$bind_address:$host_port"
echo "Jobs: $jobs_path"
echo "SQLite: $state_path/task_results.db (including SQLite sidecar files)"
echo "Redis state: $state_path/redis"
echo "Model repository (read-only): $datasets_path"
echo "Follow startup: docker logs --follow $container_name"
echo "Check health: docker inspect --format '{{.State.Health.Status}}' $container_name"
