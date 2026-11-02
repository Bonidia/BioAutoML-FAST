#!/usr/bin/env bash
set -Eeuo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
test_path="$(mktemp -d "${RUNNER_TEMP:-/tmp}/bioautoml-ci.XXXXXXXX")"
export CONTAINER_NAME="bioautoml-ci-${test_path##*.}"
export IMAGE_NAME="bioautoml-fast-ci:${test_path##*.}"
export JOBS_DIR="$test_path/jobs"
export STATE_DIR="$test_path/state"
export DATASETS_DIR="$test_path/datasets"
export MODEL_KEYS_DIR="$test_path/keys"
export MODEL_VERIFICATION_DIR="$test_path/verification"
export BIND_ADDRESS=127.0.0.1
export HOST_PORT="${CI_PORT:-18501}"
export CI_ARTIFACTS_DIR="${CI_ARTIFACTS_DIR:-$test_path/logs}"
final_container="$CONTAINER_NAME-example"
final_image="$IMAGE_NAME-example"
unset BLABLADOR_API_KEY MAILGUN_API_KEY
mkdir -p "$DATASETS_DIR" "$CI_ARTIFACTS_DIR"

cleanup() {
    local status=$?
    trap - EXIT
    for name in "$CONTAINER_NAME" "$CONTAINER_NAME-training" "$final_container"; do
        docker logs "$name" > "$CI_ARTIFACTS_DIR/$name.log" 2>&1 || true
    done
    docker rm -f "$final_container" >/dev/null 2>&1 || true
    docker image rm "$final_image" >/dev/null 2>&1 || true
    ./run-docker.sh --delete </dev/null || status=1
    # Remove temporary key copies, never the operator's source key.
    if [[ -d "$MODEL_KEYS_DIR" ]]; then
        rm -f -- "$MODEL_KEYS_DIR/signing.pem"
        rmdir -- "$MODEL_KEYS_DIR" || status=1
    fi
    if [[ -d "$MODEL_VERIFICATION_DIR" ]]; then
        rm -f -- "$MODEL_VERIFICATION_DIR/trusted_keys.json"
        rmdir -- "$MODEL_VERIFICATION_DIR" || status=1
    fi
    echo "Docker check logs: $CI_ARTIFACTS_DIR"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -n "${EXAMPLE_MODEL_SIGNING_KEY:-}" ]]; then
    mkdir -m 700 "$MODEL_KEYS_DIR"
    "${CI_PYTHON:-python3}" -c 'import os; from pathlib import Path; p = Path(os.environ["MODEL_KEYS_DIR"]) / "signing.pem"; p.touch(mode=0o600); p.write_text(os.environ["EXAMPLE_MODEL_SIGNING_KEY"].strip() + "\n")'
elif [[ -n "${CI_SIGNING_KEY_FILE:-}" ]]; then
    mkdir -m 700 "$MODEL_KEYS_DIR"
    install -m 600 "$CI_SIGNING_KEY_FILE" "$MODEL_KEYS_DIR/signing.pem"
elif [[ "${CI_REQUIRE_OFFICIAL_KEY:-0}" == 1 ]]; then
    echo 'Set the EXAMPLE_MODEL_SIGNING_KEY Actions secret before publishing an image.' >&2
    exit 1
fi
unset EXAMPLE_MODEL_SIGNING_KEY MODEL_SIGNING_KEY

./run-docker.sh --build \
    --label "org.opencontainers.image.source=${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-Bonidia/BioAutoML-FAST}" \
    --label "org.opencontainers.image.revision=${GITHUB_SHA:-local}" </dev/null
"${CI_PYTHON:-python3}" .github/check_docker.py
mkdir "$test_path/image"
docker cp "$CONTAINER_NAME:/tmp/ci-example" "$test_path/image/example"
docker build --file .github/example.Dockerfile --build-arg "BASE_IMAGE=$IMAGE_NAME" \
    --tag "$final_image" "$test_path/image"

# Start the published image with empty persistent storage and public keys only.
mkdir "$test_path/final-jobs" "$test_path/final-state"
docker run -d --name "$final_container" --security-opt no-new-privileges:true \
    --user "$(id -u):$(id -g)" \
    --cpus 2 --publish "127.0.0.1:${CI_EXAMPLE_PORT:-18502}:8501" \
    --mount "type=bind,src=$test_path/final-jobs,dst=/app/App/jobs" \
    --mount "type=bind,src=$test_path/final-state,dst=/app/App/task-results" \
    --mount "type=bind,src=$MODEL_VERIFICATION_DIR/trusted_keys.json,dst=/run/bioautoml/trusted_keys.json,readonly" \
    --env BIOAUTOML_TRUSTED_KEYS=/run/bioautoml/trusted_keys.json "$final_image"
CONTAINER_NAME="$final_container" HOST_PORT="${CI_EXAMPLE_PORT:-18502}" \
    "${CI_PYTHON:-python3}" .github/check_docker.py --example
if [[ -n "${CI_TESTED_IMAGE:-}" ]]; then
    docker tag "$final_image" "$CI_TESTED_IMAGE"
fi
