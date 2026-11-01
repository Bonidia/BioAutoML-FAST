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
export BIND_ADDRESS=127.0.0.1
export HOST_PORT="${CI_PORT:-18501}"
export CI_ARTIFACTS_DIR="${CI_ARTIFACTS_DIR:-$test_path/logs}"
unset BLABLADOR_API_KEY MAILGUN_API_KEY
mkdir -p "$DATASETS_DIR" "$CI_ARTIFACTS_DIR"

cleanup() {
    local status=$?
    trap - EXIT
    for name in "$CONTAINER_NAME" "$CONTAINER_NAME-training"; do
        docker logs "$name" > "$CI_ARTIFACTS_DIR/$name.log" 2>&1 || true
    done
    ./run-docker.sh --delete </dev/null || status=1
    # Only these disposable CI keys are removed; keep logs for diagnosis.
    if [[ -d "$MODEL_KEYS_DIR" ]]; then
        rm -f -- "$MODEL_KEYS_DIR/signing.pem" "$MODEL_KEYS_DIR/trusted_keys.json" "$MODEL_KEYS_DIR/key_id"
        rmdir -- "$MODEL_KEYS_DIR" || status=1
    fi
    echo "Docker check logs: $CI_ARTIFACTS_DIR"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cp App/datasets/references.bib "$DATASETS_DIR/references.bib"
./run-docker.sh --build \
    --label "org.opencontainers.image.source=${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-Bonidia/BioAutoML-FAST}" \
    --label "org.opencontainers.image.revision=${GITHUB_SHA:-local}" </dev/null
"${CI_PYTHON:-python3}" .github/check_docker.py
if [[ -n "${CI_TESTED_IMAGE:-}" ]]; then
    docker tag "$IMAGE_NAME" "$CI_TESTED_IMAGE"
fi
