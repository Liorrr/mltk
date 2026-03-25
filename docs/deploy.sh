#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="mltk-docs"
CONTAINER_NAME="mltk-docs"
HOST_PORT=8080

usage() {
    echo "Usage: $0 [--stop | --rebuild]"
    echo ""
    echo "  (no flag)   Build image and start container"
    echo "  --stop      Stop and remove the running container"
    echo "  --rebuild   Stop, rebuild image, and restart container"
    exit 1
}

stop_container() {
    if docker ps -q --filter "name=${CONTAINER_NAME}" | grep -q .; then
        echo "Stopping container: ${CONTAINER_NAME}"
        docker stop "${CONTAINER_NAME}"
        docker rm "${CONTAINER_NAME}"
        echo "Container stopped and removed."
    else
        echo "No running container named '${CONTAINER_NAME}' found."
    fi
}

build_image() {
    echo "Building Docker image: ${IMAGE_NAME}"
    # Build context is the project root (one level up from docs/)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
    docker build -f "${SCRIPT_DIR}/Dockerfile" -t "${IMAGE_NAME}" "${PROJECT_ROOT}"
    echo "Image built successfully."
}

start_container() {
    echo "Starting container: ${CONTAINER_NAME}"
    docker run -d \
        -p "${HOST_PORT}:80" \
        --name "${CONTAINER_NAME}" \
        --restart unless-stopped \
        "${IMAGE_NAME}"
    echo ""
    echo "mltk docs running at: http://localhost:${HOST_PORT}"
}

FLAG="${1:-}"

case "${FLAG}" in
    "")
        build_image
        start_container
        ;;
    --stop)
        stop_container
        ;;
    --rebuild)
        stop_container
        build_image
        start_container
        ;;
    *)
        usage
        ;;
esac
