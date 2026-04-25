# syntax=docker/dockerfile:1
# mltk multi-stage runtime image.
#
# Targets:
#   runtime-slim  -> ghcr.io/liorrr/mltk:latest    (python:3.12-slim + mltk[all])
#   runtime-full  -> ghcr.io/liorrr/mltk:full      (runtime-slim + Trivy 0.60.0)
#
# Build locally:
#   docker build --target runtime-slim -t mltk:slim .
#   docker build --target runtime-full -t mltk:full .
#
ARG PYTHON_VERSION=3.12

# --- slim runtime (default) --------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install mltk with all optional extras. If the published wheel is unavailable
# (for example during local dev builds from a checkout), a caller can override
# with `--build-arg MLTK_PIP_TARGET=.` to install from source.
ARG MLTK_PIP_TARGET=mltk[all]
RUN pip install --no-cache-dir "${MLTK_PIP_TARGET}"

ENTRYPOINT ["mltk"]
CMD ["--help"]

# --- full runtime (bundles Trivy for container scanning) ---------------------
FROM runtime-slim AS runtime-full

# Trivy 0.60.0 (2026-04 stable) — pin version so JSON schema is deterministic.
# Update this tag + docs/guides/container-deployment.md + CHANGELOG together.
COPY --from=aquasec/trivy:0.60.0 /usr/local/bin/trivy /usr/local/bin/trivy

ENV TRIVY_BIN=/usr/local/bin/trivy
