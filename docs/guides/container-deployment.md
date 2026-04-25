# Container Deployment

mltk ships two officially supported Docker images on GitHub Container Registry
(GHCR). Both are multi-architecture (`linux/amd64` and `linux/arm64`) and are
built from the root `Dockerfile` in this repository.

| Image | Base | Extra tooling | Use case |
| --- | --- | --- | --- |
| `ghcr.io/liorrr/mltk:latest` | `python:3.12-slim` + `mltk[all]` | none | General CLI, scans, MCP server, CI jobs |
| `ghcr.io/liorrr/mltk:full` | `latest` + Trivy 0.60.0 | `/usr/local/bin/trivy` | Container security scanning |

Version-pinned tags (`ghcr.io/liorrr/mltk:v0.11.1`, `...:v0.11.1-full`) are
published automatically on every `v*` git tag via the
`docker-publish.yml` GitHub Actions workflow.

The legacy `server/Dockerfile` is superseded by this root `Dockerfile`. Use
the root image for all new deployments.

## 1. Pulling the images

```bash
# Slim runtime (default — the one you want 90% of the time)
docker pull ghcr.io/liorrr/mltk:latest

# Full runtime with Trivy bundled for container scanning
docker pull ghcr.io/liorrr/mltk:full
```

Pin to a specific release in production:

```bash
docker pull ghcr.io/liorrr/mltk:v0.11.1
docker pull ghcr.io/liorrr/mltk:v0.11.1-full
```

## 2. Quick smoke test

```bash
docker run --rm ghcr.io/liorrr/mltk:latest --help
docker run --rm ghcr.io/liorrr/mltk:latest --version
```

The image's `ENTRYPOINT` is `mltk`, so any CLI subcommand works directly:

```bash
docker run --rm -v "$PWD":/work -w /work ghcr.io/liorrr/mltk:latest \
  scan data --path ./datasets/train.csv
```

## 3. Running a container security scan

The `:full` image bundles a pinned Trivy binary so container scans produce
deterministic JSON output across runs. Enable the end-to-end container scan
path by setting `MLTK_CONTAINER_E2E=1`.

```bash
docker run --rm \
  -e MLTK_CONTAINER_E2E=1 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$PWD":/work -w /work \
  ghcr.io/liorrr/mltk:full \
  container scan python:3.12-slim --json
```

To point mltk at a Trivy binary that lives somewhere other than the default,
override `TRIVY_BIN`:

```bash
docker run --rm -e TRIVY_BIN=/opt/trivy/bin/trivy ghcr.io/liorrr/mltk:full ...
```

## 4. docker-compose

A minimal compose file for running the mltk MCP/HTTP server alongside a
long-lived scan worker:

```yaml
services:
  mltk-server:
    image: ghcr.io/liorrr/mltk:latest
    command: ["server", "--host", "0.0.0.0", "--port", "8080"]
    ports:
      - "8080:8080"
    volumes:
      - ./workspace:/work
    working_dir: /work

  mltk-scan:
    image: ghcr.io/liorrr/mltk:full
    command: ["scan", "all", "--path", "/work"]
    environment:
      MLTK_CONTAINER_E2E: "1"
    volumes:
      - ./workspace:/work
      - /var/run/docker.sock:/var/run/docker.sock
    working_dir: /work
    depends_on:
      - mltk-server
```

Run it:

```bash
docker compose up --build
```

## 5. Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `MLTK_CONTAINER_E2E` | unset | Set to `1` to enable end-to-end container-scan flows that shell out to Trivy. Off by default so unit-test runs stay hermetic. |
| `TRIVY_BIN` | `/usr/local/bin/trivy` (in `:full`) | Absolute path to the Trivy binary mltk should invoke. Override when bundling a different Trivy version or when running the `:latest` image with a host-mounted Trivy. |
| `PYTHONDONTWRITEBYTECODE` | `1` | Set in the image; keeps `.pyc` files out of read-only layers. |
| `PYTHONUNBUFFERED` | `1` | Set in the image; flushes stdout/stderr so container logs appear immediately. |

## 6. Verifying the image

Every published image is signed with the default GitHub-issued
`GITHUB_TOKEN`-backed provenance. To inspect which Trivy version is baked in:

```bash
docker run --rm ghcr.io/liorrr/mltk:full trivy --version
```

Expected output begins with `Version: 0.60.0`.
