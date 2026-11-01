# mltk Helm chart

Installs the mltk server (`mltk server`) with a persistent SQLite volume.

**SQLite is a single writer.** `replicaCount` defaults to `1`. Do not raise it.

## Install

```bash
helm install mltk charts/mltk
```

Override the image tag to pin a release:

```bash
helm install mltk charts/mltk --set image.tag=v0.13.0
```

## Probes

The server router is mounted at `/api`:

- liveness: `GET /api/health/live`
- readiness: `GET /api/health/ready`

## KinD smoke (optional, not CI)

Requires `kind`, `helm`, and `kubectl` on PATH. This is **not** a default
CI job.

```bash
export MLTK_KIND_SMOKE=1
python charts/mltk/kind_smoke.py
```

The script creates a local cluster named `mltk-smoke`, installs the
chart, and waits until the Deployment is Available (the readiness probe on
`/api/health/ready` has passed) before deleting the cluster. It refuses to run
unless `MLTK_KIND_SMOKE=1`.

The install timeout defaults to 15 minutes. Override it with
`MLTK_KIND_SMOKE_TIMEOUT`, for example `MLTK_KIND_SMOKE_TIMEOUT=20m` for a
slower first image pull. For repeated runs, pull the image once and load it
into the KinD node before running the smoke:

```bash
docker pull ghcr.io/liorrr/mltk:v0.13.0
kind load docker-image ghcr.io/liorrr/mltk:v0.13.0 --name mltk-smoke
```
