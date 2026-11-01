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
chart, waits for the deployment, curls `/api/health`, then deletes the
cluster. It refuses to run unless `MLTK_KIND_SMOKE=1`.
