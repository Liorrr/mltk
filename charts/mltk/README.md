# mltk Helm chart

Installs the mltk server (`mltk server`) with a persistent SQLite volume.

**SQLite is a single writer.** `replicaCount` defaults to `1`. Do not raise it.

## Install

```bash
helm install mltk charts/mltk
```

Override the image tag to pin a release:

```bash
helm install mltk charts/mltk --set image.tag=v0.14.0
```

## Probes

The server router is mounted at `/api`:

- liveness: `GET /api/health/live`
- readiness: `GET /api/health/ready`

## Back up before uninstalling

Uninstalling the release deletes the PVC and the database on it.

```bash
kubectl exec deploy/mltk -- mltk server-backup --db /data/mltk_server.db --out /tmp/mltk-backup.db
POD=$(kubectl get pod -l app.kubernetes.io/instance=mltk -o jsonpath='{.items[0].metadata.name}')
kubectl cp "$POD:/tmp/mltk-backup.db" ./mltk-backup.db
helm uninstall mltk
```

The backup goes to the container's `/tmp`, so it doesn't use PVC space, and it disappears when the pod restarts.

Add `-n <namespace>` to each command if the release is not in the current namespace.

For images that do not have `server-backup` yet, when `kubectl exec deploy/mltk -- mltk server-backup --help` fails, use:

```bash
kubectl exec deploy/mltk -- python -c "import sqlite3; s = sqlite3.connect('file:/data/mltk_server.db?mode=ro', uri=True); d = sqlite3.connect('/tmp/mltk-backup.db'); s.backup(d); d.execute('PRAGMA journal_mode=DELETE'); d.close(); s.close()"
```

Then run the same `kubectl cp` step.

## Restore from a backup

```bash
helm install mltk charts/mltk
kubectl rollout status deploy/mltk
POD=$(kubectl get pod -l app.kubernetes.io/instance=mltk -o jsonpath='{.items[0].metadata.name}')
kubectl cp ./mltk-backup.db "$POD:/data/mltk_restored.db"
helm upgrade mltk charts/mltk --reuse-values --set server.dbPath=/data/mltk_restored.db
```

The `Recreate` rollout stops the pod running on the new, empty database and starts one on the restored file.

Keep `server.dbPath` set in later upgrades with `--reuse-values` or a values file; dropping it points the server back at the empty default database.

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
slower first image pull. For repeated runs, pull the image once; the script
then copies it into each fresh KinD node instead of pulling it in-cluster:

```bash
docker pull ghcr.io/liorrr/mltk:v0.14.0
MLTK_KIND_SMOKE=1 MLTK_KIND_SMOKE_PRELOAD=ghcr.io/liorrr/mltk:v0.14.0 python charts/mltk/kind_smoke.py
```

The preloaded tag must match the chart's default image (`v` + the chart's
`appVersion`), or the pod pulls the image anyway.
