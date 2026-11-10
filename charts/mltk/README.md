# mltk Helm chart

Installs the mltk server (`mltk server`) with a persistent SQLite volume.

**SQLite is a single writer.** `replicaCount` defaults to `1`. Do not raise it.

## Install

```bash
helm install mltk charts/mltk
```

Override the image tag to pin a release:

```bash
helm install mltk charts/mltk --set image.tag=v0.17.0
```

## Probes

The server router is mounted at `/api`:

- liveness: `GET /api/health/live`
- readiness: `GET /api/health/ready`

## External access and API keys

Enabling `ingress` makes the API reachable from outside the cluster. Read
endpoints require a reader key unless `MLTK_SERVER_PUBLIC_READ` is set. Set
server environment variables with the `extraEnv` value, for example:

```yaml
extraEnv:
  - name: MLTK_SERVER_PUBLIC_READ
    value: "true"
```

Create the first admin key from the deployed container. The deployment is
named `<release>-mltk` (just `<release>` when the release name already contains
`mltk`), and the default database path is `/data/mltk_server.db`:

```bash
kubectl exec deploy/<release>-mltk -- mltk server-create-key --role admin --db /data/mltk_server.db
```

## Back up before uninstalling

Uninstalling the release deletes the PVC and the database on it.

```bash
POD=$(kubectl get pod -l app.kubernetes.io/instance=mltk --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}') &&
  test -n "$POD" &&
  kubectl exec "$POD" -- mltk server-backup --out /tmp/mltk-backup.db --force &&
  kubectl cp "$POD:/tmp/mltk-backup.db" ./mltk-backup.db &&
  test -s ./mltk-backup.db &&
  helm uninstall mltk
```

`server-backup` reads the database path from the pod's `MLTK_DB_PATH`, which the chart sets to `server.dbPath`, so it backs up the database the server is actually using.

The commands are chained with `&&`, so `helm uninstall` runs only after the backup and the copy succeed. `--force` replaces a backup left in `/tmp` by an earlier run, so you never copy out a stale file.

The backup goes to the container's `/tmp`, so it doesn't use PVC space, and it disappears when the pod restarts.

Add `-n <namespace>` to each command if the release is not in the current namespace.

For images that do not have `server-backup` yet, when `kubectl exec "$POD" -- mltk server-backup --help` fails, use:

```bash
kubectl exec "$POD" -- python -c "import os, pathlib, sqlite3; s = sqlite3.connect(pathlib.Path(os.environ['MLTK_DB_PATH']).as_uri() + '?mode=ro', uri=True); d = sqlite3.connect('/tmp/mltk-backup.db'); s.backup(d); d.execute('PRAGMA journal_mode=DELETE'); d.close(); s.close()"
```

Then run the same `kubectl cp`, `test -s` and `helm uninstall` steps.

## Restore from a backup

```bash
helm install mltk charts/mltk --set server.dbPath=/data/restore-staging.db &&
  kubectl rollout status deploy/mltk &&
  POD=$(kubectl get pod -l app.kubernetes.io/instance=mltk --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}') &&
  test -n "$POD" &&
  kubectl exec "$POD" -- rm -f /data/mltk_server.db-wal /data/mltk_server.db-shm &&
  kubectl cp ./mltk-backup.db "$POD:/data/mltk_server.db" &&
  helm upgrade mltk charts/mltk --reuse-values --set server.dbPath=/data/mltk_server.db &&
  kubectl rollout status deploy/mltk
```

The first install runs the server on an empty staging database, so nothing has the default path open while the backup is copied there. The `rm -f` clears any leftover WAL files, which must never be applied to the restored file. The upgrade switches the server to the default path, and the `Recreate` rollout stops the old pod before the new one opens the restored file.

The restored data ends up at the default `server.dbPath`, so later upgrades need no extra values. The empty `/data/restore-staging.db` stays on the volume; you can delete it from the new pod with `rm -f /data/restore-staging.db*`.

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
docker pull ghcr.io/liorrr/mltk:v0.17.0
MLTK_KIND_SMOKE=1 MLTK_KIND_SMOKE_PRELOAD=ghcr.io/liorrr/mltk:v0.17.0 python charts/mltk/kind_smoke.py
```

The preloaded tag must match the chart's default image (`v` + the chart's
`appVersion`), or the pod pulls the image anyway.
