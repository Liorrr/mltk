# Container Scanning

mltk integrates with [Trivy](https://trivy.dev/) (Aqua Security) to bring
container image scanning into your pytest test suite. Scan model-serving
images for OS package vulnerabilities, language-level CVEs, exposed secrets,
and misconfigurations -- then fail the build when severity thresholds are
exceeded.

## Installation

```bash
pip install mltk[container]
```

!!! info "Trivy binary"
    The `mltk[container]` extra pulls in the `trivy-py` wrapper, which
    bundles the Trivy binary. If a system-wide `trivy` is already on
    `PATH`, mltk uses it instead. Override explicitly by setting
    `TRIVY_BIN=/path/to/trivy`.

## Quick Start

### Scan in pytest

```python
from mltk.container import (
    assert_container_vulnerabilities,
    assert_no_secrets_in_image,
)


def test_base_image_security():
    assert_container_vulnerabilities(
        "my-org/my-model-service:latest",
        max_critical=0,
        max_high=2,
    )
    assert_no_secrets_in_image("my-org/my-model-service:latest")
```

The assertions raise `AssertionError` with a structured message when a
severity threshold is exceeded, so pytest reports them like any other test
failure.

### Scan from CLI

```bash
mltk container scan alpine:3.18 --max-critical 0 --json
```

Exit code is non-zero when thresholds are exceeded, which is exactly what
CI systems need to fail a build.

## CI/CD Integration

### GitHub Actions

```yaml
- name: Scan model container for vulnerabilities
  run: mltk container scan ${{ env.IMAGE_REF }} --max-critical 0 --junit-xml scan-results.xml

- name: Upload scan results
  uses: actions/upload-artifact@v4
  with:
    name: container-scan-results
    path: scan-results.xml
```

!!! tip "Caching Trivy's vulnerability database"
    Trivy downloads its database on first run. Cache `~/.cache/trivy`
    between workflow runs to cut scan time from ~30s to <5s.

### GitLab CI

```yaml
container_scan:
  image: python:3.11-slim
  script:
    - pip install mltk[container]
    - mltk container scan $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA --max-critical 0 --junit-xml scan.xml
  artifacts:
    when: always
    reports:
      junit: scan.xml
```

## Kubernetes CronJob

Run scheduled scans against images that are already deployed. This catches
newly disclosed CVEs in images you shipped weeks ago.

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: mltk-container-scan
spec:
  schedule: "0 2 * * *"  # Daily at 02:00 UTC
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: Never
          containers:
          - name: scanner
            image: liorrr/mltk:full
            resources:
              requests:
                memory: "256Mi"
                cpu: "100m"
              limits:
                memory: "512Mi"
                cpu: "500m"
            command:
            - mltk
            - container
            - scan
            - "$(IMAGE_REF)"
            - "--max-critical"
            - "0"
            - "--junit-xml"
            - "/results/scan.xml"
            env:
            - name: IMAGE_REF
              valueFrom:
                configMapKeyRef:
                  name: mltk-config
                  key: target_image
            volumeMounts:
            - name: results
              mountPath: /results
          volumes:
          - name: results
            emptyDir: {}
```

## Prometheus Metrics

mltk exposes assertion and container-scan counters in Prometheus exposition
format. This lets you alert on "a new CRITICAL vulnerability appeared in a
deployed image" without waiting for the next CI run.

```bash
pip install mltk[metrics]
mltk server  # exposes /metrics on the server port
```

Add to your Prometheus scrape config:

```yaml
scrape_configs:
  - job_name: 'mltk'
    static_configs:
      - targets: ['localhost:8000']
```

Metrics exposed:

| Metric | Type | Labels |
|--------|------|--------|
| `mltk_assertions_total` | Counter | `status`, `category` |
| `mltk_assertion_duration_seconds` | Histogram | `category` |
| `mltk_container_scan_vulnerabilities_total` | Counter | `severity` |

!!! note "Metrics are opt-in"
    Without the `mltk[metrics]` extra, `/metrics` returns HTTP 404 with an
    install hint. The rest of the server is unaffected.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MLTK_CONTAINER_E2E` | Set to `1` to enable live container-scan tests (requires a local Docker daemon and network access). |
| `TRIVY_BIN` | Absolute path to a Trivy binary. Overrides auto-detection on `PATH` and the bundled `trivy-py` copy. |

## Troubleshooting

!!! warning "`ImportError: Trivy binary not found`"
    Install the extra (`pip install mltk[container]`) or point
    `TRIVY_BIN` at an existing Trivy binary. On macOS: `brew install
    aquasecurity/trivy/trivy`. On Debian/Ubuntu, follow the
    [official install guide](https://trivy.dev/latest/getting-started/installation/).

!!! warning "Scans fail on air-gapped hosts"
    Trivy needs to download its vulnerability database on first run.
    Either mirror the database internally and set `TRIVY_DB_REPOSITORY`,
    or pre-populate `~/.cache/trivy` from a host with network access.
