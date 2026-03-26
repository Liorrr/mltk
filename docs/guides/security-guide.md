# Security Guide

Security architecture and protections built into the mltk server.

---

## 1. SSRF (Server-Side Request Forgery) Protection

The webhook system allows users to register external URLs that receive POST notifications on test events. Without proper safeguards an attacker could register internal URLs (e.g., `http://127.0.0.1/admin`) and use the server as a proxy to access internal services.

### URL Validation (`validate_webhook_url`)

Every webhook URL is validated at two points:

1. **At registration time** (POST `/api/webhooks`) -- the route rejects invalid URLs with HTTP 422.
2. **At dispatch time** (`send_webhook`) -- the URL is re-validated before any HTTP request is made.

The validator rejects:

| Check | Examples blocked |
|-------|-----------------|
| Non-http(s) schemes | `file:///etc/passwd`, `ftp://`, `gopher://` |
| Missing/empty hostname | `https:///no-host` |
| Literal `localhost` | `http://localhost/hook`, `http://localhost.localdomain/` |
| Loopback IPs | `127.0.0.1`, `127.255.255.255`, `::1` |
| Private RFC-1918 IPs | `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16` |
| IPv6 unique local | `fc00::/7` |

### DNS Rebinding Defence

A DNS rebinding attack exploits the time gap between URL validation and the actual HTTP request. The attacker's DNS server returns a safe public IP during `validate_webhook_url()`, then switches to `127.0.0.1` by the time `send_webhook()` makes the request.

**Mitigation:** `send_webhook()` resolves the hostname via `socket.getaddrinfo()` immediately before making the HTTP request, and checks the resolved IP against the same private-network blocklist:

```python
import socket
addrinfo = socket.getaddrinfo(host, port)
resolved_ip = addrinfo[0][4][0]
if _is_private_ip(resolved_ip):
    return False  # block the request
```

This closes the TOCTOU (time-of-check-time-of-use) gap. Even if DNS returns a different IP at dispatch time, the resolved address is validated before any data is sent.

### Redirect Following Disabled

HTTP 3xx redirects are a secondary SSRF vector: the webhook target could return a `302 Found` pointing to an internal address. The `send_webhook()` function uses a custom `_NoRedirectHandler` that raises an exception on any redirect response, preventing the request from following it.

### Defence in Depth Summary

| Layer | What it catches |
|-------|----------------|
| URL validation (registration) | Malformed URLs, private IPs, localhost, bad schemes |
| URL validation (dispatch) | Same checks repeated before every send |
| DNS rebinding guard | Attacker-controlled DNS flipping to private IPs |
| Redirect blocking | 3xx responses pointing to internal addresses |

---

## 2. Authentication

### API Key Model

Write operations (POST, DELETE) require a Bearer token. Keys are generated with `mltk server-create-key` and stored as SHA-256 hashes in the `api_keys` table. The raw key is shown once at creation and never stored.

```bash
mltk server-create-key --project my-project
# mltk_aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890abcd
```

Usage:

```bash
curl -X POST http://localhost:8080/api/runs \
  -H "Authorization: Bearer mltk_your_key_here" \
  -H "Content-Type: application/json" \
  -d '...'
```

Read operations (GET) are unauthenticated by default. If your deployment requires read-auth, put the server behind a reverse proxy with its own authentication layer.

### Key Storage

Keys are hashed with SHA-256 before persistence. The database stores only the hash, so a database leak does not expose usable credentials. Each key is bound to a project name.

---

## 3. Database Security

### WAL Mode

The server enables SQLite WAL (Write-Ahead Logging) journal mode. This improves concurrent-read performance and reduces the risk of database corruption under load.

### Foreign Key Enforcement

`PRAGMA foreign_keys = ON` is set on every connection. This prevents orphan rows in the `results` table (every result must reference a valid `runs.id`).

### Migration System

Schema changes are tracked in a `schema_versions` table. Each migration is applied exactly once and recorded with a timestamp. See the [DevOps Guide](devops-guide.md) for details on the migration system.

---

## 4. Deployment Hardening Checklist

- [ ] Use HTTPS in production (TLS termination via nginx, Caddy, or cloud LB)
- [ ] Restrict the bind address (`--host 127.0.0.1`) when behind a reverse proxy
- [ ] Set rate limiting on the reverse proxy (recommended: 10 req/s burst 20)
- [ ] Store API keys in a secrets manager, not in plaintext config files
- [ ] Back up the SQLite database regularly (see DevOps Guide)
- [ ] Monitor webhook delivery failures for signs of SSRF probing
- [ ] Keep mltk updated to get the latest security patches
