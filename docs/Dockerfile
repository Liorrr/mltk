# ── Stage 1: Build ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir \
    mkdocs-material \
    mkdocstrings[python]

# Copy project files required by mkdocs build
COPY mkdocs.yml .
COPY docs/ docs/

RUN mkdocs build --strict

# ── Stage 2: Serve ─────────────────────────────────────────────────────────────
FROM nginx:alpine

COPY --from=builder /app/site /usr/share/nginx/html
COPY docs/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
