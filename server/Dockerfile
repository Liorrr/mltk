FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir "mlspec[server,cli]"
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /app/data
ENV MLTK_DB_PATH=/app/data/mltk_server.db
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8080/api/health || exit 1
# Pass --db explicitly: mlspec 0.14.0 and earlier ignore MLTK_DB_PATH.
CMD ["sh", "-c", "exec mltk server --host 0.0.0.0 --port 8080 --db \"$MLTK_DB_PATH\""]
