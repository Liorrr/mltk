FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir mltk[server]
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8080/api/health || exit 1
CMD ["mltk", "server", "--host", "0.0.0.0", "--port", "8080"]
