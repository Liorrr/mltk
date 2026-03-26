FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir mltk[server]
EXPOSE 8080
CMD ["mltk", "server", "--host", "0.0.0.0", "--port", "8080"]
