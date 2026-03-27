"""Structured logging configuration for the mltk server."""
from __future__ import annotations

import json
import logging
import os


class JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects.

    Activated when the ``MLTK_LOG_FORMAT`` environment variable is set
    to ``"json"``.  Useful for log aggregation pipelines (ELK, Datadog,
    Loki) that expect structured log lines.
    """

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        })


def setup_logging() -> None:
    """Configure root logger based on ``MLTK_LOG_FORMAT`` env var.

    When ``MLTK_LOG_FORMAT=json``, attaches a :class:`JsonFormatter` to
    the root logger so all log output is machine-parseable JSON.
    Otherwise, Python's default text formatter is used.
    """
    fmt = os.environ.get("MLTK_LOG_FORMAT", "text")
    level = os.environ.get("MLTK_LOG_LEVEL", "INFO").upper()

    if fmt == "json":
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logging.root.handlers.clear()
        logging.root.addHandler(handler)
        logging.root.setLevel(getattr(logging, level, logging.INFO))
