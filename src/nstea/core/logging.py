"""Structured logging configuration for NS-TEA.

Uses structlog for JSON-formatted, correlation-ID-aware logging.
Call ``configure_logging()`` once at application startup.
"""

from __future__ import annotations

import logging
import sys

import structlog

from nstea.config import settings


def configure_logging() -> None:
    """Set up structlog processors and stdlib logging bridge.

    After calling this, both ``structlog.get_logger()`` and stdlib
    ``logging.getLogger()`` emit structured JSON to stderr.
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer()
            if settings.log_level == "DEBUG"
            else structlog.processors.JSONRenderer(),
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # Quiet noisy libraries
    for name in ("httpx", "httpcore", "urllib3", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)


def bind_correlation_id(correlation_id: str) -> None:
    """Bind a correlation ID to the current context (thread/task-local).

    All subsequent log messages in the same context will include this ID.
    """
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)


def clear_contextvars() -> None:
    """Clear all context-local log bindings (call at end of request)."""
    structlog.contextvars.clear_contextvars()
