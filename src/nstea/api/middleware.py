"""FastAPI middleware for structured logging and correlation IDs."""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from nstea.core.logging import bind_correlation_id, clear_contextvars

import structlog

logger = structlog.stdlib.get_logger(__name__)


class CorrelationMiddleware(BaseHTTPMiddleware):
    """Inject a correlation ID into every request and log request lifecycle."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4())[:8])
        bind_correlation_id(correlation_id)

        start = time.time()
        logger.info(
            "request_start",
            method=request.method,
            path=request.url.path,
        )

        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request_error", method=request.method, path=request.url.path)
            raise
        finally:
            elapsed = time.time() - start
            logger.info(
                "request_end",
                method=request.method,
                path=request.url.path,
                status_code=getattr(response, "status_code", 500) if "response" in dir() else 500,
                elapsed=round(elapsed, 3),
            )
            clear_contextvars()

        response.headers["X-Correlation-ID"] = correlation_id
        return response
