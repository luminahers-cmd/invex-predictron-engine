from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


def _extract_user_from_request(request: Request) -> str | None:
    """Safely extract authenticated user identifier from request without logging secrets."""
    from app.core.config import get_settings

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    token = auth_header.removeprefix("Bearer ")
    if not token:
        return None
    try:
        import jwt as pyjwt

        payload = pyjwt.decode(
            token,
            get_settings().SECRET_KEY,
            algorithms=[get_settings().ALGORITHM],
            options={"verify_exp": False},
        )
        return str(payload.get("sub"))
    except Exception:
        return None


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log structured request data for every HTTP request."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        request_id = getattr(request.state, "request_id", None)
        user = _extract_user_from_request(request)

        logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "processing_time_ms": elapsed_ms,
                "user": user or "anonymous",
            },
        )
        return response


class JsonFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        import json

        timestamp = self.formatTime(record, "%Y-%m-%dT%H:%M:%S")
        log_entry: dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("request_id", "method", "path", "status_code", "processing_time_ms", "user"):
            value = getattr(record, key, None)
            if value is not None:
                log_entry[key] = value
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)
