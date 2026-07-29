from __future__ import annotations

import time
import uuid
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.config import get_settings

settings = get_settings()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Generate or propagate a unique request ID for every HTTP request.

    Reads an existing X-Request-ID header from the incoming request (e.g. from
    a load balancer) or generates a new UUID.  The ID is stored on
    ``request.state.request_id`` and returned in the response header along
    with an X-Response-Time header.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        header_name = settings.REQUEST_ID_HEADER
        request_id = request.headers.get(header_name, str(uuid.uuid4()))
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        response.headers[header_name] = request_id
        response.headers["X-Response-Time"] = f"{elapsed_ms}ms"
        return response
