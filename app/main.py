import logging
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.core.config import get_settings
from app.middleware.logging import JsonFormatter, LoggingMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.request_id import RequestIDMiddleware

settings = get_settings()

_log_handler = logging.StreamHandler(sys.stdout)
_log_handler.setFormatter(JsonFormatter())
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    handlers=[_log_handler],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: startup and shutdown hooks."""
    from predictron_engine.engine import PredictronEngine

    app.state.startup_state = "starting"
    logger.info("Application startup state set to 'starting'")

    logger.info(
        "Starting %s v%s",
        settings.APP_NAME,
        settings.APP_VERSION,
    )

    try:
        settings.validate_required()
    except RuntimeError as e:
        logger.warning("Configuration issue: %s", e)

    engine = PredictronEngine()
    app.state.predictron_engine = engine
    logger.info("PredictronEngine singleton initialized")

    app.state.startup_state = "ready"
    logger.info("Application startup state set to 'ready'")

    yield

    app.state.startup_state = "stopping"
    logger.info("Shutting down %s", settings.APP_NAME)
    app.state.predictron_engine = None


def _build_exception_handlers() -> dict[int, Any]:
    """Return a mapping of HTTP status codes to exception handler callables."""

    async def _unhandled_exception_handler(
        request: Request, exc: Exception,
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.exception(
            "Unhandled exception",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
            },
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "code": "INTERNAL_ERROR"},
        )

    async def _request_validation_error_handler(
        request: Request, exc: Exception,
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.warning(
            "Validation error",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
            },
        )
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc), "code": "VALIDATION_ERROR"},
        )

    return {
        500: _unhandled_exception_handler,
        422: _request_validation_error_handler,
    }


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        swagger_ui_parameters={"persistAuthorization": True},
    )

    # --- Middleware (executed in reverse order of addition) ---
    # Innermost: CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # RateLimitMiddleware
    app.add_middleware(RateLimitMiddleware)
    # RequestIDMiddleware – sets request_id before LoggingMiddleware reads it
    app.add_middleware(RequestIDMiddleware)
    # Outermost: LoggingMiddleware
    app.add_middleware(LoggingMiddleware)

    # --- Global exception handlers ---
    for status_code, handler in _build_exception_handlers().items():
        app.add_exception_handler(status_code, handler)  # type: ignore[arg-type]

    # --- Routes ---
    app.include_router(api_router)

    return app


app = create_app()
