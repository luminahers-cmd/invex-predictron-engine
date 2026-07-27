import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.core.config import get_settings
from app.middleware.logging import LoggingMiddleware
from app.middleware.request_id import RequestIDMiddleware

settings = get_settings()

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: startup and shutdown hooks."""
    from predictron_engine.engine import PredictronEngine

    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)

    engine = PredictronEngine()
    app.state.predictron_engine = engine
    logger.info("PredictronEngine singleton initialized")

    yield

    logger.info("Shutting down %s", settings.APP_NAME)
    app.state.predictron_engine = None


def _build_exception_handlers() -> dict[int, Any]:
    """Return a mapping of HTTP status codes to exception handler callables."""

    async def _unhandled_exception_handler(
        request: Request, exc: Exception,
    ) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "code": "INTERNAL_ERROR"},
        )

    async def _request_validation_error_handler(
        request: Request, exc: Exception,
    ) -> JSONResponse:
        logger.warning("Validation error on %s %s: %s", request.method, request.url.path, exc)
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
    )

    # --- Middleware (executed in reverse order of addition) ---
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Global exception handlers ---
    for status_code, handler in _build_exception_handlers().items():
        app.add_exception_handler(status_code, handler)  # type: ignore[arg-type]

    # --- Routes ---
    app.include_router(api_router)

    return app


app = create_app()
