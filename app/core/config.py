from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "InveX AI Backend"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    DATABASE_URL: str = "postgresql+asyncpg://postgres@localhost:5432/invex"
    DATABASE_ECHO: bool = False

    SECRET_KEY: str = "CHANGE_ME_IN_PRODUCTION"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ALGORITHM: str = "HS256"

    CORS_ORIGINS: list[str] = ["http://localhost:3000", "https://*.vercel.app"]

    PREDICTRON_ENGINE_URL: str = "http://localhost:8001"
    PREDICTRON_API_KEY: str = ""

    API_V1_PREFIX: str = "/api/v1"

    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    RATE_LIMIT_TRUSTED_PROXIES: list[str] = []
    RATE_LIMIT_MAX_TRACKED_CLIENTS: int = 100_000

    REQUEST_ID_HEADER: str = "X-Request-ID"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }

    @property
    def is_production(self) -> bool:
        """Return True when running in the production environment."""
        return self.ENVIRONMENT.strip().lower() == "production"

    def validate_required(self) -> None:
        errors: list[str] = []
        warnings: list[str] = []
        if self.is_production and self.SECRET_KEY in ("", "CHANGE_ME_IN_PRODUCTION"):
            errors.append(
                "SECRET_KEY is insecure for production — set a strong, "
                "non-default value when ENVIRONMENT=production"
            )
        elif self.SECRET_KEY == "CHANGE_ME_IN_PRODUCTION":
            warnings.append("SECRET_KEY is still set to the default value — change for production")
        if not self.DATABASE_URL:
            errors.append("DATABASE_URL is not configured")
        if self.ENVIRONMENT.strip().lower() not in ("development", "production"):
            errors.append(
                f"ENVIRONMENT must be 'development' or 'production', got {self.ENVIRONMENT!r}"
            )
        if errors:
            raise RuntimeError(
                "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )
        for w in warnings:
            import logging
            logging.getLogger(__name__).warning("Configuration warning: %s", w)


@lru_cache
def get_settings() -> Settings:
    return Settings()
