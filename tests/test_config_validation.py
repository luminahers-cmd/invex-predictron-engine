"""Tests for configuration validation (Phase 4)."""

import pytest

from app.core.config import Settings


class TestConfigValidation:
    """Configuration validation must fail fast on missing or insecure values."""

    def test_default_secret_key_does_not_raise(self):
        """The default SECRET_KEY must not raise — it's a warning, not a hard error."""
        settings = Settings(
            SECRET_KEY="CHANGE_ME_IN_PRODUCTION",
            DATABASE_URL="postgresql+asyncpg://user:pass@localhost/test",
        )
        settings.validate_required()

    def test_empty_database_url_raises_error(self):
        """An empty DATABASE_URL must trigger a validation error."""
        settings = Settings(
            SECRET_KEY="a-valid-secret-key",
            DATABASE_URL="",
        )
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            settings.validate_required()

    def test_valid_config_passes_validation(self):
        """A properly configured settings instance must pass validation."""
        settings = Settings(
            SECRET_KEY="super-secret-key-12345",
            DATABASE_URL="postgresql+asyncpg://user:pass@localhost/proddb",
        )
        settings.validate_required()

    def test_multiple_errors_reported_together(self):
        """All critical validation errors must be reported together."""
        settings = Settings(
            SECRET_KEY="CHANGE_ME_IN_PRODUCTION",
            DATABASE_URL="",
        )
        with pytest.raises(RuntimeError) as exc:
            settings.validate_required()
        msg = str(exc.value)
        assert "DATABASE_URL" in msg


class TestProductionModeEnforcement:
    """In production mode insecure defaults must fail startup validation."""

    def test_insecure_secret_key_fails_in_production(self):
        settings = Settings(
            ENVIRONMENT="production",
            SECRET_KEY="CHANGE_ME_IN_PRODUCTION",
            DATABASE_URL="postgresql+asyncpg://user:pass@localhost/proddb",
        )
        with pytest.raises(RuntimeError, match="SECRET_KEY"):
            settings.validate_required()

    def test_empty_secret_key_fails_in_production(self):
        settings = Settings(
            ENVIRONMENT="production",
            SECRET_KEY="",
            DATABASE_URL="postgresql+asyncpg://user:pass@localhost/proddb",
        )
        with pytest.raises(RuntimeError, match="SECRET_KEY"):
            settings.validate_required()

    def test_strong_secret_key_passes_in_production(self):
        settings = Settings(
            ENVIRONMENT="production",
            SECRET_KEY="a-long-random-production-secret",
            DATABASE_URL="postgresql+asyncpg://user:pass@localhost/proddb",
        )
        settings.validate_required()

    def test_default_secret_key_only_warns_in_development(self):
        settings = Settings(
            ENVIRONMENT="development",
            SECRET_KEY="CHANGE_ME_IN_PRODUCTION",
            DATABASE_URL="postgresql+asyncpg://user:pass@localhost/devdb",
        )
        settings.validate_required()

    def test_invalid_environment_is_rejected(self):
        settings = Settings(
            ENVIRONMENT="staging",
            SECRET_KEY="some-secret",
            DATABASE_URL="postgresql+asyncpg://user:pass@localhost/devdb",
        )
        with pytest.raises(RuntimeError, match="ENVIRONMENT"):
            settings.validate_required()
