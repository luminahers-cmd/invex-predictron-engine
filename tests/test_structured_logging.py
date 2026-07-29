"""Tests for structured logging (Phase 4)."""

import json
import logging

from app.middleware.logging import JsonFormatter


class TestJsonFormatter:
    """The JSON formatter must produce valid JSON with expected fields."""

    def test_format_produces_valid_json(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "Test message"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert "timestamp" in parsed

    def test_format_includes_extra_fields(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="Request completed",
            args=(),
            exc_info=None,
        )
        record.request_id = "abc-123"
        record.method = "GET"
        record.path = "/api/v1/health"
        record.status_code = 200
        record.processing_time_ms = 12.34
        record.user = "test-user"

        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["request_id"] == "abc-123"
        assert parsed["method"] == "GET"
        assert parsed["path"] == "/api/v1/health"
        assert parsed["status_code"] == 200
        assert parsed["processing_time_ms"] == 12.34
        assert parsed["user"] == "test-user"

    def test_format_omits_missing_extra_fields(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg="Simple message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "Simple message"
        assert "request_id" not in parsed
        assert "method" not in parsed

    def test_format_includes_exception_info(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=(ValueError, ValueError("test error"), None),
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "exception" in parsed
        assert "test error" in parsed["exception"]
