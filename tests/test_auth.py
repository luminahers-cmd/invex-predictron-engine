"""Tests for Phase 3 authentication and authorization."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.auth.jwt import create_access_token
from app.schemas.analysis import AnalysisDetailResponse, AnalysisListResponse


def _auth_header(user_id: str = "test-user") -> dict:
    token = create_access_token(subject=user_id)
    return {"Authorization": f"Bearer {token}"}


def _other_auth_header() -> dict:
    return _auth_header("other-user")


@pytest.fixture(autouse=True)
def _mock_persist():
    """Prevent _persist_async from connecting to a real database."""
    with patch("app.services.analysis._persist_async", new_callable=AsyncMock):
        yield


# ── Unauthorized access ──────────────────────────────────────────────


class TestUnauthorizedAccess:
    """GET endpoints must reject requests without a valid token."""

    @pytest.mark.anyio
    async def test_list_analyses_without_token(self, client):
        response = await client.get("/api/v1/analyze")
        assert response.status_code == 401
        assert response.json()["detail"] == "Not authenticated"

    @pytest.mark.anyio
    async def test_get_analysis_without_token(self, client):
        with patch(
            "app.api.analyze._get_analysis",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = await client.get("/api/v1/analyze/some-id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @pytest.mark.anyio
    async def test_list_analyses_with_invalid_token(self, client):
        headers = {"Authorization": "Bearer invalid-token"}
        response = await client.get("/api/v1/analyze", headers=headers)
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid token"

    @pytest.mark.anyio
    async def test_get_analysis_with_expired_token(self, client):
        token = create_access_token(
            subject="user", expires_delta=timedelta(seconds=-1)
        )
        response = await client.get(
            "/api/v1/analyze/some-id",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Token has expired"


# ── Authenticated access ─────────────────────────────────────────────


class TestAuthenticatedAccess:
    """GET endpoints must succeed with a valid token."""

    @pytest.mark.anyio
    async def test_list_analyses_with_token(self, client):
        mock_result = MagicMock(spec=AnalysisListResponse)
        mock_result.analyses = []
        mock_result.total = 0

        with patch(
            "app.api.analyze._list_analyses",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            response = await client.get(
                "/api/v1/analyze", headers=_auth_header()
            )
        assert response.status_code == 200
        body = response.json()
        assert "analyses" in body
        assert "total" in body

    @pytest.mark.anyio
    async def test_get_analysis_with_token(self, client):
        mock_detail = AnalysisDetailResponse(
            id="my-id",
            startup_name="MyCo",
            website="https://myco.example.com",
            description="My startup.",
            venture_score=80.0,
            market_score=75.0,
            founder_score=70.0,
            traction_score=85.0,
            confidence=0.9,
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
        )

        with patch(
            "app.api.analyze._get_analysis",
            new_callable=AsyncMock,
            return_value=mock_detail,
        ):
            response = await client.get(
                "/api/v1/analyze/my-id", headers=_auth_header()
            )
        assert response.status_code == 200
        assert response.json()["id"] == "my-id"


# ── Ownership isolation ──────────────────────────────────────────────


class TestOwnershipIsolation:
    """Users must not be able to access each other's analyses."""

    @pytest.mark.anyio
    async def test_list_analyses_only_returns_own(self, client):
        async def _list_for_user(session, user_id, offset=0, limit=20):
            assert user_id == "test-user"
            return AnalysisListResponse(analyses=[], total=0)

        with patch(
            "app.api.analyze._list_analyses",
            new_callable=AsyncMock,
            side_effect=_list_for_user,
        ):
            response = await client.get(
                "/api/v1/analyze", headers=_auth_header("test-user")
            )
        assert response.status_code == 200

    @pytest.mark.anyio
    async def test_get_analysis_ownership_gate(self, client):
        async def _get_for_user(session, analysis_id, user_id=None):
            if user_id != "test-user":
                return None
            if analysis_id != "my-analysis":
                return None
            return AnalysisDetailResponse(
                id="my-analysis",
                startup_name="MyCo",
                website="https://myco.example.com",
                description="Mine.",
                venture_score=80.0,
                market_score=75.0,
                founder_score=70.0,
                traction_score=85.0,
                confidence=0.9,
                created_at=datetime(2025, 1, 1, tzinfo=UTC),
            )

        with patch(
            "app.api.analyze._get_analysis",
            new_callable=AsyncMock,
            side_effect=_get_for_user,
        ):
            response = await client.get(
                "/api/v1/analyze/my-analysis", headers=_auth_header("other-user")
            )
        assert response.status_code == 404


# ── POST backward compatibility ──────────────────────────────────────


class TestPostBackwardCompatibility:
    """POST /analyze must remain accessible without authentication."""

    @pytest.mark.anyio
    async def test_post_without_token(self, client):
        payload = {
            "startup_name": "NoAuthCo",
            "website": "https://noauth.example.com",
            "description": "Testing that POST works without authentication.",
        }
        response = await client.post("/api/v1/analyze", json=payload)
        assert response.status_code == 200
        assert response.json()["startup_name"] == "NoAuthCo"

    @pytest.mark.anyio
    async def test_post_with_valid_token(self, client):
        payload = {
            "startup_name": "AuthCo",
            "website": "https://auth.example.com",
            "description": "Testing that POST works with authentication.",
        }
        response = await client.post(
            "/api/v1/analyze", json=payload, headers=_auth_header()
        )
        assert response.status_code == 200
        assert response.json()["startup_name"] == "AuthCo"

    @pytest.mark.anyio
    async def test_post_passes_user_id_to_service(self, client):
        from app.schemas.analysis import StartupAnalysisResponse

        payload = {
            "startup_name": "UserCo",
            "website": "https://user.example.com",
            "description": "Testing user_id threading through the service.",
        }

        mock_response = StartupAnalysisResponse(
            startup_name="UserCo",
            venture_score=50.0,
            market_score=50.0,
            founder_score=50.0,
            traction_score=50.0,
            confidence=0.5,
        )

        with patch(
            "app.api.analyze.run_analysis", new_callable=AsyncMock, return_value=mock_response
        ) as mock_run:
            await client.post(
                "/api/v1/analyze", json=payload, headers=_auth_header("alice")
            )

            mock_run.assert_called_once()
            _call_user_id = mock_run.call_args.kwargs.get("user_id")
            assert _call_user_id == "alice"
