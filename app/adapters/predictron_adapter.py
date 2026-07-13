"""Predictron Engine adapter.

This module is the ONLY integration point with the external prediction engine.
It exposes a single public function `analyze` that accepts validated startup data
and returns analysis results. The internal workings of Predictron Engine are
entirely opaque to the rest of the backend.

When the real engine is ready, replace the mock implementation inside this module.
No other file should need to change.
"""

from __future__ import annotations

import logging

import httpx

from app.core.config import get_settings
from app.schemas.analysis import StartupAnalysisRequest, StartupAnalysisResponse

logger = logging.getLogger(__name__)
settings = get_settings()


class PredictronError(Exception):
    """Raised when the Predictron Engine returns an error or is unreachable."""


async def analyze(request: StartupAnalysisRequest) -> StartupAnalysisResponse:
    """Send startup data to the Predictron Engine and return the analysis.

    Currently returns mocked data. When the engine is deployed, swap this
    implementation to make an HTTP call to ``settings.PREDICTRON_ENGINE_URL``.
    """
    try:
        return await _call_engine(request)
    except Exception as exc:
        logger.warning("Predictron engine call failed, falling back to mock: %s", exc)
        return _mock_response(request)


async def _call_engine(request: StartupAnalysisRequest) -> StartupAnalysisResponse:
    """HTTP call to the real Predictron Engine (to be enabled when ready)."""
    headers = {}
    if settings.PREDICTRON_API_KEY:
        headers["X-API-Key"] = settings.PREDICTRON_API_KEY

    payload = {
        "startup_name": request.startup_name,
        "website": str(request.website),
        "description": request.description,
        "pitch_deck_url": str(request.pitch_deck_url) if request.pitch_deck_url else None,
        "founder_linkedin_urls": [str(url) for url in request.founder_linkedin_urls],
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{settings.PREDICTRON_ENGINE_URL}/analyze",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

    return StartupAnalysisResponse(
        startup_name=request.startup_name,
        venture_score=data["venture_score"],
        market_score=data["market_score"],
        founder_score=data["founder_score"],
        traction_score=data["traction_score"],
        recommendations=data.get("recommendations", []),
        confidence=data["confidence"],
    )


def _mock_response(request: StartupAnalysisRequest) -> StartupAnalysisResponse:
    """Return deterministic mocked results for development and testing."""
    return StartupAnalysisResponse(
        startup_name=request.startup_name,
        venture_score=72.5,
        market_score=68.0,
        founder_score=75.0,
        traction_score=65.0,
        recommendations=[
            "Strengthen go-to-market strategy documentation",
            "Expand founding team with technical co-founder",
            "Validate unit economics with early customers",
        ],
        confidence=0.85,
    )


async def check_engine_health() -> bool:
    """Return True if the Predictron Engine is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.PREDICTRON_ENGINE_URL}/health")
            return resp.status_code == 200
    except Exception:
        return False
