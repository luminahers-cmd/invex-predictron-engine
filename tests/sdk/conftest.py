"""Shared fixtures for the Predictron SDK test suite."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from predictron_sdk import PredictronClient, RetryPolicy

NO_RETRY = RetryPolicy(max_retries=0)


@pytest.fixture
def no_retries() -> RetryPolicy:
    """A retry policy that never retries (fast, deterministic tests)."""
    return NO_RETRY


@pytest.fixture
def make_client():
    """Build a client wired to an in-memory ``httpx.MockTransport``."""

    def _make(handler, **kwargs: Any) -> PredictronClient:
        transport = httpx.MockTransport(handler)
        params: dict[str, Any] = {
            "base_url": "https://api.predictron.test",
            "api_key": "test-key",
            "transport": transport,
            "retry_policy": NO_RETRY,
        }
        params.update(kwargs)
        return PredictronClient(**params)

    return _make


@pytest.fixture
def venture_payload() -> dict[str, Any]:
    return {
        "id": "an-1",
        "startup_name": "Acme AI",
        "overall_score": 72.4,
        "overall_confidence": 0.81,
        "dimension_scores": [
            {
                "dimension": "market_opportunity",
                "score": 78.0,
                "rationale": "Large TAM",
            },
            {
                "dimension": "founder_quality",
                "score": 64.0,
                "rationale": "Strong team",
            },
        ],
        "decision_category": "invest",
        "conviction_level": "high",
        "recommendation_count": 3,
        "key_recommendations": ["Hire a CTO", "Expand to EU"],
        "processing_time_ms": 812.5,
        "engine_version": "0.12.1",
        "created_at": "2026-01-01T00:00:00Z",
    }


@pytest.fixture
def portfolio_payload() -> dict[str, Any]:
    return {
        "company_count": 2,
        "companies": [
            {
                "startup_name": "Alpha",
                "overall_score": 70.0,
                "overall_confidence": 0.8,
                "decision_category": "invest",
                "conviction_level": "medium",
            },
            {
                "startup_name": "Beta",
                "overall_score": 55.0,
                "overall_confidence": 0.6,
                "decision_category": "watch",
                "conviction_level": "low",
            },
        ],
        "portfolio_score": 62.5,
        "sector_distribution": [
            {"sector": "ai", "count": 2, "percentage": 100.0, "avg_score": 62.5}
        ],
        "stage_distribution": [{"stage": "seed", "count": 2, "percentage": 100.0}],
        "risk_summary": {
            "high_risk_count": 1,
            "medium_risk_count": 1,
            "low_risk_count": 0,
            "average_confidence": 0.7,
            "risk_factors": ["Concentration"],
        },
        "diversification": {
            "sector_diversity": 0.0,
            "stage_diversity": 0.0,
            "geography_diversity": 0.5,
            "overall_diversification": 0.17,
            "recommendation_count": 2,
        },
        "concentration_risk": {
            "sector_concentration": 1.0,
            "stage_concentration": 0.8,
            "score_variance": 112.5,
            "most_concentrated_sector": "ai",
            "recommendations": ["Diversify sector"],
        },
        "heatmap_data": [
            {"row_label": "Alpha", "col_label": "Beta", "value": 0.6}
        ],
        "similarity_matrix": [],
        "overall_confidence": 0.7,
        "processing_time_ms": 1200.0,
    }


@pytest.fixture
def comparison_payload() -> dict[str, Any]:
    return {
        "companies": ["Alpha", "Beta"],
        "feature_diffs": {
            "companies": ["Alpha", "Beta"],
            "feature_diffs": [
                {
                    "feature": "revenue",
                    "company_a": 100.0,
                    "company_b": 50.0,
                    "difference": 50.0,
                    "direction": "alpha_greater",
                }
            ],
        },
        "decision_diffs": {
            "companies": ["Alpha", "Beta"],
            "decision_comparison": [{"startup": "Alpha", "decision": "invest"}],
        },
        "contribution_diffs": {
            "companies": ["Alpha", "Beta"],
            "diffs": [
                {
                    "feature": "market",
                    "contributions": {"Alpha": 10.0, "Beta": 5.0},
                    "max_contribution": 10.0,
                    "min_contribution": 5.0,
                }
            ],
        },
        "signal_diffs": {
            "companies": ["Alpha", "Beta"],
            "signal_comparison": [],
        },
        "knowledge_graph_diffs": {
            "companies": ["Alpha", "Beta"],
            "graph_summary": {},
        },
        "benchmark_comparison": {
            "companies": ["Alpha", "Beta"],
            "benchmark_metrics": [],
        },
        "overall_summary": {"avg_score": 62.5, "score_range": 15.0},
    }


@pytest.fixture
def due_diligence_payload() -> dict[str, Any]:
    return {
        "report_id": "dd-1",
        "startup_name": "Acme AI",
        "executive_summary": {
            "headline": "Strong candidate",
            "overview": "Solid team, growing market.",
            "key_findings": ["Revenue growth 40% YoY"],
            "confidence_level": 0.8,
        },
        "strengths": [
            {
                "title": "Team",
                "description": "Experienced",
                "dimension": "founder",
                "severity": "high",
                "evidence": ["ref"],
            }
        ],
        "weaknesses": [
            {
                "title": "Cash burn",
                "description": "High",
                "dimension": "finance",
                "severity": "medium",
                "evidence": [],
            }
        ],
        "opportunities": [
            {
                "title": "EU expansion",
                "description": "Untapped",
                "dimension": "market",
                "impact": "high",
                "evidence": [],
            }
        ],
        "risks": [
            {
                "title": "Competition",
                "description": "Crowded",
                "dimension": "market",
                "severity": "medium",
                "mitigation": "Focus niche",
                "evidence": [],
            }
        ],
        "evidence": [
            {
                "claim": "Revenue",
                "domain": "finance",
                "source": "census",
                "trust_score": 0.9,
                "relevance_score": 1.0,
            }
        ],
        "decision_trace": {
            "category": "invest",
            "conviction": "high",
            "composite_score": 76.0,
            "margin_to_next_category": 4.0,
            "rationale_for": ["Traction"],
            "rationale_against": [],
            "missing_information": [],
        },
        "confidence": 0.78,
        "supporting_features": {"revenue_growth": 0.4},
        "benchmark_context": {
            "composite_score": 76.0,
            "benchmark_mean": 70.0,
            "benchmark_std_dev": 8.0,
            "percentile_rank": 0.77,
            "sample_size": 120,
            "category_distribution": {},
        },
        "processing_time_ms": 900.0,
        "engine_version": "0.12.1",
    }


@pytest.fixture
def search_payload() -> dict[str, Any]:
    return {
        "query": "robotics",
        "search_type": "all",
        "total": 1,
        "offset": 0,
        "limit": 20,
        "results": [
            {
                "result_type": "company",
                "id": "c-1",
                "name": "Robotics Co",
                "description": "Industrial robots",
                "score": 0.9,
                "metadata": {"industry": "robotics"},
            }
        ],
    }


@pytest.fixture
def search_by_type_payload() -> dict[str, Any]:
    return {
        "search_type": "company",
        "total": 1,
        "offset": 0,
        "limit": 20,
        "companies": [
            {
                "record_id": "c-1",
                "startup_name": "Robotics Co",
                "website": "https://roboticsco.example.com",
                "industries": ["robotics"],
                "country_code": "US",
                "headquarters": "San Francisco",
                "founded_year": 2019,
                "decision": "invest",
                "composite_score": 80.0,
                "confidence": 0.85,
            }
        ],
        "signals": [],
        "graph_nodes": [],
    }


@pytest.fixture
def batch_summary_payload() -> dict[str, Any]:
    return {
        "job_id": "job-1",
        "job_name": "demo",
        "status": "pending",
        "total_items": 2,
        "completed_items": 0,
        "failed_items": 0,
        "created_at": "2026-01-01T00:00:00Z",
    }


@pytest.fixture
def analysis_payload() -> dict[str, Any]:
    return {
        "id": "persisted-1",
        "startup_name": "Acme AI",
        "venture_score": 71.0,
        "market_score": 68.0,
        "founder_score": 77.0,
        "traction_score": 60.0,
        "recommendations": ["Build out sales"],
        "confidence": 0.75,
    }


@pytest.fixture
def health_payload() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": "0.12.1",
        "engine_reachable": True,
        "db_healthy": True,
        "startup_state": "ready",
    }


@pytest.fixture
def readiness_payload() -> dict[str, Any]:
    return {
        "status": "ready",
        "db_healthy": True,
        "engine_ready": True,
        "startup_complete": True,
    }


def json_response(status: int, payload: Any = None, **headers: str) -> httpx.Response:
    """Build an ``httpx.Response`` carrying a JSON body."""
    return httpx.Response(status, json=payload, headers=headers)


def record_calls(handler_calls, handler=None):
    """Wrap a handler to also record every request for assertions.

    ``handler`` defaults to a trivial 200 JSON response when omitted.
    """

    def _wrapped(request: httpx.Request) -> httpx.Response:
        handler_calls.append(request)
        if handler is None:
            return json_response(200, {})
        return handler(request)

    return _wrapped
