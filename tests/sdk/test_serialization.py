"""Tests for serialization: typed response parsing and payload building."""

from __future__ import annotations

import pytest

from predictron_sdk.models import (
    AnalysisList,
    BatchJobSummary,
    FullComparison,
    Health,
    PortfolioAnalysis,
    SearchResponse,
    SignalTimeline,
    VentureAnalysis,
    to_payload,
)
from predictron_sdk.transport import parse_model


@pytest.mark.parametrize(
    "payload,cls,expected",
    [
        (
            {"status": "ok", "version": "1", "engine_reachable": True},
            Health,
            lambda m: m.version == "1",
        ),
        (
            {"startup_name": "A", "overall_score": 50.0, "overall_confidence": 0.5},
            VentureAnalysis,
            lambda m: m.decision_category is None,
        ),
        (
            {"companies": ["A"]},
            FullComparison,
            lambda m: m.overall_summary == {},
        ),
        (
            {"analyses": [], "total": 3},
            AnalysisList,
            lambda m: m.total == 3,
        ),
        (
            {"jobs": [], "total": 0},
            AnalysisList,
            lambda _: True,
        ),
    ],
)
def test_parse_model_typed(payload, cls, expected) -> None:
    model = parse_model(cls, payload)
    assert isinstance(model, cls)
    assert expected(model)


def test_portfolio_analysis_deserializes_full_payload(portfolio_payload) -> None:
    model = parse_model(PortfolioAnalysis, portfolio_payload)
    assert model.company_count == 2
    assert model.companies[0].startup_name == "Alpha"
    assert model.sector_distribution[0].percentage == 100.0
    assert model.concentration_risk.sector_concentration == 1.0
    assert model.heatmap_data[0].value == 0.6


def test_search_response_deserializes(search_payload) -> None:
    model = parse_model(SearchResponse, search_payload)
    assert model.query == "robotics"
    assert model.results[0].result_type == "company"


def test_signal_timeline_deserializes() -> None:
    payload = {
        "company_id": "c1",
        "signal_count": 2,
        "signals": [
            {"signal_id": "s1", "company_id": "c1", "signal_type": "funding"},
            {"signal_id": "s2", "company_id": "c1", "signal_type": "hiring"},
        ],
        "type_counts": {"funding": 1, "hiring": 1},
        "span_days": 30.0,
        "first_signal": "2026-01-01",
        "last_signal": "2026-02-01",
    }
    model = parse_model(SignalTimeline, payload)
    assert model.signal_count == 2
    assert model.type_counts["funding"] == 1


def test_batch_summary_deserializes_enum(batch_summary_payload) -> None:
    model = parse_model(BatchJobSummary, batch_summary_payload)
    assert model.status.value == "pending"


def test_to_payload_excludes_none() -> None:
    payload = to_payload(Health(status="ok", version="1"))
    assert payload["status"] == "ok"
    assert "startup_state" not in payload or payload["startup_state"] == "unknown"


def test_to_payload_nested_models() -> None:
    from predictron_sdk.models import PortfolioRequest

    request = PortfolioRequest(
        company_names=["A", "B"],
        descriptions={"A": "thing"},
    )
    payload = to_payload(request)
    assert payload["company_names"] == ["A", "B"]
    assert payload["descriptions"] == {"A": "thing"}
    assert payload["website_urls"] == {}


def test_datetime_values_serialize_to_iso() -> None:
    payload = to_payload(
        VentureAnalysis(
            startup_name="A",
            overall_score=1.0,
            overall_confidence=0.1,
            created_at="2026-01-01T00:00:00Z",
        )
    )
    assert payload["created_at"] == "2026-01-01T00:00:00Z"


def test_round_trip_venture() -> None:
    request_payload = {
        "startup_name": "Acme",
        "description": "Description text.",
        "overall_score": 1.0,
        "overall_confidence": 0.5,
        "dimension_scores": [],
    }
    # dropped: overall_score etc. but model_validate should coerce the dict as-is
    from predictron_sdk.models import VentureAnalysis

    model = VentureAnalysis(**request_payload)
    again = VentureAnalysis.model_validate(to_payload(model))
    assert again.model_dump() == model.model_dump()


def test_serialization_failure_typed() -> None:
    from predictron_sdk.errors import SerializationError

    with pytest.raises(SerializationError):
        parse_model(Health, {"status": "ok", "version": 123})


def test_empty_object_parses_to_defaults() -> None:
    model = parse_model(SearchResponse, {})
    assert model.limit == 20


def test_nested_default_factories_are_instances() -> None:
    from predictron_sdk.models import FeatureDiffs

    model = parse_model(FullComparison, {"companies": []})
    assert type(model.feature_diffs) is FeatureDiffs
    assert type(model.decision_diffs).__name__ == "DecisionDiffs"
    assert type(model.benchmark_comparison).__name__ == "BenchmarkComparisonDetail"


def test_model_eq_dump_roundtrip() -> None:
    payload = {"startup_name": "A", "overall_score": 10.0, "overall_confidence": 0.2}
    one = VentureAnalysis.model_validate(payload)
    two = VentureAnalysis.model_validate(one.model_dump(mode="json"))
    assert one == two
