"""Feature store test fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest

from predictron_engine.feature_store.engine import FeatureEngine
from predictron_engine.feature_store.features import ALL_FEATURES
from predictron_engine.feature_store.registry import FeatureRegistry
from predictron_engine.feature_store.reports import FeatureReportBuilder
from predictron_engine.feature_store.store import FeatureStore
from predictron_engine.feature_store.validation import FeatureValidator


@dataclass
class FakeProfile:
    domain: str | None = None
    industries: list[str] = field(default_factory=list)
    headquarters: str | None = None
    country_code: str | None = None
    city: str | None = None
    region: str | None = None
    founded_year: int | None = None
    founded_date: datetime | None = None
    employee_count: int | None = None
    employee_range: str | None = None
    description: str | None = None
    legal_name: str | None = None
    status: str | None = None


@dataclass
class FakeOutcome:
    outcome_id: str = "outcome-1"
    record_id: str = "rec-1"
    outcome_events: list[Any] = field(default_factory=list)


@dataclass
class FakeRecord:
    record_id: str = "rec-1"
    startup_name: str = "TestCo"
    website: str = "https://testco.com"
    engine_version: str = "0.12.1"
    profile: FakeProfile = field(default_factory=FakeProfile)
    funding_stage_at_analysis: Any = None
    raw_data: dict[str, Any] = field(default_factory=dict)
    founder_linkedin_urls: list[str] | None = None


@dataclass
class FakeSignal:
    signal_id: str = ""
    company_id: str = "rec-1"
    signal_type: Any = None
    timestamp: datetime = field(
        default_factory=lambda: datetime(2024, 1, 1, tzinfo=UTC)
    )
    source: str = "test"
    confidence: float = 0.9
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence: Any = None
    provenance: str = ""


@dataclass
class FakeTimeline:
    company_id: str = "rec-1"
    signals: tuple[FakeSignal, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "signals",
            tuple(sorted(self.signals, key=lambda s: s.timestamp)),
        )

    @property
    def signal_count(self) -> int:
        return len(self.signals)

    @property
    def first(self) -> FakeSignal | None:
        return self.signals[0] if self.signals else None

    @property
    def last(self) -> FakeSignal | None:
        return self.signals[-1] if self.signals else None


@dataclass
class FakeGraph:
    _nodes: dict[str, Any] = field(default_factory=dict)
    _edges: list[Any] = field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0

    def degree(self, node_id: str) -> int:
        return self._nodes.get(node_id, 0)


@dataclass
class FakeGraphQueries:
    _components: list[set[str]] = field(default_factory=list)
    _neighbors: dict[str, list[str]] = field(default_factory=dict)

    def connected_components(self) -> list[set[str]]:
        return self._components

    def neighbors(self, node_id: str) -> list[str]:
        return self._neighbors.get(node_id, [])


class FakeFundingStage:
    def __init__(self, value: str) -> None:
        self.value = value


def make_signal(
    signal_type: Any = None,
    days_offset: int = 0,
    amount: float | None = None,
    confidence: float = 0.9,
    investors: list[Any] | None = None,
    company_id: str = "rec-1",
) -> FakeSignal:
    from predictron_engine.dataset.signals.model import SignalType
    st = signal_type or SignalType.FUNDING_ROUND
    meta: dict[str, Any] = {}
    if amount is not None:
        meta["amount_usd"] = amount
    if investors is not None:
        meta["investors"] = investors
    ts = datetime(2023, 1, 1, tzinfo=UTC)
    if days_offset:
        from datetime import timedelta
        ts = ts + timedelta(days=days_offset)
    return FakeSignal(
        signal_id=f"sig-{days_offset}",
        company_id=company_id,
        signal_type=st,
        timestamp=ts,
        confidence=confidence,
        metadata=meta,
    )


def make_timeline(
    company_id: str = "rec-1",
    signals: list[FakeSignal] | None = None,
) -> FakeTimeline:
    return FakeTimeline(
        company_id=company_id,
        signals=tuple(signals or []),
    )


@pytest.fixture
def fake_record() -> FakeRecord:
    return FakeRecord(
        profile=FakeProfile(
            founded_year=2020,
            industries=["SaaS"],
            country_code="US",
            employee_count=50,
        ),
        funding_stage_at_analysis=FakeFundingStage("seed"),
        raw_data={"founders": [{"name": "Alice"}, {"name": "Bob"}]},
    )


@pytest.fixture
def empty_record() -> FakeRecord:
    return FakeRecord()


@pytest.fixture
def full_timeline() -> FakeTimeline:
    from predictron_engine.dataset.signals.model import SignalType
    signals = [
        make_signal(SignalType.FUNDING_ROUND, days_offset=0, amount=1_000_000),
        make_signal(SignalType.FUNDING_ROUND, days_offset=180, amount=5_000_000),
        make_signal(SignalType.EMPLOYEE_MILESTONE, days_offset=90),
        make_signal(SignalType.ARR_MILESTONE, days_offset=270),
        make_signal(SignalType.REVENUE_MILESTONE, days_offset=360),
    ]
    return make_timeline(signals=signals)


@pytest.fixture
def registry() -> FeatureRegistry:
    reg = FeatureRegistry()
    reg.register_all(ALL_FEATURES)
    return reg


@pytest.fixture
def engine(registry: FeatureRegistry) -> FeatureEngine:
    return FeatureEngine(registry)


@pytest.fixture
def feature_store(tmp_path: Any) -> FeatureStore:
    store = FeatureStore(tmp_path / "features")
    store.initialize()
    return store


@pytest.fixture
def validator(registry: FeatureRegistry) -> FeatureValidator:
    return FeatureValidator(registry)


@pytest.fixture
def report_builder(registry: FeatureRegistry) -> FeatureReportBuilder:
    return FeatureReportBuilder(registry)


@pytest.fixture
def full_store(tmp_path: Any) -> FeatureStore:
    store = FeatureStore(tmp_path / "full_features")
    store.initialize()
    return store


@pytest.fixture
def fake_outcome() -> FakeOutcome:
    return FakeOutcome()
