"""Tests for predictron_engine.dataset.signals.builder — Project E4."""

from __future__ import annotations

from predictron_engine.dataset.models import (
    DatasetRecord,
    DecisionLabel,
    PredictionSummary,
)
from predictron_engine.dataset.signals.builder import (
    CompanySignalBuilder,
    OutcomeSignalExtractor,
    SignalBuildReport,
    record_to_company_ids,
)
from predictron_engine.dataset.signals.model import SignalType
from tests.dataset.signals_helpers import (
    make_funding_event,
    make_outcome,
)


def make_record(
    startup_name: str = "Acme Corp",
    website: str = "https://acme.example.com",
    record_id: str | None = None,
) -> DatasetRecord:
    kwargs: dict[str, object] = {
        "startup_name": startup_name,
        "website": website,
        "engine_version": "0.12.1",
        "prediction": PredictionSummary(
            decision=DecisionLabel.INVEST,
            confidence=0.75,
            composite_score=68.5,
            dimension_scores={"market": 70.0, "team": 75.0},
        ),
    }
    if record_id is not None:
        kwargs["record_id"] = record_id
    return DatasetRecord(**kwargs)


SAMPLE_OUTCOME = make_outcome(
    "rec-1",
    funding_rounds=[
        make_funding_event(
            date="2024-01-15T00:00:00+00:00",
            amount=1_000_000.0,
            investors=["Acme Capital", "Beta Partners", "Acme Capital"],
            valuation=5_000_000.0,
        ),
        make_funding_event(
            date="2024-06-01T00:00:00+00:00",
            round_type="series_a",
            amount=None,
            investors=[],
        ),
    ],
    arr=[("2024-03-01T00:00:00+00:00", 100_000.0)],
    employees=[
        ("2024-01-01T00:00:00+00:00", 10.0),
        ("2024-02-01T00:00:00+00:00", 20.0),
        ("2024-03-01T00:00:00+00:00", 15.0),
    ],
    valuations=[("2024-02-01T00:00:00+00:00", 4_000_000.0)],
)


def _extract(outcome, company_id: str = "company:a"):
    extractor = OutcomeSignalExtractor()
    signals, skips = extractor.extract(outcome, company_id)
    return signals, skips


class TestFundingExtraction:
    def test_funding_round_signal(self) -> None:
        outcome = make_outcome(
            "rec-1",
            funding_rounds=[
                make_funding_event(date="2024-01-15T00:00:00+00:00", amount=1e6)
            ],
        )
        signals, _ = _extract(outcome)
        funding = [s for s in signals if s.signal_type == SignalType.FUNDING_ROUND]
        assert len(funding) == 1
        assert funding[0].metadata["round_type"] == "seed"
        assert funding[0].metadata["round_index"] == 0
        assert funding[0].metadata["amount_usd"] == 1e6

    def test_investors_sorted_and_deduped(self) -> None:
        outcome = make_outcome(
            "rec-1",
            funding_rounds=[
                make_funding_event(
                    date="2024-01-15T00:00:00+00:00",
                    investors=["Z-Cap", "Alpha", "Alpha"],
                )
            ],
        )
        signals, _ = _extract(outcome)
        investors = [
            s for s in signals if s.signal_type == SignalType.INVESTOR_ADDED
        ]
        assert len(investors) == 2
        assert [s.metadata["investor"] for s in investors] == ["Alpha", "Z-Cap"]
        for s in investors:
            assert s.metadata["round_type"] == "seed"

    def test_round_without_date_skipped(self) -> None:
        outcome = make_outcome(
            "rec-1",
            funding_rounds=[make_funding_event(date=None, amount=1e6)],
        )
        signals, skips = _extract(outcome)
        assert signals == []
        assert skips == ["company:a: funding round 0 without date"]

    def test_confidence_values(self) -> None:
        outcome = make_outcome(
            "rec-1",
            funding_rounds=[
                make_funding_event(
                    date="2024-01-15T00:00:00+00:00",
                    investors=["Alpha"],
                    valuation=5e6,
                )
            ],
        )
        signals, _ = _extract(outcome)
        by_type = {s.signal_type: s.confidence for s in signals}
        assert by_type[SignalType.FUNDING_ROUND] == 0.9
        assert by_type[SignalType.INVESTOR_ADDED] == 0.8

    def test_provenance(self) -> None:
        outcome = make_outcome(
            "rec-1",
            funding_rounds=[make_funding_event(date="2024-01-15T00:00:00+00:00")],
        )
        signals, _ = _extract(outcome)
        for s in signals:
            assert s.provenance == "derived:funding_round"

    def test_naive_funding_date_annexed_as_utc(self) -> None:
        outcome = make_outcome(
            "rec-1",
            funding_rounds=[make_funding_event(date="2024-01-15T00:00:00")],
        )
        signals, _ = _extract(outcome)
        assert signals[0].timestamp.tzinfo is not None


class TestMilestoneExtraction:
    def test_arr_milestone(self) -> None:
        outcome = make_outcome(
            "rec-1",
            arr=[("2024-03-01T00:00:00+00:00", 100_000.0)],
        )
        signals, _ = _extract(outcome)
        arr = [s for s in signals if s.signal_type == SignalType.ARR_MILESTONE]
        assert len(arr) == 1
        assert arr[0].metadata["arr_usd"] == 100_000.0

    def test_arr_without_date_skipped(self) -> None:
        outcome = make_outcome("rec-1")
        outcome.outcome.arr_milestones.append({"arr_usd": 5.0, "source": "x"})
        signals, skips = _extract(outcome)
        assert signals == []
        assert skips == ["company:a: arr_milestone without date"]

    def test_valuation_update(self) -> None:
        outcome = make_outcome(
            "rec-1",
            valuations=[("2024-02-01T00:00:00+00:00", 4_000_000.0)],
        )
        signals, _ = _extract(outcome)
        vals = [s for s in signals if s.signal_type == SignalType.VALUATION_UPDATE]
        assert len(vals) == 1
        assert vals[0].metadata["valuation_usd"] == 4_000_000.0

    def test_employee_milestones_and_deltas(self) -> None:
        outcome = make_outcome(
            "rec-1",
            employees=[
                ("2024-01-01T00:00:00+00:00", 10.0),
                ("2024-02-01T00:00:00+00:00", 20.0),
                ("2024-03-01T00:00:00+00:00", 15.0),
            ],
        )
        signals, _ = _extract(outcome)
        milestones = [
            s for s in signals if s.signal_type == SignalType.EMPLOYEE_MILESTONE
        ]
        assert len(milestones) == 3
        growth = [
            s for s in signals if s.signal_type == SignalType.HIRING_GROWTH
        ]
        layoffs = [s for s in signals if s.signal_type == SignalType.LAYOFFS]
        assert len(growth) == 1
        assert len(layoffs) == 1
        assert growth[0].metadata == {
            "from_count": 10.0,
            "to_count": 20.0,
            "delta": 10.0,
        }
        assert layoffs[0].metadata["delta"] == -5.0

    def test_employee_flat_snapshot_no_delta(self) -> None:
        outcome = make_outcome(
            "rec-1",
            employees=[
                ("2024-01-01T00:00:00+00:00", 10.0),
                ("2024-02-01T00:00:00+00:00", 10.0),
            ],
        )
        signals, _ = _extract(outcome)
        deltas = [
            s
            for s in signals
            if s.signal_type in (SignalType.HIRING_GROWTH, SignalType.LAYOFFS)
        ]
        assert deltas == []

    def test_employee_without_date_skipped(self) -> None:
        outcome = make_outcome("rec-1")
        outcome.outcome.employee_count_history.append(
            {"count": 5.0, "source": "x"}
        )
        signals, skips = _extract(outcome)
        assert signals == []
        assert skips == ["company:a: employee snapshot without date"]


class TestTerminalExtraction:
    def test_acquisition_signal(self) -> None:
        outcome = make_outcome(
            "rec-1",
            acquisition="BigCorp",
            exit_date="2024-09-01T00:00:00+00:00",
        )
        signals, _ = _extract(outcome)
        acqs = [s for s in signals if s.signal_type == SignalType.ACQUISITION]
        assert len(acqs) == 1
        assert acqs[0].metadata["acquirer"] == "BigCorp"
        assert acqs[0].confidence == 0.95

    def test_acquisition_without_date_skipped(self) -> None:
        outcome = make_outcome("rec-1", acquisition="BigCorp")
        signals, skips = _extract(outcome)
        assert signals == []
        assert skips == ["company:a: acquisition without exit_date"]

    def test_ipo_signal(self) -> None:
        outcome = make_outcome(
            "rec-1",
            exit_date="2024-09-01T00:00:00+00:00",
            exit_type="ipo",
        )
        signals, _ = _extract(outcome)
        ips = [s for s in signals if s.signal_type == SignalType.IPO]
        assert len(ips) == 1

    def test_shutdown_signal(self) -> None:
        outcome = make_outcome(
            "rec-1", shutdown=True, shutdown_date="2024-09-01T00:00:00+00:00"
        )
        signals, _ = _extract(outcome)
        shuts = [s for s in signals if s.signal_type == SignalType.SHUTDOWN]
        assert len(shuts) == 1

    def test_shutdown_without_date_skipped(self) -> None:
        outcome = make_outcome("rec-1", shutdown=True)
        signals, skips = _extract(outcome)
        assert signals == []
        assert skips == ["company:a: shutdown without shutdown_date"]

    def test_bankruptcy_signal(self) -> None:
        outcome = make_outcome(
            "rec-1", bankruptcy=True, bankruptcy_date="2024-09-01T00:00:00+00:00"
        )
        signals, _ = _extract(outcome)
        banks = [s for s in signals if s.signal_type == SignalType.BANKRUPTCY]
        assert len(banks) == 1

    def test_bankruptcy_without_date_skipped(self) -> None:
        outcome = make_outcome("rec-1", bankruptcy=True)
        signals, skips = _extract(outcome)
        assert signals == []
        assert skips == ["company:a: bankruptcy without bankruptcy_date"]

    def test_no_external_only_signals_synthesized(self) -> None:
        from predictron_engine.dataset.signals.model import EXTERNAL_ONLY_SIGNAL_TYPES

        signals, _ = _extract(SAMPLE_OUTCOME)
        seen = {s.signal_type for s in signals}
        assert not (seen & EXTERNAL_ONLY_SIGNAL_TYPES)

    def test_empty_outcome_no_signals(self) -> None:
        outcome = make_outcome("rec-1")
        signals, skips = _extract(outcome)
        assert signals == []
        assert skips == []


class TestCompanySignalBuilder:
    def test_build_groups_by_identity_map(self) -> None:
        records = [
            make_record(startup_name="A", website="https://a.com", record_id="r1"),
            make_record(startup_name="A Dupe", website="https://a.com", record_id="r2"),
        ]
        outcomes = {
            "r1": make_outcome(
                "r1",
                funding_rounds=[
                    make_funding_event(date="2024-01-01T00:00:00+00:00")
                ],
            ),
            "r2": make_outcome(
                "r2",
                funding_rounds=[
                    make_funding_event(date="2024-06-01T00:00:00+00:00")
                ],
            ),
        }
        identity_map = {"r1": "company:x", "r2": "company:x"}
        result = CompanySignalBuilder().build(
            records, outcomes, identity_map=identity_map
        )
        assert result.report.timeline_count == 1
        assert result.report.signal_count == 2
        assert result.report.companies == ["company:x"]

    def test_build_without_outcome_no_signals(self) -> None:
        records = [make_record(record_id="r1")]
        result = CompanySignalBuilder().build(records, {})
        assert result.report.timeline_count == 0

    def test_report_skips_sorted(self) -> None:
        records = [make_record(startup_name="A", website="https://a.com", record_id="r1")]
        outcomes = {
            "r1": make_outcome("r1", acquisition="BigCo"),
        }
        identity_map = {"r1": "company:z"}
        result = CompanySignalBuilder().build(records, outcomes, identity_map=identity_map)
        assert result.report.skipped == ["company:z: acquisition without exit_date"]

    def test_report_per_type(self) -> None:
        records = [make_record(startup_name="A", website="https://a.com", record_id="r1")]
        outcomes = {
            "r1": make_outcome(
                "r1",
                funding_rounds=[
                    make_funding_event(
                        date="2024-01-01T00:00:00+00:00",
                        amount=100.0,
                        investors=["Alpha"],
                    )
                ],
            ),
        }
        identity_map = {"r1": "company:z"}
        result = CompanySignalBuilder().build(records, outcomes, identity_map=identity_map)
        assert result.report.per_type["funding_round"] == 1
        assert result.report.per_type["investor_added"] == 1

    def test_result_to_dict(self) -> None:
        result = CompanySignalBuilder().build([], {})
        d = result.to_dict()
        assert "timelines" in d
        assert d["report"]["timeline_count"] == 0

    def test_build_deterministic(self) -> None:
        records = [make_record(startup_name="A", website="https://a.com", record_id="r1")]
        outcomes = {
            "r1": make_outcome(
                "r1",
                funding_rounds=[
                    make_funding_event(date="2024-01-01T00:00:00+00:00", amount=1.0)
                ],
            ),
        }
        identity_map = {"r1": "company:z"}
        r1 = CompanySignalBuilder().build(records, outcomes, identity_map=identity_map)
        r2 = CompanySignalBuilder().build(records, outcomes, identity_map=identity_map)
        s1 = sorted(s.signal_id for t in r1.timelines.values() for s in t.signals)
        s2 = sorted(s.signal_id for t in r2.timelines.values() for s in t.signals)
        assert s1 == s2


class TestSignalBuildReport:
    def test_defaults(self) -> None:
        r = SignalBuildReport()
        assert r.timeline_count == 0
        assert r.to_dict()["timeline_count"] == 0


class TestRecordToCompanyIds:
    def test_groups_by_website(self) -> None:
        records = [
            make_record(startup_name="First", website="https://dup.example.com", record_id="r1"),
            make_record(startup_name="Second", website="https://dup.example.com", record_id="r2"),
        ]
        mapping = record_to_company_ids(records)
        assert mapping["r1"] == mapping["r2"]
        assert mapping["r1"].startswith("company:")

    def test_distinct_companies_different_ids(self) -> None:
        records = [
            make_record(startup_name="A", website="https://a.example.com", record_id="r1"),
            make_record(startup_name="B", website="https://b.example.com", record_id="r2"),
        ]
        mapping = record_to_company_ids(records)
        assert mapping["r1"] != mapping["r2"]

    def test_empty_records(self) -> None:
        assert record_to_company_ids([]) == {}
