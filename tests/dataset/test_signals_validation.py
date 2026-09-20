"""Tests for predictron_engine.dataset.signals.validation — Project E4."""

from __future__ import annotations

from predictron_engine.dataset.signals.validation import (
    SignalValidationIssue,
    SignalValidationReport,
    validate_signal,
    validate_signals,
    validate_timeline,
)
from tests.dataset.signals_helpers import make_signal, make_timeline


class TestValidateSignal:
    def test_valid_signal_no_issues(self) -> None:
        assert validate_signal(make_signal()) == []

    def test_empty_company_id(self) -> None:
        s = make_signal(company_id="")
        kinds = {i.kind for i in validate_signal(s)}
        assert "empty_company_id" in kinds

    def test_missing_source(self) -> None:
        s = make_signal(source="")
        kinds = {i.kind for i in validate_signal(s)}
        assert "missing_source" in kinds

    def test_missing_evidence(self) -> None:
        from predictron_engine.dataset.signals.model import EvidenceReference
        s = make_signal()
        s2 = type(s)(
            company_id=s.company_id,
            signal_type=s.signal_type,
            timestamp=s.timestamp,
            source=s.source,
            provenance=s.provenance,
            confidence=s.confidence,
            evidence=EvidenceReference(),
        )
        kinds = {i.kind for i in validate_signal(s2)}
        assert "missing_evidence" in kinds

    def test_issue_fields(self) -> None:
        s = make_signal(company_id="")
        (issue,) = validate_signal(s)
        assert issue.signal_id == s.signal_id
        assert issue.kind == "empty_company_id"
        assert isinstance(issue.detail, str)

    def test_to_dict(self) -> None:
        issue = SignalValidationIssue("abc", "kind", "detail")
        d = issue.to_dict()
        assert d["signal_id"] == "abc"
        assert d["kind"] == "kind"
        assert d["detail"] == "detail"


class TestValidateSignals:
    def test_valid_collection(self) -> None:
        report = validate_signals("c1", [
            make_signal(company_id="c1", at="2024-06-01T00:00:00+00:00"),
            make_signal(company_id="c1", at="2024-06-02T00:00:00+00:00"),
        ])
        assert report.is_valid

    def test_out_of_order_detected(self) -> None:
        report = validate_signals("c1", [
            make_signal(company_id="c1", at="2024-06-02T00:00:00+00:00"),
            make_signal(company_id="c1", at="2024-06-01T00:00:00+00:00"),
        ])
        assert not report.is_valid
        assert "out_of_order" in report.by_kind

    def test_duplicate_detected(self) -> None:
        s = make_signal(company_id="c1", at="2024-06-01T00:00:00+00:00")
        report = validate_signals("c1", [s, s])
        assert "duplicate_signal_id" in report.by_kind

    def test_company_mismatch_detected(self) -> None:
        report = validate_signals("c1", [
            make_signal(company_id="c2", at="2024-06-01T00:00:00+00:00"),
        ])
        assert "company_mismatch" in report.by_kind

    def test_issues_sorted(self) -> None:
        s = make_signal(company_id="c2", at="2024-06-02T00:00:00+00:00")
        s2 = make_signal(company_id="c2", at="2024-06-01T00:00:00+00:00")
        report = validate_signals("c1", [s, s2])
        kinds = [i.kind for i in report.issues]
        assert kinds == sorted(kinds)


class TestValidateTimeline:
    def test_empty_timeline_valid(self) -> None:
        tl = make_timeline("c1")
        assert validate_timeline(tl).is_valid

    def test_constructed_timeline_valid(self) -> None:
        tl = make_timeline(
            "c1",
            make_signal(company_id="c1", at="2024-06-01T00:00:00+00:00"),
            make_signal(company_id="c1", at="2024-06-02T00:00:00+00:00"),
        )
        assert validate_timeline(tl).is_valid


class TestSignalValidationReport:
    def test_is_valid_empty(self) -> None:
        assert SignalValidationReport().is_valid

    def test_issue_count_and_by_kind(self) -> None:
        report = SignalValidationReport([
            SignalValidationIssue("a", "kind_x"),
            SignalValidationIssue("b", "kind_x"),
            SignalValidationIssue("c", "kind_y"),
        ])
        assert report.issue_count == 3
        assert report.by_kind == {"kind_x": 2, "kind_y": 1}

    def test_to_dict(self) -> None:
        report = SignalValidationReport([SignalValidationIssue("a", "kind_x")])
        d = report.to_dict()
        assert d["is_valid"] is False
        assert d["issue_count"] == 1
        assert d["issues"][0]["kind"] == "kind_x"
