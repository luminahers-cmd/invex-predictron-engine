"""Tests for dataset quality checking (Project V4)."""

from __future__ import annotations

from predictron_engine.dataset.models import CompanyProfile, DatasetRecord
from predictron_engine.dataset.outcomes import (
    OutcomeRecord,
    StartupOutcome,
)
from predictron_engine.dataset.quality import (
    DatasetQualityChecker,
    QualityFinding,
    QualityReport,
    generate_quality_report,
)
from predictron_engine.dataset.store import DatasetStore
from tests.dataset.conftest import make_record


def test_clean_records_have_no_structural_findings() -> None:
    records = [
        make_record(startup_name="Acme", website="https://acme.com"),
        make_record(startup_name="Beta", website="https://beta.com"),
    ]
    report = DatasetQualityChecker().check_records(records)
    structural = [f for f in report.findings if f.check != "missing_provenance"]
    assert structural == []
    assert report.records_checked == 2


def test_missing_required_field_flagged() -> None:
    record = make_record()
    bad = DatasetRecord(
        startup_name="",
        website="https://foo.com",
        engine_version="0.12.1",
        prediction=record.prediction,
    )
    report = DatasetQualityChecker().check_records([bad])
    checks = report.by_check()
    assert checks["missing_required_field"] == 1


def test_invalid_url_flagged() -> None:
    bad = make_record(website="not-a-url")
    report = DatasetQualityChecker().check_records([bad])
    assert report.by_check()["invalid_url"] == 1


def test_duplicate_company_and_website_flagged() -> None:
    one = make_record(startup_name="Acme Inc", website="https://a.com")
    two = make_record(startup_name="Acme Inc", website="https://a.com")
    report = DatasetQualityChecker().check_records([one, two])
    checks = report.by_check()
    assert checks["duplicate_company"] == 1
    assert checks["duplicate_website"] == 1


def test_conflicting_identifiers_flagged() -> None:
    record = make_record()
    record.analysis_metadata["sec_cik"] = "00012"
    record.analysis_metadata["company_number"] = "123456"
    report = DatasetQualityChecker().check_records([record])
    assert report.by_check()["conflicting_identifiers"] == 1


def test_duplicate_domain_flagged() -> None:
    one = make_record(record_id="one").model_copy(
        update={"profile": CompanyProfile(domain="acme.com")}
    )
    two = make_record(record_id="two").model_copy(
        update={"profile": CompanyProfile(domain="acme.com")}
    )
    report = DatasetQualityChecker().check_records([one, two])
    check = report.by_check()
    assert check["duplicate_domain"] == 1


def test_duplicate_domains_with_different_casing_not_flagged_twice() -> None:
    one = make_record(record_id="one").model_copy(
        update={"profile": CompanyProfile(domain="Acme.COM")}
    )
    two = make_record(record_id="two").model_copy(
        update={"profile": CompanyProfile(domain="acme.com")}
    )
    report = DatasetQualityChecker().check_records([one, two])
    assert report.by_check().get("duplicate_domain", 0) == 1


def test_is_clean_and_to_dict() -> None:
    report = QualityReport(records_checked=0)
    assert report.is_clean() is True
    data = report.to_dict()
    assert data["clean"] is True
    assert data["finding_count"] == 0


def test_finding_to_dict() -> None:
    finding = QualityFinding("duplicate_company", "rec-1", detail="dup")
    assert finding.to_dict()["check"] == "duplicate_company"


def test_outcome_malformed_shutdown_date_without_shutdown_flag() -> None:
    from datetime import date

    outcome = OutcomeRecord(
        record_id="rec-1",
        outcome=StartupOutcome(
            shutdown=False,
            shutdown_date=date(2024, 1, 1),
        ),
    )
    report = DatasetQualityChecker().check_outcomes([outcome])
    assert report.by_check()["malformed_outcome"] == 1


def test_outcome_clean() -> None:
    outcome = OutcomeRecord(
        record_id="rec-1",
        outcome=StartupOutcome(
            shutdown=False,
            total_funding_usd=1000,
        ),
    )
    report = DatasetQualityChecker().check_outcomes([outcome])
    assert report.findings == []


def test_generate_quality_report_from_store(
    dataset_store: DatasetStore,
) -> None:
    dataset_store.save_record(make_record(startup_name="Solo"))
    report = generate_quality_report(dataset_store)
    assert report.records_checked == 1
    assert {r.record_id for r in report.findings} == {
        dataset_store.list_records()[0]
    }
