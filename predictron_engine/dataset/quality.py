"""Dataset quality checks and quality reporting (Project V4).

Automatic quality assessment performed after every population run:

  - missing required fields
  - invalid URLs
  - duplicate companies
  - duplicate websites
  - conflicting identifiers
  - malformed outcomes
  - provenance completeness

Quality checks are deterministic, non-destructive, and never fabricate
data.  Each check reports the affected record IDs and a human-readable
detail string.  A :class:`QualityReport` aggregates counts by check so
operators can triage issues after a population run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from predictron_engine.dataset.dedup import _normalize_name, _normalize_website
from predictron_engine.dataset.models import DatasetRecord
from predictron_engine.dataset.outcomes import OutcomeRecord
from predictron_engine.dataset.provenance import ProvenanceTracker
from predictron_engine.dataset.store import DatasetStore

_REQUIRED_FIELDS = ("startup_name", "website", "engine_version")

_IDENTIFIER_KEYS = (
    "sec_cik",
    "company_number",
    "crunchbase_url",
    "normalized_identifier",
)


@dataclass
class QualityFinding:
    """A single quality finding for a record."""

    check: str
    record_id: str
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "check": self.check,
            "record_id": self.record_id,
            "detail": self.detail,
        }


@dataclass
class QualityReport:
    """Aggregated quality assessment over a collection of records."""

    findings: list[QualityFinding] = field(default_factory=list)
    records_checked: int = 0

    @property
    def finding_count(self) -> int:
        return len(self.findings)

    def by_check(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.check] = counts.get(finding.check, 0) + 1
        return counts

    def is_clean(self) -> bool:
        return self.finding_count == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "records_checked": self.records_checked,
            "finding_count": self.finding_count,
            "clean": self.is_clean(),
            "by_check": self.by_check(),
            "findings": [f.to_dict() for f in self.findings],
        }


class DatasetQualityChecker:
    """Runs automatic quality checks over a population or store.

    Checks run against both ``DatasetRecord`` and ``OutcomeRecord``
    instances.  All checks are additive; they only report issues and
    never mutate records.
    """

    # ---- Public API ----

    def check_records(self, records: list[DatasetRecord]) -> QualityReport:
        """Run all record-level quality checks over a collection."""
        report = QualityReport(records_checked=len(records))
        for record in records:
            self._check_required_fields(record, report)
            self._check_website(record, report)

        self._check_duplicate_companies(records, report)
        self._check_duplicate_websites(records, report)
        self._check_duplicate_domains(records, report)
        self._check_conflicting_identifiers(records, report)

        for record in records:
            self._check_provenance_completeness(record, report)

        return report

    def check_outcomes(
        self, outcomes: list[OutcomeRecord]
    ) -> QualityReport:
        """Run outcome-level quality checks."""
        report = QualityReport(records_checked=len(outcomes))
        for outcome in outcomes:
            self._check_outcome(outcome, report)
        return report

    def check(
        self,
        store: DatasetStore,
        records: list[DatasetRecord] | None = None,
        outcomes: list[OutcomeRecord] | None = None,
    ) -> QualityReport:
        """Run all quality checks across a store's records and outcomes.

        Parameters
        ----------
        store :
            The store to check.  Records/outcomes are loaded from it when
            not supplied explicitly.
        records :
            Optional explicit list of records to check.
        outcomes :
            Optional explicit list of outcomes to check.

        Returns
        -------
        A combined QualityReport.  The number of records checked is the
        record count; outcome checks are appended to the same report.
        """
        if records is None:
            records = self._load_records(store)
        if outcomes is None:
            outcomes = self._load_outcomes(store)

        report = self.check_records(records)
        outcome_report = self.check_outcomes(outcomes)
        report.findings.extend(outcome_report.findings)
        return report

    # ---- Record-level checks ----

    def _check_required_fields(
        self, record: DatasetRecord, report: QualityReport
    ) -> None:
        for field_name in _REQUIRED_FIELDS:
            value = getattr(record, field_name, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                report.findings.append(QualityFinding(
                    "missing_required_field",
                    record.record_id,
                    detail=f"required field '{field_name}' is empty",
                ))

    def _check_website(
        self, record: DatasetRecord, report: QualityReport
    ) -> None:
        website = record.website
        if not website:
            return
        if not website.startswith(("http://", "https://")):
            report.findings.append(QualityFinding(
                "invalid_url",
                record.record_id,
                detail=f"website '{website}' lacks http(s) scheme",
            ))
            return
        if not _URL_HOST_RE.search(website):
            report.findings.append(QualityFinding(
                "invalid_url",
                record.record_id,
                detail=f"website '{website}' has no valid host",
            ))

    def _check_duplicate_companies(
        self, records: list[DatasetRecord], report: QualityReport
    ) -> None:
        name_to_ids: dict[str, list[str]] = {}
        for record in records:
            name_key = _normalize_name(record.startup_name)
            if not name_key:
                continue
            name_to_ids.setdefault(name_key, []).append(record.record_id)

        for key, ids in name_to_ids.items():
            if len(ids) > 1:
                for rid in ids[1:]:
                    report.findings.append(QualityFinding(
                        "duplicate_company",
                        rid,
                        detail=f"normalized name '{key}' shared by {sorted(ids)}",
                    ))

    def _check_duplicate_websites(
        self, records: list[DatasetRecord], report: QualityReport
    ) -> None:
        site_to_ids: dict[str, list[str]] = {}
        for record in records:
            site_key = _normalize_website(record.website)
            if not site_key:
                continue
            site_to_ids.setdefault(site_key, []).append(record.record_id)

        for key, ids in site_to_ids.items():
            if len(ids) > 1:
                for rid in ids[1:]:
                    report.findings.append(QualityFinding(
                        "duplicate_website",
                        rid,
                        detail=f"normalized website '{key}' shared by {sorted(ids)}",
                    ))

    def _check_duplicate_domains(
        self, records: list[DatasetRecord], report: QualityReport
    ) -> None:
        domain_to_ids: dict[str, list[str]] = {}
        for record in records:
            domain = record.profile.domain
            if not domain:
                continue
            domain_to_ids.setdefault(domain.lower(), []).append(record.record_id)

        for key, ids in domain_to_ids.items():
            if len(ids) > 1:
                for rid in ids[1:]:
                    report.findings.append(QualityFinding(
                        "duplicate_domain",
                        rid,
                        detail=f"canonical domain '{key}' shared by {sorted(ids)}",
                    ))

    def _check_conflicting_identifiers(
        self, records: list[DatasetRecord], report: QualityReport
    ) -> None:
        """Flag records that carry multiple distinct identifier values.

        A record that lists more than one distinct value across the
        known identifier keys (CIK, company number, Crunchbase URL,
        normalized identifier) is flagged as having conflicting
        identifiers — the operator should reconcile which is canonical.
        """
        for record in records:
            values: set[str] = set()
            for key in _IDENTIFIER_KEYS:
                value = record.analysis_metadata.get(key)
                if value:
                    values.add(str(value).strip().lower())
            if len(values) > 1:
                report.findings.append(QualityFinding(
                    "conflicting_identifiers",
                    record.record_id,
                    detail=f"multiple distinct identifier values: {sorted(values)}",
                ))

    def _check_provenance_completeness(
        self, record: DatasetRecord, report: QualityReport
    ) -> None:
        provenance = ProvenanceTracker.extract_from_record(record)
        missing: list[str] = []
        for field_name in _REQUIRED_FIELDS:
            value = getattr(record, field_name, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            if field_name not in provenance:
                missing.append(field_name)
        if missing:
            report.findings.append(QualityFinding(
                "missing_provenance",
                record.record_id,
                detail=f"no provenance for fields: {sorted(missing)}",
            ))

    # ---- Outcome-level checks ----

    def _check_outcome(
        self, outcome: OutcomeRecord, report: QualityReport
    ) -> None:
        if not outcome.record_id:
            report.findings.append(QualityFinding(
                "malformed_outcome",
                outcome.outcome_id,
                detail="outcome has empty record_id",
            ))
            return

        o = outcome.outcome
        if o is None:
            report.findings.append(QualityFinding(
                "malformed_outcome",
                outcome.outcome_id,
                detail="outcome data is missing",
            ))
            return

        if o.shutdown_date is not None and o.shutdown is not True:
            report.findings.append(QualityFinding(
                "malformed_outcome",
                outcome.outcome_id,
                detail="shutdown_date present but shutdown is not True",
            ))

        if o.bankruptcy_date is not None and o.bankruptcy is not True:
            report.findings.append(QualityFinding(
                "malformed_outcome",
                outcome.outcome_id,
                detail="bankruptcy_date present but bankruptcy is not True",
            ))

        if (
            o.bankruptcy_date is not None
            and o.shutdown_date is not None
            and o.bankruptcy_date < o.shutdown_date
        ):
            report.findings.append(QualityFinding(
                "malformed_outcome",
                outcome.outcome_id,
                detail="bankruptcy_date precedes shutdown_date",
            ))

        if (
            o.acquisition_price_usd is not None
            and o.acquisition_price_usd < 0
        ):
            report.findings.append(QualityFinding(
                "malformed_outcome",
                outcome.outcome_id,
                detail="acquisition_price_usd is negative",
            ))

        if o.total_funding_usd is not None and o.total_funding_usd < 0:
            report.findings.append(QualityFinding(
                "malformed_outcome",
                outcome.outcome_id,
                detail="total_funding_usd is negative",
            ))

        for idx, round_ in enumerate(o.funding_rounds):
            if round_.amount_usd is not None and round_.amount_usd < 0:
                report.findings.append(QualityFinding(
                    "malformed_outcome",
                    outcome.outcome_id,
                    detail=f"funding_round[{idx}] amount_usd is negative",
                ))

    # ---- Helpers ----

    @staticmethod
    def _load_records(store: DatasetStore) -> list[DatasetRecord]:
        records: list[DatasetRecord] = []
        for rid in store.list_records():
            record = store.load_record(rid)
            if record is not None:
                records.append(record)
        return records

    @staticmethod
    def _load_outcomes(store: DatasetStore) -> list[OutcomeRecord]:
        outcomes: list[OutcomeRecord] = []
        for oid in store.list_outcomes():
            outcome = store.load_outcome(oid)
            if outcome is not None:
                outcomes.append(outcome)
        return outcomes


_URL_HOST_RE = re.compile(
    r"^https?://[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?)*$",
    re.IGNORECASE,
)


def generate_quality_report(
    store: DatasetStore,
    records: list[DatasetRecord] | None = None,
    outcomes: list[OutcomeRecord] | None = None,
) -> QualityReport:
    """Generate a quality report for a store (convenience wrapper)."""
    return DatasetQualityChecker().check(store, records, outcomes)
