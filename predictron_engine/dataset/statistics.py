"""Dataset statistics generation (Part F).

Computes aggregate statistics over the records in a DatasetStore:

  - startups imported
  - sectors
  - stages
  - years
  - countries
  - missing-field report
  - duplicate report

Statistics are compiled deterministically from stored records and
enriched metadata.  Missing-field statistics use the validation
utilities; duplicate statistics use the deduplication module.

This module is additive and does not alter the existing report builder.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from predictron_engine.dataset.dedup import DeduplicationReport, find_duplicates
from predictron_engine.dataset.models import DatasetRecord
from predictron_engine.dataset.store import DatasetStore
from predictron_engine.dataset.validation_utils import validate_record_completeness


@dataclass
class MissingFieldReport:
    """Per-field missing counts across a collection of records."""

    counts: dict[str, int] = field(default_factory=dict)
    total_records: int = 0

    @property
    def missing_fields(self) -> list[str]:
        return [k for k, v in sorted(self.counts.items()) if v > 0]

    def to_dict(self) -> dict[str, object]:
        return {
            "total_records": self.total_records,
            "missing_fields": {
                k: v for k, v in sorted(self.counts.items())
            },
        }


@dataclass
class DatasetStats:
    """Aggregate statistics over a dataset store."""

    record_count: int = 0
    startup_count: int = 0
    sectors: dict[str, int] = field(default_factory=dict)
    stages: dict[str, int] = field(default_factory=dict)
    years: dict[str, int] = field(default_factory=dict)
    countries: dict[str, int] = field(default_factory=dict)
    missing_fields: MissingFieldReport = field(default_factory=MissingFieldReport)
    duplicates: list[list[str]] = field(default_factory=list)
    duplicate_record_count: int = 0
    duplicate_group_count: int = 0

    def to_dict(self) -> dict[str, object]:
        """Dict view suitable for JSON serialization."""
        duplicate_groups = [
            {"size": len(group), "record_ids": group}
            for group in self.duplicates
        ]
        return {
            "record_count": self.record_count,
            "startup_count": self.startup_count,
            "sectors": dict(sorted(self.sectors.items())),
            "stages": dict(sorted(self.stages.items())),
            "years": dict(sorted(self.years.items())),
            "countries": dict(sorted(self.countries.items())),
            "missing_fields": self.missing_fields.to_dict(),
            "duplicate_report": {
                "group_count": self.duplicate_group_count,
                "duplicate_record_count": self.duplicate_record_count,
                "groups": duplicate_groups,
            },
        }


def compute_dataset_stats(store: DatasetStore) -> DatasetStats:
    """Compute aggregate statistics for a dataset store.

    Parameters
    ----------
    store :
        The dataset store to analyse.

    Returns
    -------
    A DatasetStats object with all summary statistics.
    """
    records = _load_all_records(store)
    stats = DatasetStats(
        record_count=len(records),
        startup_count=len({r.startup_name for r in records}),
    )

    sectors: Counter[str] = Counter()
    stages: Counter[str] = Counter()
    years: Counter[str] = Counter()
    countries: Counter[str] = Counter()

    missing_counts: Counter[str] = Counter()
    for record in records:
        _tally_sector(record, sectors)
        _tally_stage(record, stages)
        _tally_year(record, years)
        _tally_country(record, countries)
        missing, _present = validate_record_completeness(record)
        for field_name in missing:
            missing_counts[field_name] += 1

    stats.sectors = dict(sectors)
    stats.stages = dict(stages)
    stats.years = dict(years)
    stats.countries = dict(countries)
    stats.missing_fields = MissingFieldReport(
        counts=dict(missing_counts),
        total_records=len(records),
    )

    if records:
        dup_report: DeduplicationReport = find_duplicates(records)
        stats.duplicate_record_count = dup_report.duplicate_record_count
        stats.duplicate_group_count = dup_report.group_count
        stats.duplicates = [
            [r.record_id for r in group] for group in dup_report.groups
        ]

    return stats


def _load_all_records(store: DatasetStore) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    for record_id in store.list_records():
        record = store.load_record(record_id)
        if record is not None:
            records.append(record)
    return records


def _tally_sector(
    record: DatasetRecord, counter: Counter[str]
) -> None:
    sector = record.analysis_metadata.get("sector")
    if sector:
        counter[str(sector)] += 1


def _tally_stage(
    record: DatasetRecord, counter: Counter[str]
) -> None:
    stage = record.analysis_metadata.get("funding_stage_at_analysis")
    if stage:
        counter[str(stage)] += 1
    else:
        counter["unknown"] += 1


def _tally_year(
    record: DatasetRecord, counter: Counter[str]
) -> None:
    counter[str(record.analysis_date.year)] += 1


def _tally_country(
    record: DatasetRecord, counter: Counter[str]
) -> None:
    country = record.analysis_metadata.get("country")
    if country:
        counter[str(country)] += 1
    else:
        counter["unknown"] += 1
