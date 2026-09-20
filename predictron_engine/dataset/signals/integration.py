"""Signal dataset integration (Project E4).

:class:`SignalDatasetManager` is the wiring that ties signals into the
existing dataset: it resolves records to canonical company identities
(same ids as the knowledge graph), builds grounded timelines from stored
outcomes, imports explicit signal JSON, persists timelines through the
:class:`DatasetStore`, validates them, and serves aggregate reports.

The manager is strictly **additive** — it never modifies records,
outcomes, evaluations, or graph objects — and every result is
deterministic for a fixed reference timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from predictron_engine.dataset.models import DatasetRecord
from predictron_engine.dataset.outcomes import OutcomeRecord
from predictron_engine.dataset.signals.aggregate import aggregate_all
from predictron_engine.dataset.signals.builder import (
    CompanySignalBuilder,
    SignalBuildResult,
    record_to_company_ids,
)
from predictron_engine.dataset.signals.importers import (
    SignalImportReport,
    import_signals_from_json,
)
from predictron_engine.dataset.signals.metrics import (
    SignalStatistics,
    compute_signal_statistics,
)
from predictron_engine.dataset.signals.reports import build_signal_dataset_report
from predictron_engine.dataset.signals.timeline import CompanyTimeline
from predictron_engine.dataset.signals.trends import TrendEngine
from predictron_engine.dataset.signals.validation import (
    SignalValidationReport,
    validate_timeline,
)
from predictron_engine.dataset.store import DatasetStore


@dataclass
class SignalImportOutcome:
    """Result of a file import that also persisted accepted signals."""

    report: SignalImportReport

    def to_dict(self) -> dict[str, object]:
        return self.report.to_dict()


class SignalDatasetManager:
    """Integrates company signals with a dataset store."""

    def __init__(
        self,
        store: DatasetStore,
        *,
        as_of: datetime | None = None,
    ) -> None:
        self._store = store
        self._as_of = as_of or datetime.now(UTC)
        self._record_map: dict[str, str] | None = None
        self._startup_name_map: dict[str, str] | None = None
        self._website_map: dict[str, str] | None = None

    # -- identity resolution ----------------------------------------

    def _load_records(self) -> list[DatasetRecord]:
        records: list[DatasetRecord] = []
        for record_id in self._store.list_records():
            record = self._store.load_record(record_id)
            if record is not None:
                records.append(record)
        return records

    def _resolve(self) -> None:
        records = self._load_records()
        record_map = record_to_company_ids(records)
        name_map: dict[str, str] = {}
        website_map: dict[str, str] = {}
        for record in records:
            company_id = record_map[record.record_id]
            if record.startup_name:
                name_map[record.startup_name] = company_id
            if record.website:
                website_map[record.website] = company_id
        self._record_map = record_map
        self._startup_name_map = name_map
        self._website_map = website_map

    def company_id_for(self, key: str) -> str | None:
        """Resolve a record id, startup name, or website to a company id."""
        if self._record_map is None:
            self._resolve()
        assert self._record_map is not None
        for lookup in (self._record_map, self._startup_name_map, self._website_map):
            if lookup is not None and key in lookup:
                return lookup[key]
        return None

    def company_map(self) -> dict[str, str]:
        """A combined mapping of every identity key to its company id."""
        if self._record_map is None:
            self._resolve()
        merged: dict[str, str] = {}
        for lookup in (self._record_map, self._startup_name_map, self._website_map):
            if lookup is not None:
                merged.update(lookup)
        return merged

    def distinct_startup_count(self) -> int:
        return len({r.startup_name for r in self._load_records()})

    # -- building -----------------------------------------------------

    def build(self) -> SignalBuildResult:
        """Build grounded timelines from stored records+outcomes.

        The result is persisted through the store (each timeline file is
        (re)written) and returned for inspection.
        """
        records = self._load_records()
        outcomes: dict[str, OutcomeRecord] = {}
        for outcome_id in self._store.list_outcomes():
            outcome = self._store.load_outcome(outcome_id)
            if outcome is not None:
                outcomes[outcome.record_id] = outcome
        result = CompanySignalBuilder().build(
            records, outcomes, identity_map=self.company_map()
        )
        for timeline in result.timelines.values():
            self._store.save_timeline(timeline)
        return result

    # -- importing ------------------------------------------------------

    def import_file(self, path: str | Path) -> SignalImportOutcome:
        """Import a JSON signal file, resolving identities and persisting."""
        company_map = self.company_map() or None
        signals, report = import_signals_from_json(path, company_map=company_map)
        for signal in signals:
            self._store.save_signal(signal)
        report.companies = sorted(
            set(report.companies) | set(self._store.list_signal_company_ids())
        )
        return SignalImportOutcome(report=report)

    # -- reading ----------------------------------------------------------

    def timeline(self, company_id: str) -> CompanyTimeline | None:
        return self._store.load_timeline(company_id)

    def all_timelines(self) -> list[CompanyTimeline]:
        timelines: list[CompanyTimeline] = []
        for company_id in self._store.list_signal_company_ids():
            timeline = self._store.load_timeline(company_id)
            if timeline is not None:
                timelines.append(timeline)
        return sorted(timelines, key=lambda t: t.company_id)

    # -- validation / statistics / reports --------------------------------

    def validate(self) -> SignalValidationReport:
        """Validate every stored timeline."""
        report = SignalValidationReport()
        for timeline in self.all_timelines():
            report.issues.extend(validate_timeline(timeline).issues)
        report.issues.sort(key=lambda issue: (issue.kind, issue.detail or ""))
        return report

    def statistics(
        self, *, as_of: datetime | None = None, expected_company_count: int | None = None
    ) -> SignalStatistics:
        expected = (
            expected_company_count
            if expected_company_count is not None
            else self.distinct_startup_count()
        )
        return compute_signal_statistics(
            self.all_timelines(),
            expected_company_count=expected,
            as_of=as_of or self._as_of,
        )

    def aggregate(self, company_id: str, *, as_of: datetime | None = None) -> dict[str, object]:
        """Bundle of aggregation metrics for one company's timeline."""
        timeline = self.timeline(company_id)
        if timeline is None:
            raise KeyError(f"no signals stored for company '{company_id}'")
        return aggregate_all(timeline, as_of=as_of or self._as_of)

    def trends(
        self, company_id: str, *, as_of: datetime | None = None
    ) -> dict[str, dict[str, object]]:
        """Trend summary for one company's timeline."""
        timeline = self.timeline(company_id)
        if timeline is None:
            raise KeyError(f"no signals stored for company '{company_id}'")
        return TrendEngine(as_of=as_of or self._as_of).summarize(timeline)

    def report(self, *, as_of: datetime | None = None) -> dict[str, object]:
        """Combined signal dataset report over stored timelines."""
        expected_ids = [value for value in self.company_map().values()]
        return build_signal_dataset_report(
            self.all_timelines(),
            as_of=as_of or self._as_of,
            expected_company_count=self.distinct_startup_count(),
            expected_company_ids=sorted(set(expected_ids)),
        )

    # -- derived views (read-only) -----------------------------------------

    def profile_highlights(
        self,
        company_id: str,
        *,
        as_of: datetime | None = None,
        limit: int = 5,
    ) -> list[str]:
        """Short, derived highlight strings for a company's profile.

        These are computed on demand and never written back into any
        :class:`CompanyProfile`.
        """
        timeline = self.timeline(company_id)
        if timeline is None:
            return []
        reference = as_of or self._as_of
        highlights: list[str] = []
        if timeline.signal_count:
            first = timeline.first
            last = timeline.last
            span = (
                f" between {first.timestamp.date().isoformat()}"
                f" and {last.timestamp.date().isoformat()}"
                if first is not None and last is not None
                else ""
            )
            highlights.append(
                f"{timeline.signal_count} signals{span} "
                f"({len(timeline.types())} signal types)"
            )
        from predictron_engine.dataset.signals.aggregate import (
            momentum_score,
            recent_activity,
        )

        highlights.append(
            f"momentum {momentum_score(timeline, as_of=reference):.3f}"
        )
        activity = recent_activity(timeline, as_of=reference)
        highlights.append("active" if activity["is_active"] else "inactive")
        funding = sum(
            1
            for s in timeline.signals
            if s.signal_type.value == "funding_round"
        )
        if funding:
            highlights.append(f"{funding} funding rounds recorded")
        return highlights[:limit]

    def graph_overview(self) -> dict[str, dict[str, object]]:
        """Per-company signal summaries keyed like graph node ids.

        Returns a mapping ``company_node_id -> overview`` that callers can
        overlay onto the knowledge graph without touching graph objects.
        """
        overview: dict[str, dict[str, object]] = {}
        for timeline in self.all_timelines():
            summary: dict[str, object] = {
                "signal_count": timeline.signal_count,
                "first_signal_at": (
                    timeline.first.timestamp.isoformat() if timeline.first else None
                ),
                "last_signal_at": (
                    timeline.last.timestamp.isoformat() if timeline.last else None
                ),
                "types": dict(sorted(timeline.types().items())),
            }
            overview[timeline.company_id] = summary
        return overview
