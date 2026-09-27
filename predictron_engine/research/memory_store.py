"""Append-only record storage for research memory — Phase 8 Sprint 6.

:class:`MemoryStore` is the ledger behind
:class:`~predictron_engine.research.memory.ResearchMemory`.
It owns the two kinds of record research memory accumulates — finished
:class:`~predictron_engine.research.execution_report.ResearchReport` values and
the :class:`~predictron_engine.research.models.Evidence` items behind them —
and it shares the same :class:`~predictron_engine.research.graph_store.MemoryIndex`
as the graph, so "the latest report" and "the newest graph fact" come from one
ordered, caller-controlled sequence.

Why reports and evidence are stored, not just the graph
-------------------------------------------------------
The graph holds *derived facts*: a company, its technology, the edge between
them.  It deliberately does not hold the artifacts those facts came from.
Memory has to keep them, because the questions research memory exists to
answer are historical: "which report told us this, what evidence backed it,
and what did we believe before?"

Append-only guarantees
----------------------
* A record is stored once, under its own content-derived identifier, and the
  value stored is the frozen object the caller supplied — memory never edits
  a report or an evidence item, and never rebuilds one.
* Re-adding a record with an *identical* payload is an idempotent no-op that
  returns the original sequence number, so replaying a pipeline cannot reorder
  or inflate anything.
* Re-adding a record identifier with a *different* payload raises
  :class:`~predictron_engine.research.exceptions.ImmutableRecordError` rather
  than overwriting what was already known.

No clock is ever read: the caller supplies ``stored_at`` on every write, and
all ordering comes from the shared index.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from predictron_engine.research.deduplication import DeduplicationIndex
from predictron_engine.research.exceptions import (
    ImmutableRecordError,
    InvalidMemoryInputError,
)
from predictron_engine.research.execution_report import ResearchReport
from predictron_engine.research.graph_store import MemoryIndex, MemoryRecordKey
from predictron_engine.research.linking import normalize_name
from predictron_engine.research.models import Evidence

#: Serialization schema version for the memory record store.
MEMORY_SCHEMA_VERSION = "1.0.0"

#: The record types this store allocates sequence numbers for.
RECORD_REPORT = "report"
RECORD_EVIDENCE = "evidence"


@dataclass(frozen=True)
class ReportWrite:
    """The outcome of writing one report.

    ``previous`` is the report already stored under the same identifier, or
    ``None`` when this write introduced it.  A non-``None`` ``previous`` with
    an equal value is a replay; an unequal one never gets this far, because
    the store rejects it as an attempted rewrite.
    """

    report: ResearchReport
    previous: ResearchReport | None
    sequence: int

    @property
    def created(self) -> bool:
        """Return whether this write introduced the report."""
        return self.previous is None

    @property
    def replayed(self) -> bool:
        """Return whether an identical report was already recorded."""
        return self.previous is not None and self.previous == self.report

    @property
    def report_id(self) -> str:
        """Return the report's own identifier."""
        return self.report.report_id

    @property
    def company_name(self) -> str:
        """Return the company the report is about."""
        return self.report.company_name

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "report_id": self.report_id,
            "company_name": self.company_name,
            "created": self.created,
            "replayed": self.replayed,
            "sequence": self.sequence,
        }


@dataclass(frozen=True)
class EvidenceWrite:
    """The outcome of writing one evidence item.

    ``duplicate_of`` is the identifier this item was recognised as a repeat
    of, or ``None`` when it was new.  The stored :class:`Evidence` value is
    always the first one seen: a later observation that resolves to the same
    observation is *recorded as a repeat*, never written over the original.
    """

    evidence: Evidence
    duplicate_of: str | None
    sequence: int
    replayed: bool = False

    @property
    def is_duplicate(self) -> bool:
        """Return whether this item repeats an *earlier, different* item."""
        return self.duplicate_of is not None

    @property
    def created(self) -> bool:
        """Return whether this write stored a new evidence item."""
        return self.duplicate_of is None and not self.replayed

    @property
    def evidence_id(self) -> str:
        """Return the evidence item's own identifier."""
        return self.evidence.evidence_id

    @property
    def topic_id(self) -> str:
        """Return the topic the evidence was collected for."""
        return self.evidence.topic_id

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "evidence_id": self.evidence_id,
            "topic_id": self.topic_id,
            "duplicate_of": self.duplicate_of,
            "is_duplicate": self.is_duplicate,
            "created": self.created,
            "replayed": self.replayed,
            "sequence": self.sequence,
        }


@dataclass(frozen=True)
class TopicRetrieval:
    """Everything memory holds for one research topic.

    The four views are returned together because a topic is only useful as a
    whole: the reports describe the run, the evidence supports it, the graph
    entities are what was concluded, and the relationships are how the
    conclusions connect.
    """

    topic_id: str
    reports: tuple[ResearchReport, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    entity_ids: tuple[str, ...] = ()
    relationship_keys: tuple[tuple[str, str, str], ...] = ()

    def __len__(self) -> int:
        """Return how many evidence items were found for the topic."""
        return len(self.evidence)

    def is_empty(self) -> bool:
        """Return whether nothing at all is recorded for the topic."""
        return not (self.reports or self.evidence or self.entity_ids)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "topic_id": self.topic_id,
            "reports": [report.report_id for report in self.reports],
            "evidence": [item.evidence_id for item in self.evidence],
            "entity_ids": list(self.entity_ids),
            "relationship_keys": [list(key) for key in self.relationship_keys],
        }


class MemoryStore:
    """The append-only ledger of reports and evidence.

    Parameters
    ----------
    index:
        The shared sequence index.  Passing the graph store's own index is
        what lets a report and the facts it produced share one ordering; a
        fresh index is created when omitted.

    Attributes
    ----------
    index:
        The shared :class:`~predictron_engine.research.graph_store.MemoryIndex`.
    deduplication:
        The duplicate index used to recognise a re-observed evidence item.
        It is shared with the owning
        :class:`~predictron_engine.research.memory.ResearchMemory` so both
        layers agree on what "the same observation" means.
    """

    def __init__(
        self,
        index: MemoryIndex | None = None,
        deduplication: DeduplicationIndex | None = None,
    ) -> None:
        if index is not None and not isinstance(index, MemoryIndex):
            raise InvalidMemoryInputError("index must be a MemoryIndex instance")
        self.index = index if index is not None else MemoryIndex()
        self.deduplication = (
            deduplication if deduplication is not None else DeduplicationIndex()
        )
        self._reports: dict[str, ResearchReport] = {}
        self._evidence: dict[str, Evidence] = {}
        self._report_evidence: dict[str, tuple[str, ...]] = {}
        self._evidence_reports: dict[str, str] = {}

    # -- Reports ------------------------------------------------------

    def add_report(
        self,
        report: ResearchReport,
        *,
        evidence: Iterable[Evidence] = (),
        stored_at: str = "",
        topic_id: str = "",
    ) -> tuple[ReportWrite, tuple[EvidenceWrite, ...]]:
        """Store ``report`` and any evidence it carries, and report what changed.

        A replayed report is a no-op.  Re-supplying the evidence the report
        already carries is accepted too, because re-running the pipeline that
        produced the report will hand the same content back; the items are
        reported as replays and nothing is written.  Evidence the report did
        *not* originally carry is still rejected: silently attaching new
        evidence to a content-addressed report would rewrite history.
        """
        if not isinstance(report, ResearchReport):
            raise InvalidMemoryInputError(
                "add_report expects a ResearchReport instance"
            )
        items = tuple(evidence)
        for item in items:
            if not isinstance(item, Evidence):
                raise InvalidMemoryInputError(
                    "report evidence must be Evidence instances"
                )
        key = self.index.allocate(
            RECORD_REPORT,
            report.report_id,
            topic_id=topic_id,
            company_name=report.company_name,
            stored_at=stored_at,
        )
        existing = self._reports.get(report.report_id)
        if existing is not None:
            if existing != report:
                raise ImmutableRecordError(
                    f"report {report.report_id} is already recorded with "
                    "different content"
                )
            return (
                ReportWrite(
                    report=existing,
                    previous=existing,
                    sequence=key.sequence,
                ),
                self._replay_evidence(report, items),
            )
        self._reports[report.report_id] = report
        writes = tuple(
            self.add_evidence(
                item,
                stored_at=stored_at,
                report_id=report.report_id,
            )
            for item in items
        )
        # Every item the report carried is linked, including one that turned
        # out to be a repeat: the report really did carry it, and losing that
        # association would make the report look like it found nothing.
        self._report_evidence[report.report_id] = tuple(
            write.evidence_id for write in writes
        )
        return (
            ReportWrite(report=report, previous=None, sequence=key.sequence),
            writes,
        )

    def _replay_evidence(
        self,
        report: ResearchReport,
        items: tuple[Evidence, ...],
    ) -> tuple[EvidenceWrite, ...]:
        """Return the recorded writes for evidence a replayed report re-offers.

        Every item must already belong to this report, and must be byte-for-byte
        the item that was recorded; anything else would be an attempt to
        extend a finished report, which is rejected rather than absorbed.
        """
        if not items:
            return ()
        recorded = self._report_evidence.get(report.report_id, ())
        writes: list[EvidenceWrite] = []
        for item in items:
            stored = self._evidence.get(item.evidence_id)
            if (
                item.evidence_id not in recorded
                or stored is None
                or stored != item
            ):
                raise ImmutableRecordError(
                    f"report {report.report_id} is already recorded; its "
                    "evidence cannot be appended afterwards"
                )
            stored_key = self.index.key_for(RECORD_EVIDENCE, item.evidence_id)
            if stored_key is None:
                raise ImmutableRecordError(
                    f"evidence {item.evidence_id} has no recorded sequence"
                )
            writes.append(
                EvidenceWrite(
                    evidence=stored,
                    duplicate_of=None,
                    sequence=stored_key.sequence,
                    replayed=True,
                )
            )
        return tuple(writes)

    def has_report(self, report_id: str) -> bool:
        """Return whether ``report_id`` is stored."""
        return report_id.strip() in self._reports

    def get_report(self, report_id: str) -> ResearchReport | None:
        """Return the stored report for ``report_id`` or ``None``."""
        return self._reports.get(report_id.strip())

    def reports(self) -> tuple[ResearchReport, ...]:
        """Return every stored report, in append order."""
        return tuple(
            self._reports[key.record_id]
            for key in self.index.keys(RECORD_REPORT)
            if key.record_id in self._reports
        )

    def reports_for_company(
        self,
        company_name: str,
    ) -> tuple[ResearchReport, ...]:
        """Return the reports about ``company_name``, in append order.

        Matching is on the normalized company name, so every spelling of a
        company that memory treats as one company yields one set of reports.
        """
        target = normalize_name(company_name)
        return tuple(
            report
            for report in self.reports()
            if normalize_name(report.company_name) == target
        )

    def reports_for_topic(self, topic_id: str) -> tuple[ResearchReport, ...]:
        """Return the reports explicitly indexed under ``topic_id``."""
        target = topic_id.strip()
        return tuple(
            self._reports[key.record_id]
            for key in self.index.keys(RECORD_REPORT)
            if key.topic_id == target and key.record_id in self._reports
        )

    def latest_report(
        self,
        company_name: str = "",
    ) -> ResearchReport | None:
        """Return the most recently appended report, optionally per company.

        "Most recent" is the highest shared-index sequence, never a
        timestamp comparison, so the answer does not depend on how a label
        happens to be formatted.
        """
        candidates = (
            self.reports_for_company(company_name)
            if company_name.strip()
            else self.reports()
        )
        return candidates[-1] if candidates else None

    def retrieve_latest(
        self,
        company_name: str = "",
        topic_id: str = "",
        limit: int = 1,
    ) -> tuple[ResearchReport, ...]:
        """Return the ``limit`` newest matching reports, newest first.

        This is the public ``retrieve_latest``: reports for a company, for a
        topic, for both, or for the whole of memory.
        """
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise InvalidMemoryInputError("limit must be an int")
        if limit < 1:
            raise InvalidMemoryInputError("limit must be at least 1")
        candidates = self.reports()
        if company_name.strip():
            # Compare the way memory compares everything else: by normalized
            # name, not by the exact string a run happened to record.  Asking
            # for "Open AI" must find the report filed under "OpenAI Inc.".
            target = normalize_name(company_name)
            candidates = tuple(
                report
                for report in candidates
                if normalize_name(report.company_name) == target
            )
        if topic_id.strip():
            target = topic_id.strip()
            candidates = tuple(
                report for report in candidates if self.covers_topic(report, target)
            )
        ordered = tuple(reversed(candidates))
        return ordered[:limit]

    def covers_topic(self, report: ResearchReport, topic_id: str) -> bool:
        """Return whether ``report`` is indexed under ``topic_id``.

        A report is associated with a topic either because the write declared
        it, or because the evidence it carried was collected for that topic.
        """
        target = topic_id.strip()
        if not target:
            return False
        key = self.index.key_for(RECORD_REPORT, report.report_id)
        if key is not None and key.topic_id == target:
            return True
        return any(
            item.topic_id == target
            for item in self.evidence_for_report(report.report_id)
        )

    # -- Evidence -----------------------------------------------------

    def add_evidence(
        self,
        evidence: Evidence,
        *,
        stored_at: str = "",
        report_id: str = "",
    ) -> EvidenceWrite:
        """Store ``evidence``, recognising it as a repeat when it is one.

        Duplicate detection is delegated to the shared
        :class:`~predictron_engine.research.deduplication.DeduplicationIndex`,
        so the store can never disagree with the rest of memory about
        whether two items are the same observation.
        """
        if not isinstance(evidence, Evidence):
            raise InvalidMemoryInputError(
                "add_evidence expects an Evidence instance"
            )
        key = self.index.allocate(
            RECORD_EVIDENCE,
            evidence.evidence_id,
            topic_id=evidence.topic_id,
            stored_at=stored_at,
        )
        duplicate_of = self.deduplication.existing_duplicate(evidence)
        if duplicate_of == evidence.evidence_id:
            # Matching itself is a replay, not a repeat of a *different*
            # observation, so it is reported as such rather than as a
            # duplicate.
            duplicate_of = None
        stored = self._evidence.get(evidence.evidence_id)
        if stored is not None:
            if stored != evidence:
                raise ImmutableRecordError(
                    f"evidence {evidence.evidence_id} is already recorded with "
                    "different content"
                )
            return EvidenceWrite(
                evidence=stored,
                duplicate_of=None,
                sequence=key.sequence,
                replayed=True,
            )
        self.deduplication.record(evidence)
        self._evidence[evidence.evidence_id] = evidence
        if report_id.strip() and duplicate_of is None:
            self._evidence_reports[evidence.evidence_id] = report_id.strip()
        return EvidenceWrite(
            evidence=evidence,
            duplicate_of=duplicate_of,
            sequence=key.sequence,
        )

    def has_evidence(self, evidence_id: str) -> bool:
        """Return whether ``evidence_id`` is stored."""
        return evidence_id.strip() in self._evidence

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        """Return the stored evidence item or ``None``."""
        return self._evidence.get(evidence_id.strip())

    def evidence(self) -> tuple[Evidence, ...]:
        """Return every stored evidence item, in append order."""
        return tuple(
            self._evidence[key.record_id]
            for key in self.index.keys(RECORD_EVIDENCE)
            if key.record_id in self._evidence
        )

    def evidence_for_topic(self, topic_id: str) -> tuple[Evidence, ...]:
        """Return the evidence collected for ``topic_id``, in append order."""
        target = topic_id.strip()
        return tuple(
            item for item in self.evidence() if item.topic_id == target
        )

    def evidence_for_report(
        self,
        report_id: str,
    ) -> tuple[Evidence, ...]:
        """Return the evidence stored alongside ``report_id``, in append order."""
        target = report_id.strip()
        return tuple(
            self._evidence[evidence_id]
            for evidence_id in self._report_evidence.get(target, ())
            if evidence_id in self._evidence
        )

    def report_for_evidence(self, evidence_id: str) -> str | None:
        """Return the first report that carried ``evidence_id`` or ``None``."""
        return self._evidence_reports.get(evidence_id.strip())

    def duplicate_count(self) -> int:
        """Return how many stored evidence items repeat another one."""
        return self.deduplication.duplicate_count()

    # -- Reporting ---------------------------------------------------

    def counts(self) -> dict[str, int]:
        """Return the deterministic record counters for the store."""
        return {
            "reports": len(self._reports),
            "evidence": len(self._evidence),
            "duplicates": self.duplicate_count(),
            "indexed": len(self.index),
        }

    def key_for(self, record_type: str, record_id: str) -> MemoryRecordKey | None:
        """Return the shared-index key for a stored record or ``None``."""
        return self.index.key_for(record_type, record_id)

    def __len__(self) -> int:
        """Return how many records are stored."""
        return len(self._reports) + len(self._evidence)

    def is_empty(self) -> bool:
        """Return whether nothing has been recorded."""
        return not self._reports and not self._evidence

    # -- Serialization ------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Serialize the store into a JSON-ready dictionary.

        Reports and evidence are emitted in append order and the shared index
        is written alongside them, so a round trip restores the original
        sequence numbering exactly.
        """
        return {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "index": self.index.to_dict(),
            "reports": [report.to_dict() for report in self.reports()],
            "evidence": [item.to_dict() for item in self.evidence()],
            "report_evidence": {
                report_id: list(evidence_ids)
                for report_id, evidence_ids in sorted(
                    self._report_evidence.items()
                )
            },
            "evidence_reports": dict(sorted(self._evidence_reports.items())),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, object],
        index: MemoryIndex | None = None,
        deduplication: DeduplicationIndex | None = None,
    ) -> MemoryStore:
        """Rebuild a store from :meth:`to_dict` output.

        ``index`` and ``deduplication`` are passed in rather than read from
        the document when the caller is restoring a whole
        :class:`~predictron_engine.research.memory.ResearchMemory`, so the
        records, the graph, and the duplicate index keep sharing one state.
        """
        if not isinstance(data, Mapping):
            raise InvalidMemoryInputError("from_dict expects a mapping")
        index_data = data.get("index")
        if index is not None:
            restored_index = index
        elif isinstance(index_data, Mapping):
            restored_index = MemoryIndex.from_dict(index_data)
        else:
            raise InvalidMemoryInputError(
                "restoring memory requires the memory index"
            )
        store = cls(index=restored_index, deduplication=deduplication)
        for entry in _require_list(data, "reports"):
            report = ResearchReport.from_dict(_require_mapping(entry))
            if restored_index.key_for(RECORD_REPORT, report.report_id) is None:
                raise InvalidMemoryInputError(
                    f"report {report.report_id} is missing from the memory index"
                )
            store._reports[report.report_id] = report
        for entry in _require_list(data, "evidence"):
            item = Evidence.from_dict(_require_mapping(entry))
            if restored_index.key_for(RECORD_EVIDENCE, item.evidence_id) is None:
                raise InvalidMemoryInputError(
                    f"evidence {item.evidence_id} is missing from the memory index"
                )
            store.deduplication.record(item)
            store._evidence[item.evidence_id] = item
        links = data.get("report_evidence", {})
        if not isinstance(links, Mapping):
            raise InvalidMemoryInputError(
                "expected mapping for 'report_evidence'"
            )
        for report_id, evidence_ids in links.items():
            if not isinstance(report_id, str) or not isinstance(
                evidence_ids, list
            ):
                raise InvalidMemoryInputError(
                    "expected string report ids and list evidence ids"
                )
            store._report_evidence[report_id] = tuple(
                item
                for item in evidence_ids
                if isinstance(item, str) and item in store._evidence
            )
        owners = data.get("evidence_reports", {})
        if not isinstance(owners, Mapping):
            raise InvalidMemoryInputError(
                "expected mapping for 'evidence_reports'"
            )
        for evidence_id, report_id in owners.items():
            if not isinstance(evidence_id, str) or not isinstance(
                report_id, str
            ):
                raise InvalidMemoryInputError(
                    "expected string evidence ids and report ids"
                )
            if evidence_id in store._evidence:
                store._evidence_reports[evidence_id] = report_id
        return store

    # -- Topic retrieval ----------------------------------------------

    def topic_records(self, topic_id: str) -> TopicRetrieval:
        """Return the reports and evidence recorded for ``topic_id``.

        The graph half of the answer — the entities and edges concluded for
        the topic — is supplied by
        :meth:`~predictron_engine.research.memory.ResearchMemory.retrieve_by_topic`,
        which combines this with the graph's own topic index.
        """
        target = topic_id.strip()
        if not target:
            raise InvalidMemoryInputError("topic_id must not be empty")
        return TopicRetrieval(
            topic_id=target,
            reports=self.reports_for_topic(target),
            evidence=self.evidence_for_topic(target),
        )


def _require_list(data: Mapping[str, object], key: str) -> list[object]:
    """Read and validate a serialized list field."""
    value = data.get(key, [])
    if not isinstance(value, list):
        raise InvalidMemoryInputError(f"expected list for '{key}'")
    return value


def _require_mapping(entry: object) -> Mapping[str, object]:
    """Validate one serialized record entry."""
    if not isinstance(entry, Mapping):
        raise InvalidMemoryInputError("expected mappings for record entries")
    return entry


__all__ = [
    "MEMORY_SCHEMA_VERSION",
    "RECORD_EVIDENCE",
    "RECORD_REPORT",
    "EvidenceWrite",
    "MemoryStore",
    "ReportWrite",
    "TopicRetrieval",
]
