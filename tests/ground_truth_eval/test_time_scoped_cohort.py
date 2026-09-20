"""Tests for the time-scoped cohort machinery (Milestone V1.4).

Covers the look-ahead guard (:func:`time_scope_bundle` /
:func:`scoped_evidence`), the deterministic cohort builder
(:func:`build_cohort_dataset`), and manifest round-tripping.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from benchmarks.cohort.build import CohortBuild, build_cohort_dataset
from benchmarks.cohort.manifest import (
    CohortManifest,
    ManifestOutcome,
    ManifestSourceRecord,
    dump_manifest,
    load_manifest,
    manifest_bytes,
    manifest_hash,
)
from benchmarks.ground_truth.schema import StartupStatus

ANALYSIS = datetime(2026, 1, 1, tzinfo=UTC)
VERIFICATION = datetime(2026, 6, 30, tzinfo=UTC).date()
CREATED_AT = datetime(2026, 9, 20, tzinfo=UTC)


def _corpus_bundle(case_id: str = "b2b_saas"):
    from predictron_engine.evidence.replay.dataset import (
        load_corpus,
        rebuild_bundle,
        resolve_dataset_path,
    )

    return rebuild_bundle(load_corpus(resolve_dataset_path(case_id)))


def _manifest() -> CohortManifest:
    outcome_on = ManifestOutcome(
        status=StartupStatus.ACQUIRED,
        verification_date=VERIFICATION,
        verified=False,
        notes="illustrative placeholder",
        sources=["tests"],
    )
    outcome_verified = ManifestOutcome(
        status=StartupStatus.OPERATING,
        verification_date=VERIFICATION,
        verified=True,
        verified_by="qa",
        sources=["official filing"],
    )
    records = [
        ManifestSourceRecord(
            company_id="b2b_saas",
            company_name="Analytix Cloud",
            description=(
                "A fictional B2B SaaS startup used by the benchmark suite for "
                "deterministic end-to-end tests of the evaluation platform."
            ),
            website="https://analytixcloud.example.com",
            evidence_corpus="b2b_saas",
            analysis_timestamp=ANALYSIS,
            outcome=outcome_on,
            metadata={"sources": ["tests"]},
        ),
        ManifestSourceRecord(
            company_id="ai_infrastructure",
            company_name="Inference Labs",
            description=(
                "A fictional AI infrastructure startup used by the benchmark "
                "suite to exercise time-scoped cohort building."
            ),
            website="https://inferencelabs.example.com",
            evidence_corpus="ai_infrastructure",
            analysis_timestamp=ANALYSIS,
            outcome=outcome_verified,
            metadata={"sources": ["tests"]},
        ),
        ManifestSourceRecord(
            company_id="mrr_only_derivation",
            company_name="SubMetrics",
            description=(
                "A fictional derivation-targeted startup with no documented "
                "outcome; it must be emitted as pending ground truth."
            ),
            website="https://submetrics.example.com",
            evidence_corpus="mrr_only_derivation",
            analysis_timestamp=ANALYSIS,
            outcome=None,
        ),
    ]
    return CohortManifest(
        dataset_name="test_cohort",
        benchmark_version="1.0",
        created_at=CREATED_AT,
        source_records=records,
    )


class TestTimeScopeBundle:
    def test_identity_when_nothing_dropped(self) -> None:
        from predictron_engine.dataset.prediction import time_scope_bundle

        bundle = _corpus_bundle()
        scoped, stats = time_scope_bundle(bundle, as_of=ANALYSIS)
        assert scoped is bundle
        assert stats.dropped_documents == 0
        assert stats.dropped_sources == 0
        assert stats.blocked_lookahead is False
        assert stats.retained_documents == len(bundle.documents)
        assert stats.retained_sources == len(bundle.sources)

    def test_drops_post_analysis_documents(self) -> None:
        from predictron_engine.dataset.prediction import time_scope_bundle

        bundle = _corpus_bundle()
        shifted = [
            doc.model_copy(update={"fetched_at": datetime(2026, 6, 1, tzinfo=UTC)})
            for doc in bundle.documents
        ]
        mutated = bundle.model_copy(
            update={
                "documents": shifted,
                "sources": bundle.sources,
                "intelligence": None,
            }
        )
        scoped, stats = time_scope_bundle(mutated, as_of=ANALYSIS)
        assert scoped is not mutated
        assert stats.dropped_documents == len(shifted)
        assert stats.blocked_lookahead is True
        assert scoped.documents == []

    def test_tie_at_analysis_instant_is_retained(self) -> None:
        from predictron_engine.dataset.prediction import time_scope_bundle

        bundle = _corpus_bundle()
        scoped, stats = time_scope_bundle(bundle, as_of=ANALYSIS)
        assert scoped is bundle
        assert stats.latest_retained_fetched_at == ANALYSIS

    def test_naive_datetime_is_normalized_to_utc(self) -> None:
        from predictron_engine.dataset.prediction import time_scope_bundle

        bundle = _corpus_bundle()
        scoped, stats = time_scope_bundle(bundle, as_of=datetime(2026, 1, 1))
        assert scoped is bundle
        assert stats.as_of == ANALYSIS


class TestScopedEvidence:
    def test_corpus_loaded_and_scoped(self) -> None:
        from predictron_engine.dataset.prediction import scoped_evidence

        bundle, stats = scoped_evidence("b2b_saas", as_of=ANALYSIS)
        assert len(bundle.documents) == stats.retained_documents
        assert stats.blocked_lookahead is False

    def test_missing_corpus_raises_scope_error(self) -> None:
        from predictron_engine.dataset.prediction import PredictionScopeError, scoped_evidence

        with pytest.raises(PredictionScopeError):
            scoped_evidence("no_such_corpus", as_of=ANALYSIS)


class TestManifest:
    def test_deterministic_bytes(self) -> None:
        first = manifest_bytes(_manifest())
        second = manifest_bytes(_manifest())
        assert first == second
        assert len(manifest_hash(_manifest())) == 64

    def test_dump_load_roundtrip(self, tmp_path: Path) -> None:
        manifest = _manifest()
        path = dump_manifest(manifest, tmp_path / "manifest.json")
        loaded = load_manifest(path)
        assert loaded == manifest

    def test_refuses_clobber(self, tmp_path: Path) -> None:
        from benchmarks.cohort.manifest import ManifestError

        path = dump_manifest(_manifest(), tmp_path / "manifest.json")
        with pytest.raises(ManifestError):
            dump_manifest(_manifest(), path)

    def test_rejects_naive_created_at(self) -> None:
        manifest = _manifest()
        naive = manifest.model_copy(update={"created_at": datetime(2026, 9, 20)})
        with pytest.raises(ValueError):
            CohortManifest.model_validate(naive.model_dump())

    def test_duplicate_company_id_rejected(self) -> None:
        manifest = _manifest()
        dup = manifest.source_records[0].model_copy(update={"company_id": "ai_infrastructure"})
        bad = manifest.model_copy(
            update={"source_records": [dup, manifest.source_records[1]]}
        )
        with pytest.raises(ValueError):
            CohortManifest.model_validate(bad.model_dump())


class TestRunnerGuard:
    def _entry(self, example_dataset):
        return next(e for e in example_dataset.entries if e.company_id == "b2b_saas")

    def test_unpinned_entry_replays_verbatim(self, example_dataset) -> None:
        from benchmarks.ground_truth_eval.runner import (
            build_entry_bundle,
            entry_evidence_scope,
        )

        entry = self._entry(example_dataset)
        assert entry.analysis_timestamp is None
        bundle = build_entry_bundle(entry)
        assert bundle.documents
        assert entry_evidence_scope(entry) is None

    def test_pinned_entry_is_time_scoped(self, example_dataset) -> None:
        from benchmarks.ground_truth_eval.runner import (
            build_entry_bundle,
            entry_evidence_scope,
        )

        entry = self._entry(example_dataset).model_copy(
            update={"analysis_timestamp": ANALYSIS}
        )
        bundle = build_entry_bundle(entry)
        stats = entry_evidence_scope(entry)
        assert stats is not None
        assert stats.blocked_lookahead is False
        assert stats.retained_documents == len(bundle.documents)

    def test_guard_blocks_future_evidence(self, example_dataset) -> None:
        from benchmarks.ground_truth_eval.runner import (
            build_entry_bundle,
            entry_evidence_scope,
        )

        entry = self._entry(example_dataset).model_copy(
            update={"analysis_timestamp": datetime(2025, 1, 1, tzinfo=UTC)}
        )
        stats = entry_evidence_scope(entry)
        assert stats is not None
        assert stats.blocked_lookahead is True
        assert build_entry_bundle(entry).documents == []


class TestCohortBuild:
    @pytest.fixture(autouse=True)
    def _real_engine(self):
        from predictron_engine.engine import PredictronEngine

        self._engine = PredictronEngine()

    def test_build_core_semantics(self) -> None:
        build = build_cohort_dataset(_manifest(), engine=self._engine)
        assert isinstance(build, CohortBuild)
        assert build.built_company_ids == ["ai_infrastructure", "b2b_saas"]
        assert build.pending_company_ids == ["mrr_only_derivation"]
        assert build.failed_company_ids == []
        assert build.lookahead_blocked_company_ids == []
        assert len(build.dataset.entries) == 2

    def test_entries_are_time_pinned(self) -> None:
        build = build_cohort_dataset(_manifest(), engine=self._engine)
        for entry in build.dataset.entries:
            assert entry.is_time_pinned is True
            assert entry.analysis_timestamp == ANALYSIS
            assert entry.historical_prediction.overall_score is not None
            assert "evidence_scope" in entry.metadata
            assert entry.metadata["manifest_hash"] == build.manifest_hash

    def test_provenance_flags(self) -> None:
        build = build_cohort_dataset(_manifest(), engine=self._engine)
        by_id = {e.company_id: e for e in build.dataset.entries}
        assert by_id["b2b_saas"].provenance.is_example is True
        assert by_id["ai_infrastructure"].provenance.is_example is False
        assert by_id["ai_infrastructure"].provenance.verified_by == "qa"

    def test_deterministic_rebuild(self) -> None:
        from benchmarks.ground_truth_eval.dataset import dataset_hash

        first = build_cohort_dataset(_manifest(), engine=self._engine)
        second = build_cohort_dataset(_manifest(), engine=self._engine)
        assert dataset_hash(first.dataset) == dataset_hash(second.dataset)

    def test_report_shape(self) -> None:
        report = build_cohort_dataset(_manifest(), engine=self._engine).to_report()
        assert report["status"] == "ok"
        assert report["entries_built"] == 2
        assert report["pending_ground_truth"] == ["mrr_only_derivation"]
        assert report["failed"] == []
        assert len(report["manifest_hash"]) == 64
        assert len(report["dataset_hash"]) == 64
        assert report["validation"]["error_count"] == 0
