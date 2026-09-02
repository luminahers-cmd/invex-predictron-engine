"""Engine integration pipeline (Part B).

Creates a deterministic analysis pipeline that:
  - loads a DatasetRecord,
  - executes the PredictronEngine on the startup,
  - extracts and stores a fresh PredictionSummary,
  - records analysis metadata,
  - never mutates the historical prediction on the DatasetRecord.

The historical prediction recorded on a DatasetRecord (whether imported
or manually entered) is immutable.  When the pipeline re-analyzes a
startup with the current engine, the fresh prediction is stored as a
separate ``AnalysisRun`` artifact so the original prediction is never
overwritten.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from predictron_engine.dataset.models import DatasetRecord, PredictionSummary


class AnalysisRun(BaseModel):
    """A fresh engine analysis of a DatasetRecord.

    Captures the current engine's prediction for a startup without
    mutating the historical prediction stored on the DatasetRecord.
    """

    run_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique analysis run identifier (UUIDv4)",
    )
    record_id: str = Field(
        ..., description="Dataset record this analysis belongs to"
    )
    prediction: PredictionSummary = Field(
        ..., description="Prediction produced by the current engine run"
    )
    engine_version: str = Field(
        ..., description="Engine version that produced this run"
    )
    benchmark_version: str | None = Field(
        default=None, description="Benchmark version used for replay, if any"
    )
    ran_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the engine was executed",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Engine wall-clock processing time in milliseconds",
    )
    pipeline_stages_completed: list[str] = Field(
        default_factory=list,
        description="Engine pipeline stages that completed",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata about this analysis run",
    )
    report_reference: str | None = Field(
        default=None,
        description=(
            "Optional reference to the full serialized Report for "
            "deep analysis (path or identifier)"
        ),
    )


class AnalysisResult(BaseModel):
    """Outcome of a single pipeline analysis."""

    run: AnalysisRun
    report_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Compact digest of the engine Report",
    )


class AnalysisPipeline:
    """Deterministic pipeline running the PredictronEngine on a DatasetRecord.

    The engine is injected via the constructor.  Without a supplied
    evidence bundle, the engine runs its normal (best-effort) evidence
    collection.  For fully deterministic offline replay, supply an
    ``EvidenceBundle`` via ``evidence_bundle``.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def analyze(
        self,
        record: DatasetRecord,
        benchmark_version: str | None = None,
        evidence_bundle: Any | None = None,
    ) -> AnalysisResult:
        """Run the engine on a DatasetRecord and return an AnalysisRun.

        The input record is never mutated.
        """
        from predictron_engine.dataset.analysis_support import build_request

        request = build_request(record)
        report = self._engine.analyze(
            request,
            evidence_bundle=evidence_bundle,
        )
        from predictron_engine.dataset.analysis_support import (
            extract_prediction_summary,
        )

        prediction = extract_prediction_summary(report)
        run = AnalysisRun(
            record_id=record.record_id,
            prediction=prediction,
            engine_version=_engine_version(report),
            benchmark_version=benchmark_version,
            processing_time_ms=_report_processing_ms(report),
            pipeline_stages_completed=_report_stages(report),
        )
        from predictron_engine.dataset.analysis_support import report_summary

        return AnalysisResult(
            run=run,
            report_summary=report_summary(report),
        )

    def analyze_all(
        self,
        records: list[DatasetRecord],
        benchmark_version: str | None = None,
        evidence_bundles: dict[str, Any] | None = None,
    ) -> list[AnalysisResult]:
        """Analyze multiple records, reusing a bundle map if supplied.

        ``evidence_bundles`` maps record_id to a pre-built EvidenceBundle
        for deterministic replay.  Missing bundles fall back to live
        evidence collection.
        """
        results: list[AnalysisResult] = []
        for record in records:
            bundle = None
            if evidence_bundles:
                bundle = evidence_bundles.get(record.record_id)
            results.append(
                self.analyze(
                    record,
                    benchmark_version=benchmark_version,
                    evidence_bundle=bundle,
                )
            )
        return results


def _engine_version(report: Any) -> str:
    metadata = getattr(report, "analysis_metadata", None)
    if metadata is not None:
        version = getattr(metadata, "engine_version", None)
        if version:
            return str(version)
    return "unknown"


def _report_processing_ms(report: Any) -> float:
    metadata = getattr(report, "analysis_metadata", None)
    if metadata is not None:
        value = getattr(metadata, "processing_time_ms", None)
        if value is not None:
            return float(value)
    return 0.0


def _report_stages(report: Any) -> list[str]:
    metadata = getattr(report, "analysis_metadata", None)
    if metadata is not None:
        stages = getattr(metadata, "pipeline_stages_completed", None)
        if isinstance(stages, list):
            return [str(s) for s in stages]
    return []
