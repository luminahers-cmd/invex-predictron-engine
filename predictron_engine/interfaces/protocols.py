"""Protocol definitions for every pipeline stage.

These protocols define the contracts that all pipeline stage implementations
must satisfy. They use structural subtyping (Protocol) rather than nominal
inheritance (ABC), which means:

  - Implementations need not inherit from anything
  - Pydantic models, dataclasses, or plain classes all qualify
  - Third-party code can satisfy the protocol without modification
  - mypy verifies implementation correctness at check time
"""

from typing import Protocol, runtime_checkable

from app.schemas.analysis import StartupAnalysisRequest
from predictron_engine.evaluation.evaluation_models import DimensionAssessment, EvaluationResult
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import (
    ConfidenceAssessment,
    EvidenceItem,
    InvestmentDecision,
    InvestmentReadiness,
    Observation,
    Recommendation,
    Report,
    ScoreResult,
    SignalRelationship,
)
from predictron_engine.models.startup import Startup


@runtime_checkable
class Normalizer(Protocol):
    """Converts an API request into the internal Startup model."""

    def normalize(self, request: StartupAnalysisRequest) -> Startup: ...


@runtime_checkable
class DataCollector(Protocol):
    """Gathers and enriches data from the normalized startup."""

    def collect(self, startup: Startup) -> CollectedData: ...


@runtime_checkable
class FeatureExtractor(Protocol):
    """Derives structured factual features from collected data."""

    def extract(self, startup: Startup, data: CollectedData) -> ExtractedFeatures: ...


@runtime_checkable
class EvidenceEngine(Protocol):
    """Gathers contextual domain knowledge from extracted features."""

    def gather(self, features: ExtractedFeatures) -> object: ...


@runtime_checkable
class ReasoningEngine(Protocol):
    """Generates explainable observations from extracted features and evidence."""

    def reason(
        self, features: ExtractedFeatures, evidence: list[EvidenceItem]
    ) -> list[Observation]: ...


@runtime_checkable
class ScoringEngine(Protocol):
    """Assigns numerical scores to analysis dimensions."""

    def score(
        self, features: ExtractedFeatures, observations: list[Observation]
    ) -> list[ScoreResult]: ...


@runtime_checkable
class RecommendationEngine(Protocol):
    """Generates actionable recommendations from analysis results."""

    def recommend(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        scores: list[ScoreResult],
        assessments: list[DimensionAssessment] | None = ...,
    ) -> list[Recommendation]: ...


@runtime_checkable
class ConfidenceEngine(Protocol):
    """Assesses confidence in each scoring dimension."""

    def assess(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        scores: list[ScoreResult],
        assessments: list[DimensionAssessment] | None = ...,
    ) -> list[ConfidenceAssessment]: ...


@runtime_checkable
class ReportBuilder(Protocol):
    """Assembles all pipeline outputs into a single Report."""

    def build(
        self,
        startup: Startup,
        features: ExtractedFeatures,
        evidence: list[EvidenceItem],
        observations: list[Observation],
        scores: list[ScoreResult],
        recommendations: list[Recommendation],
        confidence: list[ConfidenceAssessment],
        dimension_assessments: list[DimensionAssessment] | None = ...,
        decision: InvestmentDecision | None = ...,
        investment_readiness: InvestmentReadiness | None = ...,
    ) -> Report: ...


@runtime_checkable
class DecisionEngine(Protocol):
    """Synthesizes all pipeline signals into an investment decision."""

    def decide(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        scores: list[ScoreResult],
        confidence: list[ConfidenceAssessment],
        assessments: list[DimensionAssessment] | None = ...,
        signal_relationships: list[SignalRelationship] | None = ...,
    ) -> InvestmentDecision: ...


@runtime_checkable
class EvaluationEngine(Protocol):
    """Evaluates observations and evidence into structured dimension assessments."""

    def evaluate(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        evidence: list[EvidenceItem],
    ) -> EvaluationResult: ...


@runtime_checkable
class DimensionEvaluator(Protocol):
    """Protocol for dimension-specific evaluators."""

    @property
    def dimension(self) -> str: ...

    def evaluate(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        evidence: list[EvidenceItem],
    ) -> DimensionAssessment: ...
