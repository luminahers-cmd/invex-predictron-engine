"""Feature Store canonical models.

Immutable, deterministic, explainable feature snapshots and definitions.
Every feature traces back to grounded data in the dataset.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class FeatureCategory(str, Enum):
    """Deterministic feature categories."""

    COMPANY = "company"
    GROWTH = "growth"
    FOUNDER = "founder"
    FUNDING = "funding"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    SIGNALS = "signals"
    BENCHMARK = "benchmark"


class ValueType(str, Enum):
    """Canonical value types for feature values."""

    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    STRING = "string"
    LIST = "list"
    DICT = "dict"
    NONE = "none"


class FeatureStatus(str, Enum):
    """Feature computation status."""

    COMPUTED = "computed"
    MISSING_DEPENDENCY = "missing_dependency"
    COMPUTATION_ERROR = "computation_error"
    STALE = "stale"


class FeatureDefinition(BaseModel):
    """Canonical feature definition in the registry.

    Immutable once registered. Describes what a feature is,
    how to compute it, and what it depends on.
    """

    feature_id: str = Field(..., description="Unique feature identifier")
    feature_name: str = Field(..., description="Human-readable feature name")
    category: FeatureCategory = Field(..., description="Feature category")
    description: str = Field(..., description="Detailed description")
    value_type: ValueType = Field(..., description="Expected value type")
    computation_version: str = Field(
        default="1.0.0", description="Computation function version"
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Feature IDs this feature depends on",
    )
    source_fields: list[str] = Field(
        default_factory=list,
        description="Dataset fields used in computation",
    )
    min_value: float | None = Field(
        default=None, description="Expected minimum value (for validation)"
    )
    max_value: float | None = Field(
        default=None, description="Expected maximum value (for validation)"
    )
    tags: list[str] = Field(
        default_factory=list, description="Searchable tags"
    )


class EvidenceReference(BaseModel):
    """Reference to the grounded data source of a feature value."""

    source_type: str = Field(
        ..., description="Type of source: record, outcome, signal, graph, etc."
    )
    source_id: str = Field(
        ..., description="Identifier of the specific source record"
    )
    source_field: str = Field(
        default="", description="Specific field within the source"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in this evidence"
    )


class FeatureSnapshot(BaseModel):
    """Immutable snapshot of a computed feature value.

    One feature per snapshot. Never mutated after creation.
    """

    snapshot_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique snapshot identifier",
    )
    company_id: str = Field(..., description="Company this feature is about")
    feature_id: str = Field(..., description="Feature definition ID")
    feature_name: str = Field(..., description="Feature name (denormalized)")
    category: FeatureCategory = Field(..., description="Feature category")
    value: Any = Field(default=None, description="Computed feature value")
    value_type: ValueType = Field(
        default=ValueType.NONE, description="Type of the value"
    )
    status: FeatureStatus = Field(
        default=FeatureStatus.COMPUTED, description="Computation status"
    )
    computation_version: str = Field(
        default="1.0.0", description="Version of computation used"
    )
    computed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of computation",
    )
    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="How this value was computed (deterministic trace)",
    )
    evidence_references: list[EvidenceReference] = Field(
        default_factory=list, description="Grounded data references"
    )
    dependencies_resolved: list[str] = Field(
        default_factory=list,
        description="Dependency feature IDs that were resolved",
    )
    error_message: str | None = Field(
        default=None, description="Error if computation failed"
    )

    def to_dict(self) -> dict[str, Any]:
        """Deterministic serialization."""
        return {
            "snapshot_id": self.snapshot_id,
            "company_id": self.company_id,
            "feature_id": self.feature_id,
            "feature_name": self.feature_name,
            "category": self.category.value,
            "value": self.value,
            "value_type": self.value_type.value,
            "status": self.status.value,
            "computation_version": self.computation_version,
            "computed_at": self.computed_at.isoformat(),
            "provenance": self.provenance,
            "evidence_references": [
                e.model_dump() for e in self.evidence_references
            ],
            "dependencies_resolved": self.dependencies_resolved,
            "error_message": self.error_message,
        }


class CompanyFeatureSet(BaseModel):
    """Complete feature set for a single company."""

    company_id: str = Field(..., description="Company identifier")
    features: dict[str, FeatureSnapshot] = Field(
        default_factory=dict, description="Features keyed by feature_id"
    )
    build_version: str = Field(
        default="1.0.0", description="Feature store build version"
    )
    built_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of build",
    )
    record_id: str | None = Field(
        default=None, description="Source dataset record ID"
    )

    def get_feature(self, feature_id: str) -> FeatureSnapshot | None:
        """Get a feature by ID."""
        return self.features.get(feature_id)

    def has_feature(self, feature_id: str) -> bool:
        """Check if a feature exists."""
        return feature_id in self.features

    def feature_count(self) -> int:
        """Number of computed features."""
        return len(self.features)

    def computed_features(self) -> list[FeatureSnapshot]:
        """All successfully computed features."""
        return [
            f for f in self.features.values()
            if f.status == FeatureStatus.COMPUTED
        ]

    def failed_features(self) -> list[FeatureSnapshot]:
        """All failed features."""
        return [
            f for f in self.features.values()
            if f.status != FeatureStatus.COMPUTED
        ]


class FeatureStoreSnapshot(BaseModel):
    """Snapshot of the entire feature store at a point in time."""

    snapshot_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique snapshot identifier",
    )
    store_version: str = Field(
        default="1.0.0", description="Feature store schema version"
    )
    engine_version: str = Field(
        default="0.12.1", description="Predictron engine version"
    )
    feature_version: str = Field(
        default="1.0.0", description="Feature computation version"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC creation timestamp",
    )
    company_count: int = Field(default=0, description="Number of companies")
    feature_count: int = Field(default=0, description="Total feature snapshots")
    companies: dict[str, CompanyFeatureSet] = Field(
        default_factory=dict, description="Feature sets by company ID"
    )

    @model_validator(mode="after")
    def _sync_counts(self) -> FeatureStoreSnapshot:
        self.company_count = len(self.companies)
        self.feature_count = sum(
            fs.feature_count() for fs in self.companies.values()
        )
        return self

    def get_company_features(self, company_id: str) -> CompanyFeatureSet | None:
        """Get a company's feature set."""
        return self.companies.get(company_id)

    def list_companies(self) -> list[str]:
        """Sorted list of company IDs."""
        return sorted(self.companies.keys())
