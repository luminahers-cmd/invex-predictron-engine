"""Feature Engine.

Computes features deterministically with dependency resolution,
caching, and incremental recomputation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from predictron_engine.feature_store.models import (
    CompanyFeatureSet,
    EvidenceReference,
    FeatureDefinition,
    FeatureSnapshot,
    FeatureStatus,
    FeatureStoreSnapshot,
    ValueType,
)
from predictron_engine.feature_store.registry import FeatureRegistry
from predictron_engine.version import ENGINE_VERSION

STORE_VERSION = "1.0.0"


def _resolve_value_type(value: Any) -> ValueType:
    if value is None:
        return ValueType.NONE
    if isinstance(value, bool):
        return ValueType.BOOL
    if isinstance(value, int):
        return ValueType.INT
    if isinstance(value, float):
        return ValueType.FLOAT
    if isinstance(value, str):
        return ValueType.STRING
    if isinstance(value, list):
        return ValueType.LIST
    if isinstance(value, dict):
        return ValueType.DICT
    return ValueType.NONE


class FeatureEngine:
    """Deterministic feature computation engine.

    Responsibilities:
    - Compute features in dependency order
    - Cache reusable computations
    - Support incremental recomputation
    - Never modify original records
    """

    def __init__(self, registry: FeatureRegistry) -> None:
        self._registry = registry
        self._cache: dict[str, dict[str, FeatureSnapshot]] = {}
        self._build_version = STORE_VERSION

    @property
    def registry(self) -> FeatureRegistry:
        return self._registry

    def build_company_features(
        self,
        record: Any,
        *,
        timeline: Any = None,
        knowledge_graph: Any = None,
        graph_queries: Any = None,
        company_node_id: str | None = None,
        outcome: Any = None,
        benchmark_context: dict[str, Any] | None = None,
        as_of: datetime | None = None,
        record_id: str | None = None,
    ) -> CompanyFeatureSet:
        """Compute all features for a single company record.

        Features are computed in topological order. Dependencies are resolved
        automatically. Failed features are recorded with error info.
        """
        company_id = getattr(record, "record_id", None) or record_id or "unknown"
        ref_as_of = as_of or datetime.now(UTC)

        ctx: dict[str, Any] = {
            "as_of": ref_as_of,
            "timeline": timeline,
            "knowledge_graph": knowledge_graph,
            "graph_queries": graph_queries,
            "company_node_id": company_node_id,
            "company_id": company_id,
            "outcome": outcome,
            "benchmark_context": benchmark_context,
        }

        order = self._registry.topological_order()
        dep_values: dict[str, Any] = {}
        features: dict[str, FeatureSnapshot] = {}

        for feature_id in order:
            defn = self._registry.get(feature_id)
            computor = self._registry.get_computor(feature_id)
            if defn is None or computor is None:
                continue

            feature_deps = resolved_feature_deps(
                defn.dependencies, features
            )
            all_deps_met = all(
                fid in dep_values for fid in defn.dependencies
                if fid != feature_id
            )

            if not all_deps_met:
                snapshot = self._make_error_snapshot(
                    company_id, defn, "Missing dependency",
                    feature_deps,
                )
                features[feature_id] = snapshot
                continue

            try:
                value, evidence = computor(record, dep_values, ctx)
                snapshot = FeatureSnapshot(
                    company_id=company_id,
                    feature_id=defn.feature_id,
                    feature_name=defn.feature_name,
                    category=defn.category,
                    value=value,
                    value_type=_resolve_value_type(value),
                    status=FeatureStatus.COMPUTED,
                    computation_version=defn.computation_version,
                    computed_at=ref_as_of,
                    provenance={
                        "engine_version": ENGINE_VERSION,
                        "feature_version": self._build_version,
                        "computation_version": defn.computation_version,
                    },
                    evidence_references=[
                        EvidenceReference(
                            source_type=e.get("source_type", ""),
                            source_id=e.get("source_id", ""),
                            source_field=e.get("source_field", ""),
                        )
                        for e in evidence
                    ],
                    dependencies_resolved=feature_deps,
                )
                features[feature_id] = snapshot
                if value is not None:
                    dep_values[feature_id] = value
            except Exception as exc:
                snapshot = self._make_error_snapshot(
                    company_id, defn, str(exc), feature_deps,
                )
                features[feature_id] = snapshot

        return CompanyFeatureSet(
            company_id=company_id,
            features=features,
            build_version=self._build_version,
            built_at=ref_as_of,
            record_id=record.record_id if hasattr(record, "record_id") else record_id,
        )

    def build_feature_store(
        self,
        records: list[Any],
        **kwargs: Any,
    ) -> FeatureStoreSnapshot:
        """Build a complete feature store snapshot for multiple companies."""
        companies: dict[str, CompanyFeatureSet] = {}
        for record in records:
            company_id = getattr(record, "record_id", None) or "unknown"
            feature_set = self.build_company_features(record, **kwargs)
            companies[company_id] = feature_set

        total_features = sum(fs.feature_count() for fs in companies.values())
        return FeatureStoreSnapshot(
            store_version=STORE_VERSION,
            engine_version=ENGINE_VERSION,
            feature_version=self._build_version,
            created_at=datetime.now(UTC),
            company_count=len(companies),
            feature_count=total_features,
            companies=companies,
        )

    def recompute_feature(
        self,
        feature_id: str,
        company_id: str,
        existing_features: dict[str, FeatureSnapshot],
        record: Any,
        **kwargs: Any,
    ) -> FeatureSnapshot | None:
        """Incrementally recompute a single feature."""
        defn = self._registry.get(feature_id)
        computor = self._registry.get_computor(feature_id)
        if defn is None or computor is None:
            return None

        dep_values: dict[str, Any] = {}
        for fid in defn.dependencies:
            snap = existing_features.get(fid)
            if snap and snap.status == FeatureStatus.COMPUTED:
                dep_values[fid] = snap.value

        all_deps_met = all(fid in dep_values for fid in defn.dependencies)
        if not all_deps_met:
            return self._make_error_snapshot(
                company_id, defn, "Missing dependency", []
            )

        ref_as_of = kwargs.get("as_of", datetime.now(UTC))
        ctx: dict[str, Any] = {
            "as_of": ref_as_of,
            "timeline": kwargs.get("timeline"),
            "knowledge_graph": kwargs.get("knowledge_graph"),
            "graph_queries": kwargs.get("graph_queries"),
            "company_node_id": kwargs.get("company_node_id"),
            "company_id": company_id,
            "outcome": kwargs.get("outcome"),
            "benchmark_context": kwargs.get("benchmark_context"),
        }

        try:
            value, evidence = computor(record, dep_values, ctx)
            feature_deps = resolved_feature_deps(defn.dependencies, existing_features)
            return FeatureSnapshot(
                company_id=company_id,
                feature_id=defn.feature_id,
                feature_name=defn.feature_name,
                category=defn.category,
                value=value,
                value_type=_resolve_value_type(value),
                status=FeatureStatus.COMPUTED,
                computation_version=defn.computation_version,
                computed_at=ref_as_of,
                provenance={
                    "engine_version": ENGINE_VERSION,
                    "feature_version": self._build_version,
                    "computation_version": defn.computation_version,
                },
                evidence_references=[
                        EvidenceReference(
                            source_type=e.get("source_type", ""),
                            source_id=e.get("source_id", ""),
                            source_field=e.get("source_field", ""),
                        )
                        for e in evidence
                    ],
                dependencies_resolved=feature_deps,
            )
        except Exception as exc:
            return self._make_error_snapshot(
                company_id, defn, str(exc), [],
            )

    def _make_error_snapshot(
        self,
        company_id: str,
        defn: FeatureDefinition,
        error_msg: str,
        deps_resolved: list[str],
    ) -> FeatureSnapshot:
        missing_status = (
            FeatureStatus.MISSING_DEPENDENCY
            if "dependency" in error_msg.lower()
            else FeatureStatus.COMPUTATION_ERROR
        )
        return FeatureSnapshot(
            company_id=company_id,
            feature_id=defn.feature_id,
            feature_name=defn.feature_name,
            category=defn.category,
            value=None,
            value_type=ValueType.NONE,
            status=missing_status,
            computation_version=defn.computation_version,
            computed_at=datetime.now(UTC),
            error_message=error_msg,
            dependencies_resolved=deps_resolved,
        )


def resolved_feature_deps(
    dependencies: list[str],
    features: dict[str, FeatureSnapshot],
) -> list[str]:
    """Return the dependency IDs that are already resolved in the feature set."""
    return [fid for fid in dependencies if fid in features]
