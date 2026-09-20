"""Feature Registry.

Central catalog of all known feature definitions. Supports discovery,
dependency resolution, and deterministic ordering for computation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from predictron_engine.feature_store.models import (
    FeatureCategory,
    FeatureDefinition,
)

if TYPE_CHECKING:
    from predictron_engine.dataset.models import DatasetRecord

# Type alias for feature computation functions.
# Receives a dataset record, a map of dependency feature_id -> value,
# and an opaque context dict. Returns (value, evidence_references).
type FeatureComputer = Callable[
    ["DatasetRecord", dict[str, Any], dict[str, Any]],
    tuple[Any, list[dict[str, Any]]],
]


class FeatureRegistry:
    """Immutable registry of feature definitions and their computors.

    Supports:
    - Registration of feature definitions with metadata
    - Registration of computation functions
    - Dependency graph resolution
    - Deterministic topological ordering
    - Discovery by category, tag, or ID
    """

    def __init__(self) -> None:
        self._definitions: dict[str, FeatureDefinition] = {}
        self._computors: dict[str, FeatureComputer] = {}
        self._by_category: dict[FeatureCategory, list[str]] = defaultdict(list)
        self._by_tag: dict[str, list[str]] = defaultdict(list)

    # ---- Registration ----

    def register(
        self,
        definition: FeatureDefinition,
        computor: FeatureComputer,
    ) -> None:
        """Register a feature definition and its computation function."""
        if definition.feature_id in self._definitions:
            raise ValueError(
                f"Feature '{definition.feature_id}' already registered"
            )
        self._definitions[definition.feature_id] = definition
        self._computors[definition.feature_id] = computor
        self._by_category[definition.category].append(definition.feature_id)
        for tag in definition.tags:
            self._by_tag[tag].append(definition.feature_id)

    def register_all(
        self,
        definitions: list[tuple[FeatureDefinition, FeatureComputer]],
    ) -> None:
        """Register multiple features at once."""
        for definition, computor in definitions:
            self.register(definition, computor)

    # ---- Lookup ----

    def get(self, feature_id: str) -> FeatureDefinition | None:
        """Look up a feature definition by ID."""
        return self._definitions.get(feature_id)

    def get_computor(self, feature_id: str) -> FeatureComputer | None:
        """Look up a feature's computation function."""
        return self._computors.get(feature_id)

    def has(self, feature_id: str) -> bool:
        """Check if a feature is registered."""
        return feature_id in self._definitions

    def list_all(self) -> list[FeatureDefinition]:
        """All definitions in deterministic order."""
        return [self._definitions[fid] for fid in sorted(self._definitions)]

    def list_by_category(self, category: FeatureCategory) -> list[FeatureDefinition]:
        """All definitions in a category."""
        ids = sorted(self._by_category.get(category, []))
        return [self._definitions[fid] for fid in ids]

    def list_by_tag(self, tag: str) -> list[FeatureDefinition]:
        """All definitions with a given tag."""
        ids = sorted(self._by_tag.get(tag, []))
        return [self._definitions[fid] for fid in ids]

    def list_ids(self) -> list[str]:
        """Sorted list of all feature IDs."""
        return sorted(self._definitions.keys())

    def count(self) -> int:
        """Total registered features."""
        return len(self._definitions)

    # ---- Dependency Graph ----

    def dependency_graph(self) -> dict[str, list[str]]:
        """Map of feature_id -> [dependency feature_ids]."""
        return {
            fid: list(defn.dependencies)
            for fid, defn in sorted(self._definitions.items())
        }

    def all_dependencies(self, feature_id: str) -> list[str]:
        """Transitive dependency list in topological order.

        Includes the feature itself at the end.
        """
        visited: set[str] = set()
        result: list[str] = []

        def _visit(fid: str) -> None:
            if fid in visited:
                return
            visited.add(fid)
            defn = self._definitions.get(fid)
            if defn is not None:
                for dep in defn.dependencies:
                    _visit(dep)
            result.append(fid)

        _visit(feature_id)
        return result

    def topological_order(self) -> list[str]:
        """Deterministic topological ordering of all features.

        Features with no dependencies come first.
        """
        in_degree: dict[str, int] = {fid: 0 for fid in self._definitions}
        for fid, defn in self._definitions.items():
            for dep in defn.dependencies:
                if dep in in_degree:
                    in_degree[fid] += 1

        queue = sorted(
            [fid for fid, deg in in_degree.items() if deg == 0]
        )
        result: list[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for fid, defn in self._definitions.items():
                if node in defn.dependencies:
                    in_degree[fid] -= 1
                    if in_degree[fid] == 0:
                        queue.append(fid)
            queue.sort()

        return result

    def validate_dependencies(self) -> list[str]:
        """Check for broken dependency references.

        Returns list of error messages (empty = valid).
        """
        errors: list[str] = []
        for fid, defn in sorted(self._definitions.items()):
            for dep in defn.dependencies:
                if dep not in self._definitions:
                    errors.append(
                        f"Feature '{fid}' depends on unknown feature '{dep}'"
                    )
        return errors

    def missing_features(self, requested: list[str]) -> list[str]:
        """Return requested features not in the registry."""
        return sorted(set(requested) - set(self._definitions.keys()))

    def to_dict(self) -> dict[str, Any]:
        """Deterministic serialization of the full registry."""
        return {
            "feature_count": self.count(),
            "categories": {
                cat.value: sorted(ids)
                for cat, ids in sorted(self._by_category.items())
            },
            "features": {
                fid: {
                    "feature_id": defn.feature_id,
                    "feature_name": defn.feature_name,
                    "category": defn.category.value,
                    "description": defn.description,
                    "value_type": defn.value_type.value,
                    "computation_version": defn.computation_version,
                    "dependencies": defn.dependencies,
                    "source_fields": defn.source_fields,
                    "tags": defn.tags,
                }
                for fid, defn in sorted(self._definitions.items())
            },
        }
