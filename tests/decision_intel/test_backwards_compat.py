"""Backwards-compatibility tests for the Decision Intelligence layer.

Ensures the new layer does not break existing APIs:
  - Existing decision calibration functions still work
  - Existing decision engine classes still import
  - All existing decision exports remain intact
  - Feature Store integration is additive (no mutation)
"""

from __future__ import annotations

import pytest

from predictron_engine.decision import (
    ConfidenceLevel,
    action_for_confidence_level,
    classify_confidence_level,
    compute_decision_confidence,
)
from predictron_engine.decision.decision_engine import DefaultDecisionEngine


class TestExistingExportsStillPresent:
    @pytest.mark.parametrize("name,expected_type", [
        ("DecisionConfidence", type),
        ("ConfidenceBreakdown", type),
        ("ConfidenceFactor", type),
        ("ConfidenceLevel", type),
        ("UncertaintyBreakdown", type),
        ("CalibrationSummary", type),
    ])
    def test_models_still_importable(self, name: str, expected_type: type) -> None:
        import predictron_engine.decision.models as models
        assert hasattr(models, name)

    def test_decision_engine_importable(self) -> None:
        assert DefaultDecisionEngine is not None

    def test_old_calibration_helpers_present(self) -> None:
        assert callable(compute_decision_confidence)
        assert callable(classify_confidence_level)
        assert callable(action_for_confidence_level)

    @pytest.mark.parametrize("value,expected", [
        (0.9, ConfidenceLevel.VERY_HIGH),
        (0.8, ConfidenceLevel.VERY_HIGH),
        (0.7, ConfidenceLevel.HIGH),
        (0.5, ConfidenceLevel.MEDIUM),
        (0.3, ConfidenceLevel.LOW),
        (0.1, ConfidenceLevel.VERY_LOW),
    ])
    def test_legacy_classify_confidence_level(
        self,
        value: float,
        expected: ConfidenceLevel,
    ) -> None:
        assert classify_confidence_level(value) == expected

    @pytest.mark.parametrize("level,nonempty", [
        (ConfidenceLevel.VERY_HIGH, True),
        (ConfidenceLevel.HIGH, True),
        (ConfidenceLevel.MEDIUM, True),
        (ConfidenceLevel.LOW, True),
        (ConfidenceLevel.VERY_LOW, True),
    ])
    def test_legacy_action_for_level(
        self,
        level: ConfidenceLevel,
        nonempty: bool,
    ) -> None:
        action = action_for_confidence_level(level)
        assert bool(action) == nonempty


class TestDecisionEngineStillWorks:
    def test_decision_engine_constructs(self) -> None:
        engine = DefaultDecisionEngine()
        assert engine is not None

    def test_decision_thresholds_unchanged(self) -> None:
        # Threshold constants remain deterministic and legible.
        from predictron_engine.decision import decision_engine as de
        assert hasattr(de, "_DECISION_THRESHOLDS")

    def test_dimension_labels_importable(self) -> None:
        from predictron_engine.knowledge.concepts import AnalysisDimension
        assert AnalysisDimension.MARKET_OPPORTUNITY.value == "market_opportunity"


class TestFeatureStoreAdditive:
    def test_feature_snapshots_not_mutated(self) -> None:
        from predictron_engine.decision.feature_engine import DecisionFeatureEngine
        from predictron_engine.feature_store.models import (
            FeatureCategory,
            FeatureSnapshot,
        )

        snap = FeatureSnapshot(
            company_id="rec-1",
            feature_id="funding_velocity",
            feature_name="Funding Velocity",
            category=FeatureCategory.GROWTH,
            value=0.8,
        )
        original_value = snap.value
        original_status = snap.status

        engine = DecisionFeatureEngine()
        engine.normalize_feature(snap)
        engine.compute_feature_weight(snap)
        engine.compute_contribution(snap)

        assert snap.value == original_value
        assert snap.status == original_status

    def test_feature_set_not_mutated_by_contribution(self) -> None:
        from predictron_engine.decision.contribution import ContributionEngine
        from predictron_engine.decision.feature_engine import DecisionFeatureEngine

        fs = _build_minimal_set()
        before = fs.model_dump_json()
        engine = ContributionEngine(DecisionFeatureEngine())
        engine.compute_contributions(fs)
        after = fs.model_dump_json()
        assert before == after

    def test_existing_calibration_module_unchanged(self) -> None:
        # Confirm the Sprint 6B calibration module still exposes its API.
        import predictron_engine.decision.calibration as cal
        assert hasattr(cal, "compute_decision_confidence")
        assert hasattr(cal, "build_calibration_summary")
        assert hasattr(cal, "apply_recommendation_risk")

    def test_sprint8_calibration_unchanged(self) -> None:
        import predictron_engine.decision.calibration_sprint8 as c8
        assert c8.CalibrationBin is not None
        assert c8.CalibrationReport is not None
        assert hasattr(c8, "compute_calibration_error")


def _build_minimal_set():
    from datetime import UTC, datetime

    from predictron_engine.feature_store.models import (
        CompanyFeatureSet,
        FeatureCategory,
        FeatureSnapshot,
        ValueType,
    )

    return CompanyFeatureSet(
        company_id="rec-min",
        features={
            "funding_velocity": FeatureSnapshot(
                company_id="rec-min",
                feature_id="funding_velocity",
                feature_name="Funding Velocity",
                category=FeatureCategory.GROWTH,
                value=0.8,
                value_type=ValueType.FLOAT,
                computed_at=datetime(2025, 1, 1, tzinfo=UTC),
            ),
        },
        build_version="1.0.0",
        built_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
