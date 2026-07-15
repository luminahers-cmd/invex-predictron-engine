"""Tests for CompetitionEvidenceProvider."""

from predictron_engine.evidence.evidence_models import EvidenceItem
from predictron_engine.evidence.providers.competition_provider import (
    CompetitionEvidenceProvider,
)
from predictron_engine.models.extracted_features import ExtractedFeatures


class TestCompetitionEvidenceProvider:
    def test_returns_list(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures(market_concentration="fragmented")
        result = provider.gather(features)
        assert isinstance(result, list)

    def test_returns_evidence_items(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures(market_concentration="fragmented")
        result = provider.gather(features)
        assert all(isinstance(item, EvidenceItem) for item in result)

    def test_domain_is_competition(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures(market_concentration="concentrated")
        result = provider.gather(features)
        for item in result:
            assert item.domain == "competition"

    def test_concentration_fragmented(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures(market_concentration="fragmented")
        result = provider.gather(features)
        cats = {item.category for item in result}
        assert "concentration" in cats
        assert "strategy" in cats

    def test_concentration_concentrated(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures(market_concentration="concentrated")
        result = provider.gather(features)
        statements = [item.statement for item in result]
        assert any("concentrated" in s.lower() for s in statements)

    def test_concentration_dominated(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures(market_concentration="dominated")
        result = provider.gather(features)
        cats = {item.category for item in result}
        assert "concentration" in cats
        assert "strategy" in cats

    def test_concentration_moderately_concentrated(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures(
            market_concentration="moderately_concentrated"
        )
        result = provider.gather(features)
        assert len(result) >= 1

    def test_density_sparse(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures(competitive_density="sparse")
        result = provider.gather(features)
        assert any(item.category == "density" for item in result)

    def test_density_moderate(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures(competitive_density="moderate")
        result = provider.gather(features)
        assert any(item.category == "density" for item in result)

    def test_density_dense(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures(competitive_density="dense")
        result = provider.gather(features)
        assert any(item.category == "density" for item in result)

    def test_density_hyper_competitive(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures(competitive_density="hyper_competitive")
        result = provider.gather(features)
        assert any(item.category == "density" for item in result)

    def test_network_effects_strong(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures(
            network_effect_competition="strong_network_effects"
        )
        result = provider.gather(features)
        assert any(item.category == "network_effects" for item in result)

    def test_network_effects_moderate(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures(
            network_effect_competition="moderate_network_effects"
        )
        result = provider.gather(features)
        assert any(item.category == "network_effects" for item in result)

    def test_network_effects_none(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures(
            network_effect_competition="no_network_effects"
        )
        result = provider.gather(features)
        assert any(item.category == "network_effects" for item in result)

    def test_barriers_to_entry(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures(
            barriers_to_entry=["regulatory_compliance", "capital_requirements"]
        )
        result = provider.gather(features)
        barrier_items = [
            item for item in result if item.category == "barriers_to_entry"
        ]
        assert len(barrier_items) == 2

    def test_barriers_to_entry_source(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures(barriers_to_entry=["high_capital"])
        result = provider.gather(features)
        barrier_items = [
            item for item in result if item.category == "barriers_to_entry"
        ]
        assert all("competition.py:barriers" in item.source for item in barrier_items)

    def test_moats(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures(
            competitive_moat_indicators=["proprietary_data", "patent_portfolio"]
        )
        result = provider.gather(features)
        moat_items = [
            item for item in result if item.category == "moat"
        ]
        assert len(moat_items) == 2

    def test_switching_costs(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures(
            switching_cost_signals=["integration_lock_in", "data_portability"]
        )
        result = provider.gather(features)
        sw_items = [
            item for item in result if item.category == "switching_costs"
        ]
        assert len(sw_items) == 2

    def test_multiple_signals_combined(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures(
            market_concentration="concentrated",
            competitive_density="dense",
            network_effect_competition="strong_network_effects",
            barriers_to_entry=["regulatory"],
            competitive_moat_indicators=["proprietary_data"],
            switching_cost_signals=["integration_lock_in"],
        )
        result = provider.gather(features)
        cats = {item.category for item in result}
        assert "concentration" in cats
        assert "density" in cats
        assert "network_effects" in cats
        assert "barriers_to_entry" in cats
        assert "moat" in cats
        assert "switching_costs" in cats

    def test_no_competition_fields_returns_empty(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures()
        result = provider.gather(features)
        assert result == []

    def test_non_features_returns_empty(self):
        provider = CompetitionEvidenceProvider()
        result = provider.gather("not features")
        assert result == []

    def test_none_returns_empty(self):
        provider = CompetitionEvidenceProvider()
        result = provider.gather(None)
        assert result == []

    def test_unknown_concentration_returns_no_concentration(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures(market_concentration="unknown")
        result = provider.gather(features)
        assert not any(item.category == "concentration" for item in result)

    def test_unknown_density_returns_no_density(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures(competitive_density="unknown")
        result = provider.gather(features)
        assert not any(item.category == "density" for item in result)

    def test_source_traceable(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures(
            market_concentration="fragmented",
            competitive_density="sparse",
        )
        result = provider.gather(features)
        for item in result:
            assert item.source.startswith("knowledge/competition.py:")

    def test_relevance_score_set(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures(market_concentration="fragmented")
        result = provider.gather(features)
        for item in result:
            assert item.relevance_score == 1.0

    def test_relevance_score_for_barriers(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures(barriers_to_entry=["regulatory"])
        result = provider.gather(features)
        for item in result:
            assert item.relevance_score == 0.8

    def test_empty_barriers_list(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures(barriers_to_entry=[])
        result = provider.gather(features)
        assert not any(item.category == "barriers_to_entry" for item in result)

    def test_empty_moats_list(self):
        provider = CompetitionEvidenceProvider()
        features = ExtractedFeatures(competitive_moat_indicators=[])
        result = provider.gather(features)
        assert not any(item.category == "moat" for item in result)
