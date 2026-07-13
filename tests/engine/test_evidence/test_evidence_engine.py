from predictron_engine.evidence.evidence_engine import DefaultEvidenceEngine
from predictron_engine.evidence.evidence_models import EvidenceSet


class TestDefaultEvidenceEngine:
    def test_returns_evidence_set(self, rich_features):
        engine = DefaultEvidenceEngine()
        result = engine.gather(rich_features)
        assert isinstance(result, EvidenceSet)

    def test_collects_items_from_providers(self, rich_features):
        engine = DefaultEvidenceEngine()
        result = engine.gather(rich_features)
        assert len(result.items) > 0

    def test_multiple_providers_contribute(self, rich_features):
        engine = DefaultEvidenceEngine()
        result = engine.gather(rich_features)
        assert result.provider_count >= 3

    def test_provider_count_matches_contributing(self, rich_features):
        engine = DefaultEvidenceEngine()
        result = engine.gather(rich_features)
        contributing_domains = {item.domain for item in result.items}
        assert result.provider_count == len(contributing_domains)

    def test_coverage_computed(self, rich_features):
        engine = DefaultEvidenceEngine()
        result = engine.gather(rich_features)
        assert result.feature_coverage > 0.0

    def test_minimal_features_low_coverage(self, minimal_features):
        engine = DefaultEvidenceEngine()
        result = engine.gather(minimal_features)
        assert result.feature_coverage == 0.0

    def test_minimal_features_fewer_items(self, minimal_features):
        engine = DefaultEvidenceEngine()
        result = engine.gather(minimal_features)
        assert len(result.items) == 0

    def test_all_items_are_evidence_items(self, rich_features):
        engine = DefaultEvidenceEngine()
        result = engine.gather(rich_features)
        for item in result.items:
            assert hasattr(item, "domain")
            assert hasattr(item, "category")
            assert hasattr(item, "statement")
            assert hasattr(item, "source")

    def test_custom_providers_injected(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        class StubProvider:
            def gather(self, features):
                from predictron_engine.evidence.evidence_models import EvidenceItem

                return [
                    EvidenceItem(
                        domain="stub",
                        category="test",
                        statement="Stub evidence.",
                        source="stub",
                    )
                ]

        engine = DefaultEvidenceEngine(providers=[StubProvider()])
        features = ExtractedFeatures(industry="fintech")
        result = engine.gather(features)
        assert len(result.items) == 1
        assert result.items[0].domain == "stub"
        assert result.provider_count == 1

    def test_empty_providers_returns_empty(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        engine = DefaultEvidenceEngine(providers=[])
        features = ExtractedFeatures(industry="fintech")
        result = engine.gather(features)
        assert len(result.items) == 0
        assert result.provider_count == 0

    def test_failed_provider_skipped(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        class FailingProvider:
            def gather(self, features):
                raise RuntimeError("Provider failed")

        class GoodProvider:
            def gather(self, features):
                from predictron_engine.evidence.evidence_models import EvidenceItem

                return [
                    EvidenceItem(
                        domain="good",
                        category="test",
                        statement="Good evidence.",
                        source="good",
                    )
                ]

        engine = DefaultEvidenceEngine(
            providers=[FailingProvider(), GoodProvider()]
        )
        features = ExtractedFeatures(industry="fintech")
        result = engine.gather(features)
        assert len(result.items) == 1
        assert result.provider_count == 1

    def test_saas_features_produce_comprehensive_evidence(self, saas_features):
        engine = DefaultEvidenceEngine()
        result = engine.gather(saas_features)
        domains = {item.domain for item in result.items}
        assert "industry" in domains
        assert "business_model" in domains
        assert "funding_stage" in domains
        assert "technology" in domains
        assert "geography" in domains

    def test_evidence_is_deterministic(self, rich_features):
        engine = DefaultEvidenceEngine()
        result_a = engine.gather(rich_features)
        result_b = engine.gather(rich_features)
        assert len(result_a.items) == len(result_b.items)
        for a, b in zip(result_a.items, result_b.items):
            assert a.statement == b.statement
            assert a.domain == b.domain
