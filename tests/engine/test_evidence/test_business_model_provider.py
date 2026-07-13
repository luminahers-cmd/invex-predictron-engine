from predictron_engine.evidence.providers.business_model_provider import (
    BusinessModelEvidenceProvider,
)


class TestBusinessModelEvidenceProvider:
    def test_returns_evidence_for_known_model(self, rich_features):
        provider = BusinessModelEvidenceProvider()
        items = provider.gather(rich_features)
        assert len(items) > 0
        assert all(item.domain == "business_model" for item in items)

    def test_saas_model(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        features = ExtractedFeatures(business_model="saas")
        provider = BusinessModelEvidenceProvider()
        items = provider.gather(features)
        assert len(items) > 0
        categories = {item.category for item in items}
        assert "unit_economics" in categories

    def test_returns_empty_for_no_model(self, minimal_features):
        provider = BusinessModelEvidenceProvider()
        items = provider.gather(minimal_features)
        assert items == []

    def test_unknown_model_returns_empty(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        features = ExtractedFeatures(business_model="nonexistent_model")
        provider = BusinessModelEvidenceProvider()
        items = provider.gather(features)
        assert items == []

    def test_evidence_sources_populated(self, rich_features):
        provider = BusinessModelEvidenceProvider()
        items = provider.gather(rich_features)
        for item in items:
            assert item.source.startswith("knowledge/taxonomies.py:")

    def test_marketplace_model(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        features = ExtractedFeatures(business_model="marketplace")
        provider = BusinessModelEvidenceProvider()
        items = provider.gather(features)
        assert len(items) > 0
        statements = [item.statement for item in items]
        assert any("network effects" in s.lower() for s in statements)
