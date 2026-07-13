from predictron_engine.evidence.providers.industry_provider import (
    IndustryEvidenceProvider,
)


class TestIndustryEvidenceProvider:
    def test_returns_evidence_for_known_industry(self, rich_features):
        provider = IndustryEvidenceProvider()
        items = provider.gather(rich_features)
        assert len(items) > 0
        assert all(item.domain == "industry" for item in items)

    def test_fintech_industry(self, industry_only_features):
        provider = IndustryEvidenceProvider()
        items = provider.gather(industry_only_features)
        assert len(items) > 0
        categories = {item.category for item in items}
        assert "regulatory" in categories
        assert "market_size" in categories

    def test_returns_empty_for_no_industry(self, minimal_features):
        provider = IndustryEvidenceProvider()
        items = provider.gather(minimal_features)
        assert items == []

    def test_healthtech_industry(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        features = ExtractedFeatures(industry="healthtech")
        provider = IndustryEvidenceProvider()
        items = provider.gather(features)
        assert len(items) > 0
        categories = {item.category for item in items}
        assert "regulatory" in categories

    def test_unknown_industry_returns_empty(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        features = ExtractedFeatures(industry="nonexistent_industry")
        provider = IndustryEvidenceProvider()
        items = provider.gather(features)
        assert items == []

    def test_evidence_sources_populated(self, rich_features):
        provider = IndustryEvidenceProvider()
        items = provider.gather(rich_features)
        for item in items:
            assert item.source.startswith("knowledge/taxonomies.py:")

    def test_all_items_have_relevance(self, rich_features):
        provider = IndustryEvidenceProvider()
        items = provider.gather(rich_features)
        for item in items:
            assert 0.0 <= item.relevance_score <= 1.0
