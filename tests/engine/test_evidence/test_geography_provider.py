from predictron_engine.evidence.providers.geography_provider import (
    GeographyEvidenceProvider,
)


class TestGeographyEvidenceProvider:
    def test_returns_evidence_for_known_region(self, rich_features):
        provider = GeographyEvidenceProvider()
        items = provider.gather(rich_features)
        assert len(items) > 0
        assert all(item.domain == "geography" for item in items)

    def test_north_america(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        features = ExtractedFeatures(geography="north_america")
        provider = GeographyEvidenceProvider()
        items = provider.gather(features)
        assert len(items) > 0
        categories = {item.category for item in items}
        assert "market_size" in categories

    def test_falls_back_to_hq_region(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        features = ExtractedFeatures(headquarters_region="europe")
        provider = GeographyEvidenceProvider()
        items = provider.gather(features)
        assert len(items) > 0
        statements = [item.statement for item in items]
        assert any("gdpr" in s.lower() for s in statements)

    def test_returns_empty_for_no_region(self, minimal_features):
        provider = GeographyEvidenceProvider()
        items = provider.gather(minimal_features)
        assert items == []

    def test_unknown_region_returns_empty(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        features = ExtractedFeatures(geography="nonexistent_region")
        provider = GeographyEvidenceProvider()
        items = provider.gather(features)
        assert items == []

    def test_evidence_sources_populated(self, rich_features):
        provider = GeographyEvidenceProvider()
        items = provider.gather(rich_features)
        for item in items:
            assert item.source.startswith("knowledge/geography:")
