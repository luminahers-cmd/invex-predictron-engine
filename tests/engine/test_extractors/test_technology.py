from predictron_engine.extraction.extractors.technology import TechnologyExtractor
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures


class TestTechnologyExtractor:
    def test_returns_extracted_features(self, sample_startup, sample_collected_data):
        result = TechnologyExtractor().extract(sample_startup, sample_collected_data)
        assert isinstance(result, ExtractedFeatures)

    def test_tech_from_domain(self, sample_startup):
        data = CollectedData(
            startup_name="ShopCo",
            website_domain="shopco.vercel.app",
        )
        result = TechnologyExtractor().extract(sample_startup, data)
        assert "vercel" in result.technology_stack

    def test_tech_from_enrichment_signals(self, sample_startup):
        data = CollectedData(
            startup_name="EnrichCo",
            enrichment_signals={"primary_tech": "python"},
        )
        result = TechnologyExtractor().extract(sample_startup, data)
        assert "python" in result.technology_stack

    def test_no_tech_signals(self, sample_startup, sample_collected_data):
        result = TechnologyExtractor().extract(sample_startup, sample_collected_data)
        assert result.technology_stack == []

    def test_deduplicates_tech_signals(self, sample_startup):
        data = CollectedData(
            startup_name="DupCo",
            website_domain="dupco.shopify.com",
            enrichment_signals={"stack": "shopify"},
        )
        result = TechnologyExtractor().extract(sample_startup, data)
        assert result.technology_stack.count("shopify") == 1
