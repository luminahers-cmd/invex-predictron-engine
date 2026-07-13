from predictron_engine.extraction.extractors.metadata import MetadataExtractor
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures


class TestMetadataExtractor:
    def test_returns_extracted_features(self, sample_startup, sample_collected_data):
        result = MetadataExtractor().extract(sample_startup, sample_collected_data)
        assert isinstance(result, ExtractedFeatures)

    def test_has_pitch_deck_from_data(self, sample_startup):
        data = CollectedData(
            startup_name="TestCo",
            has_pitch_deck=True,
        )
        result = MetadataExtractor().extract(sample_startup, data)
        assert result.has_pitch_deck is True

    def test_keywords_extracted(self, sample_startup, sample_collected_data):
        result = MetadataExtractor().extract(sample_startup, sample_collected_data)
        assert isinstance(result.key_keywords, list)
        assert len(result.key_keywords) > 0

    def test_completeness_with_all_signals(self, sample_startup):
        data = CollectedData(
            startup_name="FullCo",
            website_domain="fullco.example.com",
            description_tokens=["full", "co"],
            description_word_count=2,
            has_website=True,
            has_pitch_deck=True,
            founder_count=2,
            url_metadata={"website_url": "https://fullco.example.com"},
            enrichment_signals={"score": 0.9},
        )
        result = MetadataExtractor().extract(sample_startup, data)
        assert result.data_completeness == 1.0

    def test_completeness_with_no_signals(self, sample_startup):
        data = CollectedData(
            startup_name="EmptyCo",
            has_website=False,
        )
        result = MetadataExtractor().extract(sample_startup, data)
        assert result.data_completeness == 0.0

    def test_completeness_with_partial_signals(
        self, sample_startup, sample_collected_data
    ):
        result = MetadataExtractor().extract(sample_startup, sample_collected_data)
        assert 0.0 < result.data_completeness < 1.0
