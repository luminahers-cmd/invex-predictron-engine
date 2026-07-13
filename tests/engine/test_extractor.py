from predictron_engine.extraction.extractor import DefaultFeatureExtractor
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup

"""Tests for the DefaultFeatureExtractor."""


class TestDefaultFeatureExtractor:
    """Unit tests for the extraction stage."""

    def test_extract_returns_features(self, sample_startup, sample_collected_data):
        extractor = DefaultFeatureExtractor()
        result = extractor.extract(sample_startup, sample_collected_data)

        assert isinstance(result, ExtractedFeatures)

    def test_description_length_populated(
        self, sample_startup, sample_collected_data
    ):
        extractor = DefaultFeatureExtractor()
        result = extractor.extract(sample_startup, sample_collected_data)

        assert result.description_length == len(sample_startup.description)

    def test_has_pitch_deck_from_collected_data(
        self, sample_startup, sample_collected_data
    ):
        extractor = DefaultFeatureExtractor()
        result = extractor.extract(sample_startup, sample_collected_data)

        assert result.has_pitch_deck == sample_collected_data.has_pitch_deck

    def test_founder_profile_count_from_collected_data(
        self, sample_startup, sample_collected_data
    ):
        extractor = DefaultFeatureExtractor()
        result = extractor.extract(sample_startup, sample_collected_data)

        assert result.founder_profile_count == sample_collected_data.founder_count

    def test_keywords_extracted(self, sample_startup, sample_collected_data):
        extractor = DefaultFeatureExtractor()
        result = extractor.extract(sample_startup, sample_collected_data)

        assert isinstance(result.key_keywords, list)
        assert len(result.key_keywords) > 0

    def test_industry_classification_from_description(
        self, sample_collected_data
    ):
        startup = Startup(
            name="FintechCo",
            website="https://fintech.example.com",
            description="A fintech company providing payments and banking solutions.",
        )
        extractor = DefaultFeatureExtractor()
        result = extractor.extract(startup, sample_collected_data)

        assert result.industry is not None

    def test_business_model_classification(self, sample_collected_data):
        startup = Startup(
            name="SaaS",
            website="https://saas.example.com",
            description="An enterprise saas subscription platform for crm.",
        )
        extractor = DefaultFeatureExtractor()
        result = extractor.extract(startup, sample_collected_data)

        assert result.business_model == "saas"

    def test_data_completeness_non_negative(
        self, sample_startup, sample_collected_data
    ):
        extractor = DefaultFeatureExtractor()
        result = extractor.extract(sample_startup, sample_collected_data)

        assert 0.0 <= result.data_completeness <= 1.0
