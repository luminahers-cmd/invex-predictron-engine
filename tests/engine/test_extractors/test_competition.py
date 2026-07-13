from predictron_engine.extraction.extractors.competition import CompetitionExtractor
from predictron_engine.models.extracted_features import ExtractedFeatures


class TestCompetitionExtractor:
    def test_returns_extracted_features(self, sample_startup, sample_collected_data):
        result = CompetitionExtractor().extract(sample_startup, sample_collected_data)
        assert isinstance(result, ExtractedFeatures)

    def test_returns_default_features(self, sample_startup, sample_collected_data):
        result = CompetitionExtractor().extract(sample_startup, sample_collected_data)
        assert result.industry is None
        assert result.business_model is None
        assert result.funding_stage is None
        assert result.technology_stack == []
        assert result.key_keywords == []
