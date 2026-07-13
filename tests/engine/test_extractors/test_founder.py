from predictron_engine.extraction.extractors.founder import FounderExtractor
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup


class TestFounderExtractor:
    def test_returns_extracted_features(self, sample_startup, sample_collected_data):
        result = FounderExtractor().extract(sample_startup, sample_collected_data)
        assert isinstance(result, ExtractedFeatures)

    def test_founder_count_from_collected_data(self, sample_startup):
        data = CollectedData(
            startup_name="TestCo",
            founder_count=3,
        )
        result = FounderExtractor().extract(sample_startup, data)
        assert result.founder_profile_count == 3

    def test_founder_count_zero(self, sample_startup, sample_collected_data):
        result = FounderExtractor().extract(sample_startup, sample_collected_data)
        assert result.founder_profile_count == 0

    def test_team_size_from_description(self, sample_collected_data):
        startup = Startup(
            name="TeamCo",
            website="https://teamco.example.com",
            description="A startup with a team of 15 people building great products.",
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert result.team_size_indicator == "11-50"

    def test_team_size_small(self, sample_collected_data):
        startup = Startup(
            name="SmallCo",
            website="https://smallco.example.com",
            description="A startup with a team of 5 people building great products.",
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert result.team_size_indicator == "1-10"

    def test_team_size_not_detected(self, sample_startup, sample_collected_data):
        result = FounderExtractor().extract(sample_startup, sample_collected_data)
        assert result.team_size_indicator is None
