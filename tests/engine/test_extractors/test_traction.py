from predictron_engine.extraction.extractors.traction import TractionExtractor
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup


class TestTractionExtractor:
    def test_returns_extracted_features(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert isinstance(result, ExtractedFeatures)

    def test_seed_stage_detected(self, sample_collected_data):
        startup = Startup(
            name="SeedCo",
            website="https://seedco.example.com",
            description="A seed-stage startup that recently raised an angel round.",
        )
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.funding_stage == "seed"

    def test_series_a_detected(self, sample_collected_data):
        startup = Startup(
            name="ACo",
            website="https://aco.example.com",
            description="A company that closed its series a round last quarter.",
        )
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.funding_stage == "series_a"

    def test_revenue_detected(self, sample_collected_data):
        startup = Startup(
            name="RevCo",
            website="https://revco.example.com",
            description="A profitable company with strong recurring revenue and MRR.",
        )
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.has_revenue is True

    def test_no_revenue_signal(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.has_revenue is None

    def test_no_stage_detected(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.funding_stage is None
