from predictron_engine.extraction.extractors.company import CompanyExtractor
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup


class TestCompanyExtractor:
    def test_returns_extracted_features(self, sample_startup, sample_collected_data):
        result = CompanyExtractor().extract(sample_startup, sample_collected_data)
        assert isinstance(result, ExtractedFeatures)

    def test_description_length(self, sample_startup, sample_collected_data):
        result = CompanyExtractor().extract(sample_startup, sample_collected_data)
        assert result.description_length == len(sample_startup.description)

    def test_founded_year_detected(self, sample_collected_data):
        startup = Startup(
            name="OldCo",
            website="https://oldco.example.com",
            description="OldCo was founded in 2015 and builds great products.",
        )
        result = CompanyExtractor().extract(startup, sample_collected_data)
        assert result.founded_year == 2015

    def test_founded_year_not_detected(self, sample_startup, sample_collected_data):
        result = CompanyExtractor().extract(sample_startup, sample_collected_data)
        assert result.founded_year is None

    def test_region_north_america(self, sample_collected_data):
        startup = Startup(
            name="USCo",
            website="https://usco.example.com",
            description="A San Francisco startup serving North American markets.",
        )
        result = CompanyExtractor().extract(startup, sample_collected_data)
        assert result.headquarters_region == "north_america"

    def test_region_europe(self, sample_collected_data):
        startup = Startup(
            name="EuroCo",
            website="https://euroco.example.com",
            description="A London-based company building for the European market.",
        )
        result = CompanyExtractor().extract(startup, sample_collected_data)
        assert result.headquarters_region == "europe"

    def test_region_not_detected(self, sample_startup, sample_collected_data):
        result = CompanyExtractor().extract(sample_startup, sample_collected_data)
        assert result.headquarters_region is None

    def test_only_populates_its_fields(self, sample_startup, sample_collected_data):
        result = CompanyExtractor().extract(sample_startup, sample_collected_data)
        assert result.industry is None
        assert result.business_model is None
        assert result.funding_stage is None
        assert result.technology_stack == []
