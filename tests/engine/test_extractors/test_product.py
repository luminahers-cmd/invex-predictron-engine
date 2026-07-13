from predictron_engine.extraction.extractors.product import ProductExtractor
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup


class TestProductExtractor:
    def test_returns_extracted_features(self, sample_startup, sample_collected_data):
        result = ProductExtractor().extract(sample_startup, sample_collected_data)
        assert isinstance(result, ExtractedFeatures)

    def test_tech_stack_from_description(self, sample_collected_data):
        startup = Startup(
            name="TechCo",
            website="https://techco.example.com",
            description="Built with Python, React, and deployed on AWS using Docker.",
        )
        result = ProductExtractor().extract(startup, sample_collected_data)
        assert "python" in result.technology_stack
        assert "react" in result.technology_stack
        assert "aws" in result.technology_stack
        assert "docker" in result.technology_stack

    def test_no_tech_stack_when_none_mentioned(
        self, sample_startup, sample_collected_data
    ):
        result = ProductExtractor().extract(sample_startup, sample_collected_data)
        assert result.technology_stack == []

    def test_only_populates_its_fields(self, sample_startup, sample_collected_data):
        result = ProductExtractor().extract(sample_startup, sample_collected_data)
        assert result.industry is None
        assert result.business_model is None
        assert result.funding_stage is None
