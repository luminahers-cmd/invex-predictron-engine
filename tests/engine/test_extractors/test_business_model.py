from predictron_engine.extraction.extractors.business_model import BusinessModelExtractor
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup


class TestBusinessModelExtractor:
    def test_returns_extracted_features(self, sample_startup, sample_collected_data):
        result = BusinessModelExtractor().extract(
            sample_startup, sample_collected_data
        )
        assert isinstance(result, ExtractedFeatures)

    def test_saas_classification(self, sample_collected_data):
        startup = Startup(
            name="SaaSCo",
            website="https://saasco.example.com",
            description="An enterprise saas platform with monthly subscription pricing.",
        )
        result = BusinessModelExtractor().extract(startup, sample_collected_data)
        assert result.business_model == "saas"

    def test_marketplace_classification(self, sample_collected_data):
        startup = Startup(
            name="MarketCo",
            website="https://marketco.example.com",
            description="A two-sided marketplace with platform fee and take rate model.",
        )
        result = BusinessModelExtractor().extract(startup, sample_collected_data)
        assert result.business_model == "marketplace"

    def test_customer_type_b2b(self, sample_collected_data):
        startup = Startup(
            name="BizCo",
            website="https://bizco.example.com",
            description="An enterprise B2B platform for businesses and organizations.",
        )
        result = BusinessModelExtractor().extract(startup, sample_collected_data)
        assert result.customer_type == "b2b"

    def test_customer_type_b2c(self, sample_collected_data):
        startup = Startup(
            name="ConsCo",
            website="https://consco.example.com",
            description="A consumer app for personal use by individual end users.",
        )
        result = BusinessModelExtractor().extract(startup, sample_collected_data)
        assert result.customer_type == "b2c"

    def test_no_match_returns_none(self, sample_startup, sample_collected_data):
        result = BusinessModelExtractor().extract(
            sample_startup, sample_collected_data
        )
        assert result.business_model is None
        assert result.customer_type is None
