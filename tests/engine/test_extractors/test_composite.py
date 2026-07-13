from predictron_engine.extraction.composite import CompositeExtractor
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup


class TestCompositeExtractor:
    def test_returns_extracted_features(self, sample_startup, sample_collected_data):
        result = CompositeExtractor().extract(sample_startup, sample_collected_data)
        assert isinstance(result, ExtractedFeatures)

    def test_merges_industry_from_market_extractor(self, sample_collected_data):
        startup = Startup(
            name="FinCo",
            website="https://finco.example.com",
            description="A fintech startup providing banking and payments solutions.",
        )
        result = CompositeExtractor().extract(startup, sample_collected_data)
        assert result.industry == "fintech"

    def test_merges_business_model(self, sample_collected_data):
        startup = Startup(
            name="SaaSCo",
            website="https://saasco.example.com",
            description="An enterprise saas subscription platform for crm.",
        )
        result = CompositeExtractor().extract(startup, sample_collected_data)
        assert result.business_model == "saas"

    def test_merges_funding_stage(self, sample_collected_data):
        startup = Startup(
            name="SeedCo",
            website="https://seedco.example.com",
            description="A seed-stage startup with a great team and an MVP.",
        )
        result = CompositeExtractor().extract(startup, sample_collected_data)
        assert result.funding_stage == "seed"

    def test_merges_description_length(self, sample_startup, sample_collected_data):
        result = CompositeExtractor().extract(sample_startup, sample_collected_data)
        assert result.description_length == len(sample_startup.description)

    def test_merges_keywords(self, sample_startup, sample_collected_data):
        result = CompositeExtractor().extract(sample_startup, sample_collected_data)
        assert isinstance(result.key_keywords, list)
        assert len(result.key_keywords) > 0

    def test_merges_pitch_deck_flag(self, sample_startup, sample_collected_data):
        result = CompositeExtractor().extract(sample_startup, sample_collected_data)
        assert result.has_pitch_deck == sample_collected_data.has_pitch_deck

    def test_merges_founder_count(self, sample_startup):
        data = CollectedData(
            startup_name="TestCo",
            founder_count=3,
        )
        result = CompositeExtractor().extract(sample_startup, data)
        assert result.founder_profile_count == 3

    def test_completeness_non_negative(self, sample_startup, sample_collected_data):
        result = CompositeExtractor().extract(sample_startup, sample_collected_data)
        assert 0.0 <= result.data_completeness <= 1.0

    def test_custom_extractors_replace_defaults(
        self, sample_startup, sample_collected_data
    ):
        class StubExtractor:
            def extract(
                self, startup: Startup, data: CollectedData
            ) -> ExtractedFeatures:
                return ExtractedFeatures(
                    industry="custom_industry", founded_year=2021
                )

        composite = CompositeExtractor(extractors=[StubExtractor()])
        result = composite.extract(sample_startup, sample_collected_data)
        assert result.industry == "custom_industry"
        assert result.founded_year == 2021

    def test_multiple_extractors_contribute(self, sample_collected_data):
        startup = Startup(
            name="FullCo",
            website="https://fullco.example.com",
            description=(
                "FullCo is a fintech saas startup founded in 2020 "
                "that provides subscription banking solutions. "
                "The company has a team of 12 people and raised a seed round."
            ),
        )
        result = CompositeExtractor().extract(startup, sample_collected_data)

        assert result.industry == "fintech"
        assert result.business_model == "saas"
        assert result.funding_stage == "seed"
        assert result.founded_year == 2020
        assert result.description_length > 0
        assert len(result.key_keywords) > 0

    def test_extractors_run_in_order(self, sample_startup, sample_collected_data):
        order: list[str] = []

        class TrackingExtractor:
            def __init__(self, name: str) -> None:
                self._name = name

            def extract(
                self, startup: Startup, data: CollectedData
            ) -> ExtractedFeatures:
                order.append(self._name)
                return ExtractedFeatures()

        extractors = [
            TrackingExtractor("first"),
            TrackingExtractor("second"),
            TrackingExtractor("third"),
        ]
        composite = CompositeExtractor(extractors=extractors)
        composite.extract(sample_startup, sample_collected_data)
        assert order == ["first", "second", "third"]

    def test_nlp_service_injected(self, sample_startup, sample_collected_data):
        class StubNlp:
            def extract_entities(self, text: str) -> list[str]:
                return []

            def extract_keywords(self, text: str, top_n: int = 10) -> list[str]:
                return ["stub_keyword"]

        composite = CompositeExtractor(nlp_service=StubNlp())
        result = composite.extract(sample_startup, sample_collected_data)
        assert "stub_keyword" in result.key_keywords
