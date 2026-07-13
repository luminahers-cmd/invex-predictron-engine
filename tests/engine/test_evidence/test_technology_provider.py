from predictron_engine.evidence.providers.technology_provider import (
    TechnologyEvidenceProvider,
)


class TestTechnologyEvidenceProvider:
    def test_returns_evidence_for_known_tech(self, rich_features):
        provider = TechnologyEvidenceProvider()
        items = provider.gather(rich_features)
        assert len(items) > 0
        assert all(item.domain == "technology" for item in items)

    def test_python_evidence(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        features = ExtractedFeatures(technology_stack=["python"])
        provider = TechnologyEvidenceProvider()
        items = provider.gather(features)
        assert len(items) > 0
        categories = {item.category for item in items}
        assert "ecosystem" in categories

    def test_returns_empty_for_no_tech(self, minimal_features):
        provider = TechnologyEvidenceProvider()
        items = provider.gather(minimal_features)
        assert items == []

    def test_multiple_techs(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        features = ExtractedFeatures(technology_stack=["python", "aws", "react"])
        provider = TechnologyEvidenceProvider()
        items = provider.gather(features)
        assert len(items) >= 6
        sources = {item.source for item in items}
        assert len(sources) >= 3

    def test_unknown_tech_skipped(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        features = ExtractedFeatures(technology_stack=["unknown_tech_xyz"])
        provider = TechnologyEvidenceProvider()
        items = provider.gather(features)
        assert items == []

    def test_evidence_sources_populated(self, rich_features):
        provider = TechnologyEvidenceProvider()
        items = provider.gather(rich_features)
        for item in items:
            assert item.source.startswith("knowledge/tech:")
