from predictron_engine.evidence.providers.stage_provider import StageEvidenceProvider


class TestStageEvidenceProvider:
    def test_returns_evidence_for_known_stage(self, rich_features):
        provider = StageEvidenceProvider()
        items = provider.gather(rich_features)
        assert len(items) > 0
        assert all(item.domain == "funding_stage" for item in items)

    def test_seed_stage(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        features = ExtractedFeatures(funding_stage="seed")
        provider = StageEvidenceProvider()
        items = provider.gather(features)
        assert len(items) > 0
        categories = {item.category for item in items}
        assert "expectations" in categories

    def test_returns_empty_for_no_stage(self, minimal_features):
        provider = StageEvidenceProvider()
        items = provider.gather(minimal_features)
        assert items == []

    def test_unknown_stage_returns_empty(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        features = ExtractedFeatures(funding_stage="nonexistent_stage")
        provider = StageEvidenceProvider()
        items = provider.gather(features)
        assert items == []

    def test_includes_stage_context(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        features = ExtractedFeatures(funding_stage="series_a")
        provider = StageEvidenceProvider()
        items = provider.gather(features)
        categories = {item.category for item in items}
        assert "context" in categories

    def test_series_a_stage(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        features = ExtractedFeatures(funding_stage="series_a")
        provider = StageEvidenceProvider()
        items = provider.gather(features)
        assert len(items) > 0
        statements = [item.statement for item in items]
        assert any("product-market fit" in s.lower() for s in statements)
