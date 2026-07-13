from predictron_engine.evidence.evidence_models import EvidenceItem, EvidenceSet


class TestEvidenceItem:
    def test_creation(self):
        item = EvidenceItem(
            domain="industry",
            category="market_size",
            statement="Market is large.",
            source="knowledge/taxonomies.py:fintech",
        )
        assert item.domain == "industry"
        assert item.category == "market_size"
        assert item.relevance_score == 1.0

    def test_default_relevance(self):
        item = EvidenceItem(
            domain="test",
            category="test",
            statement="test",
            source="test",
        )
        assert item.relevance_score == 1.0

    def test_custom_relevance(self):
        item = EvidenceItem(
            domain="test",
            category="test",
            statement="test",
            source="test",
            relevance_score=0.5,
        )
        assert item.relevance_score == 0.5


class TestEvidenceSet:
    def test_empty_set(self):
        es = EvidenceSet()
        assert es.items == []
        assert es.provider_count == 0
        assert es.feature_coverage == 0.0

    def test_with_items(self):
        items = [
            EvidenceItem(
                domain="industry",
                category="test",
                statement="test",
                source="test",
            )
        ]
        es = EvidenceSet(items=items, provider_count=1, feature_coverage=0.5)
        assert len(es.items) == 1
        assert es.provider_count == 1
        assert es.feature_coverage == 0.5
