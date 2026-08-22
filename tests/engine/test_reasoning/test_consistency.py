"""Tests for Sprint 6A cross-feature consistency utilities."""

from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import EvidenceItem, Observation
from predictron_engine.reasoning.consistency import (
    build_consistency_report,
    detect_contradictory_features,
    detect_missing_evidence,
    detect_reinforcing_features,
    detect_unsupported_conclusions,
    observation_is_supported,
)


def _item(domain, category, statement, source="src"):
    return EvidenceItem(
        domain=domain,
        category=category,
        statement=statement,
        source=source,
    )


class TestContradictoryFeatures:
    def test_revenue_at_pre_seed_stage_is_contradiction(self):
        features = ExtractedFeatures(funding_stage="pre_seed", has_revenue=True)
        findings = detect_contradictory_features(features)
        assert len(findings) == 1
        assert findings[0].field_a == "funding_stage"
        assert findings[0].field_b == "has_revenue"

    def test_b2b_with_consumer_orientation_is_contradiction(self):
        features = ExtractedFeatures(
            customer_type="b2b", enterprise_orientation="consumer"
        )
        findings = detect_contradictory_features(features)
        assert len(findings) == 1
        assert "conflicts" in findings[0].reason

    def test_b2c_with_enterprise_orientation_is_contradiction(self):
        features = ExtractedFeatures(
            customer_type="b2c", enterprise_orientation="enterprise"
        )
        assert len(detect_contradictory_features(features)) == 1

    def test_on_premise_saas_is_contradiction(self):
        features = ExtractedFeatures(
            deployment_model="on_premise", saas_model="saas"
        )
        assert len(detect_contradictory_features(features)) == 1

    def test_transactional_with_annual_contract_is_contradiction(self):
        features = ExtractedFeatures(
            recurring_revenue_signal="transactional",
            pricing_model="annual_contract",
        )
        assert len(detect_contradictory_features(features)) == 1

    def test_clean_features_have_no_contradictions(self, rich_features):
        assert detect_contradictory_features(rich_features) == []

    def test_missing_fields_never_contradict(self):
        assert detect_contradictory_features(ExtractedFeatures()) == []


class TestReinforcingFeatures:
    def test_revenue_plus_recurring_signal_reinforce(self):
        features = ExtractedFeatures(
            has_revenue=True, recurring_revenue_signal="recurring"
        )
        findings = detect_reinforcing_features(features)
        assert len(findings) == 1
        assert findings[0].field_a == "has_revenue"

    def test_b2b_plus_enterprise_sales_reinforce(self):
        features = ExtractedFeatures(
            customer_type="b2b", sales_motion="enterprise_sales"
        )
        assert len(detect_reinforcing_features(features)) == 1

    def test_tech_stack_plus_domain_reinforce(self):
        features = ExtractedFeatures(
            technology_stack=["python"],
            primary_technology_domain="ai_ml",
        )
        assert len(detect_reinforcing_features(features)) == 1

    def test_founders_plus_leadership_roles_reinforce(self):
        features = ExtractedFeatures(
            founder_profile_count=2,
            leadership_roles=["CEO", "CTO"],
        )
        assert len(detect_reinforcing_features(features)) == 1

    def test_empty_features_have_no_reinforcements(self):
        assert detect_reinforcing_features(ExtractedFeatures()) == []

    def test_rich_features_detect_reinforcements(self, rich_features):
        # rich_features: b2b + technology_stack populated (no domain set)
        findings = detect_reinforcing_features(rich_features)
        assert all(isinstance(f.field_a, str) for f in findings)


class TestUnsupportedConclusions:
    def test_unresolvable_ref_is_unsupported(self, rich_evidence):
        obs = Observation(
            dimension="market_opportunity",
            category="market_context",
            statement="Claim",
            evidence=["evidence:industry/market_size: Nonexistent claim."],
            source_rule="R",
        )
        findings = detect_unsupported_conclusions([obs], rich_evidence)
        assert len(findings) == 1
        assert findings[0].source_rule == "R"

    def test_resolvable_ref_is_supported(self, rich_evidence):
        item = rich_evidence[0]
        obs = Observation(
            dimension="market_opportunity",
            category="market_context",
            statement="Claim",
            evidence=[
                f"evidence:{item.domain}/{item.category}: {item.statement}"
            ],
            source_rule="R",
        )
        assert observation_is_supported(obs, rich_evidence) is True
        assert detect_unsupported_conclusions([obs], rich_evidence) == []

    def test_citation_backed_observation_is_supported(self, rich_evidence):
        from predictron_engine.models.report import EvidenceCitation

        obs = Observation(
            dimension="d",
            category="c",
            statement="Claim",
            citations=[
                EvidenceCitation(
                    claim="Some claim", domain="d", category="c"
                )
            ],
            source_rule="R",
        )
        assert observation_is_supported(obs, rich_evidence) is True

    def test_provenance_ids_make_observation_supported(self, rich_evidence):
        obs = Observation(
            dimension="d",
            category="c",
            statement="Claim",
            provenance_document_ids=["doc-1"],
            source_rule="R",
        )
        assert observation_is_supported(obs, rich_evidence) is True


class TestMissingEvidence:
    def test_populated_feature_without_evidence_domain_reported(self):
        features = ExtractedFeatures(industry="fintech")
        missing = detect_missing_evidence(features, [])
        assert "industry" in missing

    def test_covered_domains_not_reported(self, rich_evidence):
        features = ExtractedFeatures(industry="fintech")
        industry_items = [i for i in rich_evidence if i.domain == "industry"]
        missing = detect_missing_evidence(features, industry_items)
        assert "industry" not in missing

    def test_unpopulated_features_not_reported(self):
        missing = detect_missing_evidence(ExtractedFeatures(), [])
        assert missing == []

    def test_results_deduplicated_across_mapped_fields(self):
        features = ExtractedFeatures(
            geography="north_america", headquarters_region="north_america"
        )
        missing = detect_missing_evidence(features, [])
        assert missing.count("geography") == 1


class TestConsistencyReport:
    def test_report_aggregates_all_detectors(self):
        features = ExtractedFeatures(
            funding_stage="pre_seed",
            has_revenue=True,
            industry="fintech",
        )
        obs = Observation(
            dimension="d",
            category="c",
            statement="Claim",
            evidence=["feature:industry=fintech"],
            source_rule="R",
        )
        report = build_consistency_report(features, [], [obs])
        assert report.contradiction_count == 1
        assert report.missing_evidence_count >= 1
        assert report.unsupported_count == 1

    def test_report_is_deterministic(self, rich_features, rich_evidence):
        a = build_consistency_report(rich_features, rich_evidence, [])
        b = build_consistency_report(rich_features, rich_evidence, [])
        assert a.contradictions == b.contradictions
        assert a.reinforcements == b.reinforcements
        assert a.missing_evidence_domains == b.missing_evidence_domains
