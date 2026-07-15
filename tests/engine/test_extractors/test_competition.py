"""Comprehensive tests for CompetitionExtractor — Intelligence Sprint 7.

Covers 22 dimensions of competitive intelligence extraction:
  - Direct competitor signals
  - Indirect competitor signals
  - Incumbent signals
  - Market concentration
  - Competitive density
  - Fragmentation signals
  - Winner-take-most signals
  - Network effect competition
  - Switching cost signals
  - Differentiation signals
  - Competitive moat indicators
  - Barriers to entry
  - Substitute product signals
  - Platform dependency
  - Ecosystem dependency
  - Open-source competition
  - Regulatory competition
  - Geographic competition
  - Pricing pressure
  - Competitive keywords
  - Competition confidence
"""

from __future__ import annotations

from predictron_engine.extraction.extractors.competition import (
    CompetitionExtractor,
)
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_startup(desc: str) -> Startup:
    return Startup(
        name="TestCo",
        website="https://testco.example.com",
        description=desc,
        pitch_deck_url=None,
        founder_linkedin_urls=[],
        raw_data={},
    )


def _make_data() -> CollectedData:
    return CollectedData(
        startup_name="TestCo",
        website_domain="testco.example.com",
        description_tokens=[],
        description_word_count=0,
        has_website=True,
        has_pitch_deck=False,
        founder_count=0,
        url_metadata={},
        enrichment_signals={},
    )


def _extract(desc: str) -> ExtractedFeatures:
    return CompetitionExtractor().extract(_make_startup(desc), _make_data())


# ---------------------------------------------------------------------------
# Basic sanity
# ---------------------------------------------------------------------------


class TestCompetitionExtractorBasic:
    def test_returns_extracted_features(self):
        result = _extract("A B2B SaaS platform.")
        assert isinstance(result, ExtractedFeatures)

    def test_returns_default_features_for_empty_description(self):
        result = _extract("A company.")
        assert result.direct_competitor_signals == []
        assert result.indirect_competitor_signals == []
        assert result.incumbent_signals == []
        assert result.market_concentration is None
        assert result.competitive_density is None
        assert result.fragmentation_signals == []
        assert result.winner_take_most_signals == []
        assert result.network_effect_competition is None
        assert result.switching_cost_signals == []
        assert result.differentiation_signals == []
        assert result.competitive_moat_indicators == []
        assert result.barriers_to_entry == []
        assert result.substitute_product_signals == []
        assert result.platform_dependency == []
        assert result.ecosystem_dependency == []
        assert result.open_source_competition == []
        assert result.regulatory_competition == []
        assert result.geographic_competition == []
        assert result.pricing_pressure == []
        assert result.competitive_keywords == []
        assert result.competition_confidence == 0.0

    def test_does_not_touch_non_competition_fields(self):
        result = _extract("A B2B SaaS platform for analytics.")
        assert result.industry is None
        assert result.business_model is None
        assert result.funding_stage is None
        assert result.technology_stack == []
        assert result.key_keywords == []


# ---------------------------------------------------------------------------
# Direct competitor signals
# ---------------------------------------------------------------------------


class TestDirectCompetitorSignals:
    def test_competitor_mention(self):
        result = _extract("We compete with established players in the market.")
        assert len(result.direct_competitor_signals) > 0

    def test_versus(self):
        result = _extract("A modern solution versus legacy platforms.")
        assert len(result.direct_competitor_signals) > 0

    def test_alternatives_to(self):
        result = _extract(
            "A faster alternative to traditional enterprise software."
        )
        assert len(result.direct_competitor_signals) > 0

    def test_competing_with(self):
        result = _extract("Competing with major cloud providers.")
        assert len(result.direct_competitor_signals) > 0

    def test_rival(self):
        result = _extract("We outperform our rivals in speed.")
        assert len(result.direct_competitor_signals) > 0

    def test_head_to_head(self):
        result = _extract("Head to head comparison with market leaders.")
        assert len(result.direct_competitor_signals) > 0

    def test_market_leader(self):
        result = _extract("Taking share from the market leader.")
        assert len(result.direct_competitor_signals) > 0

    def test_no_competitor_signals(self):
        result = _extract("A great product for everyone.")
        assert result.direct_competitor_signals == []


# ---------------------------------------------------------------------------
# Indirect competitor signals
# ---------------------------------------------------------------------------


class TestIndirectCompetitorSignals:
    def test_adjacent_market(self):
        result = _extract(
            "We also compete in the adjacent market space."
        )
        assert len(result.indirect_competitor_signals) > 0

    def test_traditional_solution(self):
        result = _extract(
            "Replacing the traditional solution used by most companies."
        )
        assert len(result.indirect_competitor_signals) > 0

    def test_legacy_solution(self):
        result = _extract(
            "Our legacy system approach is outdated."
        )
        assert len(result.indirect_competitor_signals) > 0

    def test_spreadsheets(self):
        result = _extract("Teams still using spreadsheets for tracking.")
        assert len(result.indirect_competitor_signals) > 0

    def test_manual_process(self):
        result = _extract(
            "Eliminating the manual process of data entry."
        )
        assert len(result.indirect_competitor_signals) > 0

    def test_homegrown(self):
        result = _extract(
            "Replacing homegrown solutions with our platform."
        )
        assert len(result.indirect_competitor_signals) > 0

    def test_point_solution(self):
        result = _extract(
            "Our platform replaces multiple point solutions."
        )
        assert len(result.indirect_competitor_signals) > 0


# ---------------------------------------------------------------------------
# Incumbent signals
# ---------------------------------------------------------------------------


class TestIncumbentSignals:
    def test_disrupting_incumbent(self):
        result = _extract(
            "Disrupting the incumbent players in the industry."
        )
        assert len(result.incumbent_signals) > 0

    def test_replacing_legacy(self):
        result = _extract(
            "Replacing legacy systems across enterprises."
        )
        assert len(result.incumbent_signals) > 0

    def test_legacy_system(self):
        result = _extract(
            "The legacy system cannot handle modern demands."
        )
        assert len(result.incumbent_signals) > 0

    def test_outdated_approach(self):
        result = _extract(
            "An outdated approach that wastes engineering time."
        )
        assert len(result.incumbent_signals) > 0

    def test_inefficient_process(self):
        result = _extract(
            "The inefficient process costs companies millions."
        )
        assert len(result.incumbent_signals) > 0

    def test_modern_alternative(self):
        result = _extract(
            "A modern alternative to old legacy tools."
        )
        assert len(result.incumbent_signals) > 0


# ---------------------------------------------------------------------------
# Market concentration
# ---------------------------------------------------------------------------


class TestMarketConcentration:
    def test_fragmented(self):
        result = _extract(
            "The market is fragmented with many small players "
            "and no clear leader."
        )
        assert result.market_concentration == "fragmented"

    def test_moderately_concentrated(self):
        result = _extract(
            "A few major players dominate with several tiers."
        )
        assert result.market_concentration == "moderately_concentrated"

    def test_concentrated(self):
        result = _extract(
            "The market is dominated by two main players."
        )
        assert result.market_concentration == "concentrated"

    def test_dominated(self):
        result = _extract(
            "A monopoly controlled by one company."
        )
        assert result.market_concentration == "dominated"

    def test_unknown_concentration(self):
        result = _extract("A cloud platform for analytics.")
        assert result.market_concentration is None


# ---------------------------------------------------------------------------
# Competitive density
# ---------------------------------------------------------------------------


class TestCompetitiveDensity:
    def test_sparse(self):
        result = _extract(
            "A blue ocean opportunity with no competitors."
        )
        assert result.competitive_density == "sparse"

    def test_dense(self):
        result = _extract(
            "Operating in a highly competitive and intense market."
        )
        assert result.competitive_density == "dense"

    def test_hyper_competitive(self):
        result = _extract(
            "An extremely crowded and cutthroat market."
        )
        assert result.competitive_density == "hyper_competitive"

    def test_unknown_density(self):
        result = _extract("A cloud platform for analytics.")
        assert result.competitive_density is None


# ---------------------------------------------------------------------------
# Fragmentation signals
# ---------------------------------------------------------------------------


class TestFragmentationSignals:
    def test_fragmented_space(self):
        result = _extract(
            "This fragmented space has many small players."
        )
        assert len(result.fragmentation_signals) > 0

    def test_no_clear_winner(self):
        result = _extract("No clear winner has emerged yet.")
        assert len(result.fragmentation_signals) > 0

    def test_consolidation_opportunity(self):
        result = _extract(
            "A consolidation opportunity in this market."
        )
        assert len(result.fragmentation_signals) > 0

    def test_no_fragmentation(self):
        result = _extract("A cloud platform for analytics.")
        assert result.fragmentation_signals == []


# ---------------------------------------------------------------------------
# Winner-take-most signals
# ---------------------------------------------------------------------------


class TestWinnerTakeMostSignals:
    def test_winner_take_all(self):
        result = _extract(
            "This is a winner-take-all market."
        )
        assert len(result.winner_take_most_signals) > 0

    def test_winner_take_most(self):
        result = _extract(
            "Winner-take-most dynamics apply here."
        )
        assert len(result.winner_take_most_signals) > 0

    def test_flywheel_effect(self):
        result = _extract(
            "The flywheel effect creates winner dynamics."
        )
        assert len(result.winner_take_most_signals) > 0

    def test_scale_advantage(self):
        result = _extract(
            "Scale advantage drives competitive outcomes."
        )
        assert len(result.winner_take_most_signals) > 0

    def test_critical_mass(self):
        result = _extract(
            "Achieving critical mass before competitors."
        )
        assert len(result.winner_take_most_signals) > 0

    def test_no_winner_take_most(self):
        result = _extract("A cloud platform for analytics.")
        assert result.winner_take_most_signals == []


# ---------------------------------------------------------------------------
# Network effect competition
# ---------------------------------------------------------------------------


class TestNetworkEffectCompetition:
    def test_strong_network_effects(self):
        result = _extract(
            "Strong network effects create a durable moat. "
            "The network effect drives viral growth."
        )
        assert result.network_effect_competition == "strong_network_effects"

    def test_moderate_network_effects(self):
        result = _extract(
            "The ecosystem and community drive engagement."
        )
        assert result.network_effect_competition == "moderate_network_effects"

    def test_no_network_effects(self):
        result = _extract(
            "A standalone single player tool for individual use."
        )
        assert result.network_effect_competition == "no_network_effects"

    def test_unknown_network_effects(self):
        result = _extract("A cloud platform for analytics.")
        assert result.network_effect_competition is None


# ---------------------------------------------------------------------------
# Switching cost signals
# ---------------------------------------------------------------------------


class TestSwitchingCostSignals:
    def test_high_switching_cost(self):
        result = _extract(
            "High switching costs protect our installed base."
        )
        assert len(result.switching_cost_signals) > 0

    def test_vendor_lock_in(self):
        result = _extract(
            "Deep integration creates vendor lock-in."
        )
        assert len(result.switching_cost_signals) > 0

    def test_deep_integration(self):
        result = _extract(
            "Deeply integrated into customer workflows."
        )
        assert len(result.switching_cost_signals) > 0

    def test_multi_year_contract(self):
        result = _extract(
            "Multi-year contracts ensure revenue stability."
        )
        assert len(result.switching_cost_signals) > 0

    def test_no_switching_costs(self):
        result = _extract("A cloud platform for analytics.")
        assert result.switching_cost_signals == []


# ---------------------------------------------------------------------------
# Differentiation signals
# ---------------------------------------------------------------------------


class TestDifferentiationSignals:
    def test_proprietary_technology(self):
        result = _extract(
            "Our proprietary technology sets us apart."
        )
        assert len(result.differentiation_signals) > 0

    def test_unique_approach(self):
        result = _extract(
            "A unique approach to data processing."
        )
        assert len(result.differentiation_signals) > 0

    def test_patent_protection(self):
        result = _extract(
            "Protected by 5 patents on our core algorithm."
        )
        assert len(result.differentiation_signals) > 0

    def test_ai_differentiation(self):
        result = _extract(
            "AI-powered insights that competitors cannot match."
        )
        assert len(result.differentiation_signals) > 0

    def test_purpose_built(self):
        result = _extract(
            "A purpose-built platform for healthcare."
        )
        assert len(result.differentiation_signals) > 0

    def test_differentiated(self):
        result = _extract(
            "Our differentiated solution outperforms alternatives."
        )
        assert len(result.differentiation_signals) > 0

    def test_no_differentiation(self):
        result = _extract("A cloud platform for analytics.")
        assert result.differentiation_signals == []


# ---------------------------------------------------------------------------
# Competitive moat indicators
# ---------------------------------------------------------------------------


class TestCompetitiveMoatIndicators:
    def test_data_moat(self):
        result = _extract(
            "Our data moat grows with every customer."
        )
        assert len(result.competitive_moat_indicators) > 0

    def test_network_effect_moat(self):
        result = _extract(
            "Network effects create a defensible moat."
        )
        assert len(result.competitive_moat_indicators) > 0

    def test_regulatory_moat(self):
        result = _extract(
            "Regulatory approval provides a competitive advantage."
        )
        assert len(result.competitive_moat_indicators) > 0

    def test_scale_moat(self):
        result = _extract(
            "Scale economies create a durable advantage."
        )
        assert len(result.competitive_moat_indicators) > 0

    def test_hard_to_replicate(self):
        result = _extract(
            "Our technology is hard to replicate."
        )
        assert len(result.competitive_moat_indicators) > 0

    def test_compounding_advantage(self):
        result = _extract(
            "Compounding advantage grows over time."
        )
        assert len(result.competitive_moat_indicators) > 0

    def test_no_moat(self):
        result = _extract("A cloud platform for analytics.")
        assert result.competitive_moat_indicators == []


# ---------------------------------------------------------------------------
# Barriers to entry
# ---------------------------------------------------------------------------


class TestBarriersToEntry:
    def test_regulatory_barrier(self):
        result = _extract(
            "Regulatory barriers protect our position."
        )
        assert len(result.barriers_to_entry) > 0

    def test_fda_approval(self):
        result = _extract(
            "FDA-cleared medical device technology."
        )
        assert len(result.barriers_to_entry) > 0

    def test_compliance_certification(self):
        result = _extract(
            "SOC 2 and HIPAA compliance required."
        )
        assert len(result.barriers_to_entry) > 0

    def test_capital_intensity(self):
        result = _extract(
            "Capital-intensive R&D requires significant funding."
        )
        assert len(result.barriers_to_entry) > 0

    def test_technical_complexity(self):
        result = _extract(
            "Deep technical complexity in our core engine."
        )
        assert len(result.barriers_to_entry) > 0

    def test_ip_barrier(self):
        result = _extract(
            "Patent-protected technology with strong IP."
        )
        assert len(result.barriers_to_entry) > 0

    def test_time_to_build(self):
        result = _extract(
            "Years to build this level of capability."
        )
        assert len(result.barriers_to_entry) > 0

    def test_no_barriers(self):
        result = _extract("A cloud platform for analytics.")
        assert result.barriers_to_entry == []


# ---------------------------------------------------------------------------
# Substitute product signals
# ---------------------------------------------------------------------------


class TestSubstituteProductSignals:
    def test_replaces_spreadsheets(self):
        result = _extract(
            "Replaces spreadsheets for team collaboration."
        )
        assert len(result.substitute_product_signals) > 0

    def test_replaces_email(self):
        result = _extract(
            "Replaces email for project management."
        )
        assert len(result.substitute_product_signals) > 0

    def test_replaces_manual(self):
        result = _extract(
            "Replaces manual data entry workflows."
        )
        assert len(result.substitute_product_signals) > 0

    def test_alternative_to(self):
        result = _extract(
            "A better alternative to existing tools."
        )
        assert len(result.substitute_product_signals) > 0

    def test_instead_of(self):
        result = _extract(
            "Instead of legacy tools, use our platform."
        )
        assert len(result.substitute_product_signals) > 0

    def test_no_substitutes(self):
        result = _extract("A cloud platform for analytics.")
        assert result.substitute_product_signals == []


# ---------------------------------------------------------------------------
# Platform dependency
# ---------------------------------------------------------------------------


class TestPlatformDependency:
    def test_aws_dependency(self):
        result = _extract(
            "Built on AWS with deep cloud integration."
        )
        assert len(result.platform_dependency) > 0

    def test_gcp_dependency(self):
        result = _extract(
            "GCP-native platform leveraging Google services."
        )
        assert len(result.platform_dependency) > 0

    def test_azure_dependency(self):
        result = _extract(
            "Azure-dependent deployment for enterprise clients."
        )
        assert len(result.platform_dependency) > 0

    def test_shopify_dependency(self):
        result = _extract(
            "Shopify app partner serving e-commerce merchants."
        )
        assert len(result.platform_dependency) > 0

    def test_salesforce_dependency(self):
        result = _extract(
            "Salesforce-built integration for CRM workflows."
        )
        assert len(result.platform_dependency) > 0

    def test_no_platform_dependency(self):
        result = _extract("A cloud platform for analytics.")
        assert result.platform_dependency == []


# ---------------------------------------------------------------------------
# Ecosystem dependency
# ---------------------------------------------------------------------------


class TestEcosystemDependency:
    def test_ecosystem_play(self):
        result = _extract(
            "An ecosystem play within the developer community."
        )
        assert len(result.ecosystem_dependency) > 0

    def test_platform_ecosystem(self):
        result = _extract(
            "Built within the platform ecosystem."
        )
        assert len(result.ecosystem_dependency) > 0

    def test_integrates_with(self):
        result = _extract(
            "Integrates with 50+ tools and platforms."
        )
        assert len(result.ecosystem_dependency) > 0

    def test_compatible_with(self):
        result = _extract(
            "Compatible with major cloud providers."
        )
        assert len(result.ecosystem_dependency) > 0

    def test_connector(self):
        result = _extract(
            "A connector for enterprise data pipelines."
        )
        assert len(result.ecosystem_dependency) > 0

    def test_no_ecosystem_dependency(self):
        result = _extract("A cloud platform for analytics.")
        assert result.ecosystem_dependency == []


# ---------------------------------------------------------------------------
# Open-source competition
# ---------------------------------------------------------------------------


class TestOpenSourceCompetition:
    def test_open_source_alternative(self):
        result = _extract(
            "Competing with open-source alternatives."
        )
        assert len(result.open_source_competition) > 0

    def test_github_stars(self):
        result = _extract(
            "The open-source version has 14,000 GitHub stars."
        )
        assert len(result.open_source_competition) > 0

    def test_community_edition(self):
        result = _extract(
            "Free community edition with premium features."
        )
        assert len(result.open_source_competition) > 0

    def test_open_source_core(self):
        result = _extract(
            "Open-source-first with commercial features."
        )
        assert len(result.open_source_competition) > 0

    def test_open_source_moat(self):
        result = _extract(
            "Open-source strategy creates a competitive moat."
        )
        assert len(result.open_source_competition) > 0

    def test_contributing_devs(self):
        result = _extract(
            "850 contributing developers on our platform."
        )
        assert len(result.open_source_competition) > 0

    def test_no_open_source_competition(self):
        result = _extract("A cloud platform for analytics.")
        assert result.open_source_competition == []


# ---------------------------------------------------------------------------
# Regulatory competition
# ---------------------------------------------------------------------------


class TestRegulatoryCompetition:
    def test_regulatory_advantage(self):
        result = _extract(
            "Our regulatory advantage protects our position."
        )
        assert len(result.regulatory_competition) > 0

    def test_compliance_first(self):
        result = _extract(
            "A compliance-first platform for financial services."
        )
        assert len(result.regulatory_competition) > 0

    def test_regulatory_expertise(self):
        result = _extract(
            "Deep regulatory expertise across 35 countries."
        )
        assert len(result.regulatory_competition) > 0

    def test_licensed_certified(self):
        result = _extract(
            "Licensed and certified by financial regulators."
        )
        assert len(result.regulatory_competition) > 0

    def test_compliance_platform(self):
        result = _extract(
            "A compliance platform for SOC 2 and GDPR."
        )
        assert len(result.regulatory_competition) > 0

    def test_no_regulatory_competition(self):
        result = _extract("A cloud platform for analytics.")
        assert result.regulatory_competition == []


# ---------------------------------------------------------------------------
# Geographic competition
# ---------------------------------------------------------------------------


class TestGeographicCompetition:
    def test_global_competition(self):
        result = _extract(
            "Competing on a global landscape with rivals."
        )
        assert len(result.geographic_competition) > 0

    def test_regional_leader(self):
        result = _extract(
            "Our regional leader position in Europe."
        )
        assert len(result.geographic_competition) > 0

    def test_international_expansion(self):
        result = _extract(
            "International expansion into new markets."
        )
        assert len(result.geographic_competition) > 0

    def test_multi_country(self):
        result = _extract(
            "Operating across 35 countries worldwide."
        )
        assert len(result.geographic_competition) > 0

    def test_us_europe(self):
        result = _extract(
            "Operations across US and Europe."
        )
        assert len(result.geographic_competition) > 0

    def test_no_geographic_competition(self):
        result = _extract("A cloud platform for analytics.")
        assert result.geographic_competition == []


# ---------------------------------------------------------------------------
# Pricing pressure
# ---------------------------------------------------------------------------


class TestPricingPressure:
    def test_price_competition(self):
        result = _extract(
            "Intense price competition in this market."
        )
        assert len(result.pricing_pressure) > 0

    def test_price_war(self):
        result = _extract(
            "A race to the bottom in pricing."
        )
        assert len(result.pricing_pressure) > 0

    def test_affordable(self):
        result = _extract(
            "An affordable alternative for small businesses."
        )
        assert len(result.pricing_pressure) > 0

    def test_low_cost(self):
        result = _extract(
            "A low-cost solution compared to enterprise tools."
        )
        assert len(result.pricing_pressure) > 0

    def test_free_tier(self):
        result = _extract(
            "Free tier with limited features available."
        )
        assert len(result.pricing_pressure) > 0

    def test_freemium(self):
        result = _extract(
            "Freemium model with premium upgrade."
        )
        assert len(result.pricing_pressure) > 0

    def test_no_pricing_pressure(self):
        result = _extract("A cloud platform for analytics.")
        assert result.pricing_pressure == []


# ---------------------------------------------------------------------------
# Competitive keywords
# ---------------------------------------------------------------------------


class TestCompetitiveKeywords:
    def test_market_dynamics(self):
        result = _extract(
            "Understanding the competitive landscape dynamics."
        )
        assert "competitive landscape" in result.competitive_keywords

    def test_positioning(self):
        result = _extract(
            "Our unique positioning creates competitive advantage."
        )
        assert "competitive advantage" in result.competitive_keywords

    def test_disruption(self):
        result = _extract(
            "We are causing disruption in the traditional industry."
        )
        assert "disruption" in result.competitive_keywords

    def test_moat(self):
        result = _extract(
            "Our moat is built on switching cost and data."
        )
        assert "moat" in result.competitive_keywords
        assert "switching cost" in result.competitive_keywords

    def test_competition_level(self):
        result = _extract(
            "A fragmented blue ocean market opportunity."
        )
        assert "blue ocean" in result.competitive_keywords

    def test_incumbent_in_keywords(self):
        result = _extract(
            "Displacing the incumbent with a new platform."
        )
        assert "incumbent" in result.competitive_keywords

    def test_no_competitive_keywords(self):
        result = _extract("A cloud platform for analytics.")
        assert result.competitive_keywords == []


# ---------------------------------------------------------------------------
# Competition confidence
# ---------------------------------------------------------------------------


class TestCompetitionConfidence:
    def test_high_confidence_competitive(self):
        desc = (
            "Competing directly with market leaders in a highly "
            "competitive space. High switching costs create vendor "
            "lock-in. Our proprietary technology and data moat "
            "provide defensible advantage. Regulatory barriers "
            "protect our position. Strong network effects drive "
            "winner-take-most dynamics."
        )
        result = _extract(desc)
        assert result.competition_confidence > 0.4

    def test_moderate_confidence(self):
        desc = (
            "A compliance platform replacing legacy systems. "
            "SOC 2 and GDPR certification required."
        )
        result = _extract(desc)
        assert 0.1 < result.competition_confidence < 0.6

    def test_low_confidence(self):
        desc = "A cloud platform for analytics."
        result = _extract(desc)
        assert result.competition_confidence == 0.0

    def test_confidence_scales_with_signals(self):
        desc_sparse = (
            "A blue ocean opportunity with no competitors. "
            "First mover in an untapped market."
        )
        desc_dense = (
            "Competing directly with market leaders in a highly "
            "competitive and intense market. Rival competitors "
            "and alternatives to our platform. Disrupting incumbent "
            "players with high switching costs and vendor lock-in. "
            "Winner-take-all dynamics. Our data moat and "
            "proprietary technology create defensible advantage. "
            "Regulatory barriers protect our position. "
            "Replaces spreadsheets and manual processes. "
            "Built on AWS deep integration. Open-source "
            "competition with 14,000 GitHub stars. Price "
            "competition from low-cost alternatives. "
            "Competitive landscape and market dynamics."
        )
        result_sparse = _extract(desc_sparse)
        result_dense = _extract(desc_dense)
        assert result_dense.competition_confidence > result_sparse.competition_confidence

    def test_confidence_capped_at_one(self):
        desc = (
            "Competing directly with market leaders. Rival "
            "competitors and alternatives. Disrupting incumbent "
            "players. High switching costs and vendor lock-in. "
            "Winner-take-all dynamics. Our data moat and "
            "proprietary technology. Regulatory barriers. "
            "Replaces spreadsheets. Built on AWS. Open-source "
            "competition. Price competition. "
            "Competitive landscape dynamics. Market leader "
            "controls. Winner-take-most. Network effect moat. "
            "Patent-protected IP. Years to build. "
            "Instead of email. Platform ecosystem. "
            "GitHub stars. Compliance platform. "
            "International expansion across 35 countries. "
            "Free tier available."
        )
        result = _extract(desc)
        assert result.competition_confidence <= 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_very_short_description(self):
        result = _extract("AI tools.")
        assert isinstance(result, ExtractedFeatures)

    def test_very_long_description(self):
        desc = "A platform. " * 500
        result = _extract(desc)
        assert isinstance(result, ExtractedFeatures)

    def test_case_insensitivity(self):
        result = _extract(
            "COMPETING DIRECTLY with MARKET LEADERS."
        )
        assert len(result.direct_competitor_signals) > 0

    def test_multiple_signals_same_dimension(self):
        desc = (
            "Competing with rivals. Head to head with market "
            "leader. Competing against alternatives."
        )
        result = _extract(desc)
        assert len(result.direct_competitor_signals) >= 2

    def test_all_dimensions_populated(self):
        desc = (
            "Competing directly with market leaders in a fragmented "
            "space with many small players. Replacing legacy systems "
            "and the old approach. Winner-take-all "
            "dynamics. Strong network effects create a moat. High "
            "switching costs with deep integration. Proprietary "
            "technology with patent protection. Regulatory barriers "
            "including SOC 2 certification. Replaces spreadsheets "
            "and manual processes. Built on AWS with platform "
            "ecosystem. Open-source competition with community "
            "edition. Regulatory advantage. International expansion "
            "across 35 countries. Price competition from low-cost "
            "alternatives. Competitive landscape and market dynamics."
        )
        result = _extract(desc)
        assert len(result.direct_competitor_signals) > 0
        assert len(result.indirect_competitor_signals) > 0
        assert len(result.incumbent_signals) > 0
        assert result.market_concentration is not None
        assert len(result.fragmentation_signals) > 0
        assert len(result.winner_take_most_signals) > 0
        assert result.network_effect_competition is not None
        assert len(result.switching_cost_signals) > 0
        assert len(result.differentiation_signals) > 0
        assert len(result.competitive_moat_indicators) > 0
        assert len(result.barriers_to_entry) > 0
        assert len(result.substitute_product_signals) > 0
        assert len(result.platform_dependency) > 0
        assert len(result.ecosystem_dependency) > 0
        assert len(result.open_source_competition) > 0
        assert len(result.regulatory_competition) > 0
        assert len(result.geographic_competition) > 0
        assert len(result.pricing_pressure) > 0
        assert len(result.competitive_keywords) > 0
        assert result.competition_confidence > 0.5
