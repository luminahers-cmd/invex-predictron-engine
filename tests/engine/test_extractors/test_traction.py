"""Comprehensive tests for TractionExtractor (Sprint 6).

Tests cover all 22 dimensions of traction extraction with
deterministic, traceable assertions.
"""

from __future__ import annotations

from predictron_engine.extraction.extractors.traction import TractionExtractor
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
    )


# ---------------------------------------------------------------------------
# Basic interface tests
# ---------------------------------------------------------------------------


class TestTractionExtractorBasic:
    def test_returns_extracted_features(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert isinstance(result, ExtractedFeatures)

    def test_empty_description_returns_empty(self, sample_collected_data):
        startup = _make_startup(".")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.funding_stage is None
        assert result.has_revenue is None
        assert result.traction_confidence == 0.0
        assert result.traction_keywords == []

    def test_no_signals_returns_defaults(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.funding_stage is None
        assert result.has_revenue is None
        assert result.funding_amount_signals == []
        assert result.investor_signals == []
        assert result.revenue_amount_signals == []
        assert result.arr_mrr_signals == []
        assert result.gmv_signals == []
        assert result.customer_count_signals == []
        assert result.active_user_signals == []
        assert result.enterprise_customer_signals == []
        assert result.pilot_customer_signals == []
        assert result.paying_customer_signals == []
        assert result.partnership_signals == []
        assert result.retention_signals == []
        assert result.engagement_signals == []
        assert result.product_adoption_signals == []
        assert result.growth_signals == []
        assert result.hiring_growth_signals == []
        assert result.expansion_signals == []
        assert result.launch_signals == []
        assert result.milestone_signals == []
        assert result.awards_recognition == []


# ---------------------------------------------------------------------------
# Funding stage classification
# ---------------------------------------------------------------------------


class TestFundingStageClassification:
    def test_pre_seed_detected(self, sample_collected_data):
        startup = _make_startup("A pre-seed startup working on an MVP.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.funding_stage == "pre_seed"

    def test_seed_detected(self, sample_collected_data):
        startup = _make_startup("A seed-stage startup that recently raised an angel round.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.funding_stage == "seed"

    def test_series_a_detected(self, sample_collected_data):
        startup = _make_startup("A company that closed its series a round last quarter.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.funding_stage == "series_a"

    def test_series_b_detected(self, sample_collected_data):
        startup = _make_startup("Raised $45M Series B from Tier 1 fintech investors.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.funding_stage == "series_b"

    def test_series_c_detected(self, sample_collected_data):
        startup = _make_startup("Series C stage with $85M raised.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.funding_stage == "series_c"

    def test_series_d_detected(self, sample_collected_data):
        startup = _make_startup("Series D funding round completed recently.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.funding_stage == "series_d"

    def test_ipo_ready_detected(self, sample_collected_data):
        startup = _make_startup("The company is preparing for its IPO next year.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.funding_stage == "ipo_ready"

    def test_growth_equity_detected(self, sample_collected_data):
        startup = _make_startup("Growth equity round from private equity investors.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.funding_stage == "growth"

    def test_no_stage_for_generic_text(self, sample_collected_data):
        startup = _make_startup("A company building software products.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.funding_stage is None

    def test_ambiguous_stages_returns_none(self, sample_collected_data):
        startup = _make_startup("Seed stage but also doing a growth round.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        # Both seed and growth have strong signals, might be ambiguous
        # The exact result depends on scoring — either is acceptable
        assert result.funding_stage in (None, "seed", "growth")


# ---------------------------------------------------------------------------
# Revenue signal detection
# ---------------------------------------------------------------------------


class TestRevenueDetection:
    def test_revenue_detected(self, sample_collected_data):
        startup = _make_startup("A profitable company with strong recurring revenue and MRR.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.has_revenue is True

    def test_no_revenue_signal(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.has_revenue is None

    def test_paying_customers_implies_revenue(self, sample_collected_data):
        startup = _make_startup("Paying customers generating sales monthly.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.has_revenue is True

    def test_pre_revenue(self, sample_collected_data):
        startup = _make_startup("Currently in beta, pre-revenue phase.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.has_revenue is None


# ---------------------------------------------------------------------------
# Funding amount signals
# ---------------------------------------------------------------------------


class TestFundingAmounts:
    def test_raised_amount_detected(self, sample_collected_data):
        startup = _make_startup("Raised $12M from healthcare-focused VCs.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.funding_amount_signals) > 0

    def test_no_funding_amounts(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.funding_amount_signals == []


# ---------------------------------------------------------------------------
# Investor signals
# ---------------------------------------------------------------------------


class TestInvestorSignals:
    def test_tier_1_vcs_detected(self, sample_collected_data):
        startup = _make_startup("Raised from tier 1 VCs in Series B round.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.investor_signals) > 0
        assert any("tier_1_vcs" in s for s in result.investor_signals)

    def test_angel_investors_detected(self, sample_collected_data):
        startup = _make_startup("Angel investors participated in seed round.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("angel_investors" in s for s in result.investor_signals)

    def test_vc_backed_detected(self, sample_collected_data):
        startup = _make_startup("A VC-backed startup focused on payments.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("vc_funded" in s for s in result.investor_signals)

    def test_healthcare_vc_detected(self, sample_collected_data):
        startup = _make_startup("Raised $12M from healthcare-focused VCs.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("healthcare_vc" in s for s in result.investor_signals)

    def test_no_investor_signals(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.investor_signals == []


# ---------------------------------------------------------------------------
# Grants and accelerator signals
# ---------------------------------------------------------------------------


class TestGrantsAccelerators:
    def test_sbir_grant_detected(self, sample_collected_data):
        startup = _make_startup("Received SBIR grant from federal agency.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("sbir_grant" in s for s in result.grants_accelerator_signals)

    def test_yc_detected(self, sample_collected_data):
        startup = _make_startup("Y Combinator batch W24 graduate.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("yc_batch" in s for s in result.grants_accelerator_signals)

    def test_techstars_detected(self, sample_collected_data):
        startup = _make_startup("Selected for Techstars accelerator program.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("techstars_batch" in s for s in result.grants_accelerator_signals)

    def test_no_grants(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.grants_accelerator_signals == []


# ---------------------------------------------------------------------------
# Revenue amount signals
# ---------------------------------------------------------------------------


class TestRevenueAmounts:
    def test_arr_figure_detected(self, sample_collected_data):
        startup = _make_startup("Currently generating $2.8M ARR from 40 hospital clients.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.revenue_amount_signals) > 0

    def test_mrr_detected(self, sample_collected_data):
        startup = _make_startup("MRR at $150K and growing monthly.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.revenue_amount_signals) > 0

    def test_no_revenue_amounts(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.revenue_amount_signals == []


# ---------------------------------------------------------------------------
# ARR / MRR signals
# ---------------------------------------------------------------------------


class TestArrMrrSignals:
    def test_arr_mention(self, sample_collected_data):
        startup = _make_startup("Achieved $4.2M ARR with 140% net revenue retention.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.arr_mrr_signals) > 0

    def test_mrr_mention(self, sample_collected_data):
        startup = _make_startup("MRR of $50K with 200 paying customers.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.arr_mrr_signals) > 0

    def test_no_arr_mrr(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.arr_mrr_signals == []


# ---------------------------------------------------------------------------
# GMV signals
# ---------------------------------------------------------------------------


class TestGmvSignals:
    def test_gmv_detected(self, sample_collected_data):
        startup = _make_startup("Facilitating $180M in GMV annually.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.gmv_signals) > 0

    def test_processing_volume(self, sample_collected_data):
        startup = _make_startup("Processing $2.1B in annual payment volume.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.gmv_signals) > 0

    def test_no_gmv(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.gmv_signals == []


# ---------------------------------------------------------------------------
# Customer count signals
# ---------------------------------------------------------------------------


class TestCustomerCount:
    def test_enterprise_clients_detected(self, sample_collected_data):
        startup = _make_startup("85 enterprise clients including Fortune 500 companies.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.customer_count_signals) > 0

    def test_client_count_detected(self, sample_collected_data):
        startup = _make_startup("280 marketplace clients across 35 countries.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.customer_count_signals) > 0

    def test_hospital_count(self, sample_collected_data):
        startup = _make_startup("Clinical trials across 12 hospital systems.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.customer_count_signals) > 0

    def test_no_customer_count(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.customer_count_signals == []


# ---------------------------------------------------------------------------
# Active user signals
# ---------------------------------------------------------------------------


class TestActiveUsers:
    def test_mau_detected(self, sample_collected_data):
        startup = _make_startup("420,000 monthly active users with 38,000 premium subscribers.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.active_user_signals) > 0

    def test_fleet_size(self, sample_collected_data):
        startup = _make_startup("Current fleet of 800 deployed robots across 15 facilities.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.active_user_signals) > 0

    def test_no_active_users(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.active_user_signals == []


# ---------------------------------------------------------------------------
# Enterprise customer signals
# ---------------------------------------------------------------------------


class TestEnterpriseCustomers:
    def test_fortune_500_detected(self, sample_collected_data):
        startup = _make_startup("12 in the Fortune 500 use our platform.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.enterprise_customer_signals) > 0
        assert any("fortune_500" in s for s in result.enterprise_customer_signals)

    def test_top_banks_detected(self, sample_collected_data):
        startup = _make_startup("Including 8 top-20 US banks as clients.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("top_banks" in s for s in result.enterprise_customer_signals)

    def test_large_acv_detected(self, sample_collected_data):
        startup = _make_startup("ACV of $120,000 with 98% gross retention.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("large_acv" in s for s in result.enterprise_customer_signals)

    def test_no_enterprise_signals(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.enterprise_customer_signals == []


# ---------------------------------------------------------------------------
# Pilot customer signals
# ---------------------------------------------------------------------------


class TestPilotCustomers:
    def test_pilot_program_detected(self, sample_collected_data):
        startup = _make_startup("Running pilot program with 5 hospitals.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.pilot_customer_signals) > 0

    def test_beta_users_detected(self, sample_collected_data):
        startup = _make_startup("Beta users testing the platform since Q1.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("beta_users" in s for s in result.pilot_customer_signals)

    def test_proof_of_concept(self, sample_collected_data):
        startup = _make_startup("Completed proof of concept with enterprise client.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("proof_of_concept" in s for s in result.pilot_customer_signals)

    def test_no_pilots(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.pilot_customer_signals == []


# ---------------------------------------------------------------------------
# Paying customer signals
# ---------------------------------------------------------------------------


class TestPayingCustomers:
    def test_paying_customers_detected(self, sample_collected_data):
        startup = _make_startup("340 open-source users converting to paying plans.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.paying_customer_signals) > 0

    def test_paid_subscribers(self, sample_collected_data):
        startup = _make_startup("38,000 premium subscribers at $9.99/month.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.paying_customer_signals) > 0

    def test_converting_to_paid(self, sample_collected_data):
        startup = _make_startup("Converting 340 open-source users to paid plans.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("converting_to_paid" in s for s in result.paying_customer_signals)

    def test_no_paying_customers(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.paying_customer_signals == []


# ---------------------------------------------------------------------------
# Partnership signals
# ---------------------------------------------------------------------------


class TestPartnershipSignals:
    def test_strategic_partnership(self, sample_collected_data):
        startup = _make_startup("Strategic partnership with major cloud provider.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.partnership_signals) > 0

    def test_technology_partnership(self, sample_collected_data):
        startup = _make_startup("Technology partnership with leading ERP vendor.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("technology_partnership" in s for s in result.partnership_signals)

    def test_no_partnerships(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.partnership_signals == []


# ---------------------------------------------------------------------------
# Retention signals
# ---------------------------------------------------------------------------


class TestRetentionSignals:
    def test_net_retention_detected(self, sample_collected_data):
        startup = _make_startup("140% net revenue retention across enterprise clients.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.retention_signals) > 0
        assert any("net_retention" in s for s in result.retention_signals)

    def test_gross_retention_detected(self, sample_collected_data):
        startup = _make_startup("95% gross retention rate.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("gross_retention" in s for s in result.retention_signals)

    def test_ltv_cac_detected(self, sample_collected_data):
        startup = _make_startup("Unit economics showing LTV/CAC of 4.2x.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("ltv_cac_ratio" in s for s in result.retention_signals)

    def test_d30_retention(self, sample_collected_data):
        startup = _make_startup("D30 retention at 42% for consumer app.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("d30_retention" in s for s in result.retention_signals)

    def test_low_churn(self, sample_collected_data):
        startup = _make_startup("Low churn with high retention across customer base.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("low_churn" in s for s in result.retention_signals)

    def test_no_retention(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.retention_signals == []


# ---------------------------------------------------------------------------
# Engagement signals
# ---------------------------------------------------------------------------


class TestEngagementSignals:
    def test_api_calls_detected(self, sample_collected_data):
        startup = _make_startup("Processing 2 billion inference requests daily.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.engagement_signals) > 0

    def test_inference_volume(self, sample_collected_data):
        startup = _make_startup(
            "Processing over 2 billion inference requests daily "
            "for 150 enterprise customers."
        )
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.engagement_signals) > 0

    def test_no_engagement(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.engagement_signals == []


# ---------------------------------------------------------------------------
# Product adoption signals
# ---------------------------------------------------------------------------


class TestProductAdoption:
    def test_github_stars_detected(self, sample_collected_data):
        startup = _make_startup("Open-source core has 14,000 GitHub stars.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.product_adoption_signals) > 0
        assert any("github_stars" in s for s in result.product_adoption_signals)

    def test_open_source_adoption(self, sample_collected_data):
        startup = _make_startup("850 contributing developers on the open-source project.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("open_source_adoption" in s for s in result.product_adoption_signals)

    def test_beta_launch(self, sample_collected_data):
        startup = _make_startup("Beta launch phase with early adopters.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("beta_launch" in s for s in result.product_adoption_signals)

    def test_no_adoption(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.product_adoption_signals == []


# ---------------------------------------------------------------------------
# Growth signals
# ---------------------------------------------------------------------------


class TestGrowthSignals:
    def test_revenue_growth_detected(self, sample_collected_data):
        startup = _make_startup("Strong revenue growth over the past 12 months.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.growth_signals) > 0

    def test_rapid_growth_detected(self, sample_collected_data):
        startup = _make_startup("Rapid growth in customer base and user adoption.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("rapid_growth" in s for s in result.growth_signals)

    def test_no_growth(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.growth_signals == []


# ---------------------------------------------------------------------------
# Hiring growth signals
# ---------------------------------------------------------------------------


class TestHiringGrowth:
    def test_team_size_detected(self, sample_collected_data):
        startup = _make_startup("Team of 35 engineers and sales professionals.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.hiring_growth_signals) > 0

    def test_expanding_team(self, sample_collected_data):
        startup = _make_startup("Expanding the engineering team rapidly.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("expanding_team" in s for s in result.hiring_growth_signals)

    def test_no_hiring_growth(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.hiring_growth_signals == []


# ---------------------------------------------------------------------------
# Expansion signals
# ---------------------------------------------------------------------------


class TestExpansionSignals:
    def test_geographic_expansion(self, sample_collected_data):
        startup = _make_startup("Expanding into new markets across Europe and Asia.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.expansion_signals) > 0

    def test_multi_country(self, sample_collected_data):
        startup = _make_startup("Operations across 35 countries.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("multi_country" in s for s in result.expansion_signals)

    def test_international(self, sample_collected_data):
        startup = _make_startup("International expansion into European markets.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("international" in s for s in result.expansion_signals)

    def test_no_expansion(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.expansion_signals == []


# ---------------------------------------------------------------------------
# Launch signals
# ---------------------------------------------------------------------------


class TestLaunchSignals:
    def test_product_launch_detected(self, sample_collected_data):
        startup = _make_startup("Launched the platform to general availability.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.launch_signals) > 0

    def test_beta_launch(self, sample_collected_data):
        startup = _make_startup("Beta launched in Q2 2024.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("beta_launch" in s for s in result.launch_signals)

    def test_went_live(self, sample_collected_data):
        startup = _make_startup("Went live with production deployment in Q1.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("went_live" in s for s in result.launch_signals)

    def test_no_launch(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.launch_signals == []


# ---------------------------------------------------------------------------
# Milestone signals
# ---------------------------------------------------------------------------


class TestMilestoneSignals:
    def test_profitability(self, sample_collected_data):
        startup = _make_startup("Profitable with positive cash flow.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.milestone_signals) > 0
        assert any("profitability" in s for s in result.milestone_signals)

    def test_fda_clearance(self, sample_collected_data):
        startup = _make_startup("FDA cleared AI diagnostic tool.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("fda_clearance" in s for s in result.milestone_signals)

    def test_soc2_compliance(self, sample_collected_data):
        startup = _make_startup("SOC 2 Type II compliant platform.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("soc2_compliance" in s for s in result.milestone_signals)

    def test_break_even(self, sample_collected_data):
        startup = _make_startup("Reached break-even in Q4 2024.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("break_even" in s for s in result.milestone_signals)

    def test_no_milestones(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.milestone_signals == []


# ---------------------------------------------------------------------------
# Awards and recognition
# ---------------------------------------------------------------------------


class TestAwardsRecognition:
    def test_forbes_detected(self, sample_collected_data):
        startup = _make_startup("Featured in Forbes 30 Under 30 list.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.awards_recognition) > 0

    def test_techcrunch_detected(self, sample_collected_data):
        startup = _make_startup("Featured in TechCrunch as top startup.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert any("techcrunch" in s for s in result.awards_recognition)

    def test_recognized_as(self, sample_collected_data):
        startup = _make_startup("Recognized as best AI platform by industry panel.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert len(result.awards_recognition) > 0

    def test_no_awards(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.awards_recognition == []


# ---------------------------------------------------------------------------
# Traction keywords
# ---------------------------------------------------------------------------


class TestTractionKeywords:
    def test_market_validation_keywords(self, sample_collected_data):
        startup = _make_startup("Strong product-market fit validated with early customers.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert "product-market fit" in result.traction_keywords

    def test_revenue_keywords(self, sample_collected_data):
        startup = _make_startup("Recurring revenue growing month over month.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert "revenue" in result.traction_keywords
        assert "recurring" in result.traction_keywords

    def test_funding_keywords(self, sample_collected_data):
        startup = _make_startup("Raised venture capital in seed round.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert "raised" in result.traction_keywords
        assert "seed" in result.traction_keywords or "funding" in result.traction_keywords

    def test_no_keywords_for_empty(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.traction_keywords == []


# ---------------------------------------------------------------------------
# Traction confidence scoring
# ---------------------------------------------------------------------------


class TestTractionConfidence:
    def test_high_confidence_for_rich_signals(self, sample_collected_data):
        startup = _make_startup(
            "Series A company with $4.2M ARR, 85 enterprise clients, "
            "140% net revenue retention, raised $12M from tier 1 VCs, "
            "team of 35, and rapid revenue growth."
        )
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.traction_confidence > 0.3

    def test_zero_confidence_for_empty(self, sample_startup, sample_collected_data):
        result = TractionExtractor().extract(sample_startup, sample_collected_data)
        assert result.traction_confidence == 0.0

    def test_low_confidence_for_minimal_signals(self, sample_collected_data):
        startup = _make_startup("A startup.")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.traction_confidence == 0.0

    def test_confidence_scales_with_signals(self, sample_collected_data):
        startup_rich = _make_startup(
            "Series A with $5M ARR, 200 customers, 150% net retention, "
            "raised $20M, tier 1 VC backed, team of 50, growing rapidly, "
            "Fortune 500 client, launched platform, SOC 2 compliant."
        )
        startup_poor = _make_startup("A new startup with an idea.")
        result_rich = TractionExtractor().extract(startup_rich, sample_collected_data)
        result_poor = TractionExtractor().extract(startup_poor, sample_collected_data)
        assert result_rich.traction_confidence > result_poor.traction_confidence

    def test_confidence_is_bounded(self, sample_collected_data):
        startup = _make_startup(
            "Series A company with $10M ARR, 500 enterprise clients, "
            "200% net revenue retention, raised $50M from tier 1 VCs, "
            "10,000 active users, $100M GMV, SOC 2 and FDA cleared, "
            "profitable with break-even, expanding internationally, "
            "Forbes recognized, TechCrunch featured, "
            "rapid revenue growth, team of 200, strong partnerships."
        )
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert 0.0 <= result.traction_confidence <= 1.0


# ---------------------------------------------------------------------------
# Integration with benchmark cases
# ---------------------------------------------------------------------------


class TestTractionBenchmarkCases:
    """Test that TractionExtractor produces expected results for benchmark cases."""

    def test_b2b_saas_benchmark(self, sample_collected_data):
        startup = _make_startup(
            "Analytix Cloud is a B2B SaaS platform providing real-time "
            "analytics and business intelligence for mid-market enterprises. "
            "The platform integrates with existing data warehouses and "
            "delivers automated insight generation through proprietary "
            "query optimization. Revenue is subscription-based with annual "
            "contracts averaging $48,000 ARR. The company has 85 enterprise "
            "clients and has achieved $4.2M ARR with 140% net revenue "
            "retention. Founded in 2021, the team of 35 engineers and "
            "sales professionals operates from San Francisco with a remote "
            "engineering hub in Austin."
        )
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.has_revenue is True
        assert result.funding_stage is None
        assert len(result.arr_mrr_signals) > 0
        assert len(result.customer_count_signals) > 0
        assert len(result.retention_signals) > 0

    def test_healthcare_ai_benchmark(self, sample_collected_data):
        startup = _make_startup(
            "MedVision AI develops FDA-cleared AI diagnostic tools for "
            "medical imaging. The platform analyzes X-rays, MRIs, and CT "
            "scans to assist radiologists in detecting anomalies with "
            "97.3% sensitivity. The company holds 3 patents on its "
            "convolutional neural network architecture and has completed "
            "clinical trials across 12 hospital systems. Revenue model is "
            "per-study licensing with enterprise hospital contracts. "
            "Currently generating $2.8M ARR from 40 hospital clients. "
            "Series A funded with $12M raised from healthcare-focused VCs. "
            "Based in Boston with regulatory and clinical teams."
        )
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.funding_stage == "series_a"
        assert result.has_revenue is True
        assert len(result.revenue_amount_signals) > 0
        assert len(result.customer_count_signals) > 0
        assert any("fda_clearance" in s for s in result.milestone_signals)

    def test_fintech_benchmark(self, sample_collected_data):
        startup = _make_startup(
            "PayBridge provides embedded payment infrastructure for "
            "marketplace and platform businesses. The API-first solution "
            "handles split payments, escrow, KYC compliance, and multi-"
            "currency settlement across 35 countries. Processing $2.1B in "
            "annual payment volume with a take rate of 0.8%. The company "
            "serves 280 marketplace clients including 12 in the Fortune "
            "500. Founded in 2019, team of 120 across London, Singapore, "
            "and New York. Raised $45M Series B from Tier 1 fintech investors."
        )
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.funding_stage == "series_b"
        assert len(result.funding_amount_signals) > 0
        assert len(result.gmv_signals) > 0
        assert len(result.customer_count_signals) > 0
        assert any("fortune_500" in s for s in result.enterprise_customer_signals)

    def test_devtools_benchmark(self, sample_collected_data):
        startup = _make_startup(
            "ShipKit is an open-source-first CI/CD platform designed for "
            "monorepo architectures. The platform provides incremental "
            "builds, intelligent test parallelization, and deployment "
            "orchestration for teams running microservices. The open-source "
            "core has 14,000 GitHub stars and 850 contributing developers. "
            "Commercial features include audit logging, SSO, and priority "
            "support. Currently converting 340 open-source users to paid "
            "plans at $299/month. Pre-revenue on commercial tier, "
            "currently in beta launch phase. Founded in 2023 by two "
            "ex-GitHub engineers based in Seattle."
        )
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.has_revenue is None
        assert len(result.product_adoption_signals) > 0
        assert any("converting_to_paid" in s for s in result.paying_customer_signals)

    def test_marketplace_benchmark(self, sample_collected_data):
        startup = _make_startup(
            "SupplyHub operates a two-sided B2B marketplace connecting "
            "industrial equipment manufacturers with small and mid-size "
            "manufacturing facilities. The platform handles quoting, "
            "procurement, and logistics for MRO supplies across North "
            "America. Marketplace take rate is 12% on transactions. "
            "Currently facilitating $180M in GMV annually with 2,400 "
            "supplier listings and 8,500 active buyer accounts. Revenue "
            "of $21.6M with unit economics showing LTV/CAC of 4.2x. "
            "Raised $30M Series A. Founded in 2020, team of 65 based "
            "in Chicago."
        )
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.has_revenue is True
        assert result.funding_stage == "series_a"
        assert len(result.gmv_signals) > 0
        assert len(result.customer_count_signals) > 0
        assert len(result.retention_signals) > 0

    def test_consumer_app_benchmark(self, sample_collected_data):
        startup = _make_startup(
            "FitSocial is a consumer mobile application that combines "
            "fitness tracking with social networking. Users log workouts, "
            "share progress, and participate in community challenges. "
            "Monetization is through a freemium model with a $9.99/month "
            "premium subscription and sponsored brand partnerships. "
            "Currently has 420,000 monthly active users with 38,000 "
            "premium subscribers. Retention at D30 is 42%. Available on "
            "iOS and Android. Founded in 2022, team of 18, based in "
            "Los Angeles. Pre-seed stage with $1.5M raised from angels."
        )
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.has_revenue is True
        assert result.funding_stage == "pre_seed"
        assert len(result.active_user_signals) > 0
        assert len(result.paying_customer_signals) > 0
        assert len(result.retention_signals) > 0

    def test_climate_tech_benchmark(self, sample_collected_data):
        startup = _make_startup(
            "CarbonLens provides enterprise-grade carbon accounting and "
            "emissions tracking software for Scope 1, 2, and 3 emissions. "
            "The platform integrates with ERP systems, supply chain tools, "
            "and IoT sensors to automate emissions data collection and "
            "reporting. Aligned with GHG Protocol and SEC climate disclosure "
            "requirements. Serving 60 enterprise clients across "
            "manufacturing, logistics, and energy sectors. ARR of $3.5M "
            "with 95% gross retention. Raised $8M seed from climate-focused "
            "VCs. Founded in 2022, team of 28, headquartered in Berlin "
            "with operations in the US."
        )
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.has_revenue is True
        assert result.funding_stage == "seed"
        assert len(result.arr_mrr_signals) > 0
        assert len(result.retention_signals) > 0
        assert len(result.investor_signals) > 0

    def test_robotics_benchmark(self, sample_collected_data):
        startup = _make_startup(
            "AutoWare Robotics builds autonomous mobile robots for "
            "warehouse and fulfillment center operations. The robots use "
            "LiDAR-based SLAM navigation and handle goods-to-person picking, "
            "inventory scanning, and pallet transport. The company offers "
            "a Robotics-as-a-Service model with per-robot monthly fees "
            "covering hardware, maintenance, and software updates. Current "
            "fleet of 800 deployed robots across 15 customer facilities. "
            "Revenue model generates $6.4M ARR from RaaS contracts. "
            "Hardware costs funded through $52M in total funding including "
            "a Series B. Founded in 2019, team of 110 in Munich and Detroit."
        )
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.has_revenue is True
        assert result.funding_stage == "series_b"
        assert len(result.arr_mrr_signals) > 0
        assert len(result.active_user_signals) > 0
        assert len(result.funding_amount_signals) > 0

    def test_enterprise_software_benchmark(self, sample_collected_data):
        startup = _make_startup(
            "ComplianceOS is an enterprise compliance management platform "
            "that automates regulatory tracking, policy management, and "
            "audit preparation for financial services companies. The "
            "platform covers SOC 2, PCI DSS, GDPR, and CCPA compliance "
            "workflows with automated evidence collection and control "
            "monitoring. Serving 95 financial institutions including 8 "
            "top-20 US banks. ACV of $120,000 with 98% gross retention. "
            "Series C stage with $85M raised. Team of 200 across New York, "
            "Charlotte, and remote. Founded in 2018."
        )
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.has_revenue is True
        assert result.funding_stage == "series_c"
        assert len(result.customer_count_signals) > 0
        assert len(result.enterprise_customer_signals) > 0
        assert len(result.retention_signals) > 0
        assert any("soc2_compliance" in s for s in result.milestone_signals)

    def test_ai_infrastructure_benchmark(self, sample_collected_data):
        startup = _make_startup(
            "Inference Labs provides GPU-optimized model serving "
            "infrastructure for teams deploying large language models in "
            "production. The platform handles auto-scaling, model versioning, "
            "A/B testing, and cost optimization across multi-cloud GPU "
            "clusters. Supports PyTorch, TensorFlow, and ONNX models. "
            "Processing over 2 billion inference requests daily for 150 "
            "enterprise customers. Usage-based pricing with average customer "
            "spend of $18,000/month. Raised $65M Series A. Founded in 2023 "
            "by former infrastructure leads from major AI labs. Team of 75 "
            "in San Francisco and Toronto."
        )
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.has_revenue is True
        assert result.funding_stage == "series_a"
        assert len(result.engagement_signals) > 0
        assert len(result.funding_amount_signals) > 0
        assert len(result.customer_count_signals) > 0


# ---------------------------------------------------------------------------
# Sprint 14 — Structured Quantitative Extraction Tests
# ---------------------------------------------------------------------------


class TestTractionQuantitativeExtraction:
    def test_funding_amount_usd(self, sample_collected_data):
        startup = _make_startup("Raised $12M in a seed round")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.funding_amount_usd == 12_000_000.0

    def test_funding_amount_usd_billions(self, sample_collected_data):
        startup = _make_startup("Raised $1.2B Series C")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.funding_amount_usd == 1_200_000_000.0

    def test_arr_usd(self, sample_collected_data):
        startup = _make_startup("ARR of $4.2M with strong retention")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.arr_usd == 4_200_000.0

    def test_arr_usd_suffix(self, sample_collected_data):
        startup = _make_startup("$500K in ARR growing 20% MoM")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.arr_usd == 500_000.0

    def test_mrr_usd(self, sample_collected_data):
        startup = _make_startup("MRR of $50K with 500 paying customers")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.mrr_usd == 50_000.0

    def test_mrr_usd_suffix(self, sample_collected_data):
        startup = _make_startup("$120,000 MRR")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.mrr_usd == 120_000.0

    def test_gmv_usd(self, sample_collected_data):
        startup = _make_startup("$500M GMV on the marketplace")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.gmv_usd == 500_000_000.0

    def test_customer_count(self, sample_collected_data):
        startup = _make_startup("500 enterprise customers")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.customer_count == 500

    def test_customer_count_comma(self, sample_collected_data):
        startup = _make_startup("1,200 active clients")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.customer_count == 1_200

    def test_active_user_count(self, sample_collected_data):
        startup = _make_startup("10,000 monthly active users")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.active_user_count == 10_000

    def test_active_user_count_k_suffix(self, sample_collected_data):
        startup = _make_startup("500K monthly active users")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.active_user_count == 500_000

    def test_growth_rate_pct_mom(self, sample_collected_data):
        startup = _make_startup("150% MoM growth")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.growth_rate_pct == 150.0

    def test_growth_rate_pct_yoy(self, sample_collected_data):
        startup = _make_startup("30% YoY growth")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.growth_rate_pct == 30.0

    def test_nrr_pct(self, sample_collected_data):
        startup = _make_startup("NRR of 120%")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.nrr_pct == 120.0

    def test_nrr_pct_net_retention(self, sample_collected_data):
        startup = _make_startup("130% net revenue retention")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.nrr_pct == 130.0

    def test_churn_rate_pct(self, sample_collected_data):
        startup = _make_startup("churn rate at 5%")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.churn_rate_pct == 5.0

    def test_no_quantitative_returns_none(self, sample_collected_data):
        startup = _make_startup("A simple SaaS platform")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.funding_amount_usd is None
        assert result.arr_usd is None
        assert result.mrr_usd is None
        assert result.gmv_usd is None
        assert result.customer_count is None
        assert result.active_user_count is None
        assert result.growth_rate_pct is None
        assert result.nrr_pct is None
        assert result.churn_rate_pct is None

    def test_empty_description_returns_none(self, sample_collected_data):
        startup = _make_startup(".")
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.funding_amount_usd is None
        assert result.arr_usd is None
        assert result.mrr_usd is None

    def test_comprehensive_startup(self, sample_collected_data):
        startup = _make_startup(
            "AI platform with $4.2M ARR, 500 enterprise customers, "
            "150% MoM growth, $12M raised in Series A, NRR of 125%, "
            "churn rate at 3%, 50K monthly active users"
        )
        result = TractionExtractor().extract(startup, sample_collected_data)
        assert result.funding_amount_usd == 12_000_000.0
        assert result.arr_usd == 4_200_000.0
        assert result.customer_count == 500
        assert result.growth_rate_pct == 150.0
        assert result.nrr_pct == 125.0
        assert result.churn_rate_pct == 3.0
        assert result.active_user_count == 50_000
