"""Comprehensive tests for FounderExtractor — Sprint 2.

Tests all 13 extraction dimensions of founder intelligence:
  1. Founder count
  2. Team size indicator
  3. Technical vs business founder type
  4. Domain expertise signals
  5. Serial founder indicators
  6. Leadership roles
  7. Hiring signals
  8. Advisor mentions
  9. Engineering strength
 10. Product strength
 11. Founder-market fit signals
 12. Execution signals
 13. Founder confidence
"""

from predictron_engine.extraction.extractors.founder import FounderExtractor
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup


class TestFounderExtractor:
    """Core FounderExtractor tests."""

    def test_returns_extracted_features(self, sample_startup, sample_collected_data):
        result = FounderExtractor().extract(sample_startup, sample_collected_data)
        assert isinstance(result, ExtractedFeatures)

    # ------------------------------------------------------------------
    # Founder count
    # ------------------------------------------------------------------

    def test_founder_count_from_collected_data(self, sample_startup):
        data = CollectedData(startup_name="TestCo", founder_count=3)
        result = FounderExtractor().extract(sample_startup, data)
        assert result.founder_profile_count == 3

    def test_founder_count_zero(self, sample_startup, sample_collected_data):
        result = FounderExtractor().extract(sample_startup, sample_collected_data)
        assert result.founder_profile_count == 0

    def test_founder_count_one(self, sample_startup):
        data = CollectedData(startup_name="SoloCo", founder_count=1)
        result = FounderExtractor().extract(sample_startup, data)
        assert result.founder_profile_count == 1

    # ------------------------------------------------------------------
    # Team size indicator
    # ------------------------------------------------------------------

    def test_team_size_from_description(self, sample_collected_data):
        startup = Startup(
            name="TeamCo",
            website="https://teamco.example.com",
            description="A startup with a team of 15 people building great products.",
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert result.team_size_indicator == "11-50"

    def test_team_size_small(self, sample_collected_data):
        startup = Startup(
            name="SmallCo",
            website="https://smallco.example.com",
            description="A startup with a team of 5 people building great products.",
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert result.team_size_indicator == "1-10"

    def test_team_size_not_detected(self, sample_startup, sample_collected_data):
        result = FounderExtractor().extract(sample_startup, sample_collected_data)
        assert result.team_size_indicator is None

    def test_team_size_large(self, sample_collected_data):
        startup = Startup(
            name="BigCo",
            website="https://bigco.example.com",
            description="A startup with a team of 150 people across multiple offices.",
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert result.team_size_indicator == "51-200"

    def test_team_size_very_large(self, sample_collected_data):
        startup = Startup(
            name="HugeCo",
            website="https://hugeco.example.com",
            description="A startup with a team of 500 people across the world.",
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert result.team_size_indicator == "200+"

    def test_team_size_via_composition_pattern(self, sample_collected_data):
        startup = Startup(
            name="EngCo",
            website="https://engco.example.com",
            description="Founded in 2021 by two ex-GitHub engineers based in Seattle.",
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert result.team_size_indicator is None

    # ------------------------------------------------------------------
    # Technical vs business founder type
    # ------------------------------------------------------------------

    def test_team_type_technical(self, sample_collected_data):
        startup = Startup(
            name="TechCo",
            website="https://techco.example.com",
            description=(
                "Engineering team building cloud infrastructure platform with "
                "microservices architecture. Developers working on API, "
                "database, and security systems."
            ),
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert result.founder_team_type == "technical"

    def test_team_type_business(self, sample_collected_data):
        startup = Startup(
            name="BizCo",
            website="https://bizco.example.com",
            description=(
                "Sales-driven company focused on enterprise accounts and "
                "revenue growth. Business development team managing partnerships "
                "and commercial pipeline."
            ),
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert result.founder_team_type == "business"

    def test_team_type_mixed(self, sample_collected_data):
        startup = Startup(
            name="MixedCo",
            website="https://mixedco.example.com",
            description=(
                "A software platform with engineering team building products "
                "and sales team driving revenue growth and customer accounts."
            ),
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert result.founder_team_type == "mixed"

    def test_team_type_from_linkedin_cto(self, sample_collected_data):
        startup = Startup(
            name="CTOCo",
            website="https://ctoco.example.com",
            description="Building an AI platform for enterprise customers.",
            founder_linkedin_urls=[
                "https://linkedin.com/in/ctoco-ceo",
                "https://linkedin.com/in/ctoco-cto",
            ],
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert result.founder_team_type in ("technical", "mixed")

    def test_team_type_unknown_when_no_signals(self, sample_collected_data):
        startup = Startup(
            name="VagueCo",
            website="https://vagueco.example.com",
            description="A company doing things.",
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert result.founder_team_type is None

    # ------------------------------------------------------------------
    # Domain expertise signals
    # ------------------------------------------------------------------

    def test_domain_expertise_healthcare(self, sample_collected_data):
        startup = Startup(
            name="HealthCo",
            website="https://healthco.example.com",
            description=(
                "Medical imaging platform with clinical expertise in "
                "diagnostic radiology. Healthcare domain specialist team."
            ),
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert len(result.domain_expertise_signals) > 0
        assert any("healthcare" in s.lower() for s in result.domain_expertise_signals)

    def test_domain_expertise_fintech(self, sample_collected_data):
        startup = Startup(
            name="FinCo",
            website="https://finco.example.com",
            description=(
                "Fintech payments platform with financial domain expertise "
                "in banking and compliance."
            ),
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert len(result.domain_expertise_signals) > 0
        assert any("fintech" in s.lower() for s in result.domain_expertise_signals)

    def test_domain_expertise_none_detected(self, sample_startup, sample_collected_data):
        result = FounderExtractor().extract(sample_startup, sample_collected_data)
        assert result.domain_expertise_signals == []

    # ------------------------------------------------------------------
    # Serial founder indicators
    # ------------------------------------------------------------------

    def test_serial_founder_founded(self, sample_collected_data):
        startup = Startup(
            name="SerialCo",
            website="https://serialco.example.com",
            description="Founded in 2021. Previously started a SaaS company.",
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert len(result.serial_founder_indicators) > 0

    def test_serial_founder_ex_founder(self, sample_collected_data):
        startup = Startup(
            name="ExCo",
            website="https://exco.example.com",
            description=(
                "Founded by an ex-entrepreneur who co-founded multiple "
                "startups in the enterprise space."
            ),
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert len(result.serial_founder_indicators) > 0

    def test_serial_founder_none(self, sample_startup, sample_collected_data):
        result = FounderExtractor().extract(sample_startup, sample_collected_data)
        assert result.serial_founder_indicators == []

    # ------------------------------------------------------------------
    # Leadership roles
    # ------------------------------------------------------------------

    def test_leadership_roles_from_linkedin(self, sample_collected_data):
        startup = Startup(
            name="LeadCo",
            website="https://leadco.example.com",
            description="Enterprise platform.",
            founder_linkedin_urls=[
                "https://linkedin.com/in/leadco-ceo",
                "https://linkedin.com/in/leadco-cto",
                "https://linkedin.com/in/leadco-coo",
            ],
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert len(result.leadership_roles) >= 2
        assert "CEO" in result.leadership_roles
        assert "CTO" in result.leadership_roles

    def test_leadership_roles_vp(self, sample_collected_data):
        startup = Startup(
            name="VPCo",
            website="https://vpco.example.com",
            description="Tech company.",
            founder_linkedin_urls=[
                "https://linkedin.com/in/vpco-vp-eng",
            ],
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert "VP-Engineering" in result.leadership_roles

    def test_leadership_roles_empty(self, sample_startup, sample_collected_data):
        result = FounderExtractor().extract(sample_startup, sample_collected_data)
        assert result.leadership_roles == []

    # ------------------------------------------------------------------
    # Hiring signals
    # ------------------------------------------------------------------

    def test_hiring_signals_team_of(self, sample_collected_data):
        startup = Startup(
            name="HireCo",
            website="https://hireco.example.com",
            description="A startup with a team of 35 engineers and sales professionals.",
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert len(result.hiring_signals) > 0

    def test_hiring_signals_growth(self, sample_collected_data):
        startup = Startup(
            name="GrowCo",
            website="https://growco.example.com",
            description="Fast growing team with open roles in engineering.",
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert len(result.hiring_signals) > 0

    def test_hiring_signals_none(self, sample_startup, sample_collected_data):
        result = FounderExtractor().extract(sample_startup, sample_collected_data)
        assert result.hiring_signals == []

    # ------------------------------------------------------------------
    # Advisor mentions
    # ------------------------------------------------------------------

    def test_advisor_mentions_detected(self, sample_collected_data):
        startup = Startup(
            name="AdvisorCo",
            website="https://advisorco.example.com",
            description="Company with advisory board of industry experts.",
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert len(result.advisor_mentions) > 0

    def test_advisor_mentions_none(self, sample_startup, sample_collected_data):
        result = FounderExtractor().extract(sample_startup, sample_collected_data)
        assert result.advisor_mentions == []

    # ------------------------------------------------------------------
    # Engineering strength
    # ------------------------------------------------------------------

    def test_engineering_strength_strong(self, sample_collected_data):
        startup = Startup(
            name="EngStrongCo",
            website="https://engstrongco.example.com",
            description=(
                "Platform with 40 engineers building distributed systems. "
                "Open-source core with 500 GitHub stars. "
                "Cloud-native microservices architecture."
            ),
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert result.engineering_strength == "strong"

    def test_engineering_strength_moderate(self, sample_collected_data):
        startup = Startup(
            name="EngModCo",
            website="https://engmodco.example.com",
            description=(
                "Building a software platform with API-first design "
                "and microservices architecture."
            ),
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert result.engineering_strength == "moderate"

    def test_engineering_strength_none(self, sample_startup, sample_collected_data):
        result = FounderExtractor().extract(sample_startup, sample_collected_data)
        assert result.engineering_strength is None

    # ------------------------------------------------------------------
    # Product strength
    # ------------------------------------------------------------------

    def test_product_strength_strong(self, sample_collected_data):
        startup = Startup(
            name="ProdStrongCo",
            website="https://prodstrongco.example.com",
            description=(
                "Proprietary AI platform with 3 patents. Series A funded "
                "with 150 enterprise customers and FDA-cleared diagnostic tools."
            ),
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert result.product_strength == "strong"

    def test_product_strength_moderate(self, sample_collected_data):
        startup = Startup(
            name="ProdModCo",
            website="https://prodmodco.example.com",
            description=(
                "Building a software platform that is currently in beta."
            ),
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert result.product_strength == "moderate"

    def test_product_strength_none(self, sample_startup, sample_collected_data):
        result = FounderExtractor().extract(sample_startup, sample_collected_data)
        assert result.product_strength is None

    # ------------------------------------------------------------------
    # Founder-market fit signals
    # ------------------------------------------------------------------

    def test_founder_market_fit_faang(self, sample_collected_data):
        startup = Startup(
            name="FAANGCo",
            website="https://faangco.example.com",
            description=(
                "Former infrastructure leads from major AI labs building "
                "GPU-optimized model serving."
            ),
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert len(result.founder_market_fit_signals) > 0

    def test_founder_market_fit_ex_github(self, sample_collected_data):
        startup = Startup(
            name="ExGHCo",
            website="https://exghco.example.com",
            description="Founded by two ex-GitHub engineers based in Seattle.",
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert len(result.founder_market_fit_signals) > 0

    def test_founder_market_fit_none(self, sample_startup, sample_collected_data):
        result = FounderExtractor().extract(sample_startup, sample_collected_data)
        assert result.founder_market_fit_signals == []

    # ------------------------------------------------------------------
    # Execution signals
    # ------------------------------------------------------------------

    def test_execution_signals_arr(self, sample_collected_data):
        startup = Startup(
            name="ARRCo",
            website="https://arrco.example.com",
            description="SaaS platform with $4.2M ARR and 85 enterprise clients.",
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert len(result.execution_signals) >= 2

    def test_execution_signals_funding(self, sample_collected_data):
        startup = Startup(
            name="FundCo",
            website="https://fundco.example.com",
            description="Raised $45M Series B from Tier 1 investors.",
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert len(result.execution_signals) >= 1

    def test_execution_signals_none(self, sample_startup, sample_collected_data):
        result = FounderExtractor().extract(sample_startup, sample_collected_data)
        assert result.execution_signals == []

    # ------------------------------------------------------------------
    # Founder confidence
    # ------------------------------------------------------------------

    def test_confidence_zero_when_no_signals(self, sample_startup, sample_collected_data):
        result = FounderExtractor().extract(sample_startup, sample_collected_data)
        assert result.founder_confidence == 0.0

    def test_confidence_increases_with_founders(self):
        data = CollectedData(startup_name="TestCo", founder_count=2)
        startup = Startup(
            name="TestCo",
            website="https://testco.example.com",
            description="A startup with a team of 20 people.",
        )
        result = FounderExtractor().extract(startup, data)
        assert result.founder_confidence > 0.0

    def test_confidence_bounded_at_one(self):
        startup = Startup(
            name="MaxCo",
            website="https://maxco.example.com",
            description=(
                "Founded by ex-Google AI researchers. Serial entrepreneurs "
                "who previously founded multiple startups. Healthcare domain "
                "expertise with clinical background. Engineering team of 80 "
                "developers building proprietary platform with 3 patents. "
                "$12M ARR from 200 enterprise customers. Raised $65M Series B. "
                "Advisory board of industry leaders."
            ),
            founder_linkedin_urls=[
                "https://linkedin.com/in/maxco-ceo",
                "https://linkedin.com/in/maxco-cto",
                "https://linkedin.com/in/maxco-coo",
            ],
        )
        data = CollectedData(startup_name="MaxCo", founder_count=3)
        result = FounderExtractor().extract(startup, data)
        assert result.founder_confidence <= 1.0
        assert result.founder_confidence > 0.5

    # ------------------------------------------------------------------
    # Integration: full signal extraction from rich description
    # ------------------------------------------------------------------

    def test_full_signal_extraction(self):
        startup = Startup(
            name="FullCo",
            website="https://fullco.example.com",
            description=(
                "Founded in 2021 by two ex-GitHub engineers. Team of 35 "
                "engineers and sales professionals. Engineering team building "
                "cloud infrastructure platform with microservices. "
                "$4.2M ARR with 85 enterprise customers. Raised $12M Series A. "
                "Patent pending on proprietary algorithm."
            ),
            founder_linkedin_urls=[
                "https://linkedin.com/in/fullco-ceo",
                "https://linkedin.com/in/fullco-cto",
            ],
        )
        data = CollectedData(startup_name="FullCo", founder_count=2)
        result = FounderExtractor().extract(startup, data)

        assert result.founder_profile_count == 2
        assert result.team_size_indicator == "11-50"
        assert result.founder_team_type in ("technical", "mixed")
        assert len(result.serial_founder_indicators) > 0
        assert len(result.leadership_roles) >= 2
        assert len(result.execution_signals) > 0
        assert result.founder_confidence > 0.3
        assert result.engineering_strength in ("strong", "moderate")
        assert result.product_strength in ("strong", "moderate")


# ---------------------------------------------------------------------------
# Sprint 14 — Structured Quantitative Extraction Tests
# ---------------------------------------------------------------------------


class TestFounderQuantitativeExtraction:
    def test_team_size_numeric_team_of(self, sample_collected_data):
        startup = Startup(
            name="TeamCo",
            website="https://teamco.example.com",
            description="Team of 42 engineers building AI",
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert result.team_size_numeric == 42

    def test_team_size_numeric_employees(self, sample_collected_data):
        startup = Startup(
            name="TeamCo",
            website="https://teamco.example.com",
            description="employees about 25 people",
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert result.team_size_numeric == 25

    def test_team_size_numeric_person_team(self, sample_collected_data):
        startup = Startup(
            name="TeamCo",
            website="https://teamco.example.com",
            description="10-person team",
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert result.team_size_numeric == 10

    def test_team_size_numeric_engineer_team(self, sample_collected_data):
        startup = Startup(
            name="TeamCo",
            website="https://teamco.example.com",
            description="5 engineer team",
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert result.team_size_numeric == 5

    def test_team_size_numeric_none(self, sample_collected_data):
        startup = Startup(
            name="TeamCo",
            website="https://teamco.example.com",
            description="A great product with no team info",
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert result.team_size_numeric is None

    def test_team_size_numeric_large(self, sample_collected_data):
        startup = Startup(
            name="TeamCo",
            website="https://teamco.example.com",
            description="team of 200 people across 3 offices",
        )
        result = FounderExtractor().extract(startup, sample_collected_data)
        assert result.team_size_numeric == 200
