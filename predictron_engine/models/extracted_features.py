"""Structured factual attributes extracted from startup data."""

from pydantic import BaseModel, Field


class ExtractedFeatures(BaseModel):
    """Pure factual attributes derived from startup data.

    This model carries only objective, verifiable facts. No scoring,
    no subjective judgments — just structured data that downstream
    reasoning and scoring modules consume.

    Every field is nullable or defaulted. Extraction is best-effort:
    if the extractor cannot determine an attribute from available data,
    the field remains None. Downstream modules must handle missing
    features gracefully.
    """

    industry: str | None = Field(
        default=None, description="Primary industry classification"
    )
    sub_industry: str | None = Field(
        default=None, description="Secondary industry classification"
    )
    business_model: str | None = Field(
        default=None, description="Business model type (e.g. SaaS, marketplace)"
    )
    funding_stage: str | None = Field(
        default=None, description="Current or most recent funding stage"
    )
    geography: str | None = Field(
        default=None, description="Primary geographic market"
    )
    headquarters_region: str | None = Field(
        default=None, description="HQ region if determinable"
    )
    technology_stack: list[str] = Field(
        default_factory=list, description="Identified technologies in use"
    )
    customer_type: str | None = Field(
        default=None, description="Target customer segment (B2B, B2C, B2B2C)"
    )
    team_size_indicator: str | None = Field(
        default=None, description="Approximate team size range"
    )
    founded_year: int | None = Field(
        default=None, description="Year the company was founded"
    )
    has_revenue: bool | None = Field(
        default=None, description="Whether revenue generation is evident"
    )
    key_keywords: list[str] = Field(
        default_factory=list, description="Key terms extracted from description"
    )
    description_length: int = Field(
        default=0, description="Character length of description"
    )
    has_pitch_deck: bool = Field(
        default=False, description="Whether a pitch deck was provided"
    )
    founder_profile_count: int = Field(
        default=0, description="Number of founder profiles"
    )
    data_completeness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of feature fields that are non-null",
    )

    # --- Market Intelligence fields (Sprint 1) ---

    target_market: str | None = Field(
        default=None,
        description="Primary target market description",
    )
    market_maturity: str | None = Field(
        default=None,
        description="Market maturity stage (emerging, growth, mature, saturated)",
    )
    market_keywords: list[str] = Field(
        default_factory=list,
        description="Domain-specific market terms extracted from description",
    )
    market_signals: list[str] = Field(
        default_factory=list,
        description="Detected market signals",
    )
    market_characteristics: list[str] = Field(
        default_factory=list,
        description="Detected market characteristics",
    )
    enterprise_orientation: str | None = Field(
        default=None,
        description="Enterprise vs consumer orientation (enterprise, consumer, hybrid)",
    )
    industry_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the industry classification (0.0-1.0)",
    )
    customer_segment: str | None = Field(
        default=None,
        description="Specific customer segment",
    )

    # --- Founder Intelligence fields (Sprint 2) ---

    founder_team_type: str | None = Field(
        default=None,
        description="Technical vs business founder emphasis (technical, business, mixed, unknown)",
    )
    domain_expertise_signals: list[str] = Field(
        default_factory=list,
        description="Detected domain expertise indicators from founder context",
    )
    serial_founder_indicators: list[str] = Field(
        default_factory=list,
        description="Signals suggesting founders have previous startup experience",
    )
    leadership_roles: list[str] = Field(
        default_factory=list,
        description="Detected leadership roles (CEO, CTO, COO, etc.)",
    )
    hiring_signals: list[str] = Field(
        default_factory=list,
        description="Detected hiring and team growth signals",
    )
    advisor_mentions: list[str] = Field(
        default_factory=list,
        description="Detected advisor, board, or mentor mentions",
    )
    engineering_strength: str | None = Field(
        default=None,
        description="Engineering team strength indicator (strong, moderate, weak, unknown)",
    )
    product_strength: str | None = Field(
        default=None,
        description="Product capability indicator (strong, moderate, weak, unknown)",
    )
    founder_market_fit_signals: list[str] = Field(
        default_factory=list,
        description="Signals of founder-market fit alignment",
    )
    execution_signals: list[str] = Field(
        default_factory=list,
        description="Detected execution and traction signals",
    )
    founder_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Composite confidence score for founder quality (0.0-1.0)",
    )

    # --- Product Intelligence fields (Sprint 3) ---

    product_category: str | None = Field(
        default=None,
        description="Primary product category (analytics, payments, diagnostics, etc.)",
    )
    product_type: str | None = Field(
        default=None,
        description="Product form factor (platform, application, tool, api, infrastructure)",
    )
    saas_model: str | None = Field(
        default=None,
        description=(
            "Delivery model (saas, api, infrastructure, marketplace,"
            " licensing, paas, raas)"
        ),
    )
    ai_orientation: str | None = Field(
        default=None,
        description="AI relationship (ai_native, ai_enabled, non_ai, unknown)",
    )
    primary_capabilities: list[str] = Field(
        default_factory=list,
        description="Core product capabilities detected from description",
    )
    feature_signals: list[str] = Field(
        default_factory=list,
        description="Specific product features mentioned in description",
    )
    integration_ecosystem: list[str] = Field(
        default_factory=list,
        description="Detected integration points and ecosystem connections",
    )
    deployment_model: str | None = Field(
        default=None,
        description="Deployment approach (cloud, on_premise, hybrid, saas, ras, edge)",
    )
    target_workflow: str | None = Field(
        default=None,
        description="Primary workflow the product addresses",
    )
    automation_level: str | None = Field(
        default=None,
        description="Degree of automation (high, moderate, low, manual)",
    )
    product_maturity: str | None = Field(
        default=None,
        description="Product lifecycle stage (concept, beta, growth, mature, legacy)",
    )
    differentiation_signals: list[str] = Field(
        default_factory=list,
        description="Detected competitive differentiators",
    )
    defensibility_signals: list[str] = Field(
        default_factory=list,
        description="Detected moat and defensibility indicators",
    )
    technical_complexity: str | None = Field(
        default=None,
        description="Estimated technical complexity (high, moderate, low, unknown)",
    )
    scalability_indicators: list[str] = Field(
        default_factory=list,
        description="Detected scalability and growth enablers",
    )
    innovation_signals: list[str] = Field(
        default_factory=list,
        description="Detected innovation and novelty indicators",
    )
    product_keywords: list[str] = Field(
        default_factory=list,
        description="Domain-specific product terms extracted from description",
    )
    product_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Composite confidence in product extraction accuracy (0.0-1.0)",
    )

    # --- Technology Intelligence fields (Sprint 5) ---

    primary_technology_domain: str | None = Field(
        default=None,
        description=(
            "Primary technology domain classification"
            " (ai_ml, cloud_infrastructure, data_engineering, cybersecurity,"
            " web_platform, mobile, devops, fintech_infra, healthtech_infra,"
            " iot, blockchain, robotics, enterprise_software, consumer_tech)"
        ),
    )
    secondary_technology_domain: str | None = Field(
        default=None,
        description="Secondary technology domain if applicable",
    )
    programming_language_signals: list[str] = Field(
        default_factory=list,
        description="Programming language signals detected from description",
    )
    framework_signals: list[str] = Field(
        default_factory=list,
        description="Framework and library signals detected from description",
    )
    cloud_infrastructure_signals: list[str] = Field(
        default_factory=list,
        description="Cloud provider and infrastructure service signals",
    )
    api_strategy: str | None = Field(
        default=None,
        description=(
            "API architecture strategy"
            " (rest, graphql, grpc, websocket, event_driven, mixed, unknown)"
        ),
    )
    data_architecture_signals: list[str] = Field(
        default_factory=list,
        description="Data architecture signals (databases, pipelines, warehouses)",
    )
    security_signals: list[str] = Field(
        default_factory=list,
        description="Security architecture signals detected from description",
    )
    infrastructure_maturity: str | None = Field(
        default=None,
        description=(
            "Infrastructure maturity level"
            " (early, developing, mature, enterprise_grade, unknown)"
        ),
    )
    open_source_signals: list[str] = Field(
        default_factory=list,
        description="Open source involvement and community signals",
    )
    developer_tooling_signals: list[str] = Field(
        default_factory=list,
        description="Developer tooling and DX signals",
    )
    engineering_maturity: str | None = Field(
        default=None,
        description=(
            "Engineering maturity assessment"
            " (nascent, developing, established, sophisticated, unknown)"
        ),
    )
    technology_keywords: list[str] = Field(
        default_factory=list,
        description="Domain-specific technology terms extracted from description",
    )
    technology_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Composite confidence in technology extraction accuracy (0.0-1.0)",
    )

    # --- Business Model Intelligence fields (Sprint 4) ---

    secondary_business_model: str | None = Field(
        default=None,
        description="Secondary business model if hybrid (e.g. saas + marketplace)",
    )
    revenue_model: str | None = Field(
        default=None,
        description=(
            "How the company generates revenue"
            " (subscription, licensing, transaction_fee, commission,"
            " freemium, advertising, usage_based)"
        ),
    )
    pricing_model: str | None = Field(
        default=None,
        description=(
            "Pricing structure"
            " (annual_contract, per_seat, per_study, tiered,"
            " usage_based, freemium, enterprise_contract, flat_rate)"
        ),
    )
    monetization_strategy: str | None = Field(
        default=None,
        description=(
            "Overall monetization approach"
            " (subscription, licensing, transaction_fee, commission,"
            " freemium, advertising, hybrid)"
        ),
    )
    customer_acquisition_model: str | None = Field(
        default=None,
        description=(
            "Primary customer acquisition approach"
            " (product_led, sales_led, hybrid, marketplace_organic)"
        ),
    )
    sales_motion: str | None = Field(
        default=None,
        description=(
            "Type of sales process"
            " (enterprise_sales, product_led, self_serve, hybrid)"
        ),
    )
    distribution_model: str | None = Field(
        default=None,
        description=(
            "How the product reaches customers"
            " (direct, api, app_store, open_source, marketplace, partner)"
        ),
    )
    value_proposition_signals: list[str] = Field(
        default_factory=list,
        description="Detected value proposition indicators from description",
    )
    recurring_revenue_signal: str | None = Field(
        default=None,
        description=(
            "Revenue recurrence pattern"
            " (recurring, transactional, mixed, unknown)"
        ),
    )
    marketplace_dynamics: list[str] = Field(
        default_factory=list,
        description="Detected marketplace-specific characteristics",
    )
    network_effects_signals: list[str] = Field(
        default_factory=list,
        description="Detected network effect indicators",
    )
    platform_characteristics: list[str] = Field(
        default_factory=list,
        description="Detected platform-specific traits",
    )
    switching_cost_indicators: list[str] = Field(
        default_factory=list,
        description="Detected lock-in and switching cost signals",
    )
    unit_economics_indicators: list[str] = Field(
        default_factory=list,
        description="Detected unit economics signals (LTV, CAC, margins)",
    )
    business_model_maturity: str | None = Field(
        default=None,
        description=(
            "Business model lifecycle stage"
            " (nascent, early, growth, mature, evolving)"
        ),
    )
    business_model_keywords: list[str] = Field(
        default_factory=list,
        description="Domain-specific business model terms extracted",
    )
    business_model_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Composite confidence in business model extraction (0.0-1.0)",
    )

    # --- Traction Intelligence fields (Sprint 6) ---

    funding_amount_signals: list[str] = Field(
        default_factory=list,
        description="Detected funding amount signals (e.g. '$12M raised', '$45M Series B')",
    )
    investor_signals: list[str] = Field(
        default_factory=list,
        description="Detected investor type signals (e.g. 'Tier 1 VCs', 'angel investors')",
    )
    grants_accelerator_signals: list[str] = Field(
        default_factory=list,
        description="Detected grants, accelerators, or incubator signals",
    )
    revenue_amount_signals: list[str] = Field(
        default_factory=list,
        description="Detected revenue amount signals (e.g. '$4.2M ARR', '$2.8M ARR')",
    )
    arr_mrr_signals: list[str] = Field(
        default_factory=list,
        description="Detected ARR/MRR specific signals with amounts",
    )
    gmv_signals: list[str] = Field(
        default_factory=list,
        description="Detected GMV (Gross Merchandise Volume) signals",
    )
    customer_count_signals: list[str] = Field(
        default_factory=list,
        description="Detected customer/client count signals",
    )
    active_user_signals: list[str] = Field(
        default_factory=list,
        description="Detected active user signals (MAU, DAU, downloads)",
    )
    enterprise_customer_signals: list[str] = Field(
        default_factory=list,
        description="Detected enterprise customer signals (Fortune 500, large contracts)",
    )
    pilot_customer_signals: list[str] = Field(
        default_factory=list,
        description="Detected pilot or trial customer signals",
    )
    paying_customer_signals: list[str] = Field(
        default_factory=list,
        description="Detected paying customer signals",
    )
    partnership_signals: list[str] = Field(
        default_factory=list,
        description="Detected partnership signals",
    )
    retention_signals: list[str] = Field(
        default_factory=list,
        description="Detected retention signals (net retention, gross retention, churn)",
    )
    engagement_signals: list[str] = Field(
        default_factory=list,
        description="Detected engagement signals (usage frequency, session depth)",
    )
    product_adoption_signals: list[str] = Field(
        default_factory=list,
        description="Detected product adoption signals (beta users, early adopters)",
    )
    growth_signals: list[str] = Field(
        default_factory=list,
        description="Detected growth signals (MoM growth, user growth rate)",
    )
    hiring_growth_signals: list[str] = Field(
        default_factory=list,
        description="Detected hiring and team growth signals specific to traction",
    )
    expansion_signals: list[str] = Field(
        default_factory=list,
        description="Detected expansion signals (new markets, new products, geographic expansion)",
    )
    launch_signals: list[str] = Field(
        default_factory=list,
        description="Detected launch signals (product launch, market launch)",
    )
    milestone_signals: list[str] = Field(
        default_factory=list,
        description="Detected milestone signals (100th customer, first enterprise deal)",
    )
    awards_recognition: list[str] = Field(
        default_factory=list,
        description="Detected awards and recognition signals",
    )
    traction_keywords: list[str] = Field(
        default_factory=list,
        description="Domain-specific traction terms extracted from description",
    )
    traction_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Composite confidence in traction extraction accuracy (0.0-1.0)",
    )

    # --- Competitive Intelligence fields (Sprint 7) ---

    direct_competitor_signals: list[str] = Field(
        default_factory=list,
        description="Detected direct competitor mentions or references",
    )
    indirect_competitor_signals: list[str] = Field(
        default_factory=list,
        description="Detected indirect or adjacent competitor signals",
    )
    incumbent_signals: list[str] = Field(
        default_factory=list,
        description="Detected incumbent or legacy player signals",
    )
    market_concentration: str | None = Field(
        default=None,
        description=(
            "Market concentration level"
            " (fragmented, moderately_concentrated, concentrated,"
            " dominated, unknown)"
        ),
    )
    competitive_density: str | None = Field(
        default=None,
        description=(
            "Density of competitive activity"
            " (sparse, moderate, dense, hyper_competitive, unknown)"
        ),
    )
    fragmentation_signals: list[str] = Field(
        default_factory=list,
        description="Detected market fragmentation indicators",
    )
    winner_take_most_signals: list[str] = Field(
        default_factory=list,
        description="Detected winner-take-most or winner-take-all dynamics",
    )
    network_effect_competition: str | None = Field(
        default=None,
        description=(
            "Network effect competitive dynamics"
            " (strong_network_effects, moderate_network_effects,"
            " no_network_effects, unknown)"
        ),
    )
    switching_cost_signals: list[str] = Field(
        default_factory=list,
        description="Detected switching cost and lock-in competitive signals",
    )
    differentiation_signals: list[str] = Field(
        default_factory=list,
        description="Detected competitive differentiation signals",
    )
    competitive_moat_indicators: list[str] = Field(
        default_factory=list,
        description="Detected competitive moat indicators",
    )
    barriers_to_entry: list[str] = Field(
        default_factory=list,
        description="Detected barriers to entry in the market",
    )
    substitute_product_signals: list[str] = Field(
        default_factory=list,
        description="Detected substitute product or alternative solution signals",
    )
    platform_dependency: list[str] = Field(
        default_factory=list,
        description="Detected platform dependency competitive signals",
    )
    ecosystem_dependency: list[str] = Field(
        default_factory=list,
        description="Detected ecosystem dependency competitive signals",
    )
    open_source_competition: list[str] = Field(
        default_factory=list,
        description="Detected open-source competition signals",
    )
    regulatory_competition: list[str] = Field(
        default_factory=list,
        description="Detected regulatory competitive advantage signals",
    )
    geographic_competition: list[str] = Field(
        default_factory=list,
        description="Detected geographic competition signals",
    )
    pricing_pressure: list[str] = Field(
        default_factory=list,
        description="Detected pricing pressure or price competition signals",
    )
    competitive_keywords: list[str] = Field(
        default_factory=list,
        description="Domain-specific competitive terms extracted from description",
    )
    competition_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Composite confidence in competition extraction accuracy (0.0-1.0)",
    )

    # --- Risk Intelligence fields (Sprint 9) ---

    market_risk: list[str] = Field(
        default_factory=list,
        description="Detected market-related risk signals",
    )
    founder_risk: list[str] = Field(
        default_factory=list,
        description="Detected founder and team risk signals",
    )
    execution_risk: list[str] = Field(
        default_factory=list,
        description="Detected execution and delivery risk signals",
    )
    product_risk: list[str] = Field(
        default_factory=list,
        description="Detected product-related risk signals",
    )
    technology_risk: list[str] = Field(
        default_factory=list,
        description="Detected technology and technical risk signals",
    )
    business_model_risk: list[str] = Field(
        default_factory=list,
        description="Detected business model risk signals",
    )
    traction_risk: list[str] = Field(
        default_factory=list,
        description="Detected traction and growth risk signals",
    )
    competitive_risk: list[str] = Field(
        default_factory=list,
        description="Detected competitive landscape risk signals",
    )
    regulatory_risk: list[str] = Field(
        default_factory=list,
        description="Detected regulatory and legal risk signals",
    )
    operational_risk: list[str] = Field(
        default_factory=list,
        description="Detected operational risk signals",
    )
    platform_dependency_risk: list[str] = Field(
        default_factory=list,
        description="Detected platform dependency risk signals",
    )
    customer_concentration_risk: list[str] = Field(
        default_factory=list,
        description="Detected customer concentration risk signals",
    )
    hiring_risk: list[str] = Field(
        default_factory=list,
        description="Detected hiring and talent risk signals",
    )
    funding_risk: list[str] = Field(
        default_factory=list,
        description="Detected funding and financial risk signals",
    )
    scaling_risk: list[str] = Field(
        default_factory=list,
        description="Detected scaling and growth risk signals",
    )
    security_risk: list[str] = Field(
        default_factory=list,
        description="Detected security risk signals",
    )
    compliance_risk: list[str] = Field(
        default_factory=list,
        description="Detected compliance risk signals",
    )
    risk_keywords: list[str] = Field(
        default_factory=list,
        description="Domain-specific risk terms extracted from description",
    )
    risk_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Composite confidence in risk extraction accuracy (0.0-1.0)",
    )
