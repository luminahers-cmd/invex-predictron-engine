"""Comprehensive tests for TechnologyExtractor — Intelligence Sprint 5.

Covers:
  - Backward-compatible extraction (domain, enrichment, dedup)
  - Primary / secondary technology domain classification
  - Programming language signal detection
  - Framework signal detection
  - Cloud infrastructure signal detection
  - API strategy classification
  - Data architecture signal detection
  - Security signal detection
  - Infrastructure maturity assessment
  - Engineering maturity assessment
  - Open source signal detection
  - Developer tooling signal detection
  - Technology keyword extraction
  - Technology confidence scoring
  - Edge cases and precision checks
"""

from __future__ import annotations

from predictron_engine.extraction.extractors.technology import TechnologyExtractor
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup


def _make_startup(description: str) -> Startup:
    return Startup(
        name="TestCo",
        website="https://testco.example.com",
        description=description,
        pitch_deck_url=None,
        founder_linkedin_urls=[],
        raw_data={},
    )


def _make_data(
    domain: str = "testco.example.com",
    enrichment: dict[str, str | int | float | bool] | None = None,
) -> CollectedData:
    return CollectedData(
        startup_name="TestCo",
        website_domain=domain,
        enrichment_signals=enrichment or {},
    )


class TestTechnologyExtractorBasics:
    def test_returns_extracted_features(self, sample_startup, sample_collected_data):
        result = TechnologyExtractor().extract(sample_startup, sample_collected_data)
        assert isinstance(result, ExtractedFeatures)

    def test_tech_from_domain(self, sample_startup):
        data = CollectedData(
            startup_name="ShopCo",
            website_domain="shopco.vercel.app",
        )
        result = TechnologyExtractor().extract(sample_startup, data)
        assert "vercel" in result.technology_stack

    def test_tech_from_enrichment_signals(self, sample_startup):
        data = CollectedData(
            startup_name="EnrichCo",
            enrichment_signals={"primary_tech": "python"},
        )
        result = TechnologyExtractor().extract(sample_startup, data)
        assert "python" in result.technology_stack

    def test_no_tech_signals(self, sample_startup, sample_collected_data):
        result = TechnologyExtractor().extract(sample_startup, sample_collected_data)
        assert result.technology_stack == []

    def test_deduplicates_tech_signals(self, sample_startup):
        data = CollectedData(
            startup_name="DupCo",
            website_domain="dupco.shopify.com",
            enrichment_signals={"stack": "shopify"},
        )
        result = TechnologyExtractor().extract(sample_startup, data)
        assert result.technology_stack.count("shopify") == 1

    def test_new_fields_populated(self, sample_startup, sample_collected_data):
        result = TechnologyExtractor().extract(sample_startup, sample_collected_data)
        assert result.primary_technology_domain is None
        assert result.secondary_technology_domain is None
        assert result.programming_language_signals == []
        assert result.framework_signals == []
        assert result.cloud_infrastructure_signals == []
        assert result.api_strategy is None
        assert result.data_architecture_signals == []
        assert result.security_signals == []
        assert result.infrastructure_maturity == "unknown"
        assert result.open_source_signals == []
        assert result.developer_tooling_signals == []
        assert result.engineering_maturity == "unknown"
        assert result.technology_keywords == []
        assert result.technology_confidence < 0.2


class TestPrimaryTechnologyDomain:
    def test_ai_ml_domain(self):
        startup = _make_startup(
            "We build machine learning models using deep learning and PyTorch "
            "for computer vision. Our neural network architecture processes "
            "training data at scale on GPU clusters."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.primary_technology_domain == "ai_ml"

    def test_cloud_infrastructure_domain(self):
        startup = _make_startup(
            "Our cloud infrastructure platform provides Kubernetes orchestration, "
            "Terraform infrastructure as code, and container orchestration across "
            "multi-cloud environments with auto-scaling."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.primary_technology_domain == "cloud_infrastructure"

    def test_data_engineering_domain(self):
        startup = _make_startup(
            "We provide data pipeline orchestration with ETL, data warehouse "
            "management, Kafka streaming, and Airflow scheduling. Our data "
            "engineering platform integrates Snowflake and Databricks."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.primary_technology_domain == "data_engineering"

    def test_cybersecurity_domain(self):
        startup = _make_startup(
            "Our cybersecurity platform provides threat detection, vulnerability "
            "scanning, zero trust architecture, and SOC 2 compliance automation "
            "with endpoint detection and security operations."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.primary_technology_domain == "cybersecurity"

    def test_devops_domain(self):
        startup = _make_startup(
            "Our DevOps platform provides CI/CD, continuous integration and "
            "continuous deployment with build pipeline management, deployment "
            "orchestration, and infrastructure automation."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.primary_technology_domain == "devops"

    def test_fintech_infra_domain(self):
        startup = _make_startup(
            "Payment infrastructure for platforms with embedded payment, "
            "payment API, split payment, escrow, KYC compliance, and "
            "multi-currency settlement for transaction processing."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.primary_technology_domain == "fintech_infra"

    def test_no_domain_when_ambiguous(self):
        startup = _make_startup(
            "We build a platform that combines analytics and automation."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.primary_technology_domain is None

    def test_no_domain_when_weak_signals(self):
        startup = _make_startup("A small tool for tracking tasks.")
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.primary_technology_domain is None


class TestSecondaryTechnologyDomain:
    def test_secondary_domain_detected(self):
        startup = _make_startup(
            "Our machine learning platform uses deep learning neural networks "
            "with PyTorch and TensorFlow for computer vision. We deploy on "
            "Kubernetes with Terraform infrastructure as code and Docker "
            "container orchestration across multi-cloud environments."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.primary_technology_domain == "ai_ml"
        assert result.secondary_technology_domain == "cloud_infrastructure"

    def test_secondary_domain_excluded_when_same_as_primary(self):
        startup = _make_startup(
            "Our machine learning platform uses deep learning, neural networks, "
            "PyTorch, TensorFlow, computer vision, model serving, inference, "
            "training data, GPU clusters, and model deployment."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.primary_technology_domain == "ai_ml"
        # Secondary should not be ai_ml even if signals are strong
        assert result.secondary_technology_domain != "ai_ml"

    def test_no_secondary_when_weak(self):
        startup = _make_startup(
            "We use machine learning and deep learning for our AI-powered "
            "neural network models."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.primary_technology_domain == "ai_ml"
        assert result.secondary_technology_domain is None


class TestProgrammingLanguageSignals:
    def test_python_detected(self):
        startup = _make_startup("Built with Python and Django for data processing.")
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "python" in result.programming_language_signals

    def test_javascript_and_typescript(self):
        startup = _make_startup(
            "Full-stack JavaScript and TypeScript application using Node.js."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "javascript" in result.programming_language_signals
        assert "typescript" in result.programming_language_signals

    def test_rust_detected(self):
        startup = _make_startup("High-performance backend written in Rust.")
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "rust" in result.programming_language_signals

    def test_multiple_languages(self):
        startup = _make_startup(
            "Backend in Python and Go, frontend in TypeScript, mobile in Swift."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "python" in result.programming_language_signals
        assert "typescript" in result.programming_language_signals
        assert "swift" in result.programming_language_signals

    def test_no_false_positive_java_for_javascript(self):
        startup = _make_startup(
            "JavaScript application with Node.js backend."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "javascript" in result.programming_language_signals
        # "java" should NOT appear when only "javascript" is present
        assert "java" not in result.programming_language_signals

    def test_no_languages_for_vague_description(self):
        startup = _make_startup("A platform for team collaboration.")
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.programming_language_signals == []


class TestFrameworkSignals:
    def test_react_detected(self):
        startup = _make_startup("Frontend built with React and Next.js.")
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "react" in result.framework_signals

    def test_kubernetes_detected(self):
        startup = _make_startup("Deployed on Kubernetes with Docker containers.")
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "kubernetes" in result.framework_signals
        assert "docker" in result.framework_signals

    def test_ml_frameworks(self):
        startup = _make_startup(
            "Model training with PyTorch and TensorFlow, served via FastAPI."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "pytorch" in result.framework_signals
        assert "tensorflow" in result.framework_signals
        assert "fastapi" in result.framework_signals

    def test_no_frameworks_for_vague_description(self):
        startup = _make_startup("Building innovative solutions for enterprises.")
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.framework_signals == []


class TestCloudInfrastructureSignals:
    def test_aws_detected(self):
        startup = _make_startup("Hosted on AWS with EC2 and S3 storage.")
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "aws" in result.cloud_infrastructure_signals

    def test_multi_cloud(self):
        startup = _make_startup(
            "Multi-cloud deployment on AWS and GCP with Kubernetes."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "aws" in result.cloud_infrastructure_signals
        assert "gcp" in result.cloud_infrastructure_signals
        assert "kubernetes" in result.cloud_infrastructure_signals

    def test_azure_with_services(self):
        startup = _make_startup(
            "Azure cloud platform with Azure Kubernetes Service and serverless."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "azure" in result.cloud_infrastructure_signals

    def test_no_cloud_for_premise_description(self):
        startup = _make_startup("On-premise deployment with self-hosted servers.")
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.cloud_infrastructure_signals == []


class TestApiStrategy:
    def test_rest_api(self):
        startup = _make_startup(
            "REST API for data access with JSON API responses."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.api_strategy == "rest"

    def test_graphql_api(self):
        startup = _make_startup(
            "GraphQL API with schema-first design and GQL queries."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.api_strategy == "graphql"

    def test_grpc_api(self):
        startup = _make_startup(
            "High-performance gRPC API with protocol buffers."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.api_strategy == "grpc"

    def test_event_driven(self):
        startup = _make_startup(
            "Event-driven architecture with message queues and pub/sub."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.api_strategy == "event_driven"

    def test_mixed_api_strategy(self):
        startup = _make_startup(
            "REST API for external clients and GraphQL for internal services, "
            "with WebSocket connections for real-time updates."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.api_strategy == "mixed"

    def test_no_api_strategy_for_vague_description(self):
        startup = _make_startup("A platform for managing workflows.")
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.api_strategy is None


class TestDataArchitectureSignals:
    def test_postgresql_detected(self):
        startup = _make_startup("PostgreSQL database for transactional data.")
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "postgresql" in result.data_architecture_signals

    def test_multiple_databases(self):
        startup = _make_startup(
            "PostgreSQL for transactions, Redis for caching, "
            "and Elasticsearch for search."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "postgresql" in result.data_architecture_signals
        assert "redis" in result.data_architecture_signals
        assert "elasticsearch" in result.data_architecture_signals

    def test_cloud_data_warehouses(self):
        startup = _make_startup(
            "Data warehouse on Snowflake with Databricks for analytics."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "snowflake" in result.data_architecture_signals
        assert "databricks" in result.data_architecture_signals

    def test_vector_database(self):
        startup = _make_startup(
            "Vector database for similarity search with embeddings."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "vector_database" in result.data_architecture_signals

    def test_no_data_architecture_for_vague(self):
        startup = _make_startup("Simple task management tool.")
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.data_architecture_signals == []


class TestSecuritySignals:
    def test_encryption_detected(self):
        startup = _make_startup(
            "End-to-end encryption with OAuth authentication."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "encryption" in result.security_signals
        assert "authentication" in result.security_signals

    def test_compliance_frameworks(self):
        startup = _make_startup(
            "SOC 2 compliant platform with HIPAA, GDPR, and PCI DSS adherence."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "soc2" in result.security_signals
        assert "hipaa" in result.security_signals
        assert "gdpr" in result.security_signals
        assert "pci_dss" in result.security_signals

    def test_zero_trust(self):
        startup = _make_startup(
            "Zero trust architecture with SSO, MFA, and RBAC."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "zero_trust" in result.security_signals
        assert "sso" in result.security_signals
        assert "mfa" in result.security_signals
        assert "rbac" in result.security_signals

    def test_no_security_for_simple_tool(self):
        startup = _make_startup("A simple note-taking application.")
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.security_signals == []


class TestInfrastructureMaturity:
    def test_enterprise_grade(self):
        startup = _make_startup(
            "Running on Kubernetes with Terraform infrastructure as code, "
            "multi-cloud deployment, auto-scaling, service mesh, and "
            "container orchestration with GitOps."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.infrastructure_maturity == "enterprise_grade"

    def test_mature(self):
        startup = _make_startup(
            "Cloud-native platform with Docker containers, "
            "CI/CD pipelines, and monitoring. "
            "Deployed on hosted infrastructure with SSL encryption."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.infrastructure_maturity == "mature"

    def test_developing(self):
        startup = _make_startup(
            "Cloud-hosted platform with API and database, "
            "with basic authentication."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.infrastructure_maturity == "developing"

    def test_early(self):
        startup = _make_startup(
            "Simple hosted application."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.infrastructure_maturity == "early"

    def test_unknown_when_no_signals(self):
        startup = _make_startup("Building innovative solutions.")
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.infrastructure_maturity == "unknown"


class TestEngineeringMaturity:
    def test_sophisticated(self):
        startup = _make_startup(
            "Platform engineering team using GitOps, infrastructure as code, "
            "service mesh, Kubernetes, Terraform, with canary deployments "
            "and feature flags for A/B testing."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.engineering_maturity == "sophisticated"

    def test_established(self):
        startup = _make_startup(
            "CI/CD with continuous integration and deployment, Docker "
            "containers, automated testing, code review, monitoring, "
            "and observability."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.engineering_maturity == "established"

    def test_developing(self):
        startup = _make_startup(
            "Agile development with API backend and cloud deployment."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.engineering_maturity == "developing"

    def test_nascent(self):
        startup = _make_startup("Early stage with basic sprints.")
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.engineering_maturity == "nascent"

    def test_unknown_when_no_signals(self):
        startup = _make_startup("A company building products.")
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.engineering_maturity == "unknown"


class TestOpenSourceSignals:
    def test_open_source_project(self):
        startup = _make_startup(
            "Open-source CI/CD platform with 14,000 GitHub stars "
            "and 850 contributing developers."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "open_source" in result.open_source_signals
        assert "github_stars" in result.open_source_signals
        assert "github_repository" in result.open_source_signals
        assert "contributors" in result.open_source_signals

    def test_oss_first(self):
        startup = _make_startup(
            "Open-source-first developer tool with a self-hosted option."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "oss_first" in result.open_source_signals
        assert "self_hosted_option" in result.open_source_signals

    def test_no_open_source_for_proprietary(self):
        startup = _make_startup("Proprietary enterprise SaaS platform.")
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.open_source_signals == []


class TestDeveloperToolingSignals:
    def test_cli_and_sdk(self):
        startup = _make_startup(
            "Developer tool with CLI, SDK, and API documentation."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "cli" in result.developer_tooling_signals
        assert "sdk" in result.developer_tooling_signals
        assert "api_documentation" in result.developer_tooling_signals

    def test_ide_extension(self):
        startup = _make_startup(
            "VS Code extension for code generation and debugging tools."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "vscode_extension" in result.developer_tooling_signals
        assert "code_generation" in result.developer_tooling_signals
        assert "debugging_tools" in result.developer_tooling_signals

    def test_playground_and_sandbox(self):
        startup = _make_startup(
            "Interactive playground and sandbox for testing APIs."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "playground" in result.developer_tooling_signals
        assert "sandbox" in result.developer_tooling_signals

    def test_no_devtools_for_non_technical(self):
        startup = _make_startup("A marketplace for handmade goods.")
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.developer_tooling_signals == []


class TestTechnologyKeywords:
    def test_cloud_native_keywords(self):
        startup = _make_startup(
            "Cloud-native SaaS platform with microservices, serverless, "
            "and real-time AI-powered analytics."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "cloud_native" in result.technology_keywords
        assert "microservices" in result.technology_keywords
        assert "serverless" in result.technology_keywords
        assert "real_time" in result.technology_keywords
        assert "ai_powered" in result.technology_keywords
        assert "saas" in result.technology_keywords

    def test_devops_keywords(self):
        startup = _make_startup(
            "DevOps platform with GitOps, CI/CD, infrastructure as code, "
            "and observability."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "devops" in result.technology_keywords
        assert "gitops" in result.technology_keywords
        assert "ci_cd_integration" not in result.technology_keywords
        assert "observability" in result.technology_keywords

    def test_ai_keywords(self):
        startup = _make_startup(
            "LLM-powered generative AI platform with RAG and vector search."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "llm" in result.technology_keywords
        assert "generative_ai" in result.technology_keywords
        assert "rag" in result.technology_keywords
        assert "vector_search" in result.technology_keywords


class TestTechnologyConfidence:
    def test_high_confidence_rich_description(self):
        startup = _make_startup(
            "Machine learning platform built with Python, PyTorch, and "
            "TensorFlow deployed on AWS Kubernetes with Terraform. "
            "Data pipeline using Kafka and Spark for real-time inference. "
            "CI/CD with Docker containers and monitoring."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.technology_confidence >= 0.5

    def test_low_confidence_vague_description(self):
        startup = _make_startup("Building cool stuff.")
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.technology_confidence < 0.2

    def test_medium_confidence_moderate_description(self):
        startup = _make_startup(
            "Cloud platform with API and database integration."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert 0.1 <= result.technology_confidence <= 0.6


class TestEdgeCases:
    def test_empty_description(self):
        startup = _make_startup("X")
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.primary_technology_domain is None
        assert result.technology_stack == []
        assert result.technology_confidence < 0.1

    def test_case_insensitive_detection(self):
        startup = _make_startup(
            "PYTHON and PYTORCH for MACHINE LEARNING models."
        )
        result = TechnologyExtractor().extract(startup, _make_data())
        assert "python" in result.programming_language_signals
        assert "pytorch" in result.framework_signals
        assert result.primary_technology_domain == "ai_ml"

    def test_enrichment_plus_description_combined(self):
        startup = _make_startup(
            "AI-powered platform using deep learning and neural networks."
        )
        data = _make_data(
            enrichment={"primary_tech": "pytorch", "secondary_tech": "aws"},
        )
        result = TechnologyExtractor().extract(startup, data)
        assert "pytorch" in result.technology_stack
        assert "aws" in result.technology_stack

    def test_domain_hints_preserved(self):
        startup = _make_startup("E-commerce platform.")
        data = _make_data(domain="mystore.shopify.com")
        result = TechnologyExtractor().extract(startup, data)
        assert "shopify" in result.technology_stack


class TestBenchmarkCases:
    """Validate TechnologyExtractor against all 10 benchmark cases."""

    def test_b2b_saas_analytix_cloud(self):
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
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.data_architecture_signals != []
        assert result.technology_keywords != []
        assert result.technology_confidence > 0.0

    def test_healthcare_ai_medvision(self):
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
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.primary_technology_domain is not None
        assert result.technology_confidence > 0.0

    def test_fintech_paybridge(self):
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
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.primary_technology_domain == "fintech_infra"
        assert result.api_strategy is not None
        assert result.technology_confidence > 0.0

    def test_devtools_shipkit(self):
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
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.primary_technology_domain is not None
        assert len(result.open_source_signals) >= 2
        assert len(result.developer_tooling_signals) >= 1
        assert "ci_cd_integration" in result.developer_tooling_signals

    def test_marketplace_supplyhub(self):
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
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.technology_confidence > 0.0

    def test_consumer_app_fitsocial(self):
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
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.primary_technology_domain == "mobile"

    def test_climate_tech_carbonlens(self):
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
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.infrastructure_maturity is not None
        assert result.technology_confidence > 0.0
        assert len(result.technology_keywords) >= 1

    def test_robotics_autoware(self):
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
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.primary_technology_domain == "robotics"
        assert result.infrastructure_maturity is not None

    def test_enterprise_software_complianceos(self):
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
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.primary_technology_domain is not None
        assert len(result.security_signals) >= 2
        assert result.infrastructure_maturity is not None
        assert result.technology_confidence > 0.0

    def test_ai_infrastructure_inferencelabs(self):
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
        result = TechnologyExtractor().extract(startup, _make_data())
        assert result.primary_technology_domain == "ai_ml"
        assert "pytorch" in result.framework_signals
        assert "tensorflow" in result.framework_signals
        assert result.infrastructure_maturity is not None
        assert result.technology_confidence >= 0.5
