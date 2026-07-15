"""Technology extractor — rich deterministic technology intelligence extraction.

Responsibilities:
  - Primary / secondary technology domain classification via weighted keyword scoring
  - Technology stack detection from website domain, URL metadata, and enrichment data
  - Programming language signal detection
  - Framework and library signal detection
  - Cloud infrastructure signal detection
  - API strategy classification
  - Data architecture signal detection
  - Security architecture signal detection
  - Infrastructure maturity assessment
  - Open source signal detection
  - Developer tooling signal detection
  - Engineering maturity assessment
  - Technology keyword extraction
  - Technology extraction confidence scoring

Design principles:
  - Deterministic rule-based logic only (no LLMs, no ML)
  - Weighted keyword scoring with context-aware disambiguation
  - Every extracted field traceable to explicit input signals
  - Prefer "unknown" over incorrect classification
  - Modular rules that are easy to extend
  - Uses weighted evidence instead of first-match logic
  - Confidence thresholds to minimize false positives
"""

from __future__ import annotations

import re

from predictron_engine.extraction.extractors.base import BaseExtractor
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup

# ---------------------------------------------------------------------------
# Domain-based tech hints (kept from original for backward compat).
# ---------------------------------------------------------------------------

DOMAIN_TECH_HINTS: dict[str, str] = {
    "vercel": "vercel",
    "netlify": "netlify",
    "heroku": "heroku",
    "shopify": "shopify",
    "wordpress": "wordpress",
    "webflow": "webflow",
    "squarespace": "squarespace",
    "github": "github",
}

# ---------------------------------------------------------------------------
# Primary technology domain — weighted keyword scoring.
# Each keyword carries a weight (higher = stronger signal).
# ---------------------------------------------------------------------------

_TECH_DOMAIN_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "ai_ml": [
        ("machine learning", 5.0), ("deep learning", 5.0),
        ("neural network", 5.0), ("convolutional neural network", 5.0),
        ("large language model", 5.0), ("llm", 4.5),
        ("pytorch", 5.0), ("tensorflow", 5.0),
        ("model serving", 5.0), ("inference", 4.5),
        ("training data", 4.0), ("gpu cluster", 5.0),
        ("transformer", 4.0), ("nlp", 3.5),
        ("computer vision", 5.0), ("model deployment", 4.5),
        ("ai-powered", 5.0), ("ai-driven", 5.0),
        ("artificial intelligence", 5.0), ("onnx", 4.0),
        ("model versioning", 4.5), ("fine-tuning", 4.5),
        ("embedding", 3.5), ("vector database", 5.0),
        ("rag", 4.0), ("retrieval augmented", 4.5),
        ("gpu-optimized", 5.0), ("multi-cloud gpu", 5.0),
        ("ai diagnostic", 5.0), ("auto-scaling", 3.0),
    ],
    "cloud_infrastructure": [
        ("cloud infrastructure", 5.0), ("cloud-native", 5.0),
        ("multi-cloud", 5.0), ("aws", 4.5), ("gcp", 4.5),
        ("azure", 4.5), ("kubernetes", 5.0), ("docker", 4.5),
        ("terraform", 5.0), ("infrastructure as code", 5.0),
        ("cloud platform", 4.0), ("serverless", 4.5),
        ("container orchestration", 5.0), ("microservices", 4.0),
        ("service mesh", 4.5), ("cloud formation", 4.5),
        ("helm", 4.0), ("ci/cd", 3.5),
        ("deployment orchestration", 4.0), ("cloud compute", 4.0),
    ],
    "data_engineering": [
        ("data pipeline", 5.0), ("etl", 5.0), ("data warehouse", 5.0),
        ("data lake", 5.0), ("data lakehouse", 5.0),
        ("spark", 4.5), ("kafka", 5.0), ("airflow", 5.0),
        ("data engineering", 5.0), ("streaming data", 4.5),
        ("batch processing", 4.0), ("real-time data", 4.5),
        ("data quality", 4.0), ("data governance", 4.0),
        ("snowflake", 4.5), ("databricks", 5.0),
        ("bigquery", 4.5), ("redshift", 4.0),
        ("postgres", 3.5), ("postgresql", 3.5),
        ("mysql", 3.5), ("mongodb", 3.5), ("redis", 3.5),
        ("clickhouse", 4.5), ("elasticsearch", 4.0),
        ("data integration", 4.0), ("data mesh", 4.5),
    ],
    "cybersecurity": [
        ("cybersecurity", 5.0), ("information security", 5.0),
        ("security platform", 5.0), ("threat detection", 5.0),
        ("vulnerability scanning", 5.0), ("penetration testing", 5.0),
        ("soc 2", 4.0), ("zero trust", 5.0),
        ("encryption", 4.0), ("authentication", 3.5),
        ("identity management", 4.5), ("siem", 4.5),
        ("endpoint detection", 5.0), ("firewall", 3.5),
        ("intrusion detection", 5.0), ("security operations", 4.5),
        ("compliance automation", 4.0), ("gdpr compliance", 4.0),
        ("hipaa compliance", 4.0), ("pci dss", 4.0),
    ],
    "web_platform": [
        ("web platform", 5.0), ("web application", 4.5),
        ("frontend", 4.0), ("react", 4.0), ("angular", 4.0),
        ("vue", 4.0), ("next.js", 4.5), ("nextjs", 4.5),
        ("web app", 4.0), ("responsive design", 3.5),
        ("single page application", 4.5), ("progressive web app", 5.0),
        ("web framework", 4.0), ("node.js", 3.5),
        ("express", 3.0), ("django", 3.5), ("flask", 3.5),
        ("fastapi", 4.0), ("rails", 3.5),
    ],
    "mobile": [
        ("mobile app", 5.0), ("ios", 4.0), ("android", 4.0),
        ("react native", 5.0), ("flutter", 5.0),
        ("swift", 4.0), ("kotlin", 4.0), ("objective-c", 3.5),
        ("mobile development", 5.0), ("cross-platform", 3.5),
        ("mobile application", 4.5), ("app store", 3.0),
        ("mobile sdk", 4.5), ("mobile platform", 4.0),
    ],
    "devops": [
        ("devops", 5.0), ("ci/cd", 5.0), ("continuous integration", 5.0),
        ("continuous deployment", 5.0), ("build pipeline", 4.5),
        ("deployment orchestration", 5.0), ("infrastructure automation", 5.0),
        ("release management", 4.0), ("monitoring", 3.0),
        ("observability", 4.5), ("log management", 4.0),
        ("incident management", 4.0), ("site reliability", 5.0),
        ("sre", 4.5), ("infrastructure as code", 5.0),
        ("gitops", 5.0), ("platform engineering", 5.0),
    ],
    "fintech_infra": [
        ("payment infrastructure", 5.0), ("payment processing", 5.0),
        ("embedded payment", 5.0), ("fintech", 4.5),
        ("payment api", 5.0), ("transaction processing", 4.5),
        ("kyc compliance", 4.0), ("aml compliance", 4.0),
        ("settlement", 4.0), ("multi-currency", 4.0),
        ("split payment", 5.0), ("escrow", 4.0),
        ("banking as a service", 5.0), ("baas", 4.5),
        ("ledger", 3.5), ("payment rails", 4.5),
    ],
    "healthtech_infra": [
        ("medical imaging", 5.0), ("diagnostic", 4.0),
        ("clinical", 3.5), ("fda", 4.5), ("hipaa", 5.0),
        ("ehr integration", 5.0), ("electronic health record", 5.0),
        ("radiology", 5.0), ("health data", 4.5),
        ("telehealth", 4.0), ("clinical trial", 4.5),
        ("patient data", 3.5), ("healthcare platform", 4.5),
        ("medical device", 4.5), ("dicom", 5.0),
    ],
    "iot": [
        ("iot", 5.0), ("internet of things", 5.0),
        ("sensor data", 4.5), ("edge computing", 4.5),
        ("embedded systems", 5.0), ("firmware", 4.0),
        ("mqtt", 4.5), ("telemetry", 4.0),
        ("device management", 4.0), ("smart device", 3.5),
        ("connected device", 4.0), ("iot platform", 5.0),
    ],
    "blockchain": [
        ("blockchain", 5.0), ("smart contract", 5.0),
        ("solidity", 5.0), ("web3", 5.0), ("defi", 5.0),
        ("decentralized", 4.0), ("token", 3.0),
        ("nft", 4.0), ("dao", 4.0), ("distributed ledger", 5.0),
        ("crypto", 3.5), ("ethereum", 4.5),
    ],
    "robotics": [
        ("robotics", 5.0), ("autonomous mobile robot", 5.0),
        ("lidar", 5.0), ("slam", 5.0), ("sensor fusion", 5.0),
        ("path planning", 4.5), ("robot", 4.0),
        ("actuator", 4.0), ("manipulation", 3.5),
        ("perception", 3.5), ("navigation", 3.5),
        ("fleet management", 4.0), ("hardware", 2.5),
    ],
    "enterprise_software": [
        ("enterprise software", 5.0), ("erp", 5.0),
        ("crm", 4.0), ("workflow automation", 4.5),
        ("business process", 4.0), ("bpm", 4.5),
        ("enterprise resource planning", 5.0),
        ("supply chain management", 5.0),
        ("hris", 4.5), ("workforce management", 4.0),
        ("enterprise collaboration", 4.0),
        ("document management", 3.5),
    ],
    "consumer_tech": [
        ("consumer app", 5.0), ("social network", 4.5),
        ("consumer mobile", 5.0), ("fitness", 3.5),
        ("gaming", 4.0), ("entertainment", 3.5),
        ("streaming", 3.5), ("marketplace", 3.0),
        ("user engagement", 3.5), ("community", 3.0),
        ("user generated content", 4.0), ("content platform", 4.0),
    ],
}

# ---------------------------------------------------------------------------
# Programming language detection — compiled regex patterns for precision.
# ---------------------------------------------------------------------------

_PROGRAMMING_LANGUAGE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("python", re.compile(r"\bpython\b", re.I)),
    ("javascript", re.compile(r"\bjavascript\b", re.I)),
    ("typescript", re.compile(r"\btypescript\b", re.I)),
    ("java", re.compile(r"\bjava\b(?!script)", re.I)),
    ("golang", re.compile(r"\bgolang\b|\bgo\b\s+(?:lang|language)", re.I)),
    ("rust", re.compile(r"\brust\b", re.I)),
    ("c++", re.compile(r"\bc\+\+\b|\bcpp\b", re.I)),
    ("c#", re.compile(r"\bc#\b|\bcsharp\b", re.I)),
    ("swift", re.compile(r"\bswift\b", re.I)),
    ("kotlin", re.compile(r"\bkotlin\b", re.I)),
    ("ruby", re.compile(r"\bruby\b", re.I)),
    ("php", re.compile(r"\bphp\b", re.I)),
    ("scala", re.compile(r"\bscala\b", re.I)),
    ("r", re.compile(r"\b(?:language\s+r|R\s+programming)\b", re.I)),
    ("solidity", re.compile(r"\bsolidity\b", re.I)),
    ("sql", re.compile(r"\bsql\b", re.I)),
    ("graphql", re.compile(r"\bgraphql\b", re.I)),
    ("html", re.compile(r"\bhtml\b", re.I)),
    ("css", re.compile(r"\bcss\b", re.I)),
    ("shell", re.compile(r"\bshell\b|\bbash\b|\bzsh\b", re.I)),
    ("lua", re.compile(r"\blua\b", re.I)),
    ("dart", re.compile(r"\bdart\b", re.I)),
    ("matlab", re.compile(r"\bmatlab\b", re.I)),
]

# ---------------------------------------------------------------------------
# Framework / library detection — compiled regex patterns.
# ---------------------------------------------------------------------------

_FRAMEWORK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("react", re.compile(r"\breact\b", re.I)),
    ("angular", re.compile(r"\bangular\b", re.I)),
    ("vue.js", re.compile(r"\bvue(?:\.js|\.js)?\b", re.I)),
    ("next.js", re.compile(r"\bnext\.?js\b", re.I)),
    ("svelte", re.compile(r"\bsvelte\b", re.I)),
    ("node.js", re.compile(r"\bnode\.?js\b", re.I)),
    ("express", re.compile(r"\bexpress\b(?!way)", re.I)),
    ("django", re.compile(r"\bdjango\b", re.I)),
    ("flask", re.compile(r"\bflask\b", re.I)),
    ("fastapi", re.compile(r"\bfastapi\b", re.I)),
    ("spring", re.compile(r"\bspring\b(?!field)", re.I)),
    ("rails", re.compile(r"\brails\b|\bruby on rails\b", re.I)),
    ("laravel", re.compile(r"\blaravel\b", re.I)),
    ("react native", re.compile(r"\breact\s*native\b", re.I)),
    ("flutter", re.compile(r"\bflutter\b", re.I)),
    ("swiftui", re.compile(r"\bswiftui\b", re.I)),
    ("pytorch", re.compile(r"\bpytorch\b", re.I)),
    ("tensorflow", re.compile(r"\btensorflow\b", re.I)),
    ("keras", re.compile(r"\bkeras\b", re.I)),
    ("scikit-learn", re.compile(r"\bscikit[\s-]?learn\b", re.I)),
    ("hugging face", re.compile(r"\bhugging\s*face\b", re.I)),
    ("langchain", re.compile(r"\blangchain\b", re.I)),
    ("openai sdk", re.compile(r"\bopenai\b", re.I)),
    ("spark", re.compile(r"\bspark\b", re.I)),
    ("kafka", re.compile(r"\bkafka\b", re.I)),
    ("airflow", re.compile(r"\bairflow\b", re.I)),
    ("docker", re.compile(r"\bdocker\b", re.I)),
    ("kubernetes", re.compile(r"\bkubernetes\b|\bk8s\b", re.I)),
    ("terraform", re.compile(r"\bterraform\b", re.I)),
    ("ansible", re.compile(r"\bansible\b", re.I)),
    ("helm", re.compile(r"\bhelm\b", re.I)),
    ("grafana", re.compile(r"\bgrafana\b", re.I)),
    ("prometheus", re.compile(r"\bprometheus\b", re.I)),
    ("nginx", re.compile(r"\bnginx\b", re.I)),
    ("redis", re.compile(r"\bredis\b", re.I)),
    ("graphql", re.compile(r"\bgraphql\b", re.I)),
    ("grpc", re.compile(r"\bgrpc\b", re.I)),
    ("socket.io", re.compile(r"\bsocket\.?io\b", re.I)),
    ("tailwind", re.compile(r"\btailwind\b", re.I)),
    ("webpack", re.compile(r"\bwebpack\b", re.I)),
    ("vite", re.compile(r"\bvite\b", re.I)),
]

# ---------------------------------------------------------------------------
# Cloud infrastructure signals — compiled regex patterns.
# ---------------------------------------------------------------------------

_CLOUD_INFRA_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws", re.compile(r"\baws\b|\bamazon\s+web\s+services\b", re.I)),
    ("gcp", re.compile(r"\bgcp\b|\bgoogle\s+cloud\b", re.I)),
    ("azure", re.compile(r"\bazure\b", re.I)),
    ("docker", re.compile(r"\bdocker\b", re.I)),
    ("kubernetes", re.compile(r"\bkubernetes\b|\bk8s\b", re.I)),
    ("terraform", re.compile(r"\bterraform\b", re.I)),
    ("cloudflare", re.compile(r"\bcloudflare\b", re.I)),
    ("heroku", re.compile(r"\bheroku\b", re.I)),
    ("vercel", re.compile(r"\bvercel\b", re.I)),
    ("netlify", re.compile(r"\bnetlify\b", re.I)),
    ("digitalocean", re.compile(r"\bdigitalocean\b", re.I)),
    ("linode", re.compile(r"\blinode\b", re.I)),
    ("gke", re.compile(r"\bgke\b|\bgoogle\s+kubernetes\b", re.I)),
    ("eks", re.compile(r"\beks\b|\bamazon\s+kubernetes\b", re.I)),
    ("aks", re.compile(r"\baks\b|\bazure\s+kubernetes\b", re.I)),
    ("lambda", re.compile(r"\blambda\b", re.I)),
    ("cloud_run", re.compile(r"\bcloud\s+run\b", re.I)),
    ("cloud_functions", re.compile(r"\bcloud\s+functions\b", re.I)),
    ("s3", re.compile(r"\bs3\b", re.I)),
    ("ec2", re.compile(r"\bec2\b", re.I)),
    ("rds", re.compile(r"\brds\b", re.I)),
    ("cloud_storage", re.compile(r"\bcloud\s+storage\b", re.I)),
    ("cdn", re.compile(r"\bcdn\b|\bcontent\s+delivery\b", re.I)),
    ("load_balancer", re.compile(r"\bload\s+balanc\w*\b", re.I)),
    ("auto_scaling_group", re.compile(r"\bauto[\s-]?scal\w*\b", re.I)),
    ("serverless", re.compile(r"\bserverless\b", re.I)),
    ("edge_network", re.compile(r"\bedge\s+network\b", re.I)),
]

# ---------------------------------------------------------------------------
# API strategy — weighted keyword scoring.
# ---------------------------------------------------------------------------

_API_STRATEGY_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "rest": [
        ("rest api", 5.0), ("restful", 5.0), ("rest", 3.0),
        ("http api", 3.5), ("json api", 3.0),
    ],
    "graphql": [
        ("graphql", 5.0), ("graph api", 4.0), ("gql", 4.0),
        ("schema-first api", 4.0),
    ],
    "grpc": [
        ("grpc", 5.0), ("protocol buffers", 5.0), ("protobuf", 5.0),
        ("binary rpc", 4.5),
    ],
    "websocket": [
        ("websocket", 5.0), ("real-time connection", 4.0),
        ("persistent connection", 4.0), ("ws protocol", 4.5),
    ],
    "event_driven": [
        ("event-driven", 5.0), ("event driven", 5.0),
        ("message queue", 4.5), ("pub/sub", 4.5), ("pubsub", 4.5),
        ("event sourcing", 5.0), ("async messaging", 4.0),
    ],
    "mixed": [
        ("api-first", 4.0), ("api first", 4.0), ("multi-protocol", 4.5),
    ],
}

# ---------------------------------------------------------------------------
# Data architecture signals — compiled regex patterns.
# ---------------------------------------------------------------------------

_DATA_ARCHITECTURE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("postgresql", re.compile(r"\bpostgres(?:ql)?\b", re.I)),
    ("mysql", re.compile(r"\bmysql\b", re.I)),
    ("mongodb", re.compile(r"\bmongodb\b|\bmongo\b", re.I)),
    ("redis", re.compile(r"\bredis\b", re.I)),
    ("elasticsearch", re.compile(r"\belasticsearch\b|\belastic\s*search\b", re.I)),
    ("cassandra", re.compile(r"\bcassandra\b", re.I)),
    ("dynamodb", re.compile(r"\bdynamodb\b", re.I)),
    ("clickhouse", re.compile(r"\bclickhouse\b", re.I)),
    ("snowflake", re.compile(r"\bsnowflake\b", re.I)),
    ("bigquery", re.compile(r"\bbigquery\b", re.I)),
    ("redshift", re.compile(r"\bredshift\b", re.I)),
    ("databricks", re.compile(r"\bdatabricks\b", re.I)),
    ("neo4j", re.compile(r"\bneo4j\b", re.I)),
    ("firebase", re.compile(r"\bfirebase\b", re.I)),
    ("supabase", re.compile(r"\bsupabase\b", re.I)),
    ("pinecone", re.compile(r"\bpinecone\b", re.I)),
    ("weaviate", re.compile(r"\bweaviate\b", re.I)),
    ("milvus", re.compile(r"\bmilvus\b", re.I)),
    ("data_warehouse", re.compile(r"\bdata\s+warehou\w*\b", re.I)),
    ("data_lake", re.compile(r"\bdata\s+lake\b", re.I)),
    ("etl_pipeline", re.compile(r"\betl\b|\bdata\s+pipeline\b", re.I)),
    ("vector_database", re.compile(r"\bvector\s+(?:database|store|db)\b", re.I)),
    ("time_series_db", re.compile(r"\btime[\s-]?series\s+(?:database|db)\b", re.I)),
    ("event_streaming", re.compile(r"\b(event|message)\s+(?:stream|queue)\b", re.I)),
    ("cache_layer", re.compile(r"\bcach\w*\s+layer\b", re.I)),
    ("multi_tenant", re.compile(r"\bmulti[\s-]?tenant\b", re.I)),
]

# ---------------------------------------------------------------------------
# Security signals — compiled regex patterns.
# ---------------------------------------------------------------------------

_SECURITY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("encryption", re.compile(r"\bencrypt(?:ion|ed|s)?\b", re.I)),
    ("authentication", re.compile(r"\bauth(?:entication|enticate)?\b", re.I)),
    ("authorization", re.compile(r"\bauthoriz\w+\b", re.I)),
    ("oauth", re.compile(r"\boauth\b", re.I)),
    ("sso", re.compile(r"\bsso\b|\bsingle\s+sign[\s-]?on\b", re.I)),
    ("mfa", re.compile(r"\bmfa\b|\bmulti[\s-]?factor\b", re.I)),
    ("zero_trust", re.compile(r"\bzero[\s-]?trust\b", re.I)),
    ("rbac", re.compile(r"\brbac\b|\brole[\s-]?based\b", re.I)),
    ("soc2", re.compile(r"\bsoc\s*2\b", re.I)),
    ("hipaa", re.compile(r"\bhipaa\b", re.I)),
    ("gdpr", re.compile(r"\bgdpr\b", re.I)),
    ("pci_dss", re.compile(r"\bpci[\s-]?dss\b", re.I)),
    ("iso27001", re.compile(r"\biso[\s-]?27001\b", re.I)),
    ("penetration_testing", re.compile(r"\bpenetration\s+test\w*\b", re.I)),
    ("vulnerability_scanning", re.compile(r"\bvulnerability\s+scan\w*\b", re.I)),
    ("waf", re.compile(r"\bwaf\b|\bweb\s+application\s+firewall\b", re.I)),
    ("audit_logging", re.compile(r"\baudit\s*log\w*\b", re.I)),
    ("data_loss_prevention", re.compile(r"\b(?:dlp|data\s+loss\s+prevention)\b", re.I)),
    ("secret_management", re.compile(r"\bsecret\s+manag\w*\b", re.I)),
    ("key_management", re.compile(r"\bkey\s+manag\w*\b", re.I)),
    ("certificate_management", re.compile(r"\b(?:tls|ssl|certificate)\b", re.I)),
]

# ---------------------------------------------------------------------------
# Open source signals — compiled regex patterns.
# ---------------------------------------------------------------------------

_OPEN_SOURCE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("open_source", re.compile(r"\bopen[\s-]?source\b", re.I)),
    ("github_stars", re.compile(r"\b\d[\d,]*\s*(?:github\s+)?star\w*\b", re.I)),
    ("github_repository", re.compile(r"\bgithub\b", re.I)),
    ("contributors", re.compile(r"\bcontribut\w*\b", re.I)),
    ("fork", re.compile(r"\bfork(?:s|ed)?\b", re.I)),
    ("pull_request", re.compile(r"\bpull\s+request\w*\b", re.I)),
    ("open_source_license", re.compile(
        r"\b(license|licens(?:ed|ing))\b.*\b(mit|apache|gpl|bsd)\b", re.I,
    )),
    ("community_driven", re.compile(r"\bcommunity[\s-]driven\b", re.I)),
    ("developer_community", re.compile(r"\bdeveloper\s+communit\w*\b", re.I)),
    ("oss_first", re.compile(r"\bopen[\s-]?source[\s-]?first\b", re.I)),
    ("oss_core", re.compile(r"\bopen[\s-]?source\s+core\b", re.I)),
    ("internal_tool_opened", re.compile(r"\bopen[\s-]?sourced\b", re.I)),
    ("self_hosted_option", re.compile(r"\bself[\s-]?host\w*\b", re.I)),
]

# ---------------------------------------------------------------------------
# Developer tooling signals — compiled regex patterns.
# ---------------------------------------------------------------------------

_DEVTOOL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("cli", re.compile(r"\bcli\b|\bcommand[\s-]?line\b", re.I)),
    ("sdk", re.compile(r"\bsdk\b", re.I)),
    ("api_documentation", re.compile(r"\bapi\s+doc\w*\b", re.I)),
    ("developer_portal", re.compile(r"\bdeveloper\s+portal\b", re.I)),
    ("ide_extension", re.compile(r"\b(ide|editor)\s+(?:extension|plugin|integration)\b", re.I)),
    ("vscode_extension", re.compile(r"\bvs\s*code\b", re.I)),
    ("code_generation", re.compile(r"\bcode\s+generat\w*\b", re.I)),
    ("scaffolding", re.compile(r"\bscaffold\w*\b", re.I)),
    ("test_runner", re.compile(r"\btest\s+run\w*\b", re.I)),
    ("linting", re.compile(r"\blint\w*\b", re.I)),
    ("debugging_tools", re.compile(r"\bdebugg\w*\s+tool\w*\b", re.I)),
    ("profiling", re.compile(r"\bprofil\w*\b", re.I)),
    ("playground", re.compile(r"\bplayground\b", re.I)),
    ("sandbox", re.compile(r"\bsandbox\b", re.I)),
    ("starter_template", re.compile(r"\bstarter\s+(?:template|kit|boilerplate)\b", re.I)),
    ("dev_experience", re.compile(r"\bdeveloper\s+experience\b|\bdx\b", re.I)),
    ("documentation", re.compile(r"\bdocumentation\b", re.I)),
    ("ci_cd_integration", re.compile(r"\bci/cd\b|\bci\s+cd\b", re.I)),
    ("git_integration", re.compile(r"\bgithub\b|\bgitlab\b|\bbitbucket\b", re.I)),
    ("monorepo", re.compile(r"\bmonorepo\b", re.I)),
    ("plugin_architecture", re.compile(r"\bplug[\s-]?in\s+architect\w*\b", re.I)),
    ("extensible", re.compile(r"\bextensible\b", re.I)),
    ("webhook", re.compile(r"\bwebhook\b", re.I)),
]

# ---------------------------------------------------------------------------
# Technology keyword extraction — domain-specific terms.
# ---------------------------------------------------------------------------

_TECHNOLOGY_KEYWORD_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("saas", re.compile(r"\bsaas\b", re.I)),
    ("paas", re.compile(r"\bpaas\b", re.I)),
    ("iaas", re.compile(r"\biaas\b", re.I)),
    ("api_first", re.compile(r"\bapi[\s-]?first\b", re.I)),
    ("cloud_native", re.compile(r"\bcloud[\s-]?native\b", re.I)),
    ("open_source", re.compile(r"\bopen[\s-]?source\b", re.I)),
    ("ai_powered", re.compile(r"\bai[\s-]?(?:powered|driven|enabled|native)\b", re.I)),
    ("real_time", re.compile(r"\breal[\s-]?time\b", re.I)),
    ("microservices", re.compile(r"\bmicroservice\w*\b", re.I)),
    ("serverless", re.compile(r"\bserverless\b", re.I)),
    ("edge_computing", re.compile(r"\bedge\s+comput\w*\b", re.I)),
    ("containerization", re.compile(r"\bcontainer\w*\b", re.I)),
    ("infrastructure_as_code", re.compile(r"\binfrastructure\s+as\s+code\b", re.I)),
    ("devops", re.compile(r"\bdevops\b", re.I)),
    ("gitops", re.compile(r"\bgitops\b", re.I)),
    ("observability", re.compile(r"\bobservabilit\w*\b", re.I)),
    ("event_driven", re.compile(r"\bevent[\s-]driven\b", re.I)),
    ("multi_cloud", re.compile(r"\bmulti[\s-]?cloud\b", re.I)),
    ("zero_trust", re.compile(r"\bzero[\s-]?trust\b", re.I)),
    ("machine_learning", re.compile(r"\bmachine\s+learning\b", re.I)),
    ("deep_learning", re.compile(r"\bdeep\s+learning\b", re.I)),
    ("llm", re.compile(r"\bllm\b|\blarge\s+language\s+model\b", re.I)),
    ("generative_ai", re.compile(r"\bgenerative\s+ai\b", re.I)),
    ("rag", re.compile(r"\b(?:retrieval[\s-]augmented|rag)\b", re.I)),
    ("vector_search", re.compile(r"\bvector\s+search\b", re.I)),
    ("data_lakehouse", re.compile(r"\bdata\s+lakehouse\b", re.I)),
    ("digital_twin", re.compile(r"\bdigital\s+twin\b", re.I)),
    ("computer_vision", re.compile(r"\bcomputer\s+vision\b", re.I)),
    ("nlp", re.compile(r"\bnlp\b|\bnatural\s+language\b", re.I)),
    ("blockchain", re.compile(r"\bblockchain\b", re.I)),
    ("web3", re.compile(r"\bweb3\b", re.I)),
    ("iot", re.compile(r"\biot\b|\binternet\s+of\s+things\b", re.I)),
    ("embedded_systems", re.compile(r"\bembedded\s+system\w*\b", re.I)),
    ("rpa", re.compile(r"\brpa\b|\brobotic\s+process\b", re.I)),
    ("low_code", re.compile(r"\blow[\s-]?code\b", re.I)),
    ("no_code", re.compile(r"\bno[\s-]?code\b", re.I)),
]


class TechnologyExtractor(BaseExtractor):
    """Extracts rich technology intelligence from startup data.

    Produces:
      - primary_technology_domain: primary technology domain classification
      - secondary_technology_domain: secondary technology domain
      - technology_stack: identified technologies in use
      - programming_language_signals: detected programming languages
      - framework_signals: detected frameworks and libraries
      - cloud_infrastructure_signals: detected cloud/infra signals
      - api_strategy: API architecture approach
      - data_architecture_signals: data layer signals
      - security_signals: security architecture signals
      - infrastructure_maturity: infrastructure maturity level
      - open_source_signals: open source involvement signals
      - developer_tooling_signals: developer tooling signals
      - engineering_maturity: engineering maturity assessment
      - technology_keywords: domain-specific technology terms
      - technology_confidence: extraction confidence (0.0-1.0)
    """

    def extract(self, startup: Startup, data: CollectedData) -> ExtractedFeatures:
        text = startup.description
        text_lower = text.lower()

        # Domain detection from website / enrichment (backward compat)
        tech_from_domain = self._detect_tech_from_domain(data.website_domain or "")
        tech_from_enrichment = self._detect_tech_from_enrichment(data.enrichment_signals)

        # Primary / secondary technology domain
        primary_tech_domain = self._classify_tech_domain(text_lower)
        secondary_tech_domain = self._classify_secondary_tech_domain(
            text_lower, primary_tech_domain,
        )

        # Technology stack (enhanced: combine all sources)
        tech_from_description = self._extract_tech_terms(startup.description)
        combined_stack = list(
            dict.fromkeys(tech_from_domain + tech_from_enrichment + tech_from_description)
        )

        # Language, framework, cloud, API, data, security signals
        programming_languages = self._extract_programming_languages(text)
        framework_signals = self._extract_framework_signals(text)
        cloud_infra_signals = self._extract_cloud_infrastructure(text)
        api_strategy = self._classify_api_strategy(text_lower)
        data_arch_signals = self._extract_data_architecture(text)
        security_sigs = self._extract_security_signals(text)

        # Maturity assessments
        infra_maturity = self._assess_infrastructure_maturity(text_lower)
        eng_maturity = self._assess_engineering_maturity(text_lower)

        # Open source and developer tooling
        open_source_sigs = self._extract_open_source_signals(text)
        dev_tooling_sigs = self._extract_developer_tooling_signals(text)

        # Keywords and confidence
        tech_keywords = self._extract_technology_keywords(text)
        tech_confidence = self._compute_technology_confidence(
            text_lower, primary_tech_domain, combined_stack,
            programming_languages, framework_signals, cloud_infra_signals,
        )

        return ExtractedFeatures(
            primary_technology_domain=primary_tech_domain,
            secondary_technology_domain=secondary_tech_domain,
            technology_stack=combined_stack,
            programming_language_signals=programming_languages,
            framework_signals=framework_signals,
            cloud_infrastructure_signals=cloud_infra_signals,
            api_strategy=api_strategy,
            data_architecture_signals=data_arch_signals,
            security_signals=security_sigs,
            infrastructure_maturity=infra_maturity,
            open_source_signals=open_source_sigs,
            developer_tooling_signals=dev_tooling_sigs,
            engineering_maturity=eng_maturity,
            technology_keywords=tech_keywords,
            technology_confidence=tech_confidence,
        )

    # ------------------------------------------------------------------
    # Backward-compatible helpers (kept from original)
    # ------------------------------------------------------------------

    def _detect_tech_from_domain(self, domain: str) -> list[str]:
        results: list[str] = []
        for hint, tech in DOMAIN_TECH_HINTS.items():
            if hint in domain:
                results.append(tech)
        return results

    def _detect_tech_from_enrichment(
        self, signals: dict[str, str | int | float | bool],
    ) -> list[str]:
        results: list[str] = []
        for key, value in signals.items():
            if "tech" in key.lower() and isinstance(value, str):
                results.append(value)
        return results

    # ------------------------------------------------------------------
    # Primary technology domain classification
    # ------------------------------------------------------------------

    def _classify_tech_domain(self, text: str) -> str | None:
        """Classify primary technology domain using weighted keyword scoring."""
        scores: dict[str, float] = {}
        match_counts: dict[str, int] = {}

        for domain, weighted_keywords in _TECH_DOMAIN_KEYWORDS.items():
            total_score = 0.0
            matches = 0
            for keyword, weight in weighted_keywords:
                if keyword in text:
                    total_score += weight
                    matches += 1
            if total_score > 0:
                scores[domain] = total_score
                match_counts[domain] = matches

        if not scores:
            return None

        best_domain = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_domain]
        best_matches = match_counts[best_domain]

        # Require minimum score for classification
        if best_score < 4.0:
            return None

        # Check for ambiguity: if second-best is too close, return None
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            margin = best_score - sorted_scores[1]
            if margin < 2.0 and best_matches < 3:
                return None

        return best_domain

    def _classify_secondary_tech_domain(
        self, text: str, primary: str | None,
    ) -> str | None:
        """Classify secondary technology domain (must differ from primary)."""
        scores: dict[str, float] = {}

        for domain, weighted_keywords in _TECH_DOMAIN_KEYWORDS.items():
            if domain == primary:
                continue
            total_score = sum(w for kw, w in weighted_keywords if kw in text)
            if total_score > 0:
                scores[domain] = total_score

        if not scores:
            return None

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best]

        # Higher threshold for secondary to avoid weak classifications
        if best_score < 5.0:
            return None

        # Must have enough distinct signals
        matched = sum(
            1 for kw, _ in _TECH_DOMAIN_KEYWORDS[best] if kw in text
        )
        if matched < 2:
            return None

        return best

    # ------------------------------------------------------------------
    # Programming language extraction
    # ------------------------------------------------------------------

    def _extract_programming_languages(self, text: str) -> list[str]:
        """Extract programming language signals from the description."""
        found: list[str] = []
        for label, pattern in _PROGRAMMING_LANGUAGE_PATTERNS:
            if pattern.search(text) and label not in found:
                found.append(label)
        return found

    # ------------------------------------------------------------------
    # Framework signal extraction
    # ------------------------------------------------------------------

    def _extract_framework_signals(self, text: str) -> list[str]:
        """Extract framework and library signals from the description."""
        found: list[str] = []
        for label, pattern in _FRAMEWORK_PATTERNS:
            if pattern.search(text) and label not in found:
                found.append(label)
        return found

    # ------------------------------------------------------------------
    # Cloud infrastructure signal extraction
    # ------------------------------------------------------------------

    def _extract_cloud_infrastructure(self, text: str) -> list[str]:
        """Extract cloud provider and infrastructure service signals."""
        found: list[str] = []
        for label, pattern in _CLOUD_INFRA_PATTERNS:
            if pattern.search(text) and label not in found:
                found.append(label)
        return found

    # ------------------------------------------------------------------
    # API strategy classification
    # ------------------------------------------------------------------

    def _classify_api_strategy(self, text: str) -> str | None:
        """Classify API architecture strategy from description signals."""
        scores: dict[str, float] = {}

        for strategy, weighted_keywords in _API_STRATEGY_KEYWORDS.items():
            total = sum(w for kw, w in weighted_keywords if kw in text)
            if total > 0:
                scores[strategy] = total

        if not scores:
            return None

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best]

        if best_score < 3.0:
            return None

        # Check for multiple API strategies -> "mixed"
        if len(scores) > 1:
            sorted_scores = sorted(scores.values(), reverse=True)
            if sorted_scores[1] >= 3.0:
                return "mixed"

        return best

    # ------------------------------------------------------------------
    # Data architecture signal extraction
    # ------------------------------------------------------------------

    def _extract_data_architecture(self, text: str) -> list[str]:
        """Extract data architecture signals from the description."""
        found: list[str] = []
        for label, pattern in _DATA_ARCHITECTURE_PATTERNS:
            if pattern.search(text) and label not in found:
                found.append(label)
        return found

    # ------------------------------------------------------------------
    # Security signal extraction
    # ------------------------------------------------------------------

    def _extract_security_signals(self, text: str) -> list[str]:
        """Extract security architecture signals from the description."""
        found: list[str] = []
        for label, pattern in _SECURITY_PATTERNS:
            if pattern.search(text) and label not in found:
                found.append(label)
        return found

    # ------------------------------------------------------------------
    # Infrastructure maturity assessment
    # ------------------------------------------------------------------

    def _assess_infrastructure_maturity(self, text: str) -> str | None:
        """Assess infrastructure maturity level from description signals."""
        enterprise_signals = [
            "kubernetes", "terraform", "infrastructure as code",
            "multi-cloud", "auto-scaling", "service mesh",
            "container orchestration", "gitops", "zero trust",
            "soc 2", "hipaa", "iso27001", "enterprise",
        ]
        mature_signals = [
            "cloud-native", "docker", "ci/cd", "microservices",
            "monitoring", "observability", "load balanc",
            "cdn", "auto-scaling", "serverless",
        ]
        developing_signals = [
            "cloud", "hosted", "api", "database",
            "authentication", "ssl", "encryption",
        ]

        enterprise_count = sum(1 for s in enterprise_signals if s in text)
        mature_count = sum(1 for s in mature_signals if s in text)
        developing_count = sum(1 for s in developing_signals if s in text)

        if enterprise_count >= 3:
            return "enterprise_grade"
        if enterprise_count >= 1 and mature_count >= 2:
            return "enterprise_grade"
        if mature_count >= 3:
            return "mature"
        if mature_count >= 1 and developing_count >= 2:
            return "mature"
        if developing_count >= 2:
            return "developing"
        if developing_count >= 1:
            return "early"
        return "unknown"

    # ------------------------------------------------------------------
    # Engineering maturity assessment
    # ------------------------------------------------------------------

    def _assess_engineering_maturity(self, text: str) -> str | None:
        """Assess engineering maturity from description signals."""
        sophisticated_signals = [
            "platform engineering", "sre", "site reliability",
            "gitops", "infrastructure as code", "service mesh",
            "microservices", "kubernetes", "terraform",
            "canary deployment", "blue-green", "feature flag",
            "a/b testing", "chaos engineering",
        ]
        established_signals = [
            "ci/cd", "continuous integration", "continuous deployment",
            "docker", "automated testing", "code review",
            "monitoring", "observability", "logging",
            "version control", "pull request",
        ]
        developing_signals = [
            "agile", "scrum", "sprint",
            "api", "database", "cloud",
            "deployment", "pipeline",
        ]

        sophisticated_count = sum(1 for s in sophisticated_signals if s in text)
        established_count = sum(1 for s in established_signals if s in text)
        developing_count = sum(1 for s in developing_signals if s in text)

        if sophisticated_count >= 3:
            return "sophisticated"
        if sophisticated_count >= 1 and established_count >= 2:
            return "sophisticated"
        if established_count >= 3:
            return "established"
        if established_count >= 1 and developing_count >= 2:
            return "established"
        if developing_count >= 2:
            return "developing"
        if developing_count >= 1:
            return "nascent"
        return "unknown"

    # ------------------------------------------------------------------
    # Open source signal extraction
    # ------------------------------------------------------------------

    def _extract_open_source_signals(self, text: str) -> list[str]:
        """Extract open source involvement and community signals."""
        found: list[str] = []
        for label, pattern in _OPEN_SOURCE_PATTERNS:
            if pattern.search(text) and label not in found:
                found.append(label)
        return found

    # ------------------------------------------------------------------
    # Developer tooling signal extraction
    # ------------------------------------------------------------------

    def _extract_developer_tooling_signals(self, text: str) -> list[str]:
        """Extract developer tooling and DX signals."""
        found: list[str] = []
        for label, pattern in _DEVTOOL_PATTERNS:
            if pattern.search(text) and label not in found:
                found.append(label)
        return found

    # ------------------------------------------------------------------
    # Technology keyword extraction
    # ------------------------------------------------------------------

    def _extract_technology_keywords(self, text: str) -> list[str]:
        """Extract domain-specific technology terms from the description."""
        found: list[str] = []
        for label, pattern in _TECHNOLOGY_KEYWORD_PATTERNS:
            if pattern.search(text) and label not in found:
                found.append(label)
        return found

    # ------------------------------------------------------------------
    # Confidence computation
    # ------------------------------------------------------------------

    def _compute_technology_confidence(
        self,
        text: str,
        primary_domain: str | None,
        tech_stack: list[str],
        languages: list[str],
        frameworks: list[str],
        cloud_signals: list[str],
    ) -> float:
        """Compute extraction confidence based on signal density.

        High confidence requires multiple strong signals across dimensions.
        """
        # Signal density component (0-0.35)
        word_count = max(len(text.split()), 1)
        signal_density = min(word_count / 50.0, 0.35)

        # Domain classification component (0-0.25)
        domain_component = 0.25 if primary_domain is not None else 0.0

        # Technology stack component (0-0.15)
        stack_component = min(len(tech_stack) / 5.0 * 0.15, 0.15)

        # Language/framework component (0-0.15)
        lang_fw_count = len(languages) + len(frameworks)
        lang_fw_component = min(lang_fw_count / 4.0 * 0.15, 0.15)

        # Cloud infrastructure component (0-0.10)
        cloud_component = min(len(cloud_signals) / 3.0 * 0.10, 0.10)

        combined = (
            signal_density
            + domain_component
            + stack_component
            + lang_fw_component
            + cloud_component
        )

        return round(min(combined, 1.0), 2)
