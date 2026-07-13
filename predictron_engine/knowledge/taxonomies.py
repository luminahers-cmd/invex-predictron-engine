"""Industry and sector taxonomies for classification.

This module provides reusable taxonomy definitions that the extraction,
reasoning, and scoring modules reference. Taxonomies are intentionally
simple data structures — they are lookup tables, not logic.

As the system evolves, these taxonomies can be:
  - Extended with new industries and sub-classifications
  - Replaced with external taxonomy services (e.g., GICS, NAICS)
  - Loaded from configuration or database

The current implementation provides placeholder categories that can be
refined once real-world data informs which classifications matter most.
"""

from enum import Enum


class Industry(str, Enum):
    """Top-level industry classifications."""

    FINTECH = "fintech"
    HEALTHTECH = "healthtech"
    EDTECH = "edtech"
    ENTERPRISE_SAAS = "enterprise_saas"
    CONSUMER_TECH = "consumer_tech"
    ECOMMERCE = "ecommerce"
    AI_ML = "ai_ml"
    CYBERSECURITY = "cybersecurity"
    CLIMATE_TECH = "climate_tech"
    BIOTECH = "biotech"
    HARDWARE = "hardware"
    MARKETPLACE = "marketplace"
    LOGISTICS = "logistics"
    GAMING = "gaming"
    MEDIA_ENTERTAINMENT = "media_entertainment"
    OTHER = "other"


class BusinessModel(str, Enum):
    """Primary business model classifications."""

    SAAS = "saas"
    PAAS = "paas"
    MARKETPLACE = "marketplace"
    ECOMMERCE = "ecommerce"
    ADVERTISING = "advertising"
    TRANSACTIONAL = "transactional"
    LICENSING = "licensing"
    HARDWARE_PLUS_SOFTWARE = "hardware_plus_software"
    SERVICES = "services"
    OTHER = "other"


class CustomerType(str, Enum):
    """Target customer segment."""

    B2B = "b2b"
    B2C = "b2c"
    B2B2C = "b2b2c"
    B2G = "b2g"
    C2C = "c2c"


class Geography(str, Enum):
    """Primary geographic market."""

    NORTH_AMERICA = "north_america"
    EUROPE = "europe"
    ASIA_PACIFIC = "asia_pacific"
    LATIN_AMERICA = "latin_america"
    MIDDLE_EAST_AFRICA = "middle_east_africa"
    GLOBAL = "global"


# --- Taxonomy Lookup Helpers ---

INDUSTRY_KEYWORDS: dict[Industry, list[str]] = {
    Industry.FINTECH: [
        "fintech", "banking", "payments", "lending", "insurtech", "wealth",
    ],
    Industry.HEALTHTECH: [
        "healthtech", "health", "medical", "telehealth", "digital_health",
    ],
    Industry.EDTECH: [
        "edtech", "education", "learning", "training", "online_courses",
    ],
    Industry.ENTERPRISE_SAAS: [
        "enterprise", "saas", "b2b software", "crm", "erp",
    ],
    Industry.CONSUMER_TECH: [
        "consumer", "mobile app", "social", "consumer platform",
    ],
    Industry.ECOMMERCE: [
        "ecommerce", "e-commerce", "online retail", "d2c", "shopify",
    ],
    Industry.AI_ML: [
        "artificial intelligence", "machine learning", "ai", "ml",
        "deep learning",
    ],
    Industry.CYBERSECURITY: [
        "cybersecurity", "security", "infosec", "threat detection",
    ],
    Industry.CLIMATE_TECH: [
        "climate", "clean energy", "sustainability", "carbon", "green",
    ],
    Industry.BIOTECH: [
        "biotech", "biotechnology", "drug discovery", "genomics",
    ],
    Industry.HARDWARE: [
        "hardware", "iot", "embedded", "chips", "semiconductor",
    ],
    Industry.MARKETPLACE: [
        "marketplace", "platform", "two-sided", "network effects",
    ],
    Industry.LOGISTICS: [
        "logistics", "supply chain", "delivery", "fulfillment", "freight",
    ],
    Industry.GAMING: [
        "gaming", "game", "esports", "virtual reality", "metaverse",
    ],
    Industry.MEDIA_ENTERTAINMENT: [
        "media", "entertainment", "streaming", "content", "creator",
    ],
}

MODEL_KEYWORDS: dict[BusinessModel, list[str]] = {
    BusinessModel.SAAS: [
        "saas", "subscription", "monthly recurring", "annual contract",
    ],
    BusinessModel.PAAS: [
        "paas", "platform", "developer tools", "api",
    ],
    BusinessModel.MARKETPLACE: [
        "marketplace", "platform fee", "take rate", "two-sided",
    ],
    BusinessModel.ECOMMERCE: [
        "ecommerce", "online store", "retail", "direct to consumer",
    ],
    BusinessModel.ADVERTISING: [
        "advertising", "ad-supported", "ad tech", "programmatic",
    ],
    BusinessModel.TRANSACTIONAL: [
        "transaction", "per-transaction", "payment processing",
    ],
    BusinessModel.LICENSING: [
        "licensing", "license", "per-seat", "enterprise license",
    ],
    BusinessModel.HARDWARE_PLUS_SOFTWARE: [
        "hardware", "device", "sensor", "connected",
    ],
    BusinessModel.SERVICES: [
        "consulting", "professional services", "managed services",
    ],
}
