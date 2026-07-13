"""Deterministic test dataset for the Predictron Engine.

Contains 8 startup examples covering different industries, stages,
and characteristics. These are NOT benchmarks for correctness —
they are regression cases for future engine development.
"""

from __future__ import annotations

from pydantic import HttpUrl

from app.schemas.analysis import StartupAnalysisRequest

B2B_SAAS = StartupAnalysisRequest(
    startup_name="CloudSync Pro",
    website=HttpUrl("https://cloudsyncpro.example.com"),
    description=(
        "CloudSync Pro is an enterprise B2B SaaS platform providing "
        "real-time data synchronization and integration solutions for "
        "mid-market companies. Founded in 2021 with a seed round, the "
        "platform serves 150 enterprise clients with subscription-based "
        "recurring revenue. Built on Python and AWS infrastructure."
    ),
    pitch_deck_url=HttpUrl("https://cloudsyncpro.example.com/deck.pdf"),
    founder_linkedin_urls=[
        HttpUrl("https://linkedin.com/in/cto-sarah"),
    ],
)

HEALTHCARE_AI = StartupAnalysisRequest(
    startup_name="MediVision AI",
    website=HttpUrl("https://medivisionai.example.com"),
    description=(
        "MediVision AI develops AI-powered diagnostic imaging tools "
        "for radiology departments in hospitals. The platform uses deep "
        "learning to detect anomalies in medical scans. Series A stage "
        "with FDA clearance in progress. Based in Boston."
    ),
    pitch_deck_url=HttpUrl("https://medivisionai.example.com/deck.pdf"),
    founder_linkedin_urls=[
        HttpUrl("https://linkedin.com/in/dr-chen"),
        HttpUrl("https://linkedin.com/in/ml-lead"),
    ],
)

FINTECH = StartupAnalysisRequest(
    startup_name="PayFlow",
    website=HttpUrl("https://payflow.example.com"),
    description=(
        "PayFlow is a fintech startup providing embedded payment "
        "infrastructure for marketplace platforms. The API-first "
        "solution enables instant payouts and cross-border transactions. "
        "Pre-seed stage with early traction in Southeast Asia."
    ),
)

MARKETPLACE = StartupAnalysisRequest(
    startup_name="ArtisanHub",
    website=HttpUrl("https://artisanhub.example.com"),
    description=(
        "ArtisanHub is a two-sided marketplace connecting independent "
        "artisans and craftspeople with consumers seeking handmade goods. "
        "The platform takes a 15% commission on transactions. Currently "
        "seed stage with 500 sellers and 5,000 buyers."
    ),
    pitch_deck_url=HttpUrl("https://artisanhub.example.com/deck.pdf"),
    founder_linkedin_urls=[
        HttpUrl("https://linkedin.com/in/founder-maria"),
    ],
)

DEVTOOLS = StartupAnalysisRequest(
    startup_name="CodeLens",
    website=HttpUrl("https://codelens.example.com"),
    description=(
        "CodeLens is a developer tools company building an AI-powered "
        "code review and security analysis platform. The tool integrates "
        "with GitHub and GitLab to provide automated code quality "
        "assessments. Open-source core with enterprise SaaS tier."
    ),
    pitch_deck_url=HttpUrl("https://codelens.example.com/deck.pdf"),
    founder_linkedin_urls=[
        HttpUrl("https://linkedin.com/in/devfounder1"),
        HttpUrl("https://linkedin.com/in/devfounder2"),
    ],
)

CLIMATE_TECH = StartupAnalysisRequest(
    startup_name="GreenGrid",
    website=HttpUrl("https://greengrid.example.com"),
    description=(
        "GreenGrid provides climate technology solutions for optimizing "
        "renewable energy grid management. Their platform uses machine "
        "learning to predict energy demand and optimize distribution. "
        "Series A stage serving utility companies in Europe."
    ),
    pitch_deck_url=HttpUrl("https://greengrid.example.com/deck.pdf"),
    founder_linkedin_urls=[
        HttpUrl("https://linkedin.com/in/climate-ceo"),
    ],
)

CONSUMER_APP = StartupAnalysisRequest(
    startup_name="FitBuddy",
    website=HttpUrl("https://fitbuddy.example.com"),
    description=(
        "FitBuddy is a consumer fitness app that uses computer vision "
        "to provide real-time form correction during workouts. Available "
        "on iOS and Android with freemium model. Pre-seed stage with "
        "10,000 monthly active users."
    ),
)

ROBOTICS = StartupAnalysisRequest(
    startup_name="RoboCraft",
    website=HttpUrl("https://robocraft.example.com"),
    description=(
        "RoboCraft develops autonomous mobile robots for warehouse "
        "logistics and fulfillment. The robots handle picking, packing, "
        "and sorting operations. Series B stage with deployments in "
        "3 major distribution centers. Built on proprietary hardware "
        "and ROS-based software stack."
    ),
    pitch_deck_url=HttpUrl("https://robocraft.example.com/deck.pdf"),
    founder_linkedin_urls=[
        HttpUrl("https://linkedin.com/in/robotics-ceo"),
        HttpUrl("https://linkedin.com/in/robotics-cto"),
    ],
)

ALL_STARTUPS: list[StartupAnalysisRequest] = [
    B2B_SAAS,
    HEALTHCARE_AI,
    FINTECH,
    MARKETPLACE,
    DEVTOOLS,
    CLIMATE_TECH,
    CONSUMER_APP,
    ROBOTICS,
]

STARTUP_NAMES: list[str] = [
    "CloudSync Pro",
    "MediVision AI",
    "PayFlow",
    "ArtisanHub",
    "CodeLens",
    "GreenGrid",
    "FitBuddy",
    "RoboCraft",
]
