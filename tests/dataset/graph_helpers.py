"""Shared fixtures for company knowledge graph tests."""

from __future__ import annotations

from typing import Any

from predictron_engine.dataset.models import (
    CompanyProfile,
    DatasetRecord,
    PredictionSummary,
)


def make_graph_record(
    *,
    record_id: str,
    startup_name: str,
    website: str,
    metadata: dict[str, Any] | None = None,
    profile: CompanyProfile | None = None,
    industries: list[str] | None = None,
    country_code: str | None = None,
    city: str | None = None,
    region: str | None = None,
    source: str = "import",
) -> DatasetRecord:
    """Create a DatasetRecord wired for knowledge-graph extraction."""
    profile = profile or CompanyProfile()
    if industries:
        profile = profile.model_copy(
            update={"industries": list(industries)}
        )
    if country_code:
        profile = profile.model_copy(update={"country_code": country_code})
    if city:
        profile = profile.model_copy(update={"city": city})
    if region:
        profile = profile.model_copy(update={"region": region})
    return DatasetRecord(
        record_id=record_id,
        startup_name=startup_name,
        website=website,
        engine_version="0.12.1",
        prediction=PredictionSummary(
            decision="invest",
            confidence=0.8,
            composite_score=70.0,
        ),
        analysis_metadata=metadata or {},
        profile=profile,
        source=source,
    )


def small_dataset() -> list[DatasetRecord]:
    """A deterministic 6-record dataset exercising every node type.

    Records:
      a, d  -> resolve to one identity "Alpha Labs" (shared domain)
      b     -> "Beta Health" (isolated company)
      c     -> "Gamma AI" (acquired by Alpha Labs, subsidiary of a holding)
      e, f  -> two "Global Bank" companies with a country conflict
    """
    return [
        make_graph_record(
            record_id="a",
            startup_name="Alpha Labs",
            website="https://alpha.ai",
            industries=["fintech"],
            country_code="US",
            city="Austin",
            region="TX",
            metadata={
                "founders": ["Ada Lovelace"],
                "investors": ["Accel"],
                "technologies": ["pytorch", "python"],
                "products": ["Alpha Platform"],
                "sec_cik": "000123",
                "domain": "alpha.ai",
            },
        ),
        make_graph_record(
            record_id="d",
            startup_name="Alpha Labs Inc",
            website="https://alpha.ai",
            industries=["fintech"],
            country_code="US",
            city="Austin",
            region="TX",
            metadata={
                "founders": ["Ada Lovelace"],
                "investors": ["Accel"],
                "technologies": ["pytorch"],
                "products": ["Alpha Platform"],
                "domain": "alpha.ai",
            },
        ),
        make_graph_record(
            record_id="b",
            startup_name="Beta Health",
            website="https://beta.health",
            industries=["healthtech"],
            country_code="US",
            city="Boston",
            region="MA",
            metadata={
                "founders": ["Alan Turing"],
                "technologies": ["python"],
                "products": ["Beta App"],
            },
        ),
        make_graph_record(
            record_id="c",
            startup_name="Gamma AI",
            website="https://gamma.ai",
            industries=["fintech"],
            country_code="US",
            city="Denver",
            region="CO",
            metadata={
                "investors": ["Accel"],
                "technologies": ["pytorch"],
                "products": ["Gamma Suite"],
                "acquirer": "Alpha Labs",
                "parent_organization": "Galaxy Holdings",
            },
        ),
        make_graph_record(
            record_id="e",
            startup_name="Global Bank",
            website="https://global-bank-us.example",
            industries=["banking"],
            country_code="GB",
        ),
        make_graph_record(
            record_id="f",
            startup_name="Global Bank",
            website="https://global-bank-uk.example",
            industries=["banking"],
            country_code="DE",
        ),
    ]


COMPANY_IDS = {
    "alpha": "company:a+d",
    "beta": "company:b",
    "gamma": "company:c",
    "us_bank": "company:e",
    "uk_bank": "company:f",
}
