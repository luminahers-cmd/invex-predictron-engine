"""Feature definitions for the COMPANY category.

Deterministic features derived from DatasetRecord and CompanyProfile.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from predictron_engine.feature_store.models import (
    FeatureCategory,
    FeatureDefinition,
    ValueType,
)
from predictron_engine.feature_store.registry import FeatureComputer

_SECONDS_PER_DAY = 86400.0
_DAYS_PER_YEAR = 365.25


def _make_feature(
    feature_id: str,
    feature_name: str,
    description: str,
    value_type: ValueType = ValueType.FLOAT,
    dependencies: list[str] | None = None,
    source_fields: list[str] | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    tags: list[str] | None = None,
) -> FeatureDefinition:
    return FeatureDefinition(
        feature_id=feature_id,
        feature_name=feature_name,
        category=FeatureCategory.COMPANY,
        description=description,
        value_type=value_type,
        dependencies=dependencies or [],
        source_fields=source_fields or [],
        min_value=min_value,
        max_value=max_value,
        tags=tags or ["company"],
    )


def _age_years(founded_year: int | None, as_of: datetime | None = None) -> float:
    if founded_year is None:
        return 0.0
    ref = as_of or datetime.now(UTC)
    delta = ref - datetime(founded_year, 1, 1, tzinfo=UTC)
    return round(delta.total_seconds() / _SECONDS_PER_DAY / _DAYS_PER_YEAR, 4)


def _ev(source_id: str, source_field: str) -> list[dict[str, str]]:
    return [{"source_type": "record", "source_id": source_id,
             "source_field": source_field}]


COMPANY_AGE_DEFINITION = _make_feature(
    feature_id="company_age",
    feature_name="Company Age",
    description="Age of the company in years since founding",
    value_type=ValueType.FLOAT,
    source_fields=["profile.founded_year", "profile.founded_date"],
    min_value=0.0,
    tags=["company", "age"],
)


def compute_company_age(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    as_of = ctx.get("as_of")
    profile = getattr(record, "profile", None)
    founded_year = getattr(profile, "founded_year", None) if profile else None
    if founded_year is not None:
        return _age_years(founded_year, as_of), _ev(record.record_id, "profile.founded_year")
    return None, []


FUNDING_STAGE_DEFINITION = _make_feature(
    feature_id="funding_stage",
    feature_name="Funding Stage",
    description="Funding stage at analysis time",
    value_type=ValueType.STRING,
    source_fields=["funding_stage_at_analysis"],
    tags=["company", "funding", "stage"],
)


def compute_funding_stage(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    stage = getattr(record, "funding_stage_at_analysis", None)
    if stage is not None:
        val = stage.value if hasattr(stage, "value") else str(stage)
        return val, _ev(record.record_id, "funding_stage_at_analysis")
    return None, []


EMPLOYEE_BAND_DEFINITION = _make_feature(
    feature_id="employee_band",
    feature_name="Employee Band",
    description="Employee count range band",
    value_type=ValueType.STRING,
    source_fields=["profile.employee_count", "profile.employee_range"],
    tags=["company", "employees", "size"],
)


def _band_from_count(emp_count: int) -> str:
    if emp_count <= 10:
        return "1-10"
    if emp_count <= 50:
        return "11-50"
    if emp_count <= 200:
        return "51-200"
    if emp_count <= 1000:
        return "201-1000"
    return "1000+"


def compute_employee_band(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    profile = getattr(record, "profile", None)
    if not profile:
        return None, []
    emp_range = getattr(profile, "employee_range", None)
    if emp_range:
        return emp_range, _ev(record.record_id, "profile.employee_range")
    emp_count = getattr(profile, "employee_count", None)
    if emp_count is not None:
        return _band_from_count(emp_count), _ev(record.record_id, "profile.employee_count")
    return None, []


OPERATING_COUNTRY_DEFINITION = _make_feature(
    feature_id="operating_country",
    feature_name="Operating Country",
    description="Primary operating country code",
    value_type=ValueType.STRING,
    source_fields=["profile.country_code"],
    tags=["company", "geography", "country"],
)


def compute_operating_country(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    profile = getattr(record, "profile", None)
    if profile and getattr(profile, "country_code", None):
        return profile.country_code, _ev(record.record_id, "profile.country_code")
    return None, []


INDUSTRY_DEFINITION = _make_feature(
    feature_id="industry",
    feature_name="Industry",
    description="Primary industry/sector",
    value_type=ValueType.STRING,
    source_fields=["profile.industries"],
    tags=["company", "industry", "sector"],
)


def compute_industry(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    profile = getattr(record, "profile", None)
    if profile and getattr(profile, "industries", None):
        industries = profile.industries
        val = industries[0] if len(industries) == 1 else industries
        return val, _ev(record.record_id, "profile.industries")
    return None, []


COMPANY_FEATURES: list[tuple[FeatureDefinition, FeatureComputer]] = [
    (COMPANY_AGE_DEFINITION, compute_company_age),
    (FUNDING_STAGE_DEFINITION, compute_funding_stage),
    (EMPLOYEE_BAND_DEFINITION, compute_employee_band),
    (OPERATING_COUNTRY_DEFINITION, compute_operating_country),
    (INDUSTRY_DEFINITION, compute_industry),
]
