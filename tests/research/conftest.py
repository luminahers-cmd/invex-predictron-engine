"""Shared fixtures for the Research Planner tests."""

from __future__ import annotations

import pytest

from predictron_engine.research import PlannerInput, ResearchPlanner


@pytest.fixture
def planner() -> ResearchPlanner:
    """A default ResearchPlanner using the built-in rule registry."""
    return ResearchPlanner()


@pytest.fixture
def company_input() -> PlannerInput:
    """A minimal planner input: company name only."""
    return PlannerInput(company_name="Acme")
