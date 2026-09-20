"""Tests for higher-level graph queries."""

from __future__ import annotations

from predictron_engine.dataset.graph import (
    CompanyKnowledgeGraphBuilder,
    GraphQueries,
)
from tests.dataset.graph_helpers import COMPANY_IDS, small_dataset


def _queries() -> GraphQueries:
    graph = CompanyKnowledgeGraphBuilder().build(small_dataset()).graph
    return GraphQueries(graph)


class TestLookup:
    def test_lookup_by_label_is_case_insensitive(self) -> None:
        queries = _queries()
        assert queries.lookup("Ada Lovelace") == "founder:ada lovelace"
        assert queries.lookup("ada lovelace") == "founder:ada lovelace"

    def test_lookup_exact_by_id(self) -> None:
        queries = _queries()
        assert queries.lookup(COMPANY_IDS["alpha"]) == COMPANY_IDS["alpha"]

    def test_lookup_returns_none_for_missing(self) -> None:
        queries = _queries()
        assert queries.lookup("nonexistent entity") is None


class TestSimilarCompanies:
    def test_gamma_and_alpha_are_most_similar(self) -> None:
        queries = _queries()
        similar = queries.similar_companies(COMPANY_IDS["gamma"])
        assert similar
        top = similar[0]
        assert top["node_id"] == COMPANY_IDS["alpha"]
        assert top["shared_neighbor_count"] == 4

    def test_alpha_companies_top_similar(self) -> None:
        queries = _queries()
        similar = queries.similar_companies(COMPANY_IDS["alpha"])
        ids = [entry["node_id"] for entry in similar]
        assert ids[0] == COMPANY_IDS["gamma"]

    def test_similar_companies_share_attribute_types(self) -> None:
        queries = _queries()
        similar = queries.similar_companies(COMPANY_IDS["gamma"])
        top = similar[0]
        assert top["shared_by_type"]["industry"] >= 1
        assert top["shared_by_type"]["technology"] >= 1

    def test_similar_to_unknown_returns_empty(self) -> None:
        queries = _queries()
        assert queries.similar_companies("company:nope") == []


class TestCompaniesByIndustry:
    def test_fintech_companies(self) -> None:
        queries = _queries()
        fintech = queries.companies_by_industry("fintech")
        company_ids = {
            entry["node_id"]
            for entry in fintech
            if entry["node_type"] == "company"
        }
        assert company_ids == {
            COMPANY_IDS["alpha"],
            COMPANY_IDS["gamma"],
        }

    def test_banking_companies_are_global_banks(self) -> None:
        queries = _queries()
        banking = queries.companies_by_industry("banking")
        company_ids = {
            entry["node_id"]
            for entry in banking
            if entry["node_type"] == "company"
        }
        assert company_ids == {
            COMPANY_IDS["us_bank"],
            COMPANY_IDS["uk_bank"],
        }


class TestCompaniesByCountry:
    def test_us_companies(self) -> None:
        queries = _queries()
        us = queries.companies_by_country("us")
        company_ids = {
            entry["node_id"]
            for entry in us
            if entry["node_type"] == "company"
        }
        assert company_ids == {
            COMPANY_IDS["alpha"],
            COMPANY_IDS["beta"],
            COMPANY_IDS["gamma"],
        }

    def test_gb_companies_single_bank(self) -> None:
        queries = _queries()
        gb = queries.companies_by_country("gb")
        company_ids = {
            entry["node_id"]
            for entry in gb
            if entry["node_type"] == "company"
        }
        assert company_ids == {COMPANY_IDS["us_bank"]}

    def test_de_companies_single_bank(self) -> None:
        queries = _queries()
        de = queries.companies_by_country("de")
        company_ids = {
            entry["node_id"]
            for entry in de
            if entry["node_type"] == "company"
        }
        assert company_ids == {COMPANY_IDS["uk_bank"]}


class TestCompaniesUsingTechnology:
    def test_python_users(self) -> None:
        queries = _queries()
        companies = queries.companies_using_technology("python")
        company_ids = {
            entry["node_id"]
            for entry in companies
            if entry["node_type"] == "company"
        }
        assert company_ids == {
            COMPANY_IDS["alpha"],
            COMPANY_IDS["beta"],
        }

    def test_pytorch_users(self) -> None:
        queries = _queries()
        companies = queries.companies_using_technology("pytorch")
        company_ids = {
            entry["node_id"]
            for entry in companies
            if entry["node_type"] == "company"
        }
        assert company_ids == {
            COMPANY_IDS["alpha"],
            COMPANY_IDS["gamma"],
        }
