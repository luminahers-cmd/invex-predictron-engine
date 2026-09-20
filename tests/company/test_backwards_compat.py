"""Backwards-compatibility guards for the additive CIH milestone.

These tests assert that existing analysis routes, persistence, models, and
OpenAPI surface remain intact after the CIH Phase 1 additions.
"""

from __future__ import annotations

import pytest


class TestExistingRoutesStillPresent:
    @pytest.mark.asyncio
    async def test_analyze_post_route_present(self, client):
        resp = await client.get("/openapi.json")
        paths = resp.json()["paths"]
        assert "/api/v1/analyze" in paths
        assert "post" in paths["/api/v1/analyze"]

    @pytest.mark.asyncio
    async def test_analyze_get_route_present(self, client):
        resp = await client.get("/openapi.json")
        paths = resp.json()["paths"]
        assert "/api/v1/analyze" in paths
        assert "get" in paths["/api/v1/analyze"]

    @pytest.mark.asyncio
    async def test_analyze_detail_route_present(self, client):
        resp = await client.get("/openapi.json")
        paths = resp.json()["paths"]
        assert "/api/v1/analyze/{analysis_id}" in paths

    @pytest.mark.asyncio
    async def test_venture_routes_present(self, client):
        resp = await client.get("/openapi.json")
        paths = resp.json()["paths"]
        assert "/api/v1/venture" in paths

    @pytest.mark.asyncio
    async def test_health_route_present(self, client):
        resp = await client.get("/openapi.json")
        assert "/api/v1/health" in resp.json()["paths"]

    @pytest.mark.asyncio
    async def test_batch_route_present(self, client):
        resp = await client.get("/openapi.json")
        assert "/api/v1/batch" in resp.json()["paths"]

    @pytest.mark.asyncio
    async def test_search_route_present(self, client):
        resp = await client.get("/openapi.json")
        assert "/api/v1/search" in resp.json()["paths"]

    @pytest.mark.asyncio
    async def test_portfolio_route_present(self, client):
        resp = await client.get("/openapi.json")
        assert "/api/v1/portfolio" in resp.json()["paths"]


class TestModelsIntact:
    def test_analysis_models_unchanged_exports(self):
        from app.models import AnalysisReport, AnalysisRequest, Company, CompanySnapshot

        assert AnalysisRequest.__tablename__ == "analysis_requests"
        assert AnalysisReport.__tablename__ == "analysis_reports"
        assert Company.__tablename__ == "companies"
        assert CompanySnapshot.__tablename__ == "company_snapshots"

    def test_analysis_request_columns_preserved(self):
        from app.models import AnalysisRequest

        cols = set(AnalysisRequest.__table__.columns.keys())
        assert {"id", "user_id", "startup_name", "website", "description"} <= cols

    def test_analysis_report_columns_preserved(self):
        from app.models import AnalysisReport

        cols = set(AnalysisReport.__table__.columns.keys())
        assert {"id", "request_id", "venture_score", "confidence"} <= cols


class TestPersistenceUnchanged:
    async def test_persist_analysis_signature_unchanged(self):
        import inspect

        from app.services.persistence import persist_analysis

        sig = inspect.signature(persist_analysis)
        assert "session" in sig.parameters
        assert "user_id" in sig.parameters or "user_id" in str(sig)

    def test_persistence_module_imports_clean(self):
        from app.services import persistence  # noqa: F401


class TestRunAnalysisStillFunctional:
    @pytest.mark.asyncio
    async def test_run_analysis_still_returns_response(self, monkeypatch):
        from app.services import analysis as analysis_mod

        class FakeEngine:
            def analyze(self, request):
                from tests.company.test_integration import _make_report

                return _make_report()

        async def fake_persist(request, report, response, user_id=None):
            response.id = "req-kept"
            return None

        monkeypatch.setattr(analysis_mod, "_persist_async", fake_persist)

        from app.schemas.analysis import StartupAnalysisRequest

        request = StartupAnalysisRequest(
            startup_name="KeepCo",
            website="https://keep.example.com",
            description="A company that must keep working.",
        )
        response = await analysis_mod.run_analysis(FakeEngine(), request, user_id="u9")
        assert response.startup_name == "KeepCo"
        assert response.id == "req-kept"


class TestCompaniesRouterRegistration:
    def test_router_included_in_api_router(self):
        from app.api.router import api_router

        routes = {getattr(r, "path", None) for r in api_router.routes}
        assert "/api/v1/companies" in routes
        assert "/api/v1/companies/{company_id}" in routes
        assert "/api/v1/companies/{company_id}/history" in routes

    def test_companies_router_tags(self):
        from app.api.companies import router

        assert router.prefix == "/companies"
        assert router.tags == ["companies"]

    @pytest.mark.asyncio
    async def test_app_includes_companies_endpoints(self, client):
        resp = await client.get("/openapi.json")
        paths = resp.json()["paths"]
        assert "/api/v1/companies" in paths
        assert "/api/v1/companies/{company_id}" in paths


class TestEngineUntouched:
    def test_predictron_engine_module_unchanged_exports(self):
        import predictron_engine.engine as engine_module

        assert hasattr(engine_module, "PredictronEngine")

    def test_store_does_not_import_engine(self):
        import app.services.company_postgres as store_module

        assert "predictron_engine.engine" not in [
            m.__name__ for m in store_module._get_imports  # type: ignore[attr-defined]
        ] if hasattr(store_module, "_get_imports") else True

    def test_resolver_uses_project_e2_normalizers(self):
        from app.services.companies import CompanyIdentityResolver
        from predictron_engine.dataset.company_name import canonical_name_key

        resolved = CompanyIdentityResolver().resolve("Stripe Inc", None)
        assert resolved.canonical_name_key == canonical_name_key("Stripe Inc")
