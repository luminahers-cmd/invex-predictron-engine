# Changelog

All notable changes to the Predictron API are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## v1.0.0 — Production Stable

**Tag:** `v1.0.0`

The Predictron API is production-ready. This release marks the stable v1 API surface with authentication, persistence, observability, rate limiting, and a deterministic intelligence pipeline backed by 1884 passing tests and zero Ruff linter violations.

### Added
- Production release checklist completed: tests, migrations, deployment, configuration, rollback, monitoring all verified.
- Migration `0004` — composite index on `analysis_requests (user_id, created_at)` to close schema drift between models and migrations.

### Changed
- README rewritten for production audience: architecture overview, feature list, environment variable reference, migration commands, deployment notes.
- Minimum Python version documented as 3.12 (matching Dockerfile and pyproject.toml).

### Fixed
- Ruff lint: N802 (function naming), E741 (ambiguous variable name) in test suite.
- Ruff per-file-ignores extended for engine files with long regex patterns.

### Security
- Production deployment checklist added: rotate `SECRET_KEY`, scope `CORS_ORIGINS`, review rate limits.

---

## v0.17.0 — Performance & Scalability

**Tag:** `v0.17.0` — 2026-07-30

### Added
- Performance profiling of the Predictron engine pipeline (`benchmarks/profile_pipeline.py`).
- Load testing suite with concurrent analysis requests, authenticated retrieval, mixed read/write workloads, rate-limit scenarios, and throughput benchmarks.
- P95/P99 latency tracking in load test output.

### Changed
- Database queries optimized: `load_only` used in list endpoint to avoid fetching full report JSON.
- Window function (`func.count().over()`) for efficient total-count pagination.

---

## v0.16.0 — Production Hardening

**Tag:** `v0.16.0`

### Added
- `LoggingMiddleware` — structured JSON request logging with request ID, method, path, status code, processing time, and authenticated user.
- `JsonFormatter` — log formatter producing machine-parseable JSON output.
- `RequestIDMiddleware` — UUID generation or propagation from `X-Request-ID` header, returned in response headers alongside `X-Response-Time`.
- `RateLimitMiddleware` — configurable in-memory sliding-window rate limiter with exempt paths for health, readiness, and docs endpoints.
- Health (`GET /api/v1/health`) and readiness (`GET /api/v1/health/readiness`) endpoints with database, engine, and startup-state checks.
- Configuration validation (`validate_required()`) with warnings for default `SECRET_KEY`.
- OpenAPI documentation improvements — response schemas, descriptions, status codes, and security scheme registration.
- Global exception handlers for 500 (unhandled exceptions) and 422 (validation errors).
- `ReadinessResponse` and `HealthResponse` Pydantic schemas.

### Changed
- `lifespan` context manager extended to track startup state (`starting` → `ready` → `stopping`).

---

## v0.15.0 — Authentication & Authorization

**Tag:** `v0.15.0`

### Added
- JWT authentication module (`app/auth/jwt.py`) with `create_access_token`, `decode_access_token`, and FastAPI dependencies.
- `HTTPBearer` security scheme registered in OpenAPI.
- `get_current_user` — required auth dependency for protected endpoints.
- `get_current_user_optional` — optional auth for endpoints supporting both authenticated and anonymous access.
- Ownership enforcement in persistence layer — `list_analyses` and `get_analysis` filter by `user_id`.
- Auth test suite covering unauthorized access, invalid tokens, expired tokens, ownership isolation, and backward-compatible POST.

### Changed
- `GET /api/v1/analyze` and `GET /api/v1/analyze/{id}` now require authentication and return only the requesting user's analyses.
- `POST /api/v1/analyze` accepts optional `Authorization` header — authenticated users have analyses associated with their identity.

### Security
- Strict ownership gate: users cannot access each other's analyses at the query level.

---

## v0.14.0 — Persistence Layer

**Tag:** `v0.14.0`

### Added
- `AnalysisRequest` and `AnalysisReport` SQLAlchemy ORM models with `TimestampMixin`.
- `AsyncSession` engine, session factory (`AsyncSessionLocal`), and `get_db` dependency with automatic commit/rollback.
- `persist_analysis` — stores completed analyses atomically (request + report).
- `get_analysis` — retrieves a single analysis by ID with optional user ownership filter.
- `list_analyses` — paginated listing with total count via window function.
- `persistence.py` service layer with full test coverage using mocked sessions.
- Alembic migrations:
  - `0001` — initial placeholder.
  - `0002` — `analysis_requests` and `analysis_reports` tables with indexes.
  - `0003` — `user_id` column on `analysis_requests`.

### Changed
- Analysis service persists results after successful engine execution; persistence failures are logged but never propagated to the caller.

---

## v0.13.0 — API Foundation

**Tag:** `v0.13.0`

### Added
- FastAPI application factory (`create_app`) with CORS middleware.
- `POST /api/v1/analyze` — submit startup data and receive venture analysis.
- `GET /api/v1/analyze` — list persisted analyses (initial stub).
- `GET /api/v1/analyze/{id}` — get analysis detail (initial stub).
- `GET /api/v1/health` — basic health endpoint.
- `StartupAnalysisRequest` and `StartupAnalysisResponse` Pydantic schemas with input validation (min/max length, `HttpUrl`, score ranges).
- Service layer — `run_analysis` bridges API and PredictronEngine via `asyncio.to_thread()`.
- PredictronEngine singleton initialized during application lifespan and stored in `app.state`.
- Global exception handler for unhandled exceptions returning JSON `INTERNAL_ERROR` response.
- API router with `/api/v1` prefix.
- Engine adapter module (`app/adapters/predictron_adapter.py`) with HTTP client and mock fallback.

---

## v0.12.1 — Architecture Maintenance

**Tag:** `v0.12.1`

### Fixed
- Derived metrics architecture hardened to prevent circular derivation chains.
- No derivation rule reads exclusively derived fields — all rules source from canonical fields.

---

## v0.12.0 — Derived Metrics & Intelligent Extraction

**Tag:** `v0.12.0`

### Added
- `DerivedMetricsEngine` with rules for ARR, MRR, revenue per employee, LTV/CAC, funding efficiency, burn multiple, runway, and ARR per customer.
- Provenance-tracking derivation log with metric name, derived value, source fields, and confidence.
- Circular derivation detection and prevention.
- Benchmark cases for derived metric validation.

---

## v0.9.1 — Structured Quantitative Feature Extraction

**Tag:** `v0.9.1`

### Added
- Quantitative data parsers for revenue figures, funding amounts, customer counts, growth metrics, and unit economics.
- Structured extraction patterns for financial and operational data.
- Benchmark expansion with quantitative test cases.

---

## Earlier Versions

Prior releases (v0.7.0 through v0.12.0) established the Predictron engine's extraction, evidence, reasoning, scoring, evaluation, recommendation, confidence, and decision-making modules. These are documented in the `predictron_engine/ARCHITECTURE.md`.
