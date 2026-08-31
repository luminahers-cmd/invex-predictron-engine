<p align="center">
  <img src="assets/banner.png" alt="Predictron Engine Banner" width="100%" />
</p>
# Predictron API

> Production-ready REST API for venture intelligence analysis. Built on the Predictron structured reasoning engine.

The Predictron API is the backend intelligence layer behind **InveX AI**. It exposes a FastAPI HTTP interface that accepts startup information and returns structured venture analysis — scores, recommendations, and confidence metrics — powered by the Predictron engine's modular extraction, evidence, reasoning, scoring, and validation pipeline.

---

## Repository Layout

- [app/](app/) — FastAPI application code: routes, auth, middleware, schemas, services, database helpers.
- [predictron_engine/](predictron_engine/) — Core intelligence pipeline: models, extraction, evidence, reasoning, scoring, evaluation, recommendations, confidence, validation.
- [benchmarks/](benchmarks/) — Benchmark runner, report tools, validators, and expected output snapshots.
- [docs/](docs/) — Architecture, design, reasoning, evidence, and benchmark methodology documentation.
- [examples/](examples/) — Sample input and sample report artifacts.
- [tests/](tests/) — Automated test suite (2854 tests).
- [alembic/](alembic/) — Database migrations.
- [assets/](assets/) — Images and static assets.
- [.github/](.github/) — Pull request and issue templates.

---

## Features

- **Startup Analysis** — Submit startup data and receive venture scores, dimensional breakdowns, recommendations, and confidence metrics.
- **Authentication** — JWT-based bearer token auth. Optional for analysis submission; required for retrieval endpoints with ownership isolation.
- **Persistence** — Analysis requests and reports stored in PostgreSQL via SQLAlchemy async sessions. Persistent retrieval and listing with pagination.
- **Structured Logging** — JSON-formatted request logging with request ID, user, method, path, status code, and processing time.
- **Request IDs** — Every request gets a unique ID (propagated from `X-Request-ID` header or generated). Returned in response headers alongside `X-Response-Time`.
- **Rate Limiting** — Configurable in-memory sliding-window rate limiter. Health, readiness, and documentation endpoints are exempt.
- **Health & Readiness** — `GET /api/v1/health` reports database connectivity, engine status, and startup state. `GET /api/v1/health/readiness` returns 200 only when all dependencies are available.
- **OpenAPI Documentation** — Auto-generated interactive docs at `/docs` (Swagger UI) and `/redoc` (ReDoc).

---

## Architecture

```
Client
  │
  ▼
API Layer          (app/api/)        — Routes, input validation, auth
  │
  ▼
Service Layer      (app/services/)   — Business logic orchestration
  │
  ▼
Persistence Layer  (app/services/)   — Async database operations
  │
  ▼
Predictron Engine  (predictron_engine/) — Extraction, reasoning, scoring, report generation
```

The API layer accepts HTTP requests, validates inputs through Pydantic schemas, and delegates to the service layer. The service layer runs the synchronous Predictron engine in a thread pool to keep the event loop unblocked, persists results asynchronously, and returns the analysis response. Persistence failures are logged but never propagated — a successful analysis always returns its response.

---

## Prerequisites

- Python 3.12+
- PostgreSQL 16 (for persistence)
- Git

---

## Installation

```bash
git clone https://github.com/luminahers-cmd/invex-predictron-engine.git
cd invex-predictron-engine

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Environment Variables

Copy `.env` to the project root (a template is provided in the repository).

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `APP_NAME` | No | `InveX AI Backend` | Application name used in responses and logging |
| `APP_VERSION` | No | `1.0.0` | Application version returned by the health endpoint |
| `DEBUG` | No | `false` | Enable debug-level logging |
| `DATABASE_URL` | Yes | `postgresql+asyncpg://postgres:postgres@localhost:5432/invex` | PostgreSQL connection string (asyncpg driver) |
| `DATABASE_ECHO` | No | `false` | Log all SQL statements |
| `SECRET_KEY` | **Yes** | `CHANGE_ME_IN_PRODUCTION` | JWT signing key — **must be changed in production** |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | No | `60` | JWT token lifetime in minutes |
| `ALGORITHM` | No | `HS256` | JWT signing algorithm |
| `CORS_ORIGINS` | No | `["http://localhost:3000","https://*.vercel.app"]` | Allowed CORS origins (JSON array) |
| `PREDICTRON_ENGINE_URL` | No | `http://localhost:8001` | External Predictron engine endpoint (unused when engine is embedded) |
| `PREDICTRON_API_KEY` | No | `` | API key for external engine calls |
| `API_V1_PREFIX` | No | `/api/v1` | URL prefix for all API routes |
| `RATE_LIMIT_ENABLED` | No | `true` | Enable or disable rate limiting |
| `RATE_LIMIT_REQUESTS` | No | `100` | Maximum requests per window |
| `RATE_LIMIT_WINDOW_SECONDS` | No | `60` | Rate limit window in seconds |
| `REQUEST_ID_HEADER` | No | `X-Request-ID` | Header name for request ID propagation |
| `EVIDENCE_SEARCH_ENABLED` | No | `false` | Opt in to search-backed evidence discovery in the default evidence orchestrator |
| `TAVILY_API_KEY` | No | `` | Tavily Search API key — required when `EVIDENCE_SEARCH_ENABLED=true` |

---

## Database Migrations

The project uses Alembic for async PostgreSQL migrations.

```bash
# Apply all pending migrations
alembic upgrade head

# Roll back the last migration
alembic downgrade -1

# Roll back to a specific revision
alembic downgrade 0003

# View migration history and current revision
alembic history
alembic current
```

Current migration chain: `0001` (initial placeholder) → `0002` (analysis tables) → `0003` (user_id column) → `0004` (composite index).

---

## Running Tests

```bash
# Run the full test suite (2854 tests)
python -m pytest tests

# Run with verbose output
python -m pytest tests -v

# Run a specific test file
python -m pytest tests/test_analyze.py -v

# Run Ruff linter
python -m ruff check
```

---

## API Documentation

Once the server is running, interactive API documentation is available at:

- **Swagger UI** — [`/docs`](http://localhost:8000/docs)
- **ReDoc** — [`/redoc`](http://localhost:8000/redoc)

The OpenAPI schema is also available at `/openapi.json`.

---

## Deployment

```bash
# Start the API and PostgreSQL with Docker Compose
docker compose up --build

# Or run directly with uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Before deploying to production:

1. Change `SECRET_KEY` to a strong random value.
2. Set `CORS_ORIGINS` to your deployed frontend origins.
3. Set `DATABASE_URL` to your production PostgreSQL connection string.
4. Review rate limit settings for your expected traffic.
5. Ensure the database is migrated with `alembic upgrade head`.

---

## Documentation

- [Architecture](predictron_engine/ARCHITECTURE.md)
- [Design Philosophy](predictron_engine/PHILOSOPHY.md)
- [Reasoning Framework](docs/reasoning-framework.md)
- [Evidence Framework](docs/evidence-framework.md)
- [Design Principles](docs/design-principles.md)
- [Benchmark Methodology](docs/benchmark-methodology.md)
- [Examples](examples/README.md)
- [Benchmarking](benchmarks/README.md)
- [Contributing](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)

---

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

For security issues, use the process described in [SECURITY.md](SECURITY.md).

---

## License

This repository is released under the [MIT License](LICENSE).

---

## System Architecture

<p align="center">
  <img src="assets/architecture.png"
       alt="Predictron Engine Architecture"
       width="100%">
</p>

The engine is organized as a modular intelligence pipeline. Each stage has a specific responsibility and remains independently testable.

```mermaid
flowchart TD

A[Startup Data Sources]
A --> B[Information Collection]

B --> C[Market Intelligence]
B --> D[Founder Intelligence]
B --> E[Product Intelligence]
B --> F[Technology Intelligence]
B --> G[Business Model Intelligence]
B --> H[Traction Intelligence]

C --> I[Evidence Framework]
D --> I
E --> I
F --> I
G --> I
H --> I

I --> J[Reasoning Framework]

J --> K[Venture Intelligence Engine]

K --> L[Venture Scoring]

L --> M[Report Generation]

M --> N[Investor & Founder Insights]
```

The architecture separates information extraction, evidence generation, reasoning, scoring, and report generation into independent modules. This modular design allows individual components to evolve independently while maintaining a consistent and explainable reasoning pipeline.
