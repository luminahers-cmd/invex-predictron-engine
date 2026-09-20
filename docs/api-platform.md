# Venture Intelligence API Platform (Project E8)

The API platform exposes the Predictron Engine stack through a production-grade
FastAPI interface. It provides venture analysis, portfolio intelligence, company
comparison, due diligence reporting, unified search, signal analytics, and
batch processing — all as thin orchestration layers over the existing engine.

The platform contains **no frontend, no ML, and no duplicated business logic**.
Every endpoint delegates to the Predictron Engine and/or the dataset store.

---

## Architecture

```text
HTTP Request
      │
      ▼
┌───────────────────────────┐
│  Middleware stack          │
│  LoggingMiddleware (JSON)  │
│  RequestIDMiddleware       │
│  RateLimitMiddleware       │
│  CORSMiddleware            │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│  API Router (app/api/*.py) │  Request/response schemas, auth hooks
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│  Service layer (app/services/*.py) │  Orchestration, no business logic
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│  Predictron Engine          │  predictron_engine/ (ML + domain logic)
│  Dataset Store              │  predictron_engine/dataset/store.py
└───────────────────────────┘
```

### Layer responsibilities

| Layer | Location | Responsibility |
|---|---|---|
| Schemas | `app/schemas/*.py` | Pydantic v2 request/response models |
| API routers | `app/api/*.py` | HTTP mapping, auth dependencies, OpenAPI docs |
| Services | `app/services/*.py` | Orchestration and aggregation across engine calls |
| Engine | `predictron_engine/` | All analysis, scoring, reasoning, decisions, signals |

---

## Base Paths

All endpoints are mounted under `/api/v1`. Auth is optional on every new
endpoint via `get_current_user_optional` — authenticated users can be tied to
their account downstream, but none of the analytics features require it.

| Prefix | Module | Description |
|---|---|---|
| `/api/v1/analyze` | `app/api/analyze.py` | Legacy single-startup analysis (unchanged) |
| `/api/v1/health` | `app/api/health.py` | Health and readiness checks (unchanged) |
| `/api/v1/venture` | `app/api/venture.py` | Venture analysis intelligence |
| `/api/v1/portfolio` | `app/api/portfolio.py` | Multi-company portfolio analysis |
| `/api/v1/compare` | `app/api/comparison.py` | Cross-company comparison |
| `/api/v1/due-diligence` | `app/api/due_diligence.py` | Structured due diligence reports |
| `/api/v1/search` | `app/api/search.py` | Unified search across the dataset |
| `/api/v1/batch` | `app/api/batch.py` | Async batch processing |

Interactive docs are available at `/docs` (Swagger UI) and `/redoc`.

---

## Endpoints

### Venture Analysis — `/api/v1/venture`

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/venture` | Full venture analysis (scores, decision, summary) |
| POST | `/api/v1/venture/explain` | Human-readable decision explanation |
| POST | `/api/v1/venture/trace` | Full decision reasoning trace as a graph |
| POST | `/api/v1/venture/features` | Extracted feature snapshot |
| GET | `/api/v1/venture/knowledge-graph` | Knowledge graph summary from dataset |
| GET | `/api/v1/venture/signals/{company_id}` | Signal timeline for a company |
| GET | `/api/v1/venture/signals/{company_id}/trends` | Signal trend analysis |
| GET | `/api/v1/venture/signals/{company_id}/aggregation` | Aggregated signal metrics |
| POST | `/api/v1/venture/benchmark` | Score against historical benchmarks |

Example `POST /api/v1/venture`:

```json
{
  "startup_name": "Acme AI",
  "description": "Acme AI builds enterprise machine learning tooling.",
  "website_url": "https://acme.example.com",
  "pitch_deck_url": null,
  "founder_linkedin_urls": []
}
```

```json
{
  "startup_name": "Acme AI",
  "overall_score": 72.4,
  "overall_confidence": 0.81,
  "dimension_scores": [
    {"dimension": "market_opportunity", "score": 78.0, "rationale": "..."}
  ],
  "decision_category": "invest",
  "conviction_level": "high",
  "summary": "..."
}
```

### Portfolio Analysis — `/api/v1/portfolio`

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/portfolio` | Analyze a portfolio: score, distributions, risk, diversification, heatmap |
| POST | `/api/v1/portfolio/similarity` | Pairwise similarity matrix for the portfolio |

Portfolio request bodies carry `company_names` plus optional per-company
`descriptions` and `website_urls` keyed by company name.

### Company Comparison — `/api/v1/compare`

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/compare` | Compare 2+ companies across features, decisions, contributions, signals, knowledge graph, benchmarks |

Requires at least two company names. Returns dimension-by-dimension diffs,
decision comparison, contribution comparison, signal counts, knowledge graph
presence, benchmark metrics, and an overall summary with average score and
score range.

### Due Diligence — `/api/v1/due-diligence`

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/due-diligence` | Generate a full due diligence report |

The report contains an executive summary, ranked strengths/weaknesses,
opportunities, risks with mitigations, evidence list, decision trace,
confidence, benchmark context, and supporting feature metrics.

### Search — `/api/v1/search`

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/search` | Unified search (JSON body) |
| GET | `/api/v1/search?q=...` | Unified search (query string) |
| GET | `/api/v1/search/by-type/{type}?q=...` | Typed search |

Supported search types: `all` (default), `company`, `industry`, `country`,
`investor`, `founder`, `technology`, `knowledge_graph_node`, `signal_type`.
Responses include `query`, `search_type`, `total`, `offset`, `limit`, and
`results` with typed result entries.

### Batch Processing — `/api/v1/batch`

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/batch` | Submit an async batch job (returns `202`, `job_id`) |
| GET | `/api/v1/batch` | List jobs (paginated) |
| GET | `/api/v1/batch/{job_id}` | Job detail + paginated results |
| GET | `/api/v1/batch/{job_id}/progress` | Lightweight progress for polling |
| POST | `/api/v1/batch/{job_id}/cancel` | Cancel a pending/running job |

Job lifecycle: `pending` → `running` → `completed` / `failed` / `cancelled`.
Jobs track `total_items`, `completed_items`, and `progress_pct`.

---

## Authentication

All analytics endpoints accept an optional bearer token via
`get_current_user_optional` (`app/auth/jwt.py`). Passing a valid token lets
downstream persistence associate the request with a user; omitting it still
allows full anonymous use. The existing `/api/v1/analyze` GET endpoints keep
their stricter `get_current_user` dependency (401 when unauthenticated).

## Rate Limiting

An in-memory sliding-window limiter (`app/middleware/rate_limit.py`) keyed by
resolved client IP applies to all endpoints except health, readiness, and
docs/openapi. Defaults: 100 requests / 60 s per client IP.

- Configured via `RATE_LIMIT_ENABLED`, `RATE_LIMIT_REQUESTS`,
  `RATE_LIMIT_WINDOW_SECONDS`, `RATE_LIMIT_TRUSTED_PROXIES`,
  `RATE_LIMIT_MAX_TRACKED_CLIENTS`.
- `X-Forwarded-For` is only honored when the peer is a trusted proxy.
- Exceeding the limit returns `429` with `code: "RATE_LIMIT_EXCEEDED"` and a
  `Retry-After` header.

## Observability

- Every request gets a `request_id` (RequestIDMiddleware) and a structured JSON
  log line (LoggingMiddleware) containing method, path, status, processing time,
  and user.
- Health endpoints report engine reachability and DB health; `/readiness`
  returns `503` when a dependency is down.

---

## Development

```bash
# Lint
ruff check app/ tests/

# Type check
mypy app/

# Tests
pytest tests/
```

## Deployment

Standard FastAPI/uvicorn deployment:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The batch service holds jobs in memory (single-process default). For
multi-worker or multi-instance deployments, back the BatchJobStore and
similarity/graph caches with a shared store (see Future Extensions). The
dataset-backed endpoints (knowledge graph, signals, search) read from
`data/dataset/` when present and degrade gracefully to empty results when the
directory does not exist.

---

## Future Extensions

- **Persistent batch store**: move in-memory `BatchJobStore` to the existing
  SQLAlchemy async persistence layer so jobs survive restarts and scale across
  workers.
- **Tenant isolation**: enforce ownership on portfolio, comparison, and batch
  resources using existing auth scopes.
- **Streaming search**: cursor-based pagination for large datasets.
- **Webhook notifications**: emit job-completion events for batch workflows.
- **Rate-limit backends**: pluggable Redis-backed limiter for multi-instance
  deployments.