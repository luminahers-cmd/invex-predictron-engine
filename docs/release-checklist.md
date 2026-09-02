# Release Checklist

This checklist captures everything required before tagging a public release of
the Predictron API / Predictron Engine. Work through every section in order and
tick each item only after it is genuinely verified. A release must not be tagged
until all **Required** items pass.

> **Engine version:** `0.12.1` (`predictron_engine/version.py`)
> **App version:** `1.0.0` (`app/core/config.py`)
> **Checklist target:** a public GitHub release (tag + notes).

---

## 1. Tests

**Required**
- [ ] Full suite passes: `python -m pytest tests`
- [ ] Expect **3044** collected tests (verify the count matches; update README if it drifts).
- [ ] No new tests skipped without justification (`pytest -q -r s` shows no unexpected skips).
- [ ] Async tests run (`asyncio_mode = "auto"` is on; no `asyncio_mode` override warnings).

**Recommended**
- [ ] Run the suite a second time to confirm determinism (no flaky ordering).

---

## 2. Lint

**Required**
- [ ] `ruff check .` passes with zero violations.
- [ ] Ruff format-clean (`ruff format --check .`) if formatting is enforced.

---

## 3. Type checking

**Required**
- [ ] `mypy` passes (strict mode; see `pyproject.toml [tool.mypy]`).
- [ ] No new `# type: ignore` comments introduced.

---

## 4. Benchmarks

**Required**
- [ ] Committed snapshots are internally consistent.
- [ ] Snapshot diff confirms no unexpected regression among the latest pair:
      `python -m benchmarks.benchmark_report --diff <prev> <last>`
- [ ] At least two committed snapshots exist in `benchmarks/expected_outputs/`
      (CI auto-discovers the latest pair).
- [ ] `python -m benchmarks.benchmark_report --validate-only` passes.

**Recommended**
- [ ] Generate a fresh report: `python -m benchmarks.benchmark_report --output report.md`.

---

## 5. Offline evidence replay

**Required**
- [ ] Evidence replay dataset present under `benchmarks/offline_evidence/`.
- [ ] Replay provider returns byte-identical production behavior when disabled
      (`build_replay_provider()` returns `None`).
- [ ] Offline replay run works without network access:
      `python -m benchmarks.benchmark_runner --offline-replay`.
- [ ] Replay corpus schema version matches `evidence/replay/dataset.py`.

---

## 6. Security

**Required**
- [ ] `ENVIRONMENT=production` refuses to start with an insecure default
      `SECRET_KEY` (verified by `tests/test_config_validation.py`).
- [ ] No secrets committed: no real API keys, passwords, or tokens in the tree.
- [ ] `.env` is gitignored and **not** present in the commit set.
- [ ] `.dockerignore` excludes `.env`, caches, `.git`, IDE files (build context clean).
- [ ] Rate limiter memory is bounded (`RATE_LIMIT_MAX_TRACKED_CLIENTS`).
- [ ] Rate limiter X-Forwarded-For is only honoured from trusted proxies
      (`RATE_LIMIT_TRUSTED_PROXIES`) — spoofing is not exploitable.
- [ ] Report a vulnerability path documented in `SECURITY.md`.

---

## 7. Docker

**Required**
- [ ] `docker compose up --build` succeeds.
- [ ] Container builds with a clean context (no `.git/`, `.env`, caches copied in).
- [ ] Image starts and serves `/health` and `/docs`.
- [ ] `requirements.txt` is present in the image; app code copied after dependency
      install (layering correct).

---

## 8. Environment variables

**Required**
- [ ] Every variable in `.env.example` is documented in `README.md`.
- [ ] Every variable referenced in `app/core/config.py` has a default or is
      documented as required.
- [ ] `ENVIRONMENT` accepts only `development` / `production`.
- [ ] `SECRET_KEY` required (hard-fail) in production.
- [ ] `DATABASE_URL` default has no embedded password.

**Recommended**
- [ ] `EVIDENCE_SEARCH_ENABLED` / `TAVILY_API_KEY` / `EVIDENCE_REPLAY_*`
      documented consistently (they are read via `os.environ`, not pydantic).

---

## 9. Database

**Required**
- [ ] Alembic migration chain is linear and up to date: `alembic upgrade head`
      succeeds on a fresh database.
- [ ] Current migration chain `0001 → 0002 → 0003 → 0004` intact.
- [ ] Rollback path verified: `alembic downgrade -1` works.
- [ ] Models and migrations are in sync (no schema drift).

---

## 10. CI

**Required**
- [ ] `.github/workflows/ci.yml` runs on push and pull request.
- [ ] CI passes: ruff, mypy, pytest, benchmark snapshot diff.
- [ ] Benchmark regression step auto-discovers snapshots (no hardcoded versions).
- [ ] CI timeout (30 min) is sufficient for the suite.

---

## 11. Versioning

**Required**
- [ ] `ENGINE_VERSION` (`0.12.1`) is the single source of truth in
      `predictron_engine/version.py`.
- [ ] `APP_VERSION` (`1.0.0`) matches the application health endpoint.
- [ ] `pyproject.toml [project].version` matches `ENGINE_VERSION`.
- [ ] CHANGELOG documents the release with a Keep-a-Changelog / SemVer entry.

**Recommended**
- [ ] Document the relationship between `ENGINE_VERSION` and `APP_VERSION`
      (they are intentionally separate namespaces).

---

## 12. Documentation

**Required**
- [ ] `ARCHITECTURE.md` reflects the current pipeline (version-stamped current).
- [ ] `docs/architecture.md` high-level doc consistent with current pipeline.
- [ ] `README.md` links resolve; test count matches actual suite.
- [ ] `CHANGELOG.md` current.

---

## 13. Dataset validation

**Required**
- [ ] Dataset builder CLI works: `import`, `verify`, `stats`, `export`,
      `analyze`, `evaluate`.
- [ ] `predictron-dataset verify` reports a valid store (no integrity issues).
- [ ] `docs/dataset-builder.md` matches the dataset module surface.

---

## 14. Pre-tagging

**Required**
- [ ] `git status` clean (all intended work committed, no stray untracked files).
- [ ] No dead modules / obsolete artifacts in the commit set.
- [ ] No accidental `.env`, cache, or build artifact committed.
- [ ] Confirm the release tag matches the CHANGELOG version.
- [ ] Write release notes summarizing changes since the last tag.

---

*Exit criteria: every **Required** item is checked. **Recommended** items should be
addressed where practical but are not release-blocking.*
