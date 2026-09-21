"""Tests for the 0008 live prediction ledger migration.

Uses the same recording-op proxy pattern as the Phase 1/4 migration tests:
verify upgrade() issues the expected DDL, downgrade() mirrors it, and the
ORM metadata is consistent with the migration.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic"
    / "versions"
    / "0008_add_forecasts_and_events.py"
)


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, MIGRATION_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RecordingOp:
    """Stand-in for alembic.op that logs every call."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return None

        return _record


def _run_upgrade(migration_module):
    recorder = RecordingOp()
    import alembic.op as real_op

    saved = real_op.create_table, real_op.create_index, real_op.drop_table, real_op.drop_index
    real_op.create_table = recorder.create_table
    real_op.create_index = recorder.create_index
    real_op.drop_table = recorder.drop_table
    real_op.drop_index = recorder.drop_index
    try:
        migration_module.upgrade()
    finally:
        real_op.create_table, real_op.create_index = saved[:2]
        real_op.drop_table, real_op.drop_index = saved[2:]
    return recorder


def _run_downgrade(migration_module):
    recorder = RecordingOp()
    import alembic.op as real_op

    saved = real_op.create_table, real_op.create_index, real_op.drop_table, real_op.drop_index
    real_op.create_table = recorder.create_table
    real_op.create_index = recorder.create_index
    real_op.drop_table = recorder.drop_table
    real_op.drop_index = recorder.drop_index
    try:
        migration_module.downgrade()
    finally:
        real_op.create_table, real_op.create_index = saved[:2]
        real_op.drop_table, real_op.drop_index = saved[2:]
    return recorder


def test_revision_value() -> None:
    module = _load_module("migration_0008_revision")
    assert module.revision == "0008"


def test_down_revision_value() -> None:
    module = _load_module("migration_0008_down")
    assert module.down_revision == "0007"


def test_branch_labels_and_depends_on_none() -> None:
    module = _load_module("migration_0008_wiring")
    assert module.branch_labels is None
    assert module.depends_on is None


def test_upgrade_creates_forecasts_table() -> None:
    recorder = _run_upgrade(_load_module("migration_0008_up_forecasts"))
    created = {c[1][0] for c in recorder.calls if c[0] == "create_table"}
    assert "forecasts" in created
    assert "forecast_events" in created


def test_upgrade_forecasts_has_deterministic_pk() -> None:
    recorder = _run_upgrade(_load_module("migration_0008_up_pk"))
    call = next(
        c for c in recorder.calls if c[0] == "create_table" and c[1][0] == "forecasts"
    )
    cols = call[1][1:]
    pk_col = next(c for c in cols if c.name == "id")
    assert pk_col.primary_key is True
    assert pk_col.type.length == 64


def test_upgrade_forecasts_has_immutable_metadata_columns() -> None:
    recorder = _run_upgrade(_load_module("migration_0008_up_meta"))
    call = next(
        c for c in recorder.calls if c[0] == "create_table" and c[1][0] == "forecasts"
    )
    names = {c.name for c in call[1][1:]}
    assert {"prediction", "engine_version", "schema_version"} <= names
    assert "outcome_id" in names
    assert "status" in names


def test_upgrade_forecasts_indexes() -> None:
    recorder = _run_upgrade(_load_module("migration_0008_up_indexes"))
    names = {c[1][0] for c in recorder.calls if c[0] == "create_index"}
    expected = {
        "ix_forecasts_company_created",
        "ix_forecasts_status",
        "ix_forecasts_company_status",
        "ix_forecasts_due_at",
        "ix_forecasts_outcome_id",
        "uq_forecasts_snapshot_horizon",
    }
    assert expected <= names
    unique_call = next(
        c
        for c in recorder.calls
        if c[0] == "create_index" and c[1][0] == "uq_forecasts_snapshot_horizon"
    )
    assert unique_call[2].get("unique") is True


def test_upgrade_forecast_events_indexes() -> None:
    recorder = _run_upgrade(_load_module("migration_0008_up_events_indexes"))
    names = {c[1][0] for c in recorder.calls if c[0] == "create_index"}
    assert "ix_forecast_events_forecast_occurred" in names
    assert "ix_forecast_events_type" in names


def test_downgrade_drops_tables_and_indexes() -> None:
    recorder = _run_downgrade(_load_module("migration_0008_down"))
    dropped_tables = {c[1][0] for c in recorder.calls if c[0] == "drop_table"}
    assert {"forecasts", "forecast_events"} <= dropped_tables
    dropped_indexes = {c[1][0] for c in recorder.calls if c[0] == "drop_index"}
    assert {
        "ix_forecasts_company_created",
        "ix_forecasts_status",
        "ix_forecasts_company_status",
        "ix_forecasts_due_at",
        "ix_forecasts_outcome_id",
        "uq_forecasts_snapshot_horizon",
        "ix_forecast_events_forecast_occurred",
        "ix_forecast_events_type",
    } <= dropped_indexes


def test_orm_metadata_matches_migration_tables() -> None:
    from app.models import Forecast, ForecastEvent

    assert Forecast.__tablename__ == "forecasts"
    assert ForecastEvent.__tablename__ == "forecast_events"
    assert "prediction" in Forecast.__table__.columns
    assert "engine_version" in Forecast.__table__.columns
    assert "schema_version" in Forecast.__table__.columns
    assert "status" in Forecast.__table__.columns
    assert "event_type" in ForecastEvent.__table__.columns
