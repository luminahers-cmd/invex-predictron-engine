"""Add the live prediction ledger (forecasts and forecast_events) — Phase 5.

Creates the deterministic forecast lifecycle tables:

* ``forecasts`` — one frozen prediction per ``(snapshot, horizon)`` with an
  immutable ``prediction`` payload, pinned ``engine_version`` /
  ``schema_version`` metadata, and a service-owned derived ``status``.
  Re-registration is idempotent via the deterministic primary key and the
  unique ``(snapshot_id, horizon_days)`` index.
* ``forecast_events`` — an append-only lifecycle event log
  (``registered`` / ``due`` / ``resolved``) with deterministic event IDs.

This milestone is additive only — no existing table is altered and snapshots
stay immutable.

Revision: 0008
Revises: 0007
"""

import sqlalchemy as sa

from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "forecasts",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "company_id",
            sa.String(64),
            sa.ForeignKey("companies.company_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "snapshot_id",
            sa.String(64),
            sa.ForeignKey("company_snapshots.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "outcome_id",
            sa.String(64),
            sa.ForeignKey("company_outcomes.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("analysis_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(50), nullable=False),
        sa.Column("decision", sa.String(50), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("composite_score", sa.Float(), nullable=False),
        sa.Column("prediction", sa.JSON(), nullable=False),
        sa.Column("engine_version", sa.String(50), nullable=False),
        sa.Column("schema_version", sa.String(50), nullable=False),
        sa.Column("reconciled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_forecasts_company_created",
        "forecasts",
        ["company_id", "created_at"],
    )
    op.create_index("ix_forecasts_status", "forecasts", ["status"])
    op.create_index(
        "ix_forecasts_company_status",
        "forecasts",
        ["company_id", "status"],
    )
    op.create_index("ix_forecasts_due_at", "forecasts", ["due_at"])
    op.create_index("ix_forecasts_outcome_id", "forecasts", ["outcome_id"])
    op.create_index(
        "uq_forecasts_snapshot_horizon",
        "forecasts",
        ["snapshot_id", "horizon_days"],
        unique=True,
    )

    op.create_table(
        "forecast_events",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "forecast_id",
            sa.String(64),
            sa.ForeignKey("forecasts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=True),
    )
    op.create_index(
        "ix_forecast_events_forecast_occurred",
        "forecast_events",
        ["forecast_id", "occurred_at"],
    )
    op.create_index(
        "ix_forecast_events_type",
        "forecast_events",
        ["event_type"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_forecast_events_type",
        table_name="forecast_events",
    )
    op.drop_index(
        "ix_forecast_events_forecast_occurred",
        table_name="forecast_events",
    )
    op.drop_table("forecast_events")

    op.drop_index(
        "uq_forecasts_snapshot_horizon",
        table_name="forecasts",
    )
    op.drop_index("ix_forecasts_outcome_id", table_name="forecasts")
    op.drop_index("ix_forecasts_due_at", table_name="forecasts")
    op.drop_index(
        "ix_forecasts_company_status",
        table_name="forecasts",
    )
    op.drop_index("ix_forecasts_status", table_name="forecasts")
    op.drop_index(
        "ix_forecasts_company_created",
        table_name="forecasts",
    )
    op.drop_table("forecasts")
