"""Add the Continuous Learning Intelligence tables — Phase 7.

Creates the append-only learning persistence:

* ``learning_snapshots`` — recorded snapshot headers (deterministic id).
* ``learning_observations`` — canonical observation rows.
* ``learning_patterns`` — per-dimension cohort pattern rows.
* ``learning_reports`` — full serialized snapshot payloads with an integrity
  hash.

This milestone is additive only — no existing table is altered.

Revision: 0009
Revises: 0008
"""

import sqlalchemy as sa

from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "learning_snapshots",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("scope", sa.String(255), nullable=False),
        sa.Column("period_kind", sa.String(20), nullable=False),
        sa.Column("anchor_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("engine_version", sa.String(50), nullable=False),
        sa.Column("schema_version", sa.String(20), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("counts", sa.JSON(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("distributions", sa.JSON(), nullable=False),
        sa.Column("meta", sa.JSON(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_learning_snapshots_scope_anchor",
        "learning_snapshots",
        ["scope", "anchor_date"],
    )
    op.create_index(
        "ix_learning_snapshots_created",
        "learning_snapshots",
        ["created_at"],
    )

    op.create_table(
        "learning_observations",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("snapshot_id", sa.String(64), nullable=False),
        sa.Column("category", sa.String(50), nullable=False),
        sa.Column("dimension", sa.String(50), nullable=False),
        sa.Column("value", sa.String(255), nullable=False),
        sa.Column("metric", sa.String(50), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=True),
        sa.Column("delta", sa.Float(), nullable=True),
        sa.Column("direction", sa.String(20), nullable=False),
        sa.Column("baseline", sa.Float(), nullable=True),
        sa.Column("sample_size", sa.Integer(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_learning_observations_snapshot_id",
        "learning_observations",
        ["snapshot_id"],
    )
    op.create_index(
        "ix_learning_observations_category",
        "learning_observations",
        ["category"],
    )
    op.create_index(
        "ix_learning_observations_dimension",
        "learning_observations",
        ["dimension"],
    )

    op.create_table(
        "learning_patterns",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("snapshot_id", sa.String(64), nullable=False),
        sa.Column("dimension", sa.String(50), nullable=False),
        sa.Column("value", sa.String(255), nullable=False),
        sa.Column("samples", sa.Integer(), nullable=False),
        sa.Column("scoreable", sa.Integer(), nullable=False),
        sa.Column("accuracy", sa.Float(), nullable=True),
        sa.Column("precision", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("confidence_bias", sa.Float(), nullable=True),
        sa.Column("false_positive_rate", sa.Float(), nullable=True),
        sa.Column("false_negative_rate", sa.Float(), nullable=True),
        sa.Column("recommendation", sa.Text(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_learning_patterns_snapshot_id",
        "learning_patterns",
        ["snapshot_id"],
    )
    op.create_index(
        "ix_learning_patterns_dimension_value",
        "learning_patterns",
        ["dimension", "value"],
    )

    op.create_table(
        "learning_reports",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("snapshot_id", sa.String(64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_learning_reports_snapshot_id",
        "learning_reports",
        ["snapshot_id"],
    )
    op.create_index(
        "ix_learning_reports_created",
        "learning_reports",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_learning_reports_created", table_name="learning_reports")
    op.drop_index("ix_learning_reports_snapshot_id", table_name="learning_reports")
    op.drop_table("learning_reports")

    op.drop_index(
        "ix_learning_patterns_dimension_value",
        table_name="learning_patterns",
    )
    op.drop_index(
        "ix_learning_patterns_snapshot_id",
        table_name="learning_patterns",
    )
    op.drop_table("learning_patterns")

    op.drop_index(
        "ix_learning_observations_dimension",
        table_name="learning_observations",
    )
    op.drop_index(
        "ix_learning_observations_category",
        table_name="learning_observations",
    )
    op.drop_index(
        "ix_learning_observations_snapshot_id",
        table_name="learning_observations",
    )
    op.drop_table("learning_observations")

    op.drop_index(
        "ix_learning_snapshots_created",
        table_name="learning_snapshots",
    )
    op.drop_index(
        "ix_learning_snapshots_scope_anchor",
        table_name="learning_snapshots",
    )
    op.drop_table("learning_snapshots")
