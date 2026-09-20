"""Add company outcomes and prediction evaluations (CIH Phase 4).

Creates the append-only ``company_outcomes`` and ``company_evaluations``
tables that power the validity loop. This milestone is additive only — no
existing table is altered and snapshots stay immutable.

Outcomes store observed facts (with a derived verdict); evaluations are
append-only rows with deterministic ids linking a snapshot to an outcome.
Re-running the evaluation pass is therefore idempotent.

Revision: 0007
Revises: 0006
"""

import sqlalchemy as sa

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "company_outcomes",
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
            sa.ForeignKey("company_snapshots.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "source",
            sa.String(255),
            nullable=False,
            server_default="manual",
        ),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("time_horizon_days", sa.Integer(), nullable=True),
        sa.Column("outcome_data", sa.JSON(), nullable=False),
        sa.Column("verdict", sa.String(50), nullable=False),
        sa.Column(
            "verdict_reasoning",
            sa.Text(),
            nullable=False,
            server_default="",
        ),
        sa.Column(
            "status",
            sa.String(50),
            nullable=False,
            server_default="unknown",
        ),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_company_outcomes_company_created",
        "company_outcomes",
        ["company_id", "created_at"],
    )
    op.create_index(
        "ix_company_outcomes_snapshot_id",
        "company_outcomes",
        ["snapshot_id"],
    )
    op.create_index(
        "ix_company_outcomes_verdict",
        "company_outcomes",
        ["verdict"],
    )

    op.create_table(
        "company_evaluations",
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
            sa.ForeignKey("company_outcomes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("prediction", sa.JSON(), nullable=False),
        sa.Column("verdict", sa.String(50), nullable=False),
        sa.Column(
            "alignment",
            sa.String(50),
            nullable=False,
            server_default="neutral",
        ),
        sa.Column("outcome_verdict", sa.String(50), nullable=False),
        sa.Column("decision_match", sa.Boolean(), nullable=True),
        sa.Column("snapshot_confidence", sa.Float(), nullable=True),
        sa.Column("snapshot_composite_score", sa.Float(), nullable=True),
        sa.Column(
            "snapshot_created_at", sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_company_evaluations_company_created",
        "company_evaluations",
        ["company_id", "created_at"],
    )
    op.create_index(
        "ix_company_evaluations_company_verdict",
        "company_evaluations",
        ["company_id", "verdict"],
    )
    op.create_index(
        "ix_company_evaluations_outcome_id",
        "company_evaluations",
        ["outcome_id"],
    )
    op.create_index(
        "uq_company_evaluations_snapshot_outcome",
        "company_evaluations",
        ["snapshot_id", "outcome_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "uq_company_evaluations_snapshot_outcome",
        table_name="company_evaluations",
    )
    op.drop_index(
        "ix_company_evaluations_outcome_id",
        table_name="company_evaluations",
    )
    op.drop_index(
        "ix_company_evaluations_company_verdict",
        table_name="company_evaluations",
    )
    op.drop_index(
        "ix_company_evaluations_company_created",
        table_name="company_evaluations",
    )
    op.drop_table("company_evaluations")

    op.drop_index("ix_company_outcomes_verdict", table_name="company_outcomes")
    op.drop_index(
        "ix_company_outcomes_snapshot_id", table_name="company_outcomes"
    )
    op.drop_index(
        "ix_company_outcomes_company_created", table_name="company_outcomes"
    )
    op.drop_table("company_outcomes")
