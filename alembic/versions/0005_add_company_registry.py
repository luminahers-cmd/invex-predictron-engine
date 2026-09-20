"""Add company registry tables (Company Intelligence Hub — Phase 1).

Creates the durable ``companies`` identity table and the append-only
``company_snapshots`` history table. This milestone is additive only —
no existing table is altered.

Identity is deterministic, never random: ``company_id`` is a SHA-256 digest
of the canonical identity material, and ``analysis_id`` is unique on
snapshots so duplicate analysis ingests cannot create duplicate history.

Revision: 0005
Revises: 0004
"""

import sqlalchemy as sa

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "companies",
        sa.Column("company_id", sa.String(64), primary_key=True),
        sa.Column("canonical_name", sa.String(255), nullable=False),
        sa.Column("canonical_name_key", sa.String(255), nullable=False),
        sa.Column("canonical_domain", sa.String(255), nullable=True),
        sa.Column(
            "fallback_slug",
            sa.String(128),
            nullable=False,
            server_default="unknown",
        ),
        sa.Column("primary_name", sa.String(255), nullable=False),
        sa.Column("website", sa.String(500), nullable=True),
        sa.Column("latest_decision", sa.String(50), nullable=True),
        sa.Column("latest_confidence", sa.Float(), nullable=True),
        sa.Column("latest_composite_score", sa.Float(), nullable=True),
        sa.Column(
            "snapshot_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("first_seen", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen", sa.DateTime(timezone=True), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_companies_user_created",
        "companies",
        ["user_id", "created_at"],
    )
    op.create_index("ix_companies_last_seen", "companies", ["last_seen"])
    op.create_index(
        "uq_companies_canonical_domain",
        "companies",
        ["canonical_domain"],
        unique=True,
    )
    op.create_index(
        "ix_companies_canonical_name_key",
        "companies",
        ["canonical_name_key"],
    )
    op.create_index(
        "ix_companies_fallback_slug",
        "companies",
        ["fallback_slug"],
    )

    op.create_table(
        "company_snapshots",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "company_id",
            sa.String(64),
            sa.ForeignKey("companies.company_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("analysis_id", sa.String(36), nullable=False),
        sa.Column("report_id", sa.String(36), nullable=False),
        sa.Column("decision", sa.String(50), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("composite_score", sa.Float(), nullable=True),
        sa.Column("readiness_score", sa.Float(), nullable=True),
        sa.Column(
            "dimension_scores",
            sa.JSON(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_company_snapshots_company_created",
        "company_snapshots",
        ["company_id", "created_at"],
    )
    op.create_index(
        "ix_company_snapshots_company_id",
        "company_snapshots",
        ["company_id"],
    )
    op.create_index(
        "uq_company_snapshots_analysis_id",
        "company_snapshots",
        ["analysis_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "uq_company_snapshots_analysis_id",
        table_name="company_snapshots",
    )
    op.drop_index(
        "ix_company_snapshots_company_id",
        table_name="company_snapshots",
    )
    op.drop_index(
        "ix_company_snapshots_company_created",
        table_name="company_snapshots",
    )
    op.drop_table("company_snapshots")

    op.drop_index("ix_companies_fallback_slug", table_name="companies")
    op.drop_index("ix_companies_canonical_name_key", table_name="companies")
    op.drop_index("uq_companies_canonical_domain", table_name="companies")
    op.drop_index("ix_companies_last_seen", table_name="companies")
    op.drop_index("ix_companies_user_created", table_name="companies")
    op.drop_table("companies")
