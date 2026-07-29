"""Add analysis_requests and analysis_reports tables.

Revision: 0002
Revises: 0001
"""

import sqlalchemy as sa

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "analysis_requests",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("startup_name", sa.String(255), nullable=False),
        sa.Column("website", sa.String(500), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("pitch_deck_url", sa.String(500), nullable=True),
        sa.Column(
            "founder_linkedin_urls",
            sa.JSON(),
            nullable=False,
            server_default="[]",
        ),
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

    op.create_table(
        "analysis_reports",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "request_id",
            sa.String(36),
            sa.ForeignKey("analysis_requests.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("startup_name", sa.String(255), nullable=False),
        sa.Column("venture_score", sa.Float(), nullable=False),
        sa.Column("market_score", sa.Float(), nullable=False),
        sa.Column("founder_score", sa.Float(), nullable=False),
        sa.Column("traction_score", sa.Float(), nullable=False),
        sa.Column(
            "recommendations",
            sa.JSON(),
            nullable=False,
            server_default="[]",
        ),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("engine_version", sa.String(50), nullable=True),
        sa.Column("processing_time_ms", sa.Float(), nullable=True),
        sa.Column("full_report", sa.JSON(), nullable=False),
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
        "ix_analysis_requests_created_at",
        "analysis_requests",
        ["created_at"],
    )
    op.create_index(
        "ix_analysis_reports_created_at",
        "analysis_reports",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_analysis_reports_created_at", table_name="analysis_reports")
    op.drop_index(
        "ix_analysis_requests_created_at", table_name="analysis_requests"
    )
    op.drop_table("analysis_reports")
    op.drop_table("analysis_requests")
