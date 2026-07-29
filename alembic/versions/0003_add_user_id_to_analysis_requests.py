"""Add user_id column to analysis_requests.

Revision: 0003
Revises: 0002
"""

import sqlalchemy as sa

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "analysis_requests",
        sa.Column("user_id", sa.String(255), nullable=True),
    )
    op.create_index(
        "ix_analysis_requests_user_id",
        "analysis_requests",
        ["user_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_analysis_requests_user_id", table_name="analysis_requests")
    op.drop_column("analysis_requests", "user_id")
