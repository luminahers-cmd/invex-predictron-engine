"""Add composite index on (user_id, created_at) for analysis_requests.

The model defines ix_analysis_requests_user_created to optimize queries that
filter by user_id and order by created_at. This index was added to the model
but never migrated.  This migration closes the drift.

Revision ID: 0004
Revises: 0003
"""

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_analysis_requests_user_created",
        "analysis_requests",
        ["user_id", "created_at"],
        postgresql_using="btree",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_analysis_requests_user_created",
        table_name="analysis_requests",
    )
