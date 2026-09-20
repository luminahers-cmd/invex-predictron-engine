"""Backfill the company registry from legacy analyses (Phase 2, roadmap #6).

Companies analyzed *before* the registry milestone are ingested from the
existing ``analysis_reports`` / ``analysis_requests`` tables so their history
becomes first-class registry data.

The pass is idempotent: re-running ``upgrade`` adds no company or snapshot
rows and never double-increments ``snapshot_count`` (deterministic snapshot
ids + ``ON CONFLICT DO NOTHING`` appends). Company rows *created* by this
milestone are recorded in the ``company_registry_backfill`` tracking table so
``downgrade`` can remove exactly the data this milestone introduced, without
touching pre-existing registry rows.

Revision: 0006
Revises: 0005
"""

from datetime import UTC, datetime

import sqlalchemy as sa

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

_TRACKING_TABLE = "company_registry_backfill"


def upgrade() -> None:
    op.create_table(
        _TRACKING_TABLE,
        sa.Column("company_id", sa.String(64), primary_key=True),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    _apply_backfill(op.get_bind())


def downgrade() -> None:
    bind = op.get_bind()
    tracking = sa.table(_TRACKING_TABLE, sa.column("company_id", sa.String(64)))
    companies = sa.table("companies", sa.column("company_id", sa.String(64)))
    snapshots = sa.table(
        "company_snapshots",
        sa.column("company_id", sa.String(64)),
    )

    created_ids = bind.execute(sa.select(tracking.c.company_id)).scalars().all()
    for company_id in created_ids:
        bind.execute(sa.delete(snapshots).where(snapshots.c.company_id == company_id))
        bind.execute(sa.delete(companies).where(companies.c.company_id == company_id))

    op.drop_table(_TRACKING_TABLE)


def _apply_backfill(bind) -> object:
    """Run the (idempotent) data backfill and record created companies.

    Routed through ``app.services.company_backfill.backfill_sync`` so the
    migration shares the same pure mapping and identity resolution as the
    application-side backfill service.
    """
    from app.services.company_backfill import backfill_sync

    summary = backfill_sync(bind)
    if summary.created_company_ids:
        tracking = sa.table(
            _TRACKING_TABLE,
            sa.column("company_id", sa.String(64)),
            sa.column("ingested_at", sa.DateTime(timezone=True)),
        )
        bind.execute(
            tracking.insert(),
            [
                {"company_id": company_id, "ingested_at": datetime.now(UTC)}
                for company_id in summary.created_company_ids
            ],
        )
    return summary
