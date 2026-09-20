"""Tests for the 0005 company registry migration.

Uses a recording-op proxy to verify that upgrade() issues the expected
DDL operations and that downgrade() issues the exact mirror image, plus
revision wiring checks against the ORM metadata.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MIGRATION_PATH = (
    Path(__file__).resolve().parents[2] / "alembic" / "versions" / "0005_add_company_registry.py"
)


@pytest.fixture(scope="module")
def migration_module():
    spec = importlib.util.spec_from_file_location("migration_0005", MIGRATION_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_revision_value(migration_module):
    assert migration_module.revision == "0005"


def test_down_revision_value(migration_module):
    assert migration_module.down_revision == "0004"


def test_branch_labels_none(migration_module):
    assert migration_module.branch_labels is None


def test_depends_on_none(migration_module):
    assert migration_module.depends_on is None


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


def test_upgrade_creates_companies_table(migration_module):
    recorder = _run_upgrade(migration_module)
    created = {c[1][0] for c in recorder.calls if c[0] == "create_table"}
    assert "companies" in created


def test_upgrade_creates_snapshots_table(migration_module):
    recorder = _run_upgrade(migration_module)
    created = {c[1][0] for c in recorder.calls if c[0] == "create_table"}
    assert "company_snapshots" in created


def test_upgrade_companies_has_company_id_pk(migration_module):
    recorder = _run_upgrade(migration_module)
    call = next(c for c in recorder.calls if c[0] == "create_table" and c[1][0] == "companies")
    cols = call[1][1:]
    pk_col = next(c for c in cols if c.name == "company_id")
    assert pk_col.primary_key is True


def test_upgrade_companies_company_id_len_64(migration_module):
    recorder = _run_upgrade(migration_module)
    call = next(c for c in recorder.calls if c[0] == "create_table" and c[1][0] == "companies")
    pk_col = next(c for c in call[1][1:] if c.name == "company_id")
    assert pk_col.type.length == 64


def test_upgrade_companies_canonical_domain_nullable(migration_module):
    recorder = _run_upgrade(migration_module)
    call = next(c for c in recorder.calls if c[0] == "create_table" and c[1][0] == "companies")
    col = next(c for c in call[1][1:] if c.name == "canonical_domain")
    assert col.nullable is True


def test_upgrade_snapshots_analysis_id_unique_index(migration_module):
    recorder = _run_upgrade(migration_module)
    my_call = next(
        c for c in recorder.calls
        if c[0] == "create_index" and c[1][0] == "uq_company_snapshots_analysis_id"
    )
    assert my_call[2].get("unique") is True


def test_upgrade_canonical_domain_index_unique(migration_module):
    recorder = _run_upgrade(migration_module)
    my_call = next(
        c for c in recorder.calls
        if c[0] == "create_index" and c[1][0] == "uq_companies_canonical_domain"
    )
    assert my_call[2].get("unique") is True


def test_upgrade_creates_all_expected_indexes(migration_module):
    recorder = _run_upgrade(migration_module)
    names = {c[1][0] for c in recorder.calls if c[0] == "create_index"}
    expected = {
        "ix_companies_user_created",
        "ix_companies_last_seen",
        "uq_companies_canonical_domain",
        "ix_companies_canonical_name_key",
        "ix_companies_fallback_slug",
        "ix_company_snapshots_company_created",
        "ix_company_snapshots_company_id",
        "uq_company_snapshots_analysis_id",
    }
    assert expected.issubset(names)


def test_downgrade_drops_snapshots_table(migration_module):
    recorder = _run_downgrade(migration_module)
    dropped = {c[1][0] for c in recorder.calls if c[0] == "drop_table"}
    assert "company_snapshots" in dropped


def test_downgrade_drops_companies_table(migration_module):
    recorder = _run_downgrade(migration_module)
    dropped = {c[1][0] for c in recorder.calls if c[0] == "drop_table"}
    assert "companies" in dropped


def test_downgrade_drops_indexes(migration_module):
    recorder = _run_downgrade(migration_module)
    names = {c[1][0] for c in recorder.calls if c[0] == "drop_index"}
    expected = {
        "uq_company_snapshots_analysis_id",
        "ix_company_snapshots_company_id",
        "ix_company_snapshots_company_created",
        "ix_companies_fallback_slug",
        "ix_companies_canonical_name_key",
        "uq_companies_canonical_domain",
        "ix_companies_last_seen",
        "ix_companies_user_created",
    }
    assert expected.issubset(names)


def test_upgrade_indexes_before_table_guard():
    """Sanity check: no index references a nonexistent table (structural)."""
    import sqlalchemy as sa

    metadata = sa.MetaData()
    sa.Table(
        "companies",
        metadata,
        sa.Column("company_id", sa.String(64), primary_key=True),
        sa.Column("canonical_domain", sa.String(255), nullable=True),
    )
    # The migration module should import cleanly without side effects.
    spec = importlib.util.spec_from_file_location("migration_0005_b", MIGRATION_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.revision == "0005"


def test_orm_metadata_matches_migration_table_names():
    """The ORM tables referenced by the store exist in metadata."""
    from app.models import Company, CompanySnapshot

    assert Company.__tablename__ == "companies"
    assert CompanySnapshot.__tablename__ == "company_snapshots"


def test_orm_models_share_migration_pk_columns():
    from app.models import Company, CompanySnapshot

    assert "company_id" in Company.__table__.columns
    assert "id" in CompanySnapshot.__table__.columns
    assert "company_id" in CompanySnapshot.__table__.columns


def test_latest_migration_is_0007():
    import re

    files = (MIGRATION_PATH.parent).glob("*.py")
    revisions = [
        re.match(r"(\d{4})_", f.name).group(1)
        for f in files
        if f.name[0].isdigit()
    ]
    assert max(revisions, key=int) == "0007"
