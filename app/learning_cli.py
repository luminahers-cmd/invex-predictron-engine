"""``predictron-learning`` — continuous learning intelligence CLI (Phase 7).

Operator-facing companion to the Phase 7 learning layer:

* ``summary`` — live learning summary (counts, digest, calibration,
  confidence, knowledge, patterns, observations, recommendations) over the
  whole platform (repository scope);
* ``snapshot`` — compute and append-only record the deterministic learning
  snapshot to the learning tables (``learning_snapshots`` /
  ``learning_observations`` / ``learning_patterns`` / ``learning_reports``).

All analytics delegate to the deterministic engine builders in
``predictron_engine.learning``.  The CLI is intentionally thin: it opens an
async session against ``DATABASE_URL`` (mirroring ``alembic/env.py``),
computes via :class:`LearningService`, formats, and exits.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.services.learning import LearningService
from predictron_engine.learning.models import LearningPeriodKind


def _parse_as_of(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_period(value: str) -> LearningPeriodKind:
    try:
        return LearningPeriodKind(value)
    except ValueError:
        print(
            f"error: period must be one of {[p.value for p in LearningPeriodKind]}",
            file=sys.stderr,
        )
        raise


async def _run_live(command: str, *, as_of: datetime | None, json_out: bool) -> int:
    settings = get_settings()
    engine = create_async_engine(settings.DATABASE_URL)
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        service = LearningService()
        if command == "summary":
            data = await service.summary(session, as_of=as_of, scope_all=True)
        else:  # pragma: no cover - guarded by argparse
            raise AssertionError(f"unsupported live command: {command}")
        _emit_summary(data, json_out)
    await engine.dispose()
    return 0


async def _run_snapshot(
    *,
    as_of: datetime | None,
    period: LearningPeriodKind,
    json_out: bool,
) -> int:
    settings = get_settings()
    engine = create_async_engine(settings.DATABASE_URL)
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        snapshot = await LearningService().snapshot(
            session, as_of=as_of, period_kind=period
        )
        await session.commit()
        if json_out:
            print(snapshot.model_dump_json(indent=2))
        else:
            print(
                f"recorded learning snapshot {snapshot.snapshot_id} "
                f"(anchor {snapshot.anchor_date}, {snapshot.period_kind.value}, "
                f"engine {snapshot.engine_version})"
            )
            print(f"counts: {_format_counts(snapshot)}")
            print(f"content_hash: {snapshot.content_hash}")
            print("verify: ok" if snapshot.verify() else "verify: FAILED")
    await engine.dispose()
    return 0


async def _run_reports(period: LearningPeriodKind) -> int:
    settings = get_settings()
    engine = create_async_engine(settings.DATABASE_URL)
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        response = await LearningService().reports(session, period_kind=period)
        rows = response.reports
        if not rows:
            print("no recorded learning snapshots")
        for row in rows:
            print(
                f"{row['snapshot_id']} anchor={row['anchor_date']} "
                f"scope={row['scope']} engine={row['engine_version']}"
            )
    await engine.dispose()
    return 0


async def _run_report_detail(snapshot_id: str) -> int:
    settings = get_settings()
    engine = create_async_engine(settings.DATABASE_URL)
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        report = await LearningService().report_detail(session, snapshot_id=snapshot_id)
        if report is None:
            print(f"error: no learning report recorded for snapshot {snapshot_id!r}")
            await engine.dispose()
            return 1
        print(f"snapshot_id: {report.snapshot_id}")
        print(f"report_id: {report.report_id}")
        print(f"recorded_at: {report.recorded_at.isoformat()}")
        print(f"content_hash: {report.content_hash}")
        print(f"verify: {'ok' if report.verify() else 'FAILED'}")
        counts = report.payload.get("counts", {})
        if isinstance(counts, dict):
            print(_format_counts_dict(counts))
    await engine.dispose()
    return 0


def _format_counts(snapshot) -> str:
    return _format_counts_dict(dict(snapshot.counts))


def _format_counts_dict(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _emit_summary(payload, json_out: bool) -> None:
    if json_out:
        print(payload.model_dump_json(indent=2))
        return
    print(payload.model_dump_json(indent=2))


async def _main_async(args: argparse.Namespace) -> int:
    try:
        period = _parse_period(getattr(args, "period", "daily"))
    except ValueError:
        return 2
    as_of = _parse_as_of(getattr(args, "as_of", None))
    if args.command in ("summary",):
        return await _run_live(args.command, as_of=as_of, json_out=args.json)
    if args.command == "snapshot":
        return await _run_snapshot(as_of=as_of, period=period, json_out=args.json)
    if args.command == "reports":
        return await _run_reports(period)
    if args.command == "report":
        return await _run_report_detail(args.snapshot_id)
    # pragma: no cover - guarded by argparse
    raise AssertionError(f"unsupported command: {args.command}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predictron-learning",
        description="Continuous Learning Intelligence (Phase 7).",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON payloads")
    sub = parser.add_subparsers(dest="command", required=True)

    summary = sub.add_parser("summary")
    summary.add_argument("--as-of", default=None, help="ISO-8601 analysis moment")

    snap = sub.add_parser("snapshot")
    snap.add_argument("--as-of", default=None, help="ISO-8601 snapshot moment")
    snap.add_argument("--period", default="daily", help="period kind (daily/weekly/monthly)")

    reports = sub.add_parser("reports")
    reports.add_argument("--period", default="daily", help="period kind (daily/weekly/monthly)")

    report = sub.add_parser("report")
    report.add_argument("snapshot_id", help="deterministic snapshot id")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
