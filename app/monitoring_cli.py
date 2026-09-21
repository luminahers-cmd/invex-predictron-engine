"""``predictron-monitor`` — continuous intelligence & drift detection CLI.

Operator-facing companion to the Phase 6 monitoring layer:

* live views — ``summary``, ``health``, ``stale``, ``overdue``, ``reanalysis``
  over the whole platform (repository scope);
* ``snapshot`` — record the deterministic, append-only daily snapshot to the
  file-backed history (``data/monitor_history``);
* history views — ``trends`` and ``drift`` over the recorded snapshots.

All analytics delegate to the deterministic engine builders in
``predictron_engine.monitoring``.  The CLI is intentionally thin: it opens an
async session against ``DATABASE_URL`` (mirroring ``alembic/env.py``),
computes via :class:`MonitoringService`, formats, and exits.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.services.monitoring import MonitoringService
from predictron_engine.monitoring.history import MonitorHistory
from predictron_engine.monitoring.models import MonitorPeriodKind


def _parse_as_of(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


async def _open_live():
    settings = get_settings()
    engine = create_async_engine(settings.DATABASE_URL)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    return engine, factory()


async def _run_live(
    command: str, *, as_of: datetime | None, window: int | None, json_out: bool
) -> int:
    engine, session = await _open_live()
    try:
        service = MonitoringService()
        if command == "summary":
            data = await service.summary(session, as_of=as_of)
        elif command == "rolling":
            data = await service.rolling(session, as_of=as_of, window=window)
        elif command == "health":
            data = await service.health(session, as_of=as_of)
        elif command == "stale":
            data = await service.health(session, as_of=as_of, health_filter="stale")
        elif command == "overdue":
            data = await service.health(session, as_of=as_of, health_filter="overdue")
        elif command == "reanalysis":
            data = await service.reanalysis(session, as_of=as_of)
        else:  # pragma: no cover - guarded by argparse
            raise AssertionError(f"unsupported live command: {command}")
        _emit(data, json_out)
        return 0
    finally:
        await session.close()
        await engine.dispose()


async def _run_snapshot(
    *,
    as_of: datetime | None,
    window: int | None,
    keep: int | None,
    json_out: bool,
) -> int:
    settings = get_settings()
    engine, session = await _open_live()
    try:
        snapshot = await MonitoringService().repository_snapshot(
            session, as_of=as_of, window=window
        )
        path = MonitorHistory(settings.MONITOR_HISTORY_DIR).record_snapshot(
            snapshot, keep=keep
        )
        if json_out:
            print(snapshot.model_dump_json(indent=2))
        else:
            print(
                f"recorded snapshot {snapshot.snapshot_id} "
                f"({snapshot.anchor_date}, {snapshot.period_kind.value}) "
                f"-> {path}"
            )
            print(f"counts: {_format_counts(snapshot)}")
        return 0
    finally:
        await session.close()
        await engine.dispose()


def _run_history(command: str, args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        period_kind = MonitorPeriodKind(args.period)
    except ValueError:
        print(
            f"error: period must be one of {[p.value for p in MonitorPeriodKind]}",
            file=sys.stderr,
        )
        return 2
    service = MonitoringService(
        history=MonitorHistory(settings.MONITOR_HISTORY_DIR)
    )
    try:
        if command == "trends":
            data = service.trends(
                period_kind=period_kind,
                metric_keys=args.metric or None,
            )
        elif command == "drift":
            data = service.drift(
                before_id=args.before,
                after_id=args.after,
                period_kind=period_kind,
            )
        else:  # pragma: no cover - guarded by argparse
            raise AssertionError(f"unsupported history command: {command}")
        _emit(data, args.json)
        return 0
    except LookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _format_counts(snapshot) -> str:
    counts = dict(snapshot.counts)
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _emit(payload, json_out: bool) -> None:
    if json_out:
        print(payload.model_dump_json(indent=2))
        return
    if hasattr(payload, "entries"):
        _emit_health_rows(payload)
        return
    if hasattr(payload, "recommendations"):
        rows = payload.recommendations
        if not rows:
            print("no recommendations")
            return
        for item in rows:
            print(
                f"{item.forecast_id} company={item.company_id} "
                f"reasons={','.join(item.reasons)}"
            )
        return
    if hasattr(payload, "trends"):
        for trend in payload.trends:
            span = ", ".join(
                f"{point.anchor}:{point.value}" for point in trend.series
            )
            print(
                f"{trend.metric:<24} direction={trend.direction.value:<8} "
                f"from={trend.from_value} to={trend.to_value} "
                f"points=[{span}]"
            )
        return
    if hasattr(payload, "signals"):
        print(
            f"baseline={payload.baseline_id} comparison={payload.comparison_id} "
            f"periods={payload.baseline_period}->{payload.comparison_period}"
        )
        for signal in payload.signals:
            print(
                f"{signal.signal:<24} {signal.severity:<8} "
                f"delta={signal.delta} magnitude={signal.magnitude}"
            )
        return
    if hasattr(payload, "counts"):
        print(_format_counts(payload))
        for key, value in payload.metrics.items():
            if value is not None:
                print(f"{key}={value}")
        return
    print(payload.model_dump_json(indent=2))


def _emit_health_rows(payload) -> None:
    rows = payload.entries
    if not rows:
        print("no rows")
    for entry in rows:
        verdict = entry.evaluation_verdict or "-"
        outcome = entry.outcome_id or "-"
        due = entry.due_at.date().isoformat()
        print(
            f"{entry.health.value:<8} {entry.forecast_id} "
            f"company={entry.company_id} decision={entry.decision} "
            f"conf={entry.confidence:.2f} status={entry.status} "
            f"due={due} verdict={verdict} outcome={outcome}"
        )
    distribution = getattr(payload, "distribution", None)
    if distribution:
        rendered = ", ".join(
            f"{key}={value}" for key, value in sorted(distribution.items())
        )
        print(f"distribution: {rendered}")


async def _main_async(args: argparse.Namespace) -> int:
    as_of = _parse_as_of(getattr(args, "as_of", None))
    if args.command in ("summary", "rolling", "health", "stale", "overdue", "reanalysis"):
        return await _run_live(
            args.command,
            as_of=as_of,
            window=getattr(args, "window", None),
            json_out=args.json,
        )
    if args.command == "snapshot":
        return await _run_snapshot(
            as_of=as_of,
            window=getattr(args, "window", None),
            keep=args.keep,
            json_out=args.json,
        )
    return _run_history(args.command, args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predictron-monitor",
        description="Continuous Intelligence & Drift Detection (Phase 6).",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON payloads")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("summary", "rolling", "health", "stale", "overdue", "reanalysis"):
        p = sub.add_parser(name)
        p.add_argument("--as-of", default=None, help="ISO-8601 analysis moment")
        p.add_argument("--window", type=int, default=None, help="rolling window size")
    snap = sub.add_parser("snapshot")
    snap.add_argument("--as-of", default=None, help="ISO-8601 snapshot moment")
    snap.add_argument("--window", type=int, default=None)
    snap.add_argument(
        "--keep", type=int, default=None, help="file rotation cap per period"
    )

    trends = sub.add_parser("trends")
    trends.add_argument("--period", default="daily")
    trends.add_argument(
        "--metric", action="append", default=None, help="metric map key (repeatable)"
    )

    drift = sub.add_parser("drift")
    drift.add_argument("--period", default="daily")
    drift.add_argument("--before", default=None)
    drift.add_argument("--after", default=None)
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
