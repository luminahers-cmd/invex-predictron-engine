"""Signal helper CLI commands (Project E4).

Five new subcommands for the ``predictron-dataset`` entry point:

  signal-import     Import company signals from a JSON file
  signal-report     Emit a full company-signals report document
  timeline          Show per-company timeline(s)
  trend-report      Emit a deterministic trend report
  signal-validate   Validate stored signal timelines

These are fully additive — no existing command is modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from predictron_engine.dataset.store import DatasetStore

_YES = 1


def cmd_signal_import(args: argparse.Namespace) -> int:
    """Import company signals from a JSON file."""
    from predictron_engine.dataset.signals.integration import SignalDatasetManager

    store = DatasetStore(args.dataset)
    store.initialize()
    manager = SignalDatasetManager(store)
    result = manager.import_file(args.file)
    _json_out(result.to_dict())
    return _YES if result.report.rejected > 0 else 0


def cmd_signal_report(args: argparse.Namespace) -> int:
    """Emit a full company-signals report document."""
    from predictron_engine.dataset.signals.integration import SignalDatasetManager

    store = DatasetStore(args.dataset)
    store.initialize()
    manager = SignalDatasetManager(store)
    as_of = _parse_as_of(getattr(args, "as_of", None))
    data = manager.report(as_of=as_of)
    output = getattr(args, "output", None)
    if output:
        _write_json(str(output), data)
    else:
        _json_out(data)
    return 0


def cmd_timeline(args: argparse.Namespace) -> int:
    """Show per-company timeline(s)."""
    from predictron_engine.dataset.signals.integration import SignalDatasetManager
    from predictron_engine.dataset.signals.reports import build_timeline_report

    store = DatasetStore(args.dataset)
    store.initialize()
    manager = SignalDatasetManager(store)
    company_id = getattr(args, "company", None)
    if company_id:
        resolved = manager.company_id_for(company_id) or company_id
        timeline = manager.timeline(resolved)
        if timeline is None:
            print(
                f"error: no signals for company '{company_id}' "
                f"(resolved: '{resolved}')",
                file=sys.stderr,
            )
            return 1
        data = build_timeline_report(timeline)
    else:
        timelines = manager.all_timelines()
        companies = [
            {
                "company_id": t.company_id,
                "signal_count": t.signal_count,
                "first": t.first.timestamp.isoformat() if t.first else None,
                "last": t.last.timestamp.isoformat() if t.last else None,
            }
            for t in timelines
        ]
        data = {"company_count": len(companies), "companies": companies}
    _json_out(data)
    return 0


def cmd_trend_report(args: argparse.Namespace) -> int:
    """Emit a deterministic trend report."""
    from predictron_engine.dataset.signals.integration import SignalDatasetManager
    from predictron_engine.dataset.signals.trends import TrendEngine

    store = DatasetStore(args.dataset)
    store.initialize()
    manager = SignalDatasetManager(store)
    as_of = _parse_as_of(getattr(args, "as_of", None))
    company_id = getattr(args, "company", None)
    if company_id:
        resolved = manager.company_id_for(company_id) or company_id
        try:
            data: object = {
                "company_id": resolved,
                "trends": manager.trends(resolved, as_of=as_of),
            }
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    else:
        engine = TrendEngine(as_of)
        payload: dict[str, object] = {}
        for timeline in manager.all_timelines():
            payload[timeline.company_id] = engine.summarize(timeline)
        data = payload
    output = getattr(args, "output", None)
    if output:
        _write_json(str(output), data)
    else:
        _json_out(data)
    return 0


def cmd_signal_validate(args: argparse.Namespace) -> int:
    """Validate stored signal timelines."""
    from predictron_engine.dataset.signals.integration import SignalDatasetManager

    store = DatasetStore(args.dataset)
    store.initialize()
    manager = SignalDatasetManager(store)
    report = manager.validate()
    _json_out(report.to_dict())
    return 0 if report.is_valid else 1


# ── helpers ──────────────────────────────────────────────────────────


def _json_out(data: object) -> None:
    print(json.dumps(data, indent=2, default=str))


def _write_json(path: str, data: object) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8"
    )
    print(file_path)


def _parse_as_of(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).astimezone(UTC)
    except (ValueError, TypeError):
        return None


def add_signal_subparsers(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the five signal CLI subcommands on an `add_subparsers` action."""

    # signal-import
    p = sub.add_parser(
        "signal-import", help="Import company signals from a JSON file"
    )
    p.add_argument("file", type=str, help="Path to the signal JSON file")
    _add_ds(p)
    p.set_defaults(func=cmd_signal_import)

    # signal-report
    p = sub.add_parser("signal-report", help="Emit a full company-signals report")
    _add_ds(p)
    p.add_argument("--as-of", type=str, default=None, help="ISO-8601 reference time")
    p.add_argument("--output", type=str, default=None, help="Write JSON to file")
    p.set_defaults(func=cmd_signal_report)

    # timeline
    p = sub.add_parser("timeline", help="Show company signal timeline(s)")
    _add_ds(p)
    p.add_argument("--company", type=str, default=None, help="Company id or name")
    p.set_defaults(func=cmd_timeline)

    # trend-report
    p = sub.add_parser("trend-report", help="Emit a deterministic trend report")
    _add_ds(p)
    p.add_argument("--company", type=str, default=None, help="Company id or name")
    p.add_argument("--as-of", type=str, default=None, help="ISO-8601 reference time")
    p.add_argument("--output", type=str, default=None, help="Write JSON to file")
    p.set_defaults(func=cmd_trend_report)

    # signal-validate
    p = sub.add_parser("signal-validate", help="Validate stored signal timelines")
    _add_ds(p)
    p.set_defaults(func=cmd_signal_validate)


def _add_ds(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset store root directory",
    )
