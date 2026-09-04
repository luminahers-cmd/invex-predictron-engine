"""Tests for the acquisition scheduler."""

from __future__ import annotations

from pathlib import Path

import pytest

from predictron_engine.dataset.acquisition.scheduler import (
    AcquisitionScheduler,
    ImportSchedule,
)


@pytest.fixture()
def scheduler(tmp_path: Path) -> AcquisitionScheduler:
    sched = AcquisitionScheduler(tmp_path / "dataset")
    sched.initialize()
    return sched


class TestImportSchedule:
    def test_defaults(self) -> None:
        s = ImportSchedule(source_name="yc_oss")
        assert s.frequency == "weekly"
        assert s.enabled is True
        assert s.priority == 0


class TestAcquisitionScheduler:
    def test_add_and_list(self, scheduler: AcquisitionScheduler) -> None:
        scheduler.add_schedule(ImportSchedule(source_name="yc_oss", frequency="daily"))
        scheduler.add_schedule(ImportSchedule(source_name="sec_edgar", frequency="weekly"))

        schedules = scheduler.load_schedules()
        assert len(schedules) == 2

    def test_add_updates_existing(self, scheduler: AcquisitionScheduler) -> None:
        scheduler.add_schedule(ImportSchedule(source_name="yc_oss", frequency="daily"))
        scheduler.add_schedule(ImportSchedule(source_name="yc_oss", frequency="monthly"))

        schedules = scheduler.load_schedules()
        assert len(schedules) == 1
        assert schedules[0].frequency == "monthly"

    def test_remove(self, scheduler: AcquisitionScheduler) -> None:
        scheduler.add_schedule(ImportSchedule(source_name="yc_oss"))
        assert scheduler.remove_schedule("yc_oss") is True
        assert len(scheduler.load_schedules()) == 0

    def test_remove_not_found(self, scheduler: AcquisitionScheduler) -> None:
        assert scheduler.remove_schedule("nonexistent") is False

    def test_get_schedule(self, scheduler: AcquisitionScheduler) -> None:
        scheduler.add_schedule(
            ImportSchedule(source_name="yc_oss", frequency="quarterly")
        )
        s = scheduler.get_schedule("yc_oss")
        assert s is not None
        assert s.frequency == "quarterly"

    def test_get_due_imports(self, scheduler: AcquisitionScheduler) -> None:
        scheduler.add_schedule(ImportSchedule(source_name="yc_oss"))
        due = scheduler.get_due_imports()
        assert len(due) == 1

    def test_get_due_imports_disabled(self, scheduler: AcquisitionScheduler) -> None:
        scheduler.add_schedule(
            ImportSchedule(source_name="yc_oss", enabled=False)
        )
        due = scheduler.get_due_imports()
        assert len(due) == 0

    def test_mark_run(self, scheduler: AcquisitionScheduler) -> None:
        scheduler.add_schedule(ImportSchedule(source_name="yc_oss"))
        scheduler.mark_run("yc_oss")
        s = scheduler.get_schedule("yc_oss")
        assert s is not None
        assert s.last_run != ""
        assert s.next_run != ""
