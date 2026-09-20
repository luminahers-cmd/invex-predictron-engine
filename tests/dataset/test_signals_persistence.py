"""Tests for predictron_engine.dataset.signals.persistence — Project E4."""

from __future__ import annotations

import json

import pytest

from predictron_engine.dataset.signals.persistence import (
    SIGNALS_SCHEMA_VERSION,
    _decode_company_id,
    _encode_company_id,
    list_signal_company_ids,
    read_timeline,
    timeline_from_dict,
    timeline_to_dict,
    write_timeline,
)
from predictron_engine.dataset.signals.timeline import CompanyTimeline
from tests.dataset.signals_helpers import make_signal, make_timeline


class TestCompanyIdEncoding:
    def test_plain_id_unchanged_round_trip(self) -> None:
        assert _decode_company_id(_encode_company_id("company:a")) == "company:a"

    def test_encodes_colon_and_plus(self) -> None:
        encoded = _encode_company_id("company:a+b/c")
        assert ":" not in encoded
        assert "+" not in encoded
        assert "/" not in encoded

    def test_round_trip_special_chars(self) -> None:
        company_id = "company:a+b/c: d@e"
        assert _decode_company_id(_encode_company_id(company_id)) == company_id

    def test_unicode_round_trip(self) -> None:
        company_id = "company:café-ünïcode"
        assert _decode_company_id(_encode_company_id(company_id)) == company_id

    def test_empty_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _encode_company_id("")


class TestTimelineSerialization:
    def test_to_dict_structure(self) -> None:
        tl = make_timeline("company:a", make_signal())
        d = timeline_to_dict(tl)
        assert d["schema_version"] == SIGNALS_SCHEMA_VERSION
        assert d["company_id"] == "company:a"
        assert d["signal_count"] == 1
        assert len(d["signals"]) == 1

    def test_round_trip_preserves_timeline(self) -> None:
        tl = make_timeline(
            "company:a",
            make_signal(at="2024-06-01T00:00:00+00:00"),
            make_signal(at="2024-06-02T00:00:00+00:00", signal_type="ipo"),
        )
        restored = timeline_from_dict(timeline_to_dict(tl))
        assert restored.company_id == tl.company_id
        assert restored.signal_count == tl.signal_count
        assert [s.signal_id for s in restored.signals] == [
            s.signal_id for s in tl.signals
        ]

    def test_from_dict_rejects_non_list_signals(self) -> None:
        with pytest.raises(ValueError, match="must be a list"):
            timeline_from_dict({"company_id": "c", "signals": "nope"})

    def test_from_dict_rejects_non_object_signal(self) -> None:
        with pytest.raises(ValueError, match="non-object signal"):
            timeline_from_dict({"company_id": "c", "signals": [42]})


class TestWriteRead:
    def test_write_creates_signals_dir(self, tmp_path) -> None:
        tl = make_timeline("company:a", make_signal())
        path = write_timeline(tmp_path, tl)
        assert path.parent.name == "signals"
        assert path.name.endswith(".json")

    def test_read_missing_returns_none(self, tmp_path) -> None:
        assert read_timeline(tmp_path, "company:nope") is None

    def test_round_trip(self, tmp_path) -> None:
        tl = make_timeline("company:a", make_signal())
        write_timeline(tmp_path, tl)
        restored = read_timeline(tmp_path, "company:a")
        assert restored is not None
        assert restored.company_id == "company:a"
        assert restored.signal_count == 1

    def test_read_special_company_id(self, tmp_path) -> None:
        tl = make_timeline("company:a+b/c", make_signal())
        write_timeline(tmp_path, tl)
        restored = read_timeline(tmp_path, "company:a+b/c")
        assert restored is not None
        assert restored.company_id == "company:a+b/c"

    def test_write_deterministic_bytes(self, tmp_path) -> None:
        tl = make_timeline("company:a", make_signal())
        path1 = write_timeline(tmp_path, tl)
        path2 = write_timeline(tmp_path, tl)
        assert path1.read_bytes() == path2.read_bytes()

    def test_written_json_is_sorted(self, tmp_path) -> None:
        tl = make_timeline("company:a", make_signal())
        path = write_timeline(tmp_path, tl)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["schema_version"] == SIGNALS_SCHEMA_VERSION

    def test_read_faulty_file_raises(self, tmp_path) -> None:
        from predictron_engine.dataset.signals.persistence import (
            _encode_company_id,
            _signals_dir,
        )

        directory = _signals_dir(tmp_path)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{_encode_company_id('company:a')}.json").write_text(
            "[1,2,3]", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="not an object"):
            read_timeline(tmp_path, "company:a")

    def test_write_overwrites(self, tmp_path) -> None:
        old = make_timeline("company:a", make_signal())
        new = make_timeline("company:a", make_signal(at="2024-07-01T00:00:00+00:00"))
        write_timeline(tmp_path, old)
        write_timeline(tmp_path, new)
        restored = read_timeline(tmp_path, "company:a")
        assert restored is not None
        assert restored.signal_count == new.signal_count


class TestListSignalCompanyIds:
    def test_empty_when_dir_missing(self, tmp_path) -> None:
        assert list_signal_company_ids(tmp_path) == []

    def test_lists_sorted_ids(self, tmp_path) -> None:
        write_timeline(tmp_path, make_timeline("company:b", make_signal(company_id="company:b")))
        write_timeline(tmp_path, make_timeline("company:a", make_signal()))
        ids = list_signal_company_ids(tmp_path)
        assert ids == ["company:a", "company:b"]

    def test_one_per_company(self, tmp_path) -> None:
        write_timeline(tmp_path, make_timeline("company:a", make_signal()))
        write_timeline(tmp_path, make_timeline("company:b", make_signal(company_id="company:b")))
        assert len(list_signal_company_ids(tmp_path)) == 2


class TestCompanyTimelineProxy:
    def test_timeline_to_dict_via_build(self) -> None:
        tl = CompanyTimeline.build("company:a", [make_signal()])
        d = tl.to_dict()
        assert d["company_id"] == "company:a"

    def test_timeline_from_dict_via_classmethod(self) -> None:
        tl = CompanyTimeline.build("company:a", [make_signal()])
        restored = CompanyTimeline.from_dict(tl.to_dict())
        assert restored == tl
