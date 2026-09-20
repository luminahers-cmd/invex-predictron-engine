"""Tests for predictron_engine.dataset.signals.timeline — Project E4."""

from __future__ import annotations

import pytest

from predictron_engine.dataset.signals.model import SignalType
from predictron_engine.dataset.signals.timeline import (
    CompanyTimeline,
    TimelineStore,
    sort_and_dedupe,
)
from tests.dataset.signals_helpers import make_signal, make_timeline


class TestSortAndDedupe:
    def test_sorts_by_timestamp(self) -> None:
        s1 = make_signal(at="2024-06-02T00:00:00+00:00")
        s2 = make_signal(at="2024-06-01T00:00:00+00:00")
        result = sort_and_dedupe([s1, s2])
        assert result[0].timestamp < result[1].timestamp

    def test_deduplicates_by_signal_id(self) -> None:
        s = make_signal(at="2024-06-01T00:00:00+00:00")
        result = sort_and_dedupe([s, s, s])
        assert len(result) == 1

    def test_preserves_first_occurrence(self) -> None:
        s = make_signal(at="2024-06-01T00:00:00+00:00")
        result = sort_and_dedupe([s, s])
        assert result[0].signal_id == s.signal_id

    def test_empty_input(self) -> None:
        assert sort_and_dedupe([]) == []


class TestCompanyTimeline:
    def test_build_sorts_signals(self) -> None:
        s1 = make_signal(at="2024-06-02T00:00:00+00:00")
        s2 = make_signal(at="2024-06-01T00:00:00+00:00")
        tl = CompanyTimeline.build("company:a", [s1, s2])
        assert tl.signals[0].timestamp < tl.signals[1].timestamp

    def test_build_deduplicates(self) -> None:
        s = make_signal(at="2024-06-01T00:00:00+00:00")
        tl = CompanyTimeline.build("company:a", [s, s])
        assert tl.signal_count == 1

    def test_empty_timeline(self) -> None:
        tl = CompanyTimeline.build("company:a", [])
        assert tl.signal_count == 0
        assert tl.first is None
        assert tl.last is None
        assert tl.span_days == 0.0

    def test_frozen(self) -> None:
        tl = make_timeline("company:a", make_signal())
        with pytest.raises(AttributeError):
            tl.company_id = "company:b"  # type: ignore[misc]

    def test_append_returns_new(self) -> None:
        tl1 = make_timeline(
            "company:a", make_signal(at="2024-06-01T00:00:00+00:00")
        )
        tl2 = tl1.append(make_signal(at="2024-06-02T00:00:00+00:00"))
        assert tl1.signal_count == 1
        assert tl2.signal_count == 2

    def test_append_rejects_different_company(self) -> None:
        tl = make_timeline("company:a", make_signal())
        with pytest.raises(ValueError, match="cannot append"):
            tl.append(make_signal(company_id="company:b"))

    def test_merge_combines(self) -> None:
        tl1 = make_timeline(
            "company:a", make_signal(at="2024-06-01T00:00:00+00:00")
        )
        tl2 = make_timeline(
            "company:a", make_signal(at="2024-06-02T00:00:00+00:00")
        )
        merged = tl1.merge(tl2)
        assert merged.signal_count == 2

    def test_merge_deduplicates(self) -> None:
        s = make_signal(at="2024-06-01T00:00:00+00:00")
        tl1 = make_timeline("company:a", s)
        tl2 = make_timeline("company:a", s)
        merged = tl1.merge(tl2)
        assert merged.signal_count == 1

    def test_merge_rejects_different_company(self) -> None:
        tl1 = make_timeline("company:a", make_signal())
        tl2 = make_timeline("company:b", make_signal(company_id="company:b"))
        with pytest.raises(ValueError, match="cannot merge"):
            tl1.merge(tl2)

    def test_signal_count(self) -> None:
        tl = make_timeline(
            "company:a",
            make_signal(at="2024-06-01T00:00:00+00:00"),
            make_signal(at="2024-06-02T00:00:00+00:00"),
        )
        assert tl.signal_count == 2

    def test_span_days(self) -> None:
        tl = make_timeline(
            "company:a",
            make_signal(at="2024-01-01T00:00:00+00:00"),
            make_signal(at="2024-01-31T00:00:00+00:00"),
        )
        assert abs(tl.span_days - 30.0) < 0.01

    def test_has_and_find(self) -> None:
        s = make_signal(at="2024-06-01T00:00:00+00:00")
        tl = make_timeline("company:a", s)
        assert tl.has(s.signal_id)
        assert tl.find(s.signal_id) is s
        assert tl.find("nonexistent") is None

    def test_types(self) -> None:
        tl = make_timeline(
            "company:a",
            make_signal(
                at="2024-06-01T00:00:00+00:00",
                signal_type=SignalType.FUNDING_ROUND,
            ),
            make_signal(at="2024-06-02T00:00:00+00:00", signal_type=SignalType.IPO),
        )
        counts = tl.types()
        assert counts["funding_round"] == 1
        assert counts["ipo"] == 1

    def test_by_type(self) -> None:
        tl = make_timeline(
            "company:a",
            make_signal(
                at="2024-06-01T00:00:00+00:00",
                signal_type=SignalType.FUNDING_ROUND,
            ),
            make_signal(at="2024-06-02T00:00:00+00:00", signal_type=SignalType.IPO),
            make_signal(
                at="2024-06-03T00:00:00+00:00",
                signal_type=SignalType.FUNDING_ROUND,
            ),
        )
        funding = tl.by_type("funding_round")
        assert len(funding) == 2

    def test_signals_since(self) -> None:
        from tests.dataset.signals_helpers import dt

        tl = make_timeline(
            "company:a",
            make_signal(at="2024-01-01T00:00:00+00:00"),
            make_signal(at="2024-06-01T00:00:00+00:00"),
            make_signal(at="2024-12-01T00:00:00+00:00"),
        )
        since_mid = tl.signals_since(dt("2024-06-01T00:00:00+00:00"))
        assert len(since_mid) == 2


class TestTimelineStore:
    def test_put_and_get(self) -> None:
        store = TimelineStore()
        tl = store.put("company:a", [make_signal()])
        assert store.get("company:a") is tl
        assert store.count() == 1

    def test_get_missing_returns_none(self) -> None:
        store = TimelineStore()
        assert store.get("company:missing") is None

    def test_remove(self) -> None:
        store = TimelineStore()
        store.put("company:a", [make_signal()])
        assert store.remove("company:a") is True
        assert store.get("company:a") is None
        assert store.remove("company:a") is False

    def test_list_companies_sorted(self) -> None:
        store = TimelineStore()
        store.put("company:b", [make_signal(company_id="company:b")])
        store.put("company:a", [make_signal()])
        assert store.list_companies() == ["company:a", "company:b"]

    def test_add_replaces(self) -> None:
        store = TimelineStore()
        tl1 = make_timeline(
            "company:a", make_signal(at="2024-06-01T00:00:00+00:00")
        )
        tl2 = make_timeline(
            "company:a", make_signal(at="2024-06-02T00:00:00+00:00")
        )
        store.add(tl1)
        store.add(tl2)
        assert store.get("company:a").signal_count == 1

    def test_total_signals(self) -> None:
        store = TimelineStore()
        store.put("company:a", [
            make_signal(at="2024-06-01T00:00:00+00:00"),
            make_signal(at="2024-06-02T00:00:00+00:00"),
        ])
        store.put("company:b", [make_signal(company_id="company:b")])
        assert store.total_signals() == 3

    def test_all_sorted(self) -> None:
        store = TimelineStore()
        store.put("company:b", [make_signal(company_id="company:b")])
        store.put("company:a", [make_signal()])
        all_tls = store.all()
        assert [t.company_id for t in all_tls] == ["company:a", "company:b"]
