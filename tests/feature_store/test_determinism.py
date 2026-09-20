"""Determinism and reproducibility tests for feature computation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from predictron_engine.dataset.signals.model import SignalType
from tests.feature_store.conftest import (
    FakeRecord,
    make_signal,
    make_timeline,
)

AS_OF = datetime(2024, 1, 1, tzinfo=UTC)


class TestDeterminism:
    def test_same_input_same_output(self, engine, fake_record):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=0, amount=1_000_000),
            make_signal(SignalType.ARR_MILESTONE, days_offset=-30),
        ])
        kwargs = dict(timeline=tl, as_of=AS_OF)
        a = engine.build_company_features(fake_record, **kwargs)
        b = engine.build_company_features(fake_record, **kwargs)
        for fid in a.features:
            assert a.features[fid].value == b.features[fid].value, fid

    def test_different_input_different_output(self, engine):
        r1 = FakeRecord(record_id="r1")
        r2 = FakeRecord(record_id="r2")
        r2.profile.founded_year = 2010
        a = engine.build_company_features(r1, as_of=AS_OF)
        b = engine.build_company_features(r2, as_of=AS_OF)
        assert a.features["company_age"].value != b.features["company_age"].value

    def test_order_independent(self, engine, fake_record):
        signals_a = [
            make_signal(SignalType.FUNDING_ROUND, days_offset=0, amount=100),
            make_signal(SignalType.ARR_MILESTONE, days_offset=-10),
        ]
        signals_b = list(reversed(signals_a))
        fs_a = engine.build_company_features(
            fake_record, timeline=make_timeline(signals=signals_a), as_of=AS_OF,
        )
        fs_b = engine.build_company_features(
            fake_record, timeline=make_timeline(signals=signals_b), as_of=AS_OF,
        )
        for fid in fs_a.features:
            assert fs_a.features[fid].value == fs_b.features[fid].value, fid

    def test_rebuild_idempotent_over_time(self, engine, fake_record):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=0, amount=500_000),
        ])
        as_of = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        runs = []
        for _ in range(5):
            fs = engine.build_company_features(fake_record, timeline=tl, as_of=as_of)
            runs.append(
                {k: v.value for k, v in fs.features.items()}
            )
        for r in runs[1:]:
            assert r == runs[0]

    def test_timeline_signal_order_independent(self, engine, fake_record):
        signals = [
            make_signal(SignalType.FUNDING_ROUND, days_offset=0, amount=1_000_000),
            make_signal(SignalType.FUNDING_ROUND, days_offset=60, amount=500_000),
            make_signal(SignalType.ARR_MILESTONE, days_offset=-30),
        ]
        import random
        random.seed(42)
        shuffled = signals.copy()
        random.shuffle(shuffled)

        fs_orig = engine.build_company_features(
            fake_record,
            timeline=make_timeline(signals=signals),
            as_of=AS_OF,
        )
        fs_shuf = engine.build_company_features(
            fake_record,
            timeline=make_timeline(signals=shuffled),
            as_of=AS_OF,
        )
        for fid in fs_orig.features:
            assert fs_orig.features[fid].value == fs_shuf.features[fid].value, fid

    def test_deterministic_json_output(self, engine, fake_record):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=0),
        ])
        kwargs = dict(timeline=tl, as_of=AS_OF)
        a = engine.build_company_features(fake_record, **kwargs)
        b = engine.build_company_features(fake_record, **kwargs)

        def canonical(fs):
            d = fs.model_dump()
            for s in d["features"].values():
                s.pop("snapshot_id", None)
                s.pop("computed_at", None)
            return d

        assert canonical(a) == canonical(b)

    def test_frozen_snapshot_ids_unique(self, engine, fake_record):
        a = engine.build_company_features(fake_record, as_of=AS_OF)
        b = engine.build_company_features(fake_record, as_of=AS_OF)
        ids_a = set(s.snapshot_id for s in a.features.values())
        ids_b = set(s.snapshot_id for s in b.features.values())
        assert ids_a != ids_b

    def test_value_none_consistent(self, engine):
        r = FakeRecord()
        tl = make_timeline(signals=[])
        values = []
        for _ in range(3):
            fs = engine.build_company_features(
                r, timeline=tl, as_of=AS_OF,
            )
            values.append(
                {k: v.value for k, v in fs.features.items() if v.value is None}
            )
        for v in values[1:]:
            assert v == values[0]

    def test_computed_at_consistent_for_same_as_of(self, engine, fake_record):
        a = engine.build_company_features(fake_record, as_of=AS_OF)
        b = engine.build_company_features(fake_record, as_of=AS_OF)
        for fid in a.features:
            assert (
                a.features[fid].computed_at == b.features[fid].computed_at
            ), fid

    def test_snapshot_json_canonical(self, engine, full_store, fake_record):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=0),
        ])
        a = engine.build_company_features(fake_record, timeline=tl, as_of=AS_OF)
        b = engine.build_company_features(fake_record, timeline=tl, as_of=AS_OF)
        full_store.save_company_features(a)
        full_store.save_company_features(b)
        loaded = full_store.load_company_features("rec-1")
        assert loaded is not None
        original_value = a.features["total_funding"].value
        assert loaded.features["total_funding"].value == original_value

    def test_multiple_runs_same_digest(self, engine, fake_record):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=0, amount=100),
        ])
        digests = []
        for _ in range(5):
            fs = engine.build_company_features(fake_record, timeline=tl, as_of=AS_OF)
            d = fs.model_dump()
            for s in d["features"].values():
                s.pop("snapshot_id", None)
                s.pop("computed_at", None)
            digests.append(
                hashlib.sha256(
                    json.dumps(d, sort_keys=True, default=str).encode()
                ).hexdigest()
            )
        assert len(set(digests)) == 1
