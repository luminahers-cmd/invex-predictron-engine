"""Tests for the append-only, hash-verified benchmark history store."""

from __future__ import annotations

import json

import pytest

from benchmarks.ground_truth_eval.history import (
    BenchmarkHistory,
    HistoryError,
)


class TestAppend:
    def test_append_and_load(self, make_run, tmp_path) -> None:
        history = BenchmarkHistory(tmp_path)
        run = make_run(
            [
                {"company_id": "a", "overall_score": 60.0},
                {"company_id": "b", "overall_score": 40.0},
            ],
            run_id="hist_1",
        )
        path = history.append(run)
        assert path.exists()
        loaded = history.load("hist_1")
        assert loaded.result_hash == run.result_hash
        assert len(loaded.entries) == 2

    def test_refuses_duplicate(self, make_run, tmp_path) -> None:
        history = BenchmarkHistory(tmp_path)
        run = make_run([{"company_id": "a"}], run_id="dup")
        history.append(run)
        with pytest.raises(HistoryError, match="already exists"):
            history.append(run)

    def test_overwrite_optional(self, make_run, tmp_path) -> None:
        history = BenchmarkHistory(tmp_path)
        run = make_run([{"company_id": "a"}], run_id="ow")
        history.append(run)
        history.append(run, overwrite=True)
        assert history.has_run("ow")
        assert history.count == 1

    def test_metrics_stored(self, make_run, tmp_path) -> None:
        history = BenchmarkHistory(tmp_path)
        run = make_run([{"company_id": "a"}], run_id="met")
        history.append(run, metrics={"accuracy": 0.5})
        assert history.load_metrics("met") == {"accuracy": 0.5}

    def test_no_metrics(self, make_run, tmp_path) -> None:
        history = BenchmarkHistory(tmp_path)
        run = make_run([{"company_id": "a"}], run_id="nomet")
        history.append(run)
        assert history.load_metrics("nomet") is None

    def test_summary_has_metrics_flag(self, make_run, tmp_path) -> None:
        history = BenchmarkHistory(tmp_path)
        run = make_run([{"company_id": "a"}], run_id="flag")
        history.append(run, metrics={"accuracy": 0.5})
        s = history.list_summaries()[0]
        assert s.has_metrics is True

    def test_declared_hash_mismatch_rejected(self, make_run, tmp_path) -> None:
        history = BenchmarkHistory(tmp_path)
        run = make_run([{"company_id": "a"}], run_id="badhash")
        run.result_hash = "0" * 64
        with pytest.raises(HistoryError, match="result_hash"):
            history.append(run)


class TestRunPath:
    def test_safe_ids(self, tmp_path) -> None:
        history = BenchmarkHistory(tmp_path)
        for rid in ("abc", "run_0.1-t2", "A.B-C_1"):
            assert history.run_path(rid).name == f"{rid}.json"

    @pytest.mark.parametrize("rid", ["../evil", "a/b", "a\\b", "run/../x"])
    def test_unsafe_ids_rejected(self, rid: str, tmp_path) -> None:
        history = BenchmarkHistory(tmp_path)
        with pytest.raises(HistoryError, match="filename-safe"):
            history.run_path(rid)


class TestLoad:
    def test_missing_run(self, tmp_path) -> None:
        history = BenchmarkHistory(tmp_path)
        with pytest.raises(HistoryError, match="not found"):
            history.load("missing")

    def test_corrupt_json(self, make_run, tmp_path) -> None:
        history = BenchmarkHistory(tmp_path)
        run = make_run([{"company_id": "a"}], run_id="corrupt")
        history.append(run)
        history.run_path("corrupt").write_text("{not json", encoding="utf-8")
        with pytest.raises(HistoryError, match="corrupt"):
            history.load("corrupt")

    def test_wrong_kind(self, make_run, tmp_path) -> None:
        history = BenchmarkHistory(tmp_path)
        run = make_run([{"company_id": "a"}], run_id="wrongkind")
        history.append(run)
        doc = json.loads(history.run_path("wrongkind").read_text(encoding="utf-8"))
        doc["kind"] = "something_else"
        history.run_path("wrongkind").write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(HistoryError, match="kind"):
            history.load("wrongkind")

    def test_tampered_score_detected(self, make_run, tmp_path) -> None:
        history = BenchmarkHistory(tmp_path)
        run = make_run([{"company_id": "a", "overall_score": 60.0}], run_id="tamper")
        history.append(run)
        doc = json.loads(history.run_path("tamper").read_text(encoding="utf-8"))
        doc["run"]["entries"][0]["overall_score"] = 99.0
        history.run_path("tamper").write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(HistoryError, match="integrity"):
            history.load("tamper")

    def test_reordered_entries_still_valid(self, make_run, tmp_path) -> None:
        history = BenchmarkHistory(tmp_path)
        run = make_run(
            [
                {"company_id": "z", "overall_score": 60.0},
                {"company_id": "a", "overall_score": 40.0},
            ],
            run_id="order",
        )
        history.append(run)
        loaded = history.load("order")
        assert [e.company_id for e in loaded.entries] == ["a", "z"]


class TestIndex:
    def test_index_sorted(self, make_run, tmp_path) -> None:
        history = BenchmarkHistory(tmp_path)
        for i, rid in enumerate(("run_b", "run_a")):
            run = make_run([{"company_id": "a"}], run_id=rid, dataset_name=f"ds{i}")
            history.append(run)
        ids = history.list_run_ids()
        assert ids == sorted(ids)

    def test_filters_by_dataset(self, make_run, tmp_path) -> None:
        history = BenchmarkHistory(tmp_path)
        history.append(make_run([{"company_id": "a"}], run_id="ds_a", dataset_name="alpha"))
        history.append(make_run([{"company_id": "a"}], run_id="ds_b", dataset_name="beta"))
        assert history.list_run_ids(dataset_name="alpha") == ["ds_a"]

    def test_versions(self, make_run, tmp_path) -> None:
        history = BenchmarkHistory(tmp_path)
        history.append(make_run([{"company_id": "a"}], run_id="v1", benchmark_version="2.0"))
        history.append(make_run([{"company_id": "a"}], run_id="v0", benchmark_version="1.0"))
        assert history.list_benchmark_versions() == ["1.0", "2.0"]

    def test_count(self, make_run, tmp_path) -> None:
        history = BenchmarkHistory(tmp_path)
        assert history.count == 0
        history.append(make_run([{"company_id": "a"}], run_id="c1"))
        history.append(make_run([{"company_id": "a"}], run_id="c2"))
        assert history.count == 2

    def test_corrupt_index_ignored(self, make_run, tmp_path) -> None:
        history = BenchmarkHistory(tmp_path)
        history.append(make_run([{"company_id": "a"}], run_id="idx"))
        history.index_path().write_text("{bad", encoding="utf-8")
        assert history.list_summaries() == []


class TestVerifyIntegrity:
    def test_all_ok(self, make_run, tmp_path) -> None:
        history = BenchmarkHistory(tmp_path)
        history.append(make_run([{"company_id": "a"}], run_id="ok1"))
        history.append(make_run([{"company_id": "a"}], run_id="ok2"))
        assert history.verify_integrity() == {"ok1": "ok", "ok2": "ok"}

    def test_raises_on_corruption(self, make_run, tmp_path) -> None:
        history = BenchmarkHistory(tmp_path)
        history.append(make_run([{"company_id": "a"}], run_id="good"))
        history.append(make_run([{"company_id": "a", "overall_score": 5.0}], run_id="evil"))
        doc = json.loads(history.run_path("evil").read_text(encoding="utf-8"))
        doc["run"]["entries"][0]["overall_score"] = 10.0
        history.run_path("evil").write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(HistoryError, match="integrity"):
            history.verify_integrity()

    def test_runs_with_engine_versions(self, make_run, tmp_path) -> None:
        history = BenchmarkHistory(tmp_path)
        history.append(make_run([{"company_id": "a"}], run_id="e1", engine_version="0.9.0"))
        s = history.list_summaries()[0]
        assert s.engine_version == "0.9.0"
        assert s.result_hash


class TestRunSummary:
    def test_to_dict(self, make_run, tmp_path) -> None:
        history = BenchmarkHistory(tmp_path)
        run = make_run([{"company_id": "a"}], run_id="sum", dataset_name="ds")
        history.append(run, metrics={"accuracy": 1.0})
        d = history.list_summaries()[0].to_dict()
        assert d["run_id"] == "sum"
        assert d["entry_count"] == 1
        assert d["has_metrics"] is True
        assert "created_at" in d


class TestDeterminism:
    def test_stored_file_deterministic(self, make_run, tmp_path) -> None:
        history = BenchmarkHistory(tmp_path)
        run = make_run(
            [
                {"company_id": "b", "overall_score": 30.0},
                {"company_id": "a", "overall_score": 80.0},
            ],
            run_id="stable",
        )
        p1 = history.append(run)
        bytes1 = p1.read_bytes()
        history2 = BenchmarkHistory(tmp_path / "other")
        p2 = history2.append(run)
        assert p1.read_bytes() == bytes1 == p2.read_bytes()

    def test_append_twice_different_dir_same_bytes(self, make_run, tmp_path) -> None:
        run = make_run([{"company_id": "a"}], run_id="same_run")
        h1 = BenchmarkHistory(tmp_path / "h1")
        h2 = BenchmarkHistory(tmp_path / "h2")
        assert h1.append(run).read_bytes() == h2.append(run).read_bytes()
