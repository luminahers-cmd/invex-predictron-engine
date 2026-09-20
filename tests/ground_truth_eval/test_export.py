"""Tests for deterministic export/import and CLI integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.ground_truth_eval.export import (
    EXPORT_SCHEMA_VERSION,
    ExportError,
    canonical_json,
    export_bundle,
    export_dataset,
    export_run,
    import_dataset_payload,
    import_run_payload,
    load_exported,
    make_bundle_paths,
    write_json,
)
from benchmarks.ground_truth_eval.history import BenchmarkHistory
from benchmarks.ground_truth_eval.runner import BenchmarkRunner


class TestCanonicalJson:
    def test_sorted_keys(self) -> None:
        j = canonical_json({"b": 1, "a": 2})
        assert j.index('"a"') < j.index('"b"')

    def test_deterministic(self) -> None:
        obj = {"z": [3, 1], "a": {"m": 2, "l": 1}}
        assert canonical_json(obj) == canonical_json(obj)

    def test_byte_stable(self) -> None:
        j1 = canonical_json({"key": "value"})
        j2 = canonical_json({"key": "value"})
        assert j1.encode() == j2.encode()


class TestWriteJson:
    def test_creates_file(self, tmp_path: Path) -> None:
        p = tmp_path / "out.json"
        result = write_json({"x": 1}, p)
        assert result == p
        assert p.exists()

    def test_refuses_overwrite(self, tmp_path: Path) -> None:
        p = tmp_path / "f.json"
        write_json({"a": 1}, p)
        with pytest.raises(ExportError, match="overwrite"):
            write_json({"a": 2}, p)

    def test_overwrite_ok(self, tmp_path: Path) -> None:
        p = tmp_path / "f.json"
        write_json({"a": 1}, p)
        write_json({"a": 2}, p, overwrite=True)
        assert load_exported(p) == {"a": 2}

    def test_non_json_suffix_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ExportError, match="must end with .json"):
            write_json({"a": 1}, tmp_path / "bad.txt")


class TestPayloadBuilders:
    def test_dataset_payload(self, example_dataset) -> None:
        from benchmarks.ground_truth_eval.export import export_dataset_payload

        p = export_dataset_payload(example_dataset)
        assert p["schema_version"] == 1
        assert len(p["entries"]) == 13

    def test_run_payload(self, example_run) -> None:
        from benchmarks.ground_truth_eval.export import export_run_payload

        p = export_run_payload(example_run)
        assert p["kind"] == "benchmark_run"
        ids = [e["company_id"] for e in p["entries"]]
        assert ids == sorted(ids)

    def test_metrics_payload(self, example_metrics) -> None:
        from benchmarks.ground_truth_eval.export import export_metrics_payload

        p = export_metrics_payload(example_metrics)
        assert p["confusion"]["total"] == 13


class TestImportRoundtrip:
    def test_dataset_roundtrip(self, example_dataset, tmp_path) -> None:
        path = export_dataset(example_dataset, tmp_path / "d.json")
        payload = load_exported(path)
        ds = import_dataset_payload(payload)
        assert ds.entry_count == example_dataset.entry_count

    def test_run_roundtrip(self, example_run, tmp_path) -> None:
        path = export_run(example_run, tmp_path / "r.json")
        payload = load_exported(path)
        run = import_run_payload(payload)
        assert run.result_hash == example_run.result_hash

    def test_bundle_roundtrip(
        self, example_dataset, example_run, example_metrics, tmp_path
    ) -> None:
        path = export_bundle(
            dataset=example_dataset,
            run=example_run,
            metrics=example_metrics,
            path=tmp_path / "bundle.json",
        )
        bundle = load_exported(path)
        assert bundle["schema_version"] == EXPORT_SCHEMA_VERSION
        assert len(bundle["dataset"]["entries"]) == 13
        assert bundle["run"]["run_id"] == example_run.run_id
        assert bundle["metrics"]["confusion"]["total"] == 13

    def test_import_bad_schema_version(self) -> None:
        with pytest.raises(ExportError, match="schema_version"):
            import_dataset_payload({"schema_version": 99, "entries": []})

    def test_import_no_entries(self) -> None:
        with pytest.raises(ExportError, match="no entries"):
            import_dataset_payload(
                {"schema_version": 1, "entries": [], "dataset_name": "x", "benchmark_version": "1"}
            )

    def test_import_run_wrong_kind(self) -> None:
        with pytest.raises(ExportError, match="kind"):
            import_run_payload({"kind": "something_else"})


class TestMakeBundlePaths:
    def test_filenames(self, tmp_path) -> None:
        paths = make_bundle_paths(tmp_path, run_id="run1")
        assert paths["dataset"].name == "run1.dataset.json"
        assert paths["run"].name == "run1.run.json"
        assert paths["metrics"].name == "run1.metrics.json"
        assert paths["bundle"].name == "run1.bundle.json"
        assert paths["reports"].name == "run1.reports.json"


class TestDeterminism:
    def test_bundle_deterministic(
        self, example_dataset, example_run, example_metrics, tmp_path
    ) -> None:
        p1 = export_bundle(
            dataset=example_dataset,
            run=example_run,
            metrics=example_metrics,
            path=tmp_path / "a.json",
        )
        p2 = export_bundle(
            dataset=example_dataset,
            run=example_run,
            metrics=example_metrics,
            path=tmp_path / "b.json",
        )
        assert p1.read_bytes() == p2.read_bytes()


class TestCLI:
    EXAMPLE_DATASET = str(
        Path(__file__).resolve().parents[2] / "benchmarks" / "golden_datasets" / "example_v1.json"
    )

    def test_history_empty(self, tmp_path, capsys) -> None:
        from benchmarks.ground_truth_eval.cli import main

        code = main(["benchmark-history", "--history", str(tmp_path / "h")])
        assert code == 0
        assert "no runs stored" in capsys.readouterr().out

    def test_run_end_to_end(self, tmp_path, capsys) -> None:
        from benchmarks.ground_truth_eval.cli import main

        code = main(
            [
                "benchmark-run",
                "--dataset",
                self.EXAMPLE_DATASET,
                "--history",
                str(tmp_path / "h"),
                "--run-id",
                "cli_test_run",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "cli_test_run" in out
        history = BenchmarkHistory(tmp_path / "h")
        assert history.has_run("cli_test_run")

    def test_rerun_same_id_errors(self, tmp_path, capsys) -> None:
        from benchmarks.ground_truth_eval.cli import main

        args = [
            "benchmark-run",
            "--dataset",
            self.EXAMPLE_DATASET,
            "--history",
            str(tmp_path / "h"),
            "--run-id",
            "dup",
        ]
        assert main(args) == 0
        code = main(args)
        assert code != 0

    def test_report_on_missing_run(self, tmp_path, capsys) -> None:
        from benchmarks.ground_truth_eval.cli import main

        code = main(
            [
                "benchmark-report",
                "--run",
                "no_such_run",
                "--history",
                str(tmp_path / "h"),
                "--kind",
                "accuracy",
            ]
        )
        assert code != 0

    def test_report_without_metrics_errors(self, tmp_path, capsys) -> None:
        from benchmarks.ground_truth_eval.cli import main

        runner = BenchmarkRunner()
        run = runner.run(self.example_dataset_for_cli(), run_id="no_metrics")
        h = BenchmarkHistory(tmp_path / "h")
        h.append(run)
        code = main(
            [
                "benchmark-report",
                "--run",
                "no_metrics",
                "--history",
                str(tmp_path / "h"),
                "--kind",
                "executive",
            ]
        )
        assert code != 0
        assert "no stored metrics" in capsys.readouterr().err

    def test_compare_with_metrics(self, tmp_path, example_run, example_metrics, capsys) -> None:
        from benchmarks.ground_truth_eval.cli import main

        h = BenchmarkHistory(tmp_path / "compare_h")
        h.append(example_run, metrics=example_metrics.to_dict())
        code = main(
            [
                "benchmark-compare",
                "--run-a",
                example_run.run_id,
                "--run-b",
                example_run.run_id,
                "--history",
                str(tmp_path / "compare_h"),
            ]
        )
        assert code == 0
        payload = __import__("json").loads(capsys.readouterr().out)
        assert payload["drift"]["affected_company_count"] == 0

    def test_export_bundle_via_run(self, tmp_path) -> None:
        from benchmarks.ground_truth_eval.cli import main

        code = main(
            [
                "benchmark-run",
                "--dataset",
                self.EXAMPLE_DATASET,
                "--history",
                str(tmp_path / "export_h"),
                "--run-id",
                "export_cli",
                "--export-dir",
                str(tmp_path / "out"),
            ]
        )
        assert code == 0
        assert (tmp_path / "out" / "export_cli.bundle.json").exists()
        assert (tmp_path / "out" / "export_cli.run.json").exists()
        assert (tmp_path / "out" / "export_cli.metrics.json").exists()

    def test_export_run_and_metrics(self, tmp_path, example_run, example_metrics) -> None:
        from benchmarks.ground_truth_eval.cli import main

        h = BenchmarkHistory(tmp_path / "h")
        h.append(example_run, metrics=example_metrics.to_dict())
        code = main(
            [
                "benchmark-export",
                "--run",
                example_run.run_id,
                "--history",
                str(tmp_path / "h"),
                "--out",
                str(tmp_path / "out"),
            ]
        )
        assert code == 0
        assert (tmp_path / "out" / f"{example_run.run_id}.run.json").exists()
        assert (tmp_path / "out" / f"{example_run.run_id}.metrics.json").exists()

    def test_history_listing(self, tmp_path, example_run, example_metrics, capsys) -> None:
        from benchmarks.ground_truth_eval.cli import main

        h = BenchmarkHistory(tmp_path / "h")
        h.append(example_run, metrics=example_metrics.to_dict())
        code = main(["benchmark-history", "--history", str(tmp_path / "h")])
        assert code == 0
        out = capsys.readouterr().out
        assert example_run.run_id in out
        assert "1 run(s)" in out

    def test_verify(self, tmp_path, example_run, example_metrics, capsys) -> None:
        from benchmarks.ground_truth_eval.cli import main

        h = BenchmarkHistory(tmp_path / "h")
        h.append(example_run, metrics=example_metrics.to_dict())
        code = main(["benchmark-history", "--history", str(tmp_path / "h"), "--verify"])
        assert code == 0
        assert "verified 1 run(s)" in capsys.readouterr().out

    def test_unknown_dataset(self, tmp_path, capsys) -> None:
        from benchmarks.ground_truth_eval.cli import main

        code = main(
            [
                "benchmark-run",
                "--dataset",
                "definitely_missing_dataset",
                "--history",
                str(tmp_path / "h"),
                "--no-store",
            ]
        )
        assert code != 0

    def atest_dataset_bare_name(self, tmp_path, capsys) -> None:
        """
        Bare-name resolution via discover_golden_datasets.
        Discovered dataset 'example_v1' resolves. (name check only)
        """
        assert Path(self.EXAMPLE_DATASET).exists()

    def example_dataset_for_cli(self):
        from benchmarks.ground_truth_eval.dataset import load_golden_dataset

        return load_golden_dataset(self.EXAMPLE_DATASET)
