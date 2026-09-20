"""Tests for the knowledge graph CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from predictron_engine.dataset import cli
from predictron_engine.dataset.store import DatasetStore
from tests.dataset.graph_helpers import small_dataset


def _store(tmp_path) -> Path:
    path = tmp_path / "ds"
    store = DatasetStore(path)
    store.initialize()
    for record in small_dataset():
        store.save_record(record)
    return path


def _run(argv: list[str]) -> int:
    return cli.main(argv)


class TestGraphBuildCli:
    def test_graph_build_reports_node_and_edge_counts(self, tmp_path, capsys) -> None:
        dataset = _store(tmp_path)
        rc = _run(["graph-build", "--dataset", str(dataset)])
        assert rc == 0
        output = json.loads(capsys.readouterr().out)
        assert output["node_count"] == 32
        assert output["edge_count"] == 36
        assert output["identity_count"] == 5
        assert len(output["graph_key"]) == 64

    def test_graph_build_writes_output_file(self, tmp_path) -> None:
        dataset = _store(tmp_path)
        out = tmp_path / "graph.json"
        _run(
            [
                "graph-build",
                "--dataset",
                str(dataset),
                "--output",
                str(out),
            ]
        )
        assert out.exists()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["node_count"] == 32


class TestGraphReportCli:
    def test_full_report(self, tmp_path, capsys) -> None:
        dataset = _store(tmp_path)
        _run(["graph-report", "--dataset", str(dataset)])
        output = json.loads(capsys.readouterr().out)
        assert output["overview"]["node_count"] == 32
        assert (
            output["relationship_summary"]["relationships"][0]["edge_type"]
            is not None
        )

    def test_statistics_report(self, tmp_path, capsys) -> None:
        dataset = _store(tmp_path)
        _run(
            [
                "graph-report",
                "--report",
                "statistics",
                "--dataset",
                str(dataset),
            ]
        )
        output = json.loads(capsys.readouterr().out)
        assert output["connected_components"]["count"] == 2

    def test_relationship_report(self, tmp_path, capsys) -> None:
        dataset = _store(tmp_path)
        _run(
            [
                "graph-report",
                "--report",
                "relationship",
                "--dataset",
                str(dataset),
            ]
        )
        output = json.loads(capsys.readouterr().out)
        assert len(output["relationships"]) == 12


class TestGraphQueryCli:
    def test_metrics_query(self, tmp_path, capsys) -> None:
        dataset = _store(tmp_path)
        _run(["graph-query", "metrics", "--dataset", str(dataset)])
        output = json.loads(capsys.readouterr().out)
        assert output["node_count"] == 32
        assert output["edge_count"] == 36
        assert output["component_count"] == 2

    def test_neighbors_query(self, tmp_path, capsys) -> None:
        dataset = _store(tmp_path)
        _run(
            [
                "graph-query",
                "neighbors",
                "--node",
                "Alpha Labs",
                "--dataset",
                str(dataset),
            ]
        )
        output = json.loads(capsys.readouterr().out)
        labels = {entry["node"]["label"] for entry in output}
        assert "Ada Lovelace" in labels
        assert "fintech" in labels

    def test_shortest_path_query(self, tmp_path, capsys) -> None:
        dataset = _store(tmp_path)
        _run(
            [
                "graph-query",
                "shortest-path",
                "--source",
                "Alpha Labs",
                "--target",
                "Beta Health",
                "--dataset",
                str(dataset),
            ]
        )

    def test_companies_by_industry_query(self, tmp_path, capsys) -> None:
        dataset = _store(tmp_path)
        _run(
            [
                "graph-query",
                "companies-by-industry",
                "--value",
                "fintech",
                "--dataset",
                str(dataset),
            ]
        )
        output = json.loads(capsys.readouterr().out)
        company_ids = {
            entry["node_id"]
            for entry in output
            if entry["node_type"] == "company"
        }
        assert company_ids == {"company:a+d", "company:c"}

    def test_similar_companies_query(self, tmp_path, capsys) -> None:
        dataset = _store(tmp_path)
        _run(
            [
                "graph-query",
                "similar-companies",
                "--node",
                "Alpha Labs",
                "--limit",
                "3",
                "--dataset",
                str(dataset),
            ]
        )
        output = json.loads(capsys.readouterr().out)
        assert output
        assert output[0]["node_id"] == "company:c"

    def test_unknown_query_fails(self, tmp_path, capsys) -> None:
        dataset = _store(tmp_path)
        with pytest.raises(SystemExit) as exc:
            _run(
                [
                    "graph-query",
                    "bogus-query",
                    "--dataset",
                    str(dataset),
                ]
            )
        assert exc.value.code == 2
        assert "invalid choice" in capsys.readouterr().err

    def test_missing_node_fails(self, tmp_path, capsys) -> None:
        dataset = _store(tmp_path)
        rc = _run(
            [
                "graph-query",
                "neighbors",
                "--node",
                "Not A Company",
                "--dataset",
                str(dataset),
            ]
        )
        assert rc == 1
