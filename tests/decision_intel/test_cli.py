"""Tests for the Decision Intelligence CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from predictron_engine.decision.cli import main

from .conftest import build_feature_set


@pytest.fixture
def populated_feature_store(tmp_path: Any) -> Any:
    from predictron_engine.feature_store.store import FeatureStore
    store = FeatureStore(tmp_path / "features")
    store.initialize()
    fs = build_feature_set(company_id="cli-co")
    store.save_company_features(fs)
    fs2 = build_feature_set(company_id="cli-co-2")
    store.save_company_features(fs2)
    return store


@pytest.fixture
def feature_dir(populated_feature_store: Any) -> Any:
    return populated_feature_store._root


def run_cli(argv: list[str], feature_dir: Any) -> dict[str, Any] | str:
    """Run the CLI and capture JSON output or markdown text."""
    import contextlib
    import io

    argv = ["--feature-dir", str(feature_dir)] + list(argv)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(argv)
    output = buf.getvalue()
    if output.startswith("Written to"):
        path = output.split()[-1]
        return json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return output


class TestDecisionReportCLI:
    def test_report_command_returns_json(
        self,
        populated_feature_store: Any,
        feature_dir: Any,
    ) -> None:
        result = run_cli(["decision-report", "--company-id", "cli-co"], feature_dir)
        assert isinstance(result, dict)
        assert result["company_id"] == "cli-co"

    def test_report_has_contributions(
        self,
        populated_feature_store: Any,
        feature_dir: Any,
    ) -> None:
        result = run_cli(["decision-report", "--company-id", "cli-co"], feature_dir)
        assert isinstance(result, dict)
        assert "contributions" in result

    def test_report_to_output_file(self, tmp_path: Any) -> None:
        from predictron_engine.feature_store.store import FeatureStore
        fs_dir = tmp_path / "features"
        store = FeatureStore(fs_dir)
        store.initialize()
        store.save_company_features(build_feature_set(company_id="f-co"))
        out = tmp_path / "report.json"
        result = run_cli(
            ["decision-report", "--company-id", "f-co", "-o", str(out)],
            fs_dir,
        )
        assert isinstance(result, dict)

    def test_report_verdict_override(self, feature_dir: Any) -> None:
        result = run_cli(
            ["decision-report", "--company-id", "cli-co", "--verdict", "pass"],
            feature_dir,
        )
        assert isinstance(result, dict)
        assert result["decision_summary"]["verdict"] == "pass"

    def test_report_confidence_override(self, feature_dir: Any) -> None:
        result = run_cli(
            ["decision-report", "--company-id", "cli-co", "--confidence", "0.75"],
            feature_dir,
        )
        assert isinstance(result, dict)
        assert "confidence" in result["decision_summary"]

    def test_report_missing_company_exits(self, feature_dir: Any) -> None:
        with pytest.raises(SystemExit):
            run_cli(["decision-report", "--company-id", "missing"], feature_dir)


class TestDecisionTraceCLI:
    def test_trace_compact(self, feature_dir: Any) -> None:
        result = run_cli(
            ["decision-trace", "--company-id", "cli-co", "--compact"],
            feature_dir,
        )
        assert isinstance(result, str)
        assert "Overall score:" in result
        assert "Verdict:" in result

    def test_trace_json(self, feature_dir: Any) -> None:
        result = run_cli(
            ["decision-trace", "--company-id", "cli-co"],
            feature_dir,
        )
        assert isinstance(result, dict)
        assert "nodes" in result
        assert "edges" in result

    def test_trace_has_decision_node(self, feature_dir: Any) -> None:
        result = run_cli(
            ["decision-trace", "--company-id", "cli-co"],
            feature_dir,
        )
        assert isinstance(result, dict)
        node_types = {n["node_type"] for n in result["nodes"]}
        assert "decision" in node_types


class TestDecisionExplainCLI:
    def test_explain_output(self, feature_dir: Any) -> None:
        result = run_cli(
            ["decision-explain", "--company-id", "cli-co"],
            feature_dir,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_explain_contains_recommendation(self, feature_dir: Any) -> None:
        result = run_cli(
            ["decision-explain", "--company-id", "cli-co"],
            feature_dir,
        )
        assert isinstance(result, str)
        assert "Recommendation:" in result

    def test_explain_verdict_override(self, feature_dir: Any) -> None:
        result = run_cli(
            ["decision-explain", "--company-id", "cli-co", "--verdict", "watch"],
            feature_dir,
        )
        assert isinstance(result, str)


class TestDecisionCompareCLI:
    def test_compare_all(self, feature_dir: Any) -> None:
        result = run_cli(["decision-compare"], feature_dir)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_compare_selected(self, feature_dir: Any) -> None:
        result = run_cli(
            ["decision-compare", "--company-ids", "cli-co"],
            feature_dir,
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["company_id"] == "cli-co"

    def test_compare_to_file(self, tmp_path: Any, feature_dir: Any) -> None:
        out = tmp_path / "compare.json"
        result = run_cli(
            ["decision-compare", "-o", str(out)],
            feature_dir,
        )
        assert isinstance(result, list)


class TestDecisionCalibrateCLI:
    def test_calibrate_command(self, feature_dir: Any) -> None:
        result = run_cli(
            [
                "decision-calibrate",
                "--expected-conf", "0.8",
                "--actual-conf", "0.6",
                "--case-id", "case-a",
            ],
            feature_dir,
        )
        assert isinstance(result, dict)
        assert result["benchmark_case_id"] == "case-a"
        assert result["adjustment_applied"] > 0.0

    def test_calibrate_missing_args(self, feature_dir: Any) -> None:
        with pytest.raises(SystemExit):
            run_cli(["decision-calibrate"], feature_dir)

    def test_calibrate_aligned(self, feature_dir: Any) -> None:
        result = run_cli(
            [
                "decision-calibrate",
                "--expected-conf", "0.7",
                "--actual-conf", "0.7",
            ],
            feature_dir,
        )
        assert isinstance(result, dict)
        assert result["calibration_delta"] == 0.0

    def test_calibrate_with_history_file(
        self,
        tmp_path: Any,
        feature_dir: Any,
    ) -> None:
        hist = tmp_path / "history.json"
        hist.write_text(json.dumps([
            {"benchmark_case_id": "h1", "expected_confidence": 0.8, "actual_confidence": 0.5},
        ]), encoding="utf-8")
        result = run_cli(
            [
                "decision-calibrate",
                "--expected-conf", "0.8",
                "--actual-conf", "0.6",
                "--history", str(hist),
            ],
            feature_dir,
        )
        assert isinstance(result, dict)
        assert len(result["historical_calibration"]) == 1


class TestCLIHelp:
    def test_no_command_prints_help(self, feature_dir: Any) -> None:
        with pytest.raises(SystemExit):
            run_cli([], feature_dir)

    @pytest.mark.parametrize("command", [
        "decision-report",
        "decision-trace",
        "decision-explain",
        "decision-compare",
        "decision-calibrate",
    ])
    def test_known_subcommand_dispatch(self, command: str, populated_feature_store: Any) -> None:
        # -h should raise SystemExit(0) which argparse emits.
        with pytest.raises(SystemExit) as exc:
            main([command, "-h"])
        assert exc.value.code == 0
