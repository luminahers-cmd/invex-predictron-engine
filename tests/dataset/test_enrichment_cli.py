"""Tests for the enrich CLI command (Project E1)."""

from __future__ import annotations

import json

import pytest

from predictron_engine.dataset.cli import build_parser, main
from tests.dataset.conftest import make_record


class TestEnrichCLI:
    def test_parser_has_enrich(self) -> None:
        args = build_parser().parse_args(["enrich", "--dataset", "/tmp/ds"])
        assert args.command == "enrich"

    def test_enrich_empty_store(self, tmp_path) -> None:
        rc = main(["enrich", "--dataset", str(tmp_path / "ds")])
        assert rc == 0

    def test_enrich_promotes_metadata(
        self, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store_dir = tmp_path / "ds"
        from predictron_engine.dataset.store import DatasetStore

        store = DatasetStore(store_dir)
        store.initialize()
        record = make_record(
            record_id="alpha",
            website="https://alpha.com",
        ).model_copy(
            update={"analysis_metadata": {"country": "United States", "sector": "SaaS"}}
        )
        store.save_record(record)

        rc = main(["enrich", "--dataset", str(store_dir)])
        assert rc == 0

        output = json.loads(capsys.readouterr().out)
        assert output["records_processed"] == 1
        assert output["records_updated"] == 1
        assert "domain" in output["fields_populated"]
        assert "country_code" in output["fields_populated"]

        loaded = store.load_record("alpha")
        assert loaded is not None
        assert loaded.profile.country_code == "US"
        assert loaded.profile.industries == ["saas"]
        assert loaded.profile.domain == "alpha.com"

    def test_enrich_is_idempotent(
        self, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store_dir = tmp_path / "ds"
        from predictron_engine.dataset.store import DatasetStore

        store = DatasetStore(store_dir)
        store.initialize()
        store.save_record(
            make_record(record_id="beta", website="https://beta.com").model_copy(
                update={"analysis_metadata": {"country": "Canada"}}
            )
        )

        assert main(["enrich", "--dataset", str(store_dir)]) == 0
        first = json.loads(capsys.readouterr().out)
        assert first["records_updated"] == 1

        assert main(["enrich", "--dataset", str(store_dir)]) == 0
        second = json.loads(capsys.readouterr().out)
        assert second["records_updated"] == 0
