"""Tests for population configuration (Project V4)."""

from __future__ import annotations

import json

import pytest

from predictron_engine.dataset.population_config import (
    PopulationConfig,
    SourceConfig,
    build_default_config,
    load_config,
)


def test_source_config_roundtrip() -> None:
    src = SourceConfig(
        name="csv_export",
        file_paths=["/a.csv", "/b.csv"],
        directories=["/data"],
        enable_resume=False,
    )
    rebuilt = SourceConfig.from_dict(src.to_dict())
    assert rebuilt == src


def test_population_config_roundtrip_and_lookup() -> None:
    cfg = PopulationConfig(
        sources=[
            SourceConfig(name="csv_export", file_paths=["/a.csv"]),
            SourceConfig(name="yc_oss", directories=["/y"]),
        ],
        idempotent=False,
        batch_size=500,
        checkpoint_every=2,
    )
    rebuilt = PopulationConfig.from_dict(cfg.to_dict())
    assert rebuilt == cfg
    assert rebuilt.source_names() == ["csv_export", "yc_oss"]
    assert rebuilt.get("csv_export").file_paths == ["/a.csv"]
    assert rebuilt.get("missing") is None


def test_from_dict_ignores_non_string_paths() -> None:
    data = {
        "name": "x",
        "file_paths": ["ok.csv", 42, None],
        "directories": [123],
    }
    src = SourceConfig.from_dict(data)
    assert src.file_paths == ["ok.csv"]
    assert src.directories == []


def test_load_config_from_file(tmp_path) -> None:
    path = tmp_path / "config.json"
    cfg = PopulationConfig(sources=[SourceConfig(name="csv_export")])
    path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    loaded = load_config(path)
    assert loaded.source_names() == ["csv_export"]


def test_load_config_rejects_non_object(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[1,2,3]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_build_default_config_includes_registered_sources() -> None:
    cfg = build_default_config()
    names = cfg.source_names()
    assert "csv_export" in names
    assert "yc_oss" in names
