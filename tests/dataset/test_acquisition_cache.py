"""Tests for the acquisition cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from predictron_engine.dataset.acquisition.cache import AcquisitionCache


@pytest.fixture()
def cache(tmp_path: Path) -> AcquisitionCache:
    c = AcquisitionCache(tmp_path / "dataset")
    c.initialize()
    return c


@pytest.fixture()
def sample_file(tmp_path: Path) -> Path:
    p = tmp_path / "data.json"
    p.write_text('{"key": "value"}')
    return p


class TestAcquisitionCache:
    def test_store_and_lookup(self, cache: AcquisitionCache, sample_file: Path) -> None:
        entry = cache.store("test_source", sample_file)
        assert entry.source_name == "test_source"
        assert len(entry.file_hash) == 64

        found = cache.lookup("test_source", entry.file_hash)
        assert found is not None
        assert found.file_hash == entry.file_hash

    def test_has(self, cache: AcquisitionCache, sample_file: Path) -> None:
        entry = cache.store("src", sample_file)
        assert cache.has("src", entry.file_hash) is True
        assert cache.has("src", "nonexistent") is False

    def test_cached_path(self, cache: AcquisitionCache, sample_file: Path) -> None:
        entry = cache.store("src", sample_file)
        p = cache.cached_path("src", entry.file_hash)
        assert p is not None
        assert p.exists()

    def test_clear(self, cache: AcquisitionCache, sample_file: Path) -> None:
        cache.store("src_a", sample_file)
        cache.store("src_b", sample_file)

        removed = cache.clear("src_a")
        assert removed == 1

    def test_clear_all(self, cache: AcquisitionCache, sample_file: Path) -> None:
        cache.store("src_a", sample_file)
        cache.store("src_b", sample_file)
        removed = cache.clear()
        assert removed == 2

    def test_store_missing_file(self, cache: AcquisitionCache) -> None:
        with pytest.raises(FileNotFoundError):
            cache.store("src", "/nonexistent/file.txt")
