"""Acquisition file cache.

Stores downloaded source files locally so repeated acquisitions do
not re-download unchanged data.  Each cached file is stored under
a content-addressed path derived from its SHA-256 hash.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from predictron_engine.dataset.acquisition.state import compute_file_hash


@dataclass
class CacheEntry:
    """Metadata for a cached file."""

    source_name: str
    original_path: str
    cached_path: str
    file_hash: str
    cached_at: str = ""
    file_size: int = 0

    def __post_init__(self) -> None:
        if not self.cached_at:
            self.cached_at = datetime.now(UTC).isoformat()


class AcquisitionCache:
    """Local file cache for acquisition sources.

    Directory layout::

        dataset_store/acquisition/cache/
            {source_name}/
                {hash_prefix}/
                    {full_hash}.{ext}
            index.json
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._cache_dir = self._root / "acquisition" / "cache"
        self._index_path = self._cache_dir / "index.json"

    def initialize(self) -> None:
        """Create cache directories."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get_index(self) -> dict[str, CacheEntry]:
        """Load the cache index."""
        if not self._index_path.exists():
            return {}
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
            return {k: CacheEntry(**v) for k, v in data.items()}
        except (json.JSONDecodeError, TypeError):
            return {}

    def _write_index(self, index: dict[str, CacheEntry]) -> None:
        """Persist the cache index."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        data = {k: asdict(v) for k, v in index.items()}
        self._index_path.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )

    def lookup(self, source_name: str, file_hash: str) -> CacheEntry | None:
        """Look up a cached file by source name and content hash."""
        index = self.get_index()
        key = f"{source_name}:{file_hash}"
        return index.get(key)

    def store(
        self,
        source_name: str,
        file_path: Path | str,
        *,
        file_hash: str | None = None,
    ) -> CacheEntry:
        """Copy a file into the cache and record it in the index.

        Returns the CacheEntry with the cached file path.
        """
        src = Path(file_path)
        if not src.exists():
            msg = f"File not found: {src}"
            raise FileNotFoundError(msg)

        if file_hash is None:
            file_hash = compute_file_hash(src)

        ext = src.suffix or ".dat"
        hash_prefix = file_hash[:4]
        cache_subdir = self._cache_dir / source_name / hash_prefix
        cache_subdir.mkdir(parents=True, exist_ok=True)
        cached_path = cache_subdir / f"{file_hash}{ext}"
        shutil.copy2(src, cached_path)

        entry = CacheEntry(
            source_name=source_name,
            original_path=str(src),
            cached_path=str(cached_path),
            file_hash=file_hash,
            file_size=cached_path.stat().st_size,
        )

        index = self.get_index()
        key = f"{source_name}:{file_hash}"
        index[key] = entry
        self._write_index(index)
        return entry

    def has(self, source_name: str, file_hash: str) -> bool:
        """Check if a file is already cached."""
        entry = self.lookup(source_name, file_hash)
        if entry is None:
            return False
        return Path(entry.cached_path).exists()

    def cached_path(self, source_name: str, file_hash: str) -> Path | None:
        """Return the cached file path if it exists."""
        entry = self.lookup(source_name, file_hash)
        if entry is None:
            return None
        p = Path(entry.cached_path)
        return p if p.exists() else None

    def clear(self, source_name: str | None = None) -> int:
        """Remove cached files.  Returns the number of entries removed."""
        index = self.get_index()
        removed = 0
        keys_to_remove: list[str] = []
        for key, entry in index.items():
            if source_name and entry.source_name != source_name:
                continue
            cached = Path(entry.cached_path)
            if cached.exists():
                cached.unlink()
            keys_to_remove.append(key)
            removed += 1
        for key in keys_to_remove:
            del index[key]
        self._write_index(index)
        return removed
