"""Knowledge graph helper — builds a KnowledgeGraph from a DatasetStore.

All knowledge-graph endpoints share this single construction path so
graph-bearing responses stay consistent across the API surface.
"""

from __future__ import annotations

from predictron_engine.dataset.graph.builder import CompanyKnowledgeGraphBuilder
from predictron_engine.dataset.graph.store import KnowledgeGraph
from predictron_engine.dataset.store import DatasetStore


def load_records(store: DatasetStore | None) -> list[object]:
    """Load all DatasetRecords from the store (empty when store is None)."""
    if store is None:
        return []
    records: list[object] = []
    for record_id in store.list_records():
        record = store.load_record(record_id)
        if record is not None:
            records.append(record)
    return records


def build_graph(store: DatasetStore | None) -> KnowledgeGraph:
    """Build the company knowledge graph from a DatasetStore."""
    records = load_records(store)
    from predictron_engine.dataset.models import DatasetRecord

    typed: list[DatasetRecord] = [
        r for r in records if isinstance(r, DatasetRecord)
    ]
    if not typed:
        return KnowledgeGraph()
    return CompanyKnowledgeGraphBuilder().build(typed).graph
