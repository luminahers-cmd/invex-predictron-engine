"""Compatibility fixtures for the durable research-memory backend.

This module pins the golden ``1.0.0`` document: a canonical four-section
document (``schema_version``, ``index``, ``graph``, ``records``) together
with its exact bytes, frozen as the additive-compatibility baseline every
later persistence change is measured against.

Every test here is load-path only — the frozen document is reopened and
read, never mutated — so the suite stays valid when the writer later gains
sections or bumps the document version.  The fixture bytes themselves are
the contract and must never be regenerated.
"""

from __future__ import annotations

import json
from pathlib import Path

from predictron_engine.canonical import canonical_json
from predictron_engine.integrations.research_persistence import (
    PersistentResearchMemory,
)

#: The frozen golden document, stored as the exact bytes the writer produced.
GOLDEN_DOCUMENT_PATH = Path(__file__).resolve().parent / "fixtures" / "research_memory_v1.json"


# -- the fixture itself ------------------------------------------------


def test_the_golden_document_is_the_exact_canonical_encoding() -> None:
    text = GOLDEN_DOCUMENT_PATH.read_bytes().decode("utf-8")
    document = json.loads(text)
    assert canonical_json(document) == text


def test_the_golden_document_is_a_four_section_1_0_0_document() -> None:
    document = json.loads(GOLDEN_DOCUMENT_PATH.read_bytes())
    assert set(document) == {"schema_version", "index", "graph", "records"}
    assert document["schema_version"] == "1.0.0"
    assert document["index"]["schema_version"] == "1.0.0"
    assert document["graph"]["schema_version"] == "1.0.0"
    assert document["records"]["schema_version"] == "1.0.0"


# -- reopening the frozen document -------------------------------------


def test_reopening_the_golden_document_keeps_every_section_byte_identical(
    tmp_path: Path,
) -> None:
    raw = GOLDEN_DOCUMENT_PATH.read_bytes()
    golden = json.loads(raw)
    path = tmp_path / "memory.json"
    path.write_bytes(raw)

    reopened = PersistentResearchMemory(path)
    reopened.flush()

    document = json.loads(path.read_bytes())
    for section in ("index", "graph", "records"):
        assert canonical_json(document[section]) == canonical_json(golden[section])


def test_reopening_the_golden_document_reconstructs_the_reference_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.json"
    path.write_bytes(GOLDEN_DOCUMENT_PATH.read_bytes())

    handle = PersistentResearchMemory(path)

    assert handle.memory.counts() == {
        "reports": 2,
        "evidence": 1,
        "duplicates": 0,
        "indexed": 10,
        "entities": 3,
        "relationships": 1,
        "history": 0,
    }
    assert [report.report_id for report in handle.memory.reports()] == [
        "rep-1",
        "rep-2",
    ]
    assert [item.evidence_id for item in handle.memory.records.evidence()] == [
        "ev_cba43023a7d22f70"
    ]
    assert list(handle.memory.entity_ids()) == [
        "ent_company_b3afb67e9325bf5b",
        "ent_evidence_afb8531574b44427",
        "ent_technology_33c115a50f1e9704",
    ]
    assert handle.index.next_sequence() == 11
    assert handle.graph.relationship_count() == 1
    assert handle.memory.memory_key() == "graph_2f91defe8a6d4a79"
