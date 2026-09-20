# Company Knowledge Graph

## Overview

The Company Knowledge Graph (Project E3) is a deterministic graph layer over
the resolved company identities produced by Entity Resolution (Project E2).
It transforms a collection of `DatasetRecord` instances into a connected,
queryable graph: companies, their founders, investors, technologies, products,
locations, industries, domains, identifiers, and the relationships between
companies.

The graph is generated **entirely from `DatasetRecord` objects**. Every node
and edge carries provenance back to the exact records that produced it, node
IDs and serialization are deterministic (identical input always yields a
byte-identical graph), and **no relationship is ever fabricated**.

**This module does NOT modify any engine scoring, reasoning, confidence,
recommendation, or threshold behavior.** It is purely additive: it reads
already-resolved identities and builds a read model on top of them.

## Architecture

```
predictron_engine/dataset/graph/
├── model.py          # NodeType / EdgeType vocabulary, GraphNode / GraphEdge
├── store.py          # KnowledgeGraph — in-memory adjacency store
├── extract.py        # Grounded metadata extraction contract
├── builder.py        # CompanyKnowledgeGraphBuilder — deterministic builder
├── queries.py        # GraphQueries — traversal and lookup API
├── metrics.py        # GraphMetrics — structural statistics
├── reports.py        # graph_report / graph_statistics / relationship_summary
└── persistence.py    # Versioned JSON snapshot contract (no Neo4j dependency)
```

## Node Vocabulary

The graph uses exactly twelve node types (`NodeType`):

| Node Type   | Node ID pattern                 | Example           |
|-------------|---------------------------------|-------------------|
| Company     | `company:<sorted record ids>`   | `company:a+d`     |
| Founder     | `founder:<entity key>`          | `founder:ada lovelace` |
| Investor    | `investor:<entity key>`         | `investor:accel`  |
| Organization| `organization:<entity key>`     | `organization:galaxy holdings` |
| Industry    | `industry:<entity key>`         | `industry:fintech`|
| Technology  | `technology:<entity key>`       | `technology:pytorch` |
| Product     | `product:<entity key>`          | `product:alpha platform` |
| Country     | `country:<entity key>`          | `country:us`      |
| State/Region| `state:<entity key>`            | `state:tx`        |
| City        | `city:<entity key>`             | `city:austin`     |
| Domain      | `domain:<entity key>`           | `domain:alpha.ai` |
| Identifier  | `identifier:<kind>:<value>`     | `identifier:sec_cik:000123` |

Notes:

- **Company node IDs are record-derived**, not UUIDs: `company:` followed by
  the `+`-joined sorted `record_id` values of the identity. This makes node IDs
  deterministic across builds and machines.
- **Attribute node IDs** (`attribute_node_id`) lowercase-normalize the label
  via `entity_key`, so `Accel` and `accel` collapse to one node while the human
  `label` preserves the first-seen grounded spelling.
- A single record that resolves to an identity merges into that company node;
  a merged identity (`a` + `d`) is exactly one company node.

## Relationship Vocabulary

The graph uses exactly twelve relationship types (`EdgeType`):

| Relationship   | Source   | Target        | Grounded by                                   |
|----------------|----------|---------------|-----------------------------------------------|
| `FOUNDED_BY`   | Company  | Founder       | founders metadata                              |
| `INVESTED_BY`  | Company  | Investor      | investors metadata / outcome investors        |
| `LOCATED_IN`   | Company  | Country/State/City | profile country/region/city             |
| `OPERATES_IN`  | Company  | Industry      | profile industries                             |
| `USES_TECHNOLOGY` | Company | Technology | technologies metadata                      |
| `BUILDS_PRODUCT` | Company | Product     | products metadata                              |
| `HAS_DOMAIN`   | Company  | Domain        | resolved canonical/alternate/record domains   |
| `HAS_IDENTIFIER`| Company | Identifier    | identifiers resolved from metadata            |
| `ALIAS_OF`     | Company  | Company       | shared core/name keys across identities       |
| `ACQUIRED_BY`  | Company  | Company/Organization | acquirer metadata or outcome acquisition |
| `SUBSIDIARY_OF`| Company  | Company/Organization | parent-organization metadata           |
| `RELATED_TO`   | Company  | Company       | related/competitors/peers metadata            |

Direction and symmetry:

- `ALIAS_OF` and `RELATED_TO` are **symmetric**: the store canonicalizes their
  endpoints (`canonical_edge_key`) so `(a, ALIAS_OF, b)` and
  `(b, ALIAS_OF, a)` are the same edge.
- All other relationship types are directed; queries default to bidirectional
  traversal but can restrict to one direction.

## Groundedness — No Fabricated Relationships

Every edge in the graph traces back to a literal value in a record's
`analysis_metadata` or `profile` (or an outcome record):

- **Attribute edges** exist only when the value appears in the record data
  under a documented extraction key (see Metadata Contract).
- **`RELATED_TO`** is emitted only when the referenced name resolves to an
  existing company identity in the same graph; references that do not resolve
  are dropped (never guessed).
- **`ACQUIRED_BY` / `SUBSIDIARY_OF`**: a reference that resolves to a known
  company is pinned to that company node; otherwise the acquirer/parent becomes
  an `Organization` node grounded in the literal name. No name is invented.
- **`ALIAS_OF`** edges connect identities that share a normalized core name or
  name key while remaining distinct after Entity Resolution (e.g. two
  differently-registered "Global Bank" entities with conflicting countries).
  This is a *derived* edge, but it is derived from resolved identity data, so
  it never implies a merge.

`KnowledgeGraph.validate()` reports any node or edge that lacks provenance and
any edge that dangles, so a clean build is verifiable programmatically
(`GraphBuildReport.provenance_issues`).

## Determinism

Two properties make the graph reproducible:

1. **Deterministic node IDs** — company IDs come from record IDs, attribute
   IDs from normalized labels, identifier IDs from `kind:value`.
2. **Canonical serialization** — `canonical_graph_json(graph)` orders nodes by
   `node_id` and edges by canonical key, with sorted keys and no insignificant
   whitespace.

`graph_key(graph)` is the SHA-256 of the canonical JSON. Building the same
records in any order produces the same `graph_key`. This makes graph artifacts
safe to cache and diff across builds.

## Provenance

- Every `GraphNode` and `GraphEdge` carries a `sources: list[str]` of record
  IDs.
- Shared attribute nodes union provenance: the `industry:fintech` node created
  by records `a`, `c`, and `d` records all three IDs.
- Merged companies carry the full record-ID set of their identity; identifier
  edges carry the specific records that produced each identifier value.
- Provenance is stable in `GraphNode.to_dict()` (sorted, deduplicated).

## Builder Pipeline

`CompanyKnowledgeGraphBuilder.build(records, outcomes=None)`:

1. **Resolve** — runs the default `EntityResolver` over records to produce
   canonical `CompanyIdentity` objects.
2. **Emit company nodes** — one `Company` node per identity, with
   `canonical_domain`, aliases, status, founded year, employee range,
   `merge_decision`, and `record_sources` properties.
3. **Emit identity attributes** — `OPERATES_IN` (industries),
   `LOCATED_IN` (countries), `HAS_IDENTIFIER`, `HAS_DOMAIN`.
4. **Emit per-record attributes** — founders, investors, technologies,
   products, city/state/country, parent organizations, acquirers, related
   companies (record-level provenance per edge).
5. **Emit outcome edges** — when `include_outcomes=True` (default) and an
   `outcomes` mapping is provided, investors and acquirers observed in
   `OutcomeRecord` data become `INVESTED_BY` / `ACQUIRED_BY` edges.
6. **Emit alias edges** — parallel-safe, bucketed by core/name keys over
   distinct identities (never an O(I²) pairwise scan).
7. **Measure** — counts nodes/edges by type, validates provenance, computes
   the `graph_key`, and returns a `GraphBuildResult` with a `GraphBuildReport`.

`include_outcomes=False` ignores the outcomes mapping entirely, so base-index
builds depend only on records.

## Query API

`GraphQueries(graph)` provides:

| Method | Description |
|--------|-------------|
| `lookup(token)` | Resolve a node ID or case-insensitive label/core name to a node ID |
| `neighbors(node, edge_type=None, bidirectional=True)` | Adjacent nodes with connecting edge kinds |
| `shortest_path(start, end, bidirectional=True)` | BFS shortest node-ID path (or `None`) |
| `connected_components()` | Weakly connected components (union-find) |
| `similar_companies(company, limit=10)` | Companies ranked by Jaccard neighbor-set overlap with shared-by-type breakdown |
| `companies_by_industry(value)` | Company nodes linked to the industry |
| `companies_by_country(value)` | Company nodes linked to the country |
| `companies_using_technology(value)` | Company nodes using the technology |

All results are deterministic: sorted by stable keys, ties broken by
`(node_type, node_id)`.

## Metrics

`compute_graph_metrics(graph)` returns a `GraphMetrics` with:

- node / edge counts (total and per type)
- connected components: count, sizes, largest, average
- degree distribution: histogram, min / max / average, isolated nodes
- graph density: directed and undirected

## Reports

Three report builders (each deterministic, JSON-serializable):

- `build_graph_report(graph)` — overview (counts, key, connectivity), full
  metrics, relationship summary, top-connected nodes, validation result.
- `build_graph_statistics(graph)` — node/edge counts by type, degree
  distribution, component summary, density, per-relationship rows.
- `build_relationship_summary(graph)` — one row per relationship type (all
  twelve, including zero-count rows), with count, distinct sources/targets, and
  a sample of source→target labels.

The relationship summary lists the full vocabulary, making it easy to see which
relationships are currently instantiated and which are absent.

## Persistence

`graph/persistence.py` defines a versioned JSON snapshot contract
(`GRAPH_SCHEMA_VERSION = 1`):

- `graph_to_dict(graph)` / `graph_from_dict(dict)` — strict round-trip.
- `canonical_graph_json(graph)` — canonical, hashable JSON.
- `write_graph(graph, path)` / `read_graph(path)` — file persistence.

The in-memory graph is the working store; the snapshot is the persistence
boundary. A future backend (Neo4j, object store, columnar warehouse) needs only
to round-trip this document — no database driver is required today.

## CLI

Commands are exposed on the existing `predictron-dataset` console script:

```
predictron-dataset graph-build [--include-outcomes] [--output graph.json]
    Build the graph and print a GraphBuildReport (or write to --output).

predictron-dataset graph-report [--report full|statistics|relationship] [--output ...]
    Emit one of the three report documents.

predictron-dataset graph-query <query> [--node N] [--edge-type T]
    [--source S --target T] [--value V] [--limit L] [--directed]
    Run a read-only query. Queries: neighbors, shortest-path,
    connected-components, similar-companies, companies-by-industry,
    companies-by-country, companies-using-technology, metrics.
```

## Metadata Contract

Attribute extraction reads only the following keys from
`record.analysis_metadata` (see `graph/extract.py`):

| Node kind  | Metadata keys |
|------------|---------------|
| Founders   | `founders`, `founder`, `founder_name`, `founder_names`, `cofounders`, `co_founders`, `ceo` |
| Investors  | `investors`, `investor`, `investor_name`, `lead_investor`, `lead_investors`, `backers`, `funding_investors` |
| Technologies | `technologies`, `technology`, `tech_stack`, `technology_stack`, `stack`, `technologies_used` |
| Products   | `products`, `product`, `product_name`, `product_names` |
| Organizations | `organizations`, `organization`, `parent_organization`, `parent_company`, `parent`, `parent_org` |
| Acquirer   | `acquirer`, `acquired_by`, `acquiring_company`, `acquisition_parent`, `purchaser` |
| Related    | `related_companies`, `related`, `competitors`, `peers`, `partners` |

Profile fields (`profile.industries`, `.country_code`, `.city`, `.region`,
`.domain`) and website-derived domains are also grounded inputs. Values are
normalized into entity keys but never inflected or interpolated.

## Scalability

The design targets millions of edges without external infrastructure:

- **Adjacency-store** — `KnowledgeGraph` keeps `set`-based out/in adjacency
  maps: O(1) node/edge existence, degree, and neighbor iteration.
- **Instantiated (not materialized) relationships** — company-to-company edges
  (`ALIAS_OF`, `RELATED_TO`) are only emitted for pairs that actually share a
  resolved name/core key, so the graph does not blow up with dense clique-free
  attribute space; shared attribute nodes are single nodes with combined
  provenance, not N×N company edges.
- **Bucketed alias emission** — alias pairs are produced per name-key bucket,
  avoiding a full O(I²) identity scan while still covering all sharing pairs.
- **Canonical hashing** — `graph_key` is computed over an ordered adjacency
  dump, so identical graphs hash identically with no precomputed key drift.
- **Multi-pass safe** — builds are pure functions of their inputs
  (`records`, optional `outcomes`), enabling incremental/partitioned builds
  later without changing identities.

For very large corpora, the recommended evolution is to persist the canonical
snapshot to an object store and, only if interactive traversal is required,
load it into a graph database — the snapshot contract keeps that migration
additive.

## Relation to Entity Resolution

The graph consumes `EntityResolver` output and the optional `OutcomeRecord`
stream:

- Resolved identities become Company nodes; merge decisions are recorded in
  node properties (`merge_decision`), tying the graph back to E2.
- `ALIAS_OF` encodes the same logical company across entities that E2
  deliberately kept distinct (country-conflict splits), giving a "related
  entities" view without overriding resolution.
- Outcomes contribute only *observed* investor/acquisition facts, never
  predictions.

## Recommended Next Milestone

1. **Graph lineage** — persist per-build `graph_key` plus input manifest to
   enable reproducible rebuilds and drift detection.
2. **Cycle-aware ranking** — degree + PageRank-style centrality for the query
   API (deterministic, in-memory).
3. **Full-text attribute search** — index labels via an inverted map to make
   `lookup` sub-linear on very large attribute spaces.
4. **Graph database backend** — implement the snapshot contract for Neo4j
   (or an object store) behind a `GraphBackend` protocol without changing the
   builder, queries, or reports.