"""Search service — unified search across the venture intelligence stack."""

from __future__ import annotations

from app.schemas.search import (
    CompanySearchResult,
    KnowledgeGraphNodeSearchResult,
    SearchByTypeResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    SignalSearchResult,
)
from predictron_engine.dataset.store import DatasetStore


def search(
    request: SearchRequest,
    store: DatasetStore | None = None,
) -> SearchResponse:
    """Execute a search across the venture intelligence stack."""
    query = request.query.lower().strip()
    results: list[SearchResultItem] = []

    if request.search_type in ("all", "company"):
        results.extend(_search_companies(query, store))

    if request.search_type in ("all", "signal_type"):
        results.extend(_search_signals(query, store))

    if request.search_type in ("all", "knowledge_graph_node"):
        results.extend(_search_graph_nodes(query, store))

    if request.search_type in ("all", "industry"):
        results.extend(_search_industry(query, store))

    if request.search_type in ("all", "country"):
        results.extend(_search_country(query, store))

    total = len(results)
    page = results[request.offset : request.offset + request.limit]

    return SearchResponse(
        query=request.query,
        search_type=request.search_type,
        total=total,
        offset=request.offset,
        limit=request.limit,
        results=page,
    )


def search_by_type(
    search_type: str,
    query: str,
    offset: int = 0,
    limit: int = 20,
    store: DatasetStore | None = None,
) -> SearchByTypeResponse:
    """Execute a type-specific search."""
    query_lower = query.lower().strip()
    companies: list[CompanySearchResult] = []
    signals: list[SignalSearchResult] = []
    graph_nodes: list[KnowledgeGraphNodeSearchResult] = []

    if search_type == "company":
        companies = _search_companies_typed(query_lower, store)
    elif search_type == "signal_type":
        signals = _search_signals_typed(query_lower, store)
    elif search_type in ("knowledge_graph_node", "investor", "founder", "technology"):
        graph_nodes = _search_graph_nodes_typed(query_lower, store)
    elif search_type == "industry":
        companies = _search_companies_typed(query_lower, store)
    elif search_type == "country":
        companies = _search_companies_typed(query_lower, store)

    total = len(companies) + len(signals) + len(graph_nodes)

    return SearchByTypeResponse(
        search_type=search_type,
        total=total,
        offset=offset,
        limit=limit,
        companies=companies[offset : offset + limit],
        signals=signals[offset : offset + limit],
        graph_nodes=graph_nodes[offset : offset + limit],
    )


def _search_companies(
    query: str, store: DatasetStore | None
) -> list[SearchResultItem]:
    """Search companies by name, description, or industry."""
    items: list[SearchResultItem] = []
    if store is None:
        return items

    for rid in store.list_records():
        rec = store.load_record(rid)
        if rec is None:
            continue

        match_score = 0.0
        if query in rec.startup_name.lower():
            match_score = 0.9
        elif rec.profile.description and query in rec.profile.description.lower():
            match_score = 0.5
        elif any(query in ind.lower() for ind in rec.profile.industries):
            match_score = 0.7
        elif rec.profile.headquarters and query in rec.profile.headquarters.lower():
            match_score = 0.6
        elif rec.profile.country_code and query in rec.profile.country_code.lower():
            match_score = 0.6

        if match_score > 0:
            items.append(
                SearchResultItem(
                    result_type="company",
                    id=rec.record_id,
                    name=rec.startup_name,
                    description=rec.profile.description or "",
                    score=match_score,
                    metadata={
                        "website": rec.website,
                        "industries": rec.profile.industries,
                        "country_code": rec.profile.country_code,
                        "decision": rec.prediction.decision.value,
                    },
                )
            )

    items.sort(key=lambda x: x.score, reverse=True)
    return items


def _search_companies_typed(
    query: str, store: DatasetStore | None
) -> list[CompanySearchResult]:
    """Search companies returning typed results."""
    results: list[CompanySearchResult] = []
    if store is None:
        return results

    for rid in store.list_records():
        rec = store.load_record(rid)
        if rec is None:
            continue

        match = False
        if query in rec.startup_name.lower():
            match = True
        elif any(query in ind.lower() for ind in rec.profile.industries):
            match = True
        elif rec.profile.country_code and query in rec.profile.country_code.lower():
            match = True
        elif rec.profile.headquarters and query in rec.profile.headquarters.lower():
            match = True

        if match:
            results.append(
                CompanySearchResult(
                    record_id=rec.record_id,
                    startup_name=rec.startup_name,
                    website=rec.website,
                    industries=rec.profile.industries,
                    country_code=rec.profile.country_code,
                    headquarters=rec.profile.headquarters,
                    founded_year=rec.profile.founded_year,
                    decision=rec.prediction.decision.value,
                    composite_score=rec.prediction.composite_score,
                    confidence=rec.prediction.confidence,
                )
            )

    return results


def _search_signals(
    query: str, store: DatasetStore | None
) -> list[SearchResultItem]:
    """Search signals by type or metadata."""
    items: list[SearchResultItem] = []
    if store is None:
        return items

    for company_id in store.list_signal_company_ids():
        timeline = store.load_timeline(company_id)
        if timeline is None:
            continue

        for sig in timeline.signals:
            if (
                query in sig.signal_type.value
                or query in sig.source.lower()
                or query in company_id.lower()
            ):
                items.append(
                    SearchResultItem(
                        result_type="signal",
                        id=sig.signal_id,
                        name=f"{sig.signal_type.value} ({company_id})",
                        description=f"{sig.source} @ {sig.timestamp.date().isoformat()}",
                        score=0.7,
                        metadata={
                            "company_id": company_id,
                            "signal_type": sig.signal_type.value,
                            "source": sig.source,
                        },
                    )
                )

    return items


def _search_signals_typed(
    query: str, store: DatasetStore | None
) -> list[SignalSearchResult]:
    """Search signals returning typed results."""
    results: list[SignalSearchResult] = []
    if store is None:
        return results

    for company_id in store.list_signal_company_ids():
        timeline = store.load_timeline(company_id)
        if timeline is None:
            continue

        for sig in timeline.signals:
            if (
                query in sig.signal_type.value
                or query in sig.source.lower()
                or query in company_id.lower()
            ):
                results.append(
                    SignalSearchResult(
                        signal_id=sig.signal_id,
                        company_id=company_id,
                        signal_type=sig.signal_type.value,
                        timestamp=sig.timestamp.isoformat(),
                        source=sig.source,
                        confidence=sig.confidence,
                        metadata=dict(sig.metadata),
                    )
                )

    return results


def _search_graph_nodes(
    query: str, store: DatasetStore | None
) -> list[SearchResultItem]:
    """Search knowledge graph nodes."""
    items: list[SearchResultItem] = []
    if store is None:
        return items

    try:
        from app.services.graph import build_graph
        from predictron_engine.dataset.graph.queries import GraphQueries

        graph = build_graph(store)
        gq = GraphQueries(graph)

        for node in graph.nodes():
            if (
                query in node.label.lower()
                or query in node.node_id.lower()
                or query in node.node_type.value
            ):
                neighbors = gq.neighbors(node.node_id, bidirectional=True)
                connected = [
                    n.get("node", {}).get("label", "")
                    for n in neighbors
                    if n.get("node", {}).get("node_type") == "company"
                ]
                items.append(
                    SearchResultItem(
                        result_type="knowledge_graph_node",
                        id=node.node_id,
                        name=node.label,
                        description=f"Type: {node.node_type.value}",
                        score=0.7,
                        metadata={
                            "node_type": node.node_type.value,
                            "connected_companies": connected[:5],
                        },
                    )
                )
    except Exception:
        pass

    return items


def _search_graph_nodes_typed(
    query: str, store: DatasetStore | None
) -> list[KnowledgeGraphNodeSearchResult]:
    """Search graph nodes returning typed results."""
    results: list[KnowledgeGraphNodeSearchResult] = []
    if store is None:
        return results

    try:
        from app.services.graph import build_graph
        from predictron_engine.dataset.graph.queries import GraphQueries

        graph = build_graph(store)
        gq = GraphQueries(graph)

        for node in graph.nodes():
            if (
                query in node.label.lower()
                or query in node.node_id.lower()
                or query in node.node_type.value
            ):
                neighbors = gq.neighbors(node.node_id, bidirectional=True)
                connected = [
                    n.get("node", {}).get("label", "")
                    for n in neighbors
                    if n.get("node", {}).get("node_type") == "company"
                ]
                results.append(
                    KnowledgeGraphNodeSearchResult(
                        node_id=node.node_id,
                        node_type=node.node_type.value,
                        label=node.label,
                        properties=node.properties,
                        sources=node.sources,
                        connected_companies=connected,
                    )
                )
    except Exception:
        pass

    return results


def _search_industry(
    query: str, store: DatasetStore | None
) -> list[SearchResultItem]:
    """Search by industry."""
    return _search_companies(query, store)


def _search_country(
    query: str, store: DatasetStore | None
) -> list[SearchResultItem]:
    """Search by country."""
    return _search_companies(query, store)
