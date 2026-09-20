"""Search API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Unified search request across the venture intelligence stack."""

    query: str = Field(
        ..., min_length=1, max_length=500, description="Search query string"
    )
    search_type: str = Field(
        default="all",
        description=(
            "Search type: all, company, industry, country, investor, "
            "founder, technology, knowledge_graph_node, signal_type"
        ),
    )
    offset: int = Field(default=0, ge=0, description="Pagination offset")
    limit: int = Field(
        default=20, ge=1, le=100, description="Pagination limit"
    )


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class SearchResultItem(BaseModel):
    """Single search result."""

    result_type: str = Field(
        ..., description="Type of result: company, signal, node, record, etc."
    )
    id: str = Field(..., description="Result identifier")
    name: str = Field(default="", description="Display name")
    description: str = Field(default="", description="Brief description")
    score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Relevance score"
    )
    metadata: dict[str, object] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Paginated search response."""

    query: str = ""
    search_type: str = "all"
    total: int = Field(default=0, ge=0)
    offset: int = 0
    limit: int = 20
    results: list[SearchResultItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Specific search results
# ---------------------------------------------------------------------------


class CompanySearchResult(BaseModel):
    """Company search result."""

    record_id: str = ""
    startup_name: str = ""
    website: str = ""
    industries: list[str] = Field(default_factory=list)
    country_code: str | None = None
    headquarters: str | None = None
    founded_year: int | None = None
    decision: str = ""
    composite_score: float = Field(default=0.0)
    confidence: float = Field(default=0.0)


class SignalSearchResult(BaseModel):
    """Signal search result."""

    signal_id: str = ""
    company_id: str = ""
    signal_type: str = ""
    timestamp: str = ""
    source: str = ""
    confidence: float = 1.0
    metadata: dict[str, object] = Field(default_factory=dict)


class KnowledgeGraphNodeSearchResult(BaseModel):
    """Knowledge graph node search result."""

    node_id: str = ""
    node_type: str = ""
    label: str = ""
    properties: dict[str, object] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)
    connected_companies: list[str] = Field(default_factory=list)


class SearchByTypeResponse(BaseModel):
    """Type-specific search response."""

    search_type: str = ""
    total: int = 0
    offset: int = 0
    limit: int = 20
    companies: list[CompanySearchResult] = Field(default_factory=list)
    signals: list[SignalSearchResult] = Field(default_factory=list)
    graph_nodes: list[KnowledgeGraphNodeSearchResult] = Field(
        default_factory=list
    )
