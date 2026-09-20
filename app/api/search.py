"""Search API — unified search across the venture intelligence stack."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query, status

from app.schemas.search import (
    SearchByTypeResponse,
    SearchRequest,
    SearchResponse,
)
from app.services.search import search, search_by_type

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


@router.post(
    "",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search venture intelligence",
    description=(
        "Unified search across companies, industries, countries, investors, "
        "founders, technology, knowledge graph nodes, and signal types."
    ),
    responses={
        200: {"description": "Search completed"},
        422: {"description": "Validation error"},
    },
)
async def search_endpoint(
    request: SearchRequest,
) -> SearchResponse:
    from pathlib import Path

    from predictron_engine.dataset.store import DatasetStore

    data_dir = Path("data") / "dataset"
    store = DatasetStore(str(data_dir)) if data_dir.exists() else None
    return search(request, store=store)


@router.get(
    "",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search (GET)",
    description="Search using GET parameters for convenience.",
    responses={
        200: {"description": "Search completed"},
    },
)
async def search_get(
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    type: str = Query(
        default="all",
        alias="search_type",
        description="Search type filter",
    ),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    limit: int = Query(default=20, ge=1, le=100, description="Pagination limit"),
) -> SearchResponse:
    from pathlib import Path

    from predictron_engine.dataset.store import DatasetStore

    data_dir = Path("data") / "dataset"
    store = DatasetStore(str(data_dir)) if data_dir.exists() else None

    request = SearchRequest(
        query=q,
        search_type=type,
        offset=offset,
        limit=limit,
    )
    return search(request, store=store)


@router.get(
    "/by-type/{search_type}",
    response_model=SearchByTypeResponse,
    status_code=status.HTTP_200_OK,
    summary="Search by type",
    description="Search filtered to a specific type: company, industry, country, etc.",
    responses={
        200: {"description": "Search completed"},
    },
)
async def search_by_type_endpoint(
    search_type: str,
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
) -> SearchByTypeResponse:
    from pathlib import Path

    from predictron_engine.dataset.store import DatasetStore

    data_dir = Path("data") / "dataset"
    store = DatasetStore(str(data_dir)) if data_dir.exists() else None
    return search_by_type(search_type, q, offset=offset, limit=limit, store=store)
