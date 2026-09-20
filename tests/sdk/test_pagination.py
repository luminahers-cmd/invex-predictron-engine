"""Tests for pagination helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from predictron_sdk.models import SearchResultItem
from predictron_sdk.pagination import (
    MAX_PAGE_LIMIT,
    Page,
    Paginator,
    PaginatorError,
    page_from_offset,
)


def make_fetcher(pages: list[list[int]]) -> Callable:
    """Fetch pages of ints based on an offset token; 3 items per page."""

    def fetch(token: str | int | None, limit: int) -> Page[int]:
        offset = 0 if token is None else int(token)
        chunk = pages[0][offset : offset + limit]
        return page_from_offset(chunk, len(pages[0]), offset, limit)

    return fetch


def test_page_basics() -> None:
    page = Page(items=(1, 2), total=2, limit=20, has_more=False)
    assert len(page) == 2
    assert list(page) == [1, 2]
    assert page.next_token is None


def test_page_is_sequence() -> None:
    page = Page(items=(1, 2), total=2, limit=2)
    assert page[0] == 1
    assert tuple(page) == (1, 2)


def test_page_frozen() -> None:
    page = Page(items=(1,), total=1, limit=20)
    with pytest.raises(Exception):
        page.total = 5  # type: ignore[misc]


def test_page_from_offset_has_more() -> None:
    page = page_from_offset([1, 2, 3], total=9, offset=0, limit=3)
    assert page.has_more is True
    assert page.next_token == 3


def test_page_from_offset_terminal() -> None:
    page = page_from_offset([1, 2, 3], total=3, offset=0, limit=3)
    assert page.has_more is False
    assert page.next_token is None


def test_page_from_offset_empty_page_no_more() -> None:
    page = page_from_offset([], total=0, offset=0, limit=20)
    assert page.has_more is False


def test_page_from_offset_edge_partial() -> None:
    page = page_from_offset([1, 2], total=5, offset=3, limit=3)
    assert page.has_more is False
    assert page.next_token is None


def test_paginator_iterates_all_pages() -> None:
    paginator = Paginator(make_fetcher([list(range(9))]), limit=3)
    pages = [page for page in paginator]
    assert len(pages) == 3
    assert [p.total for p in pages] == [9, 9, 9]


def test_paginator_items_flattened() -> None:
    paginator = Paginator(make_fetcher([list(range(9))]), limit=3)
    assert list(paginator.items()) == list(range(9))


def test_paginator_all() -> None:
    paginator = Paginator(make_fetcher([list(range(9))]), limit=3)
    assert paginator.all() == list(range(9))


def test_paginator_all_max_pages() -> None:
    paginator = Paginator(make_fetcher([list(range(9))]), limit=3)
    assert paginator.all(max_pages=2) == [0, 1, 2, 3, 4, 5]
    assert paginator.all(max_pages=0) == []


def test_paginator_re_iterable() -> None:
    base = [list(range(6))]
    paginator = Paginator(make_fetcher(base), limit=2)
    assert list(paginator.items()) == list(range(6))
    # Re-iteration returns a fresh traversal from the start.
    assert list(paginator.items()) == list(range(6))


def test_paginator_single_page() -> None:
    paginator = Paginator(make_fetcher([[1, 2]]), limit=20)
    assert list(paginator.items()) == [1, 2]


def test_paginator_empty_result() -> None:
    paginator = Paginator(make_fetcher([[]]), limit=20)
    assert list(paginator.items()) == []


def test_paginator_stop_iteration() -> None:
    paginator = Paginator(make_fetcher([[1]]), limit=20)
    first = next(paginator)
    assert len(first.items) == 1
    with pytest.raises(StopIteration):
        next(paginator)


def test_paginator_guard_invalid_limit() -> None:
    with pytest.raises(ValueError):
        Paginator(make_fetcher([[]]), limit=0)
    with pytest.raises(ValueError):
        Paginator(make_fetcher([[]]), limit=MAX_PAGE_LIMIT + 1)


def test_paginator_cursor_style() -> None:
    server_pages = [
        {"cursor": "next-1", "items": [1, 2]},
        {"cursor": None, "items": [3]},
    ]
    state = {"i": 0}

    def fetch(token: str | int | None, limit: int) -> Page[int]:
        page = server_pages[state["i"]]
        state["i"] += 1
        return Page(
            items=tuple(page["items"]),
            total=3,
            limit=limit,
            has_more=page["cursor"] is not None,
            next_token=page["cursor"],
        )

    paginator = Paginator(fetch, limit=10)
    assert list(paginator.items()) == [1, 2, 3]


def test_paginator_token_start() -> None:
    seen: list[Any] = []

    def fetch(token: str | int | None, limit: int) -> Page[int]:
        seen.append(token)
        return page_from_offset([1], total=1, offset=0, limit=limit)

    paginator = Paginator(fetch, limit=10, start=7)
    next(paginator)
    assert seen[0] == 7


def test_paginator_is_exhausted_flag() -> None:
    paginator = Paginator(make_fetcher([[1]]), limit=20)
    assert paginator.is_exhausted is False
    next(paginator)
    assert paginator.is_exhausted is True


def test_paginator_page_count() -> None:
    paginator = Paginator(make_fetcher([list(range(6))]), limit=2)
    list(paginator)
    assert paginator.page_count == 3


@pytest.mark.parametrize(
    "items,total,offset,limit,expected_next",
    [
        ([1], 5, 0, 2, 1),
        ([1, 2], 5, 0, 2, 2),
        ([], 0, 0, 20, None),
        ([5], 1, 0, 20, None),
    ],
)
def test_page_from_offset_matrix(items, total, offset, limit, expected_next) -> None:
    page = page_from_offset(items, total, offset, limit)
    assert page.next_token == expected_next


def test_page_fields_exposed() -> None:
    page = page_from_offset([10, 11], 4, 0, 2)
    assert page.total == 4
    assert page.limit == 2
    assert page.next_token == 2


def test_paginator_typed_items() -> None:
    items = [SearchResultItem(result_type="company", id=str(i), name=f"n{i}") for i in range(5)]

    def fetch(token: str | int | None, limit: int) -> Page[SearchResultItem]:
        offset = 0 if token is None else int(token)
        chunk = items[offset : offset + limit]
        return page_from_offset(chunk, len(items), offset, limit)

    paginator: Paginator[SearchResultItem] = Paginator(fetch, limit=2)
    collected = paginator.all()
    assert [item.name for item in collected] == ["n0", "n1", "n2", "n3", "n4"]


def test_paginator_error_class() -> None:
    assert issubclass(PaginatorError, Exception)


def test_paginator_large_dataset_determinism() -> None:
    data = list(range(50))
    paginator = Paginator(make_fetcher([data]), limit=7)
    assert paginator.all() == data
