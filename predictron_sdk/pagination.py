"""Automatic pagination helpers for the Predictron SDK.

``Paginator`` exposes the Python iterator protocol over any paginated API
response. It supports both offset/limit pagination (analysis list, search,
batch jobs) and cursor-based pagination (batch job results).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Generic, Self, TypeVar

__all__ = [
    "Page",
    "Paginator",
    "DEFAULT_PAGE_LIMIT",
    "MAX_PAGE_LIMIT",
    "PaginatorError",
]

DEFAULT_PAGE_LIMIT = 20
MAX_PAGE_LIMIT = 100

T = TypeVar("T")

#: A pagination token — either an integer offset or an opaque cursor string.
Token = str | int | None


class PaginatorError(Exception):
    """Raised when pagination state cannot be advanced safely."""


@dataclass(frozen=True, slots=True)
class Page(Generic[T]):
    """A single page of results from a paginated endpoint."""

    items: Sequence[T] = field(default_factory=tuple)
    total: int = 0
    limit: int = DEFAULT_PAGE_LIMIT
    has_more: bool = False
    next_token: Token = None

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

    def __getitem__(self, index: int) -> T:
        return self.items[index]

    def __repr__(self) -> str:
        return (
            f"<Page items={len(self.items)} total={self.total} "
            f"has_more={self.has_more}>"
        )


#: Protocol for functions that fetch a single page.
PageFetcher = Callable[[Token, int], Page[T]]


class Paginator(Generic[T], Iterator[Page[T]]):
    """Iterate over every page of a paginated API endpoint.

    ``fetcher`` receives the current pagination token and the requested page
    size and must return a :class:`Page`. Iteration stops when
    ``page.has_more`` is ``False`` or when ``page.next_token`` is ``None``.
    """

    def __init__(
        self,
        fetcher: PageFetcher[T],
        limit: int = DEFAULT_PAGE_LIMIT,
        *,
        start: Token = None,
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if limit > MAX_PAGE_LIMIT:
            raise ValueError(f"limit must be <= {MAX_PAGE_LIMIT}")
        self._fetcher = fetcher
        self._limit = limit
        self._start = start
        self._next: Token = start
        self._done = False
        self._page_count = 0

    def __iter__(self) -> Self:
        self._next = self._start
        self._done = False
        self._page_count = 0
        return self

    def __next__(self) -> Page[T]:
        if self._done:
            raise StopIteration
        page = self._fetcher(self._next, self._limit)
        self._page_count += 1
        next_token = page.next_token
        if not page.has_more or next_token is None:
            self._done = True
        self._next = next_token
        return page

    def items(self) -> Iterator[T]:
        """Yield individual items across every page."""
        for page in self:
            yield from page.items

    def all(self, *, max_pages: int | None = None) -> list[T]:
        """Collect every item across every page into a single list."""
        if max_pages is not None and max_pages <= 0:
            return []
        collected: list[T] = []
        page_count = 0
        for page in self:
            page_count += 1
            collected.extend(page.items)
            if max_pages is not None and page_count >= max_pages:
                break
        return collected

    @property
    def page_count(self) -> int:
        """Number of pages fetched so far."""
        return self._page_count

    @property
    def is_exhausted(self) -> bool:
        """Whether iteration has reached the final page."""
        return self._done


def page_from_offset(
    items: Sequence[T],
    total: int,
    offset: int,
    limit: int,
) -> Page[T]:
    """Build a :class:`Page` computing the next offset from the item count."""
    next_offset = offset + len(items)
    has_more = next_offset < total and bool(items)
    return Page(
        items=items,
        total=total,
        limit=limit,
        has_more=has_more,
        next_token=next_offset if has_more else None,
    )
