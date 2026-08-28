"""Deterministic HTML cleaner built on BeautifulSoup.

Removes JavaScript, CSS, navigation, footer, cookie banners, ads, and
hidden elements while preserving headings, paragraphs, lists, tables,
and meaningful text. Output text is whitespace-normalized.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from bs4 import BeautifulSoup, Tag

from predictron_engine.evidence.exceptions import CleanError
from predictron_engine.evidence.models import CleanResult

_NON_CONTENT_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "iframe",
        "frame",
        "frameset",
        "object",
        "embed",
        "applet",
        "canvas",
        "svg",
        "audio",
        "video",
        "source",
        "picture",
        "form",
        "input",
        "button",
        "select",
        "textarea",
        "label",
        "map",
        "area",
        "track",
        "meta",
        "link",
        "base",
    }
)

_CHROME_TAGS = frozenset({"header", "footer", "nav", "aside"})

_CHROME_PATTERN = re.compile(
    r"(^|[\s\-_])(cookie|consent|cookiebar|gdpr|banner|ad|ads|advert|advertisement|"
    r"adsbox|adslot|sponsor|sponsored|popup|pop-up|modal|overlay|newsletter|subscribe|"
    r"share-buttons|social-icons|social-links|social-buttons|social-share|comments|"
    r"sidebar|related|recommended|site-footer|footer|navbar|nav-menu|main-menu|"
    r"navigation|breadcrumb|pagination|carousel|slider|toolbar|login|signup|masthead|"
    r"site-header|header)([\s\-_]|$)",
    re.IGNORECASE,
)

_HIDDEN_STYLE = re.compile(r"(?:display|visibility)\s*:\s*(?:none|hidden)\b", re.IGNORECASE)

_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

_TEXT_CONTAINERS = {
    "p": "paragraph",
    "blockquote": "quote",
    "figcaption": "caption",
    "dt": "term",
    "dd": "detail",
    "caption": "caption",
}

_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Collapse all whitespace runs into single spaces."""
    return _WS.sub(" ", text).strip()


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _attr(tag: Tag, name: str) -> str:
    """Return a tag attribute as plain text, coercing bs4 list values."""
    value = tag.get(name)
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value or "")


def _is_hidden(tag: Tag) -> bool:
    if tag.has_attr("hidden"):
        return True
    if _attr(tag, "aria-hidden").lower().strip() == "true":
        return True
    style = _attr(tag, "style")
    return bool(_HIDDEN_STYLE.search(style))


def _matches_chrome(tag: Tag) -> bool:
    tag_id = _attr(tag, "id")
    classes = " ".join(_attr(tag, "class").split())
    return bool(_CHROME_PATTERN.search(tag_id)) or bool(_CHROME_PATTERN.search(classes))


class HtmlCleaner:
    """Cleans raw HTML into normalized readable text."""

    def __init__(self, min_length: int = 0) -> None:
        self._min_length = min_length

    def clean(self, html: str, *, url: str | None = None) -> CleanResult:
        """Clean raw HTML, returning a CleanResult (never raises for page content)."""
        if not html or not html.strip():
            return CleanResult(empty=True)

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:  # noqa: BLE001 - deterministic, defensive
            raise CleanError(f"Failed to parse HTML: {exc}") from exc

        title = self._extract_title(soup)
        self._remove_chrome(soup)
        root = self._content_root(soup)

        pieces = list(self._walk(root))
        text = "\n\n".join(_dedupe(list(piece for _, piece in pieces)))
        word_count = len(text.split())

        return CleanResult(
            title=title,
            text=text,
            headings=[piece for kind, piece in pieces if kind == "heading"],
            paragraphs=[piece for kind, piece in pieces if kind == "paragraph"],
            list_items=[piece for kind, piece in pieces if kind == "list"],
            table_rows=[piece for kind, piece in pieces if kind == "table"],
            word_count=word_count,
            empty=word_count == 0,
        )

    def _extract_title(self, soup: BeautifulSoup) -> str:
        for title_tag in soup.find_all("title"):
            text = _normalize(title_tag.get_text())
            if text:
                return text
        for meta_tag in soup.find_all("meta"):
            if _attr(meta_tag, "content").strip() and (
                _attr(meta_tag, "property") == "og:title"
                or _attr(meta_tag, "name") == "twitter:title"
            ):
                return _normalize(_attr(meta_tag, "content"))
        h1 = soup.find("h1")
        if isinstance(h1, Tag):
            text = _normalize(h1.get_text(separator=" ", strip=True))
            if text:
                return text
        return ""

    def _remove_chrome(self, soup: BeautifulSoup) -> None:
        for tag in soup.find_all(list(_NON_CONTENT_TAGS)):
            tag.decompose()
        for tag in soup.find_all(list(_CHROME_TAGS)):
            tag.decompose()
        for tag in soup.find_all(role="navigation"):
            tag.decompose()
        for tag in list(soup.find_all(True)):
            if tag.parent is not None and (_is_hidden(tag) or _matches_chrome(tag)):
                tag.decompose()

    def _content_root(self, soup: BeautifulSoup) -> Tag:
        main = soup.find("main") or soup.find(role="main") or soup.find("article")
        if isinstance(main, Tag):
            return main
        body = soup.find("body")
        if isinstance(body, Tag):
            return body
        return soup

    def _walk(self, node: Tag) -> Iterator[tuple[str, str]]:
        name = node.name or ""
        if name in _HEADING_TAGS:
            text = self._text(node)
            if text:
                yield "heading", text
            return
        if name == "li":
            text = self._text(node)
            if text:
                yield "list", text
            return
        if name == "tr":
            cells = [self._text(cell) for cell in node.find_all(["td", "th"], recursive=False)]
            cells = [cell for cell in cells if cell]
            if cells:
                yield "table", " | ".join(cells)
            return
        if name in _TEXT_CONTAINERS:
            kind = _TEXT_CONTAINERS[name]
            text = self._text(node)
            if text and len(text) >= self._min_length:
                yield kind, text
            return

        for child in node.find_all(recursive=False):
            if isinstance(child, Tag):
                yield from self._walk(child)

    @staticmethod
    def _text(tag: Tag) -> str:
        return _normalize(tag.get_text(separator=" ", strip=True))
