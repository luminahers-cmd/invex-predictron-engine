"""Tests for the deterministic HTML cleaner."""

from __future__ import annotations

from predictron_engine.evidence.cleaner import HtmlCleaner
from predictron_engine.evidence.models import CleanResult


class TestHtmlCleaner:
    def setup_method(self) -> None:
        self.cleaner = HtmlCleaner()

    def test_removes_script_and_style(self) -> None:
        html = (
            "<html><head><style>.x{display:none}</style></head>"
            "<body><script>alert('x')</script><main><p>Hello world.</p></main></body></html>"
        )
        result = self.cleaner.clean(html)
        assert "alert" not in result.text
        assert "Hello world." in result.text

    def test_removes_navigation_and_footer(self) -> None:
        html = (
            "<body><nav>Home About</nav><main><p>Content</p></main>"
            "<footer>© 2026</footer></body>"
        )
        result = self.cleaner.clean(html)
        assert "Home" not in result.text
        assert "©" not in result.text
        assert "Content" in result.text

    def test_removes_cookie_banner_and_ads(self) -> None:
        html = (
            '<body><div id="cookie-banner">We use cookies to improve your experience.</div>'
            '<div class="advertisement">Sponsored content here</div>'
            "<main><p>Real content.</p></main></body>"
        )
        result = self.cleaner.clean(html)
        assert "cookie" not in result.text.lower()
        assert "sponsored" not in result.text.lower()
        assert "Real content." in result.text

    def test_removes_hidden_elements(self) -> None:
        html = (
            "<body><main>"
            "<p>Visible text.</p>"
            '<p style="display:none">Hidden text one.</p>'
            "<p hidden>Hidden text two.</p>"
            '<p aria-hidden="true">Hidden text three.</p>'
            "</main></body>"
        )
        result = self.cleaner.clean(html)
        assert "Visible text." in result.text
        assert "Hidden" not in result.text

    def test_preserves_headings_paragraphs_lists_tables(self) -> None:
        html = (
            "<main>"
            "<h1>Title</h1><h2>Subtitle</h2>"
            "<p>Paragraph one.</p>"
            "<ul><li>First item</li><li>Second item</li></ul>"
            "<table><tr><th>Metric</th><th>Value</th></tr>"
            "<tr><td>Customers</td><td>120</td></tr></table>"
            "</main>"
        )
        result = self.cleaner.clean(html)
        assert result.headings == ["Title", "Subtitle"]
        assert result.paragraphs == ["Paragraph one."]
        assert result.list_items == ["First item", "Second item"]
        assert any("Customers" in row and "120" in row for row in result.table_rows)
        assert "Title" in result.text

    def test_extracts_title_from_title_tag(self) -> None:
        html = (
            "<html><head><title>Example — Official Site</title></head>"
            "<body><main><h1>Ignored</h1></main></body></html>"
        )
        assert self.cleaner.clean(html).title == "Example — Official Site"

    def test_title_falls_back_to_og_title(self) -> None:
        html = (
            '<html><head><meta property="og:title" content="Open Graph Title"></head>'
            "<body><main><p>Text</p></main></body></html>"
        )
        assert self.cleaner.clean(html).title == "Open Graph Title"

    def test_normalizes_whitespace(self) -> None:
        html = "<body><main><p>Line one.\n\n   Line two.    Line three.</p></main></body>"
        result = self.cleaner.clean(html)
        assert "Line one. Line two. Line three." in result.text

    def test_empty_html_returns_empty_result(self) -> None:
        result = self.cleaner.clean("")
        assert isinstance(result, CleanResult)
        assert result.empty is True
        assert result.text == ""

    def test_no_text_content_marks_empty(self) -> None:
        html = '<body><main><div class="ad">Sponsored</div></main></body>'
        result = self.cleaner.clean(html)
        assert result.empty is True

    def test_avoids_duplicate_list_item_content(self) -> None:
        html = "<body><main><ul><li><p>Item content</p></li></ul></main></body>"
        result = self.cleaner.clean(html)
        assert result.list_items == ["Item content"]
        assert result.paragraphs == []
        assert result.text.count("Item content") == 1

    def test_main_is_preferred_content_root(self) -> None:
        html = (
            "<body><div><p>Outside main</p></div>"
            "<main><p>Inside main</p></main></body>"
        )
        result = self.cleaner.clean(html)
        assert "Inside main" in result.text
        assert "Outside main" not in result.text

    def test_invalid_html_does_not_raise(self) -> None:
        result = self.cleaner.clean("<main><p>Unclosed paragraph")
        assert "Unclosed paragraph" in result.text

    def test_word_count_reported(self) -> None:
        html = "<body><main><p>One two three four.</p></main></body>"
        result = self.cleaner.clean(html)
        assert result.word_count == 4
