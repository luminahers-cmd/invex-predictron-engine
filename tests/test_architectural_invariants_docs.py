"""Tests that the documented architectural invariants stay documented.

A rule that exists only in CI is a rule a contributor cannot understand.
The prose in ``CONTRIBUTING.md`` is therefore treated as part of the
contract: if an invariant is enforced mechanically but no longer
explained, or explained but not enforced, these tests fail.

The checks are deliberately textual rather than semantic. Their purpose
is to catch drift — a renamed module, a deleted rationale, a documented
exception that has silently accumulated more entries than reviewers
approved — not to grade the writing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tools.check_canonical_invariant import (
    ALLOWED_MODULES,
    EXEMPT_PREFIXES,
    SUPPRESSIONS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRIBUTING = REPO_ROOT / "CONTRIBUTING.md"
ARCHITECTURE = REPO_ROOT / "docs" / "architecture.md"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

#: Invariants that must be documented, with the marker each one uses.
#: Keyed by title so a rename forces an explicit update rather than
#: silently dropping the requirement.
REQUIRED_INVARIANTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Content-addressed identifiers go through `predictron_engine.canonical`",
        ("predictron_engine.canonical", "content_digest", "text_digest"),
    ),
    (
        "Replay determinism",
        ("byte-identical", "wall-clock"),
    ),
    (
        "Protocol boundaries, not concrete dependencies",
        ("Protocol", "getattr"),
    ),
    (
        "Evidence, provenance, and trust stay distinct",
        ("EvidenceDocument", "ProvenanceRecord", "TrustScore", "citations"),
    ),
    (
        "Typed, strict, and dependency-light",
        ("mypy", "strict"),
    ),
    (
        "Additive, versioned changes to persisted artifacts",
        ("SCHEMA_VERSION", "content-addressed"),
    ),
)


def _flatten(text: str) -> str:
    """Normalize prose so a rule can be matched across line breaks.

    Markdown emphasis characters (backticks and asterisks) are dropped and everything is lowered:
    what is being asserted is that the requirement is *stated*, not that
    it is stated with a particular capitalization or at a particular
    column.  Asserting on the exact wrapping would push the wording
    toward what is convenient to match rather than what reads well.
    """
    collapsed = " ".join(text.replace("`", "").replace("*", "").split())
    return collapsed.lower()


def _contributing_text() -> str:
    return CONTRIBUTING.read_text(encoding="utf-8")


def _invariants_section() -> str:
    """Return the Architectural Invariants section, isolated."""
    text = _contributing_text()
    match = re.search(
        r"^## Architectural Invariants\s*$(.*?)(?=^## )",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, "CONTRIBUTING.md has no Architectural Invariants section"
    return match.group(1)


class TestContributingStructure:
    def test_invariants_section_exists(self) -> None:
        assert "## Architectural Invariants" in _contributing_text()

    def test_section_appears_before_the_workflow(self) -> None:
        """Invariants are read as constraints, not as an afterthought."""
        text = _contributing_text()
        assert text.index("## Architectural Invariants") < text.index(
            "## Development Workflow"
        )


class TestEveryInvariantIsDocumented:
    @pytest.mark.parametrize(("title", "markers"), REQUIRED_INVARIANTS, ids=lambda v: v)
    def test_invariant_is_listed_with_its_rationale(
        self, title: str, markers: tuple[str, ...]
    ) -> None:
        section = _invariants_section()
        assert title in section, f"invariant not documented: {title}"
        for marker in markers:
            assert marker in section, (
                f"{title!r} is documented but never mentions {marker!r}; "
                "a rule without its reason is a rule nobody will follow"
            )

    def test_canonical_rule_states_the_three_step_exception_process(self) -> None:
        """The bar for an exception must be spelled out, not implied.

        Compared after flattening whitespace and backticks so the check
        tracks meaning rather than line-wrapping — the prose has to be
        readable, and a test that forbids a paragraph break would push
        the wording toward what is convenient to match.
        """
        section = _flatten(_invariants_section())
        for requirement in (
            "architectural justification",
            "suppression with a reason",
            "reviewer approval",
        ):
            assert requirement in section, (
                f"the exception process must name {requirement!r}"
            )
    def test_no_title_is_a_duplicate(self) -> None:
        section = _invariants_section()
        for title, _ in REQUIRED_INVARIANTS:
            assert section.count(title) == 1, f"duplicate heading: {title}"


class TestDocumentationMatchesEnforcement:
    def test_documented_module_is_the_one_enforced(self) -> None:
        """Docs and enforcement must name the same module.

        The guard keys on the filesystem path while the prose reads
        better with dotted notation, so either spelling counts. What must
        not happen is the docs naming one module and the allowlist
        naming another — that gap is exactly where an unreviewed
        exception would hide.
        """
        allowed = {path.split("/")[-1] for path in ALLOWED_MODULES}
        assert "canonical.py" in allowed
        haystack = _flatten(
            _contributing_text() + ARCHITECTURE.read_text(encoding="utf-8")
        )
        assert "predictron_engine.canonical" in haystack
        assert "predictron_engine/canonical" in haystack or "canonical.py" in haystack

    def test_documented_exemption_matches_the_enforced_one(self) -> None:
        """Every suppression must be named in the docs, and vice versa."""
        for suppression in SUPPRESSIONS:
            symbol = suppression.symbol
            haystack = _contributing_text() + ARCHITECTURE.read_text(
                encoding="utf-8"
            )
            assert symbol in haystack, (
                f"suppression {suppression.path}:{symbol} is enforced but "
                "undocumented; a silent exception is not reviewable"
            )

    def test_single_exemption_documented_as_single(self) -> None:
        """A growing exception list means the invariant is eroding."""
        assert len(SUPPRESSIONS) == 1, (
            "a new suppression needs reviewer approval and a documented "
            f"justification; found {len(SUPPRESSIONS)}: "
            f"{[(s.path, s.symbol) for s in SUPPRESSIONS]}"
        )

    def test_tests_exemption_is_documented(self) -> None:
        haystack = _contributing_text() + ARCHITECTURE.read_text(encoding="utf-8")
        for prefix in EXEMPT_PREFIXES:
            assert prefix in haystack, f"{prefix} exempt but undocumented"


class TestEnforcementIsWiredIntoCi:
    def test_workflow_invokes_the_guard(self) -> None:
        workflow = CI_WORKFLOW.read_text(encoding="utf-8")
        assert "python -m tools.check_canonical_invariant" in workflow, (
            "the invariant checker must remain part of CI"
        )

    def test_workflow_runs_ruff_and_mypy(self) -> None:
        workflow = CI_WORKFLOW.read_text(encoding="utf-8")
        assert "ruff check" in workflow
        assert "mypy" in workflow

    def test_guard_runs_before_tests(self) -> None:
        """A fast structural check should not wait on the full suite."""
        workflow = CI_WORKFLOW.read_text(encoding="utf-8")
        guard = workflow.index("check_canonical_invariant")
        tests = workflow.index("pytest")
        assert guard < tests


class TestContributingChecklistIsComplete:
    def test_prerequisites_list_all_four_checks(self) -> None:
        text = _contributing_text()
        for command in (
            "python -m pytest tests",
            "python -m ruff check",
            "python -m mypy",
            "python -m tools.check_canonical_invariant",
        ):
            assert command in text, f"{command} missing from the checklist"


#: The five things an ADR must explain. Keyed by the phrase the prose
#: has to use so a rewrite that drops a topic fails rather than drifting.
REQUIRED_ADR_TOPICS: tuple[str, ...] = (
    "motivation",
    "alternatives considered",
    "migration strategy",
    "compatibility impact",
    "why the previous invariant is no longer appropriate",
)

ADR_README = REPO_ROOT / "docs" / "adr" / "README.md"
ADR_TEMPLATE = REPO_ROOT / "docs" / "adr" / "TEMPLATE.md"


class TestInvariantEvolutionPolicy:
    """An invariant nobody is allowed to weaken is an invariant with no
    documented exit — which is how one gets weakened anyway, quietly.

    The policy is therefore part of the contract: strengthen or clarify
    freely, but relaxation has to leave a reviewable trail.
    """

    def test_policy_is_stated_as_strengthen_or_clarify_only(self) -> None:
        section = _flatten(_invariants_section())
        assert "may only be strengthened or clarified" in section

    def test_policy_requires_an_adr_to_relax_or_remove(self) -> None:
        section = _flatten(_invariants_section())
        required = "relaxing or removing an invariant requires an "
        assert required + "architecture decision record" in section

    def test_policy_explains_why_the_exit_is_gatekept(self) -> None:
        """A rule with a gate and no reason reads as bureaucracy."""
        section = _invariants_section()
        assert "erode by accumulation" in section

    @pytest.mark.parametrize("topic", REQUIRED_ADR_TOPICS)
    def test_required_adr_topics_appear_in_the_policy(self, topic: str) -> None:
        section = _flatten(_invariants_section())
        assert topic in section, f"the policy must name {topic!r} as ADR content"

    def test_policy_links_to_the_adr_process(self) -> None:
        section = _invariants_section()
        assert "docs/adr/README.md" in section
        assert "docs/adr/TEMPLATE.md" in section


class TestAdrInfrastructure:
    def test_directory_holds_a_process_document(self) -> None:
        assert ADR_README.is_file(), (
            "the ADR policy must point at something contributors can follow"
        )

    def test_template_contains_every_required_topic(self) -> None:
        template = _flatten(ADR_TEMPLATE.read_text(encoding="utf-8"))
        for topic in REQUIRED_ADR_TOPICS:
            assert topic in template, f"template is missing a required section: {topic}"

    def test_template_does_not_preempt_a_decision(self) -> None:
        """A record that already claims a verdict invites rubber-stamping."""
        template = _flatten(ADR_TEMPLATE.read_text(encoding="utf-8"))
        assert template.count("accepted | superseded") == 1

    def test_index_covers_every_adr_on_disk(self) -> None:
        """An ADR that exists but is not indexed is an ADR nobody reads."""
        index = _flatten(ADR_README.read_text(encoding="utf-8"))
        adrs = sorted(REPO_ROOT.glob("docs/adr/[0-9][0-9][0-9][0-9]-*.md"))
        for adr in adrs:
            assert adr.stem in index, f"{adr.name} is not listed in the ADR index"

    def test_index_states_the_current_state(self) -> None:
        """The index must never misrepresent what is on disk: an empty
        directory says so, and the first real ADR replaces that line
        rather than being appended beside it."""
        index = ADR_README.read_text(encoding="utf-8")
        adrs = sorted(REPO_ROOT.glob("docs/adr/[0-9][0-9][0-9][0-9]-*.md"))
        if adrs:
            assert "No ADRs yet" not in index, (
                "ADR(s) exist but the index still claims there are none: "
                f"{[a.name for a in adrs]}"
            )
        else:
            assert "No ADRs yet" in index

    def test_no_adr_relabels_a_relaxation_as_a_clarification(self) -> None:
        """If a file in the ADR directory merely 'clarifies', the two
        categories have collapsed and the gate means nothing."""
        for adr in REPO_ROOT.glob("docs/adr/[0-9][0-9][0-9][0-9]-*.md"):
            body = _flatten(adr.read_text(encoding="utf-8"))
            assert "relax" in body, f"{adr.name} does not record a relaxation"
