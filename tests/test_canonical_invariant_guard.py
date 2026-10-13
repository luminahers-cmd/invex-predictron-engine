"""Tests for the canonical-hashing architectural invariant guard.

The guard is only useful if it fires on real violations and stays quiet
on legitimate code.  A CI check that produces false positives gets
deleted, so both halves are tested here: the rules must catch a
reimplemented digest, and they must not fire on a thin wrapper, on
human-readable pretty-printing, or on ``__hash__``.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import tools.check_canonical_invariant as guard
from tools.check_canonical_invariant import (
    ALLOWED_MODULES,
    SUPPRESSIONS,
    UNPARSEABLE_SENTINEL,
    Suppression,
    _is_digest_name,
    _ModuleChecker,
    check_repo,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _rules_for(source: str, path: str = "predictron_engine/example.py") -> set[str]:
    """Return the rule names triggered by a source snippet."""
    checker = _ModuleChecker(path)
    checker.visit(ast.parse(source))
    return {f.rule for f in checker.finish()}


class TestRepositoryIsClean:
    def test_no_violations_in_the_repository(self) -> None:
        findings = check_repo(REPO_ROOT)
        assert findings == [], "\n".join(f.render() for f in findings)

    def test_every_module_is_parseable(self) -> None:
        """A module the guard cannot read is a module it cannot protect."""
        findings = check_repo(REPO_ROOT)
        assert [f for f in findings if f.rule == "UNPARSEABLE"] == []


class TestHashPrimitiveRule:
    def test_flags_direct_sha256(self) -> None:
        assert "HASH_PRIMITIVE" in _rules_for("x = hashlib.sha256(b'a')")

    @pytest.mark.parametrize(
        "constructor",
        ["md5", "sha1", "sha512", "blake2b", "sha3_256"],
    )
    def test_flags_every_hash_constructor(self, constructor: str) -> None:
        assert "HASH_PRIMITIVE" in _rules_for(f"x = hashlib.{constructor}(b'a')")

    def test_flags_from_import(self) -> None:
        assert "HASH_PRIMITIVE" in _rules_for("from hashlib import sha256")

    def test_flags_hexdigest(self) -> None:
        assert "HEXDIGEST" in _rules_for("x = hashlib.sha256(b'a').hexdigest()")


class TestSortedJsonRule:
    def test_flags_compact_canonical_dialect(self) -> None:
        source = 'json.dumps(p, sort_keys=True, separators=(",", ":"))'
        assert "SORTED_JSON" in _rules_for(source)

    def test_ignores_pretty_printed_report(self) -> None:
        """An ``indent=2`` dump is a human-readable report, not an encoding."""
        source = "json.dumps(p, indent=2, sort_keys=True)"
        assert "SORTED_JSON" not in _rules_for(source)

    def test_ignores_sort_keys_false(self) -> None:
        source = 'json.dumps(p, sort_keys=False, separators=(",", ":"))'
        assert "SORTED_JSON" not in _rules_for(source)

    def test_flags_json_dump_too(self) -> None:
        source = 'json.dump(p, handle, sort_keys=True, separators=(",", ":"))'
        assert "SORTED_JSON" in _rules_for(source)


class TestBuiltinHashRule:
    def test_flags_builtin_hash(self) -> None:
        assert "BUILTIN_HASH" in _rules_for("x = hash(some_string)")

    def test_exempts_dunder_hash(self) -> None:
        """``__hash__`` returns a salted int CPython never serializes."""
        source = "class C:\n    def __hash__(self):\n        return hash(self.x)"
        assert _rules_for(source) == set()


class TestUnwrappedHelperRule:
    def test_flags_digest_named_helper_that_hashes_directly(self) -> None:
        source = (
            "def _thing_hash(payload):\n"
            "    return hashlib.sha256(str(payload).encode()).hexdigest()"
        )
        assert "UNWRAPPED_DIGEST_HELPER" in _rules_for(source)

    def test_accepts_thin_wrapper(self) -> None:
        source = (
            "from predictron_engine.canonical import content_digest\n"
            "def _thing_hash(payload):\n"
            "    return content_digest(payload)"
        )
        assert _rules_for(source) == set()

    def test_accepts_wrapper_using_an_import_alias(self) -> None:
        """Aliasing the import does not stop it being a thin wrapper."""
        source = (
            "from predictron_engine.canonical import content_digest as _cd\n"
            "def _thing_hash(payload):\n"
            "    return _cd(payload)"
        )
        assert _rules_for(source) == set()

    def test_ignores_helper_that_only_hashes_via_canonical(self) -> None:
        """Both conditions are required: digest-shaped name *and* hashing."""
        source = (
            "from predictron_engine.canonical import text_digest\n"
            "def _fingerprint(payload):\n"
            "    return text_digest(str(payload))"
        )
        assert "UNWRAPPED_DIGEST_HELPER" not in _rules_for(source)

    @pytest.mark.parametrize(
        "name",
        ["_normalise_text_for_hash", "_rotate", "_build_payload"],
    )
    def test_digest_name_matcher_rejects_preprocessing_names(self, name: str) -> None:
        """``..._for_hash`` describes an input, not a function that hashes."""
        assert not _is_digest_name(name)

    @pytest.mark.parametrize(
        "name",
        [
            "_evidence_digest",
            "_content_hash",
            "_snapshot_id",
            "compute_hash",
            "contentFingerprint",
        ],
    )
    def test_digest_name_matcher_catches_real_helpers(self, name: str) -> None:
        assert _is_digest_name(name)

    def test_broad_name_match_alone_does_not_fire_the_rule(self) -> None:
        """``_recommendation_fingerprint`` is digest-shaped but hashes nothing.

        It returns a normalized dedup key.  Name matching only nominates
        candidates; the hashing condition is what keeps the rule quiet on
        code that merely has a suggestive name.
        """
        source = (
            "def _recommendation_fingerprint(rec):\n"
            "    return rec.action.strip().lower()"
        )
        assert _rules_for(source) == set()


class TestSuppressionDiscipline:
    def test_every_suppression_has_a_reason(self) -> None:
        for suppression in SUPPRESSIONS:
            assert suppression.reason.strip(), (
                f"{suppression.path}:{suppression.symbol} has no reason; the "
                "reason is the part a reviewer needs to read"
            )

    def test_suppressions_target_real_files(self) -> None:
        for suppression in SUPPRESSIONS:
            assert (REPO_ROOT / suppression.path).exists(), suppression.path

    def test_allowed_modules_exist(self) -> None:
        for module in ALLOWED_MODULES:
            assert (REPO_ROOT / module).exists(), module

    def test_unreasoned_suppression_points_at_the_adr_policy(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A missing reason is a proposed relaxation, so the message
        should say where the record goes, not just which field is blank.

        This is the only moment a contributor reads the guidance — in
        the failure output, while the context is fresh — so the ADR
        requirement belongs there rather than only in the prose.
        """
        proposed = Suppression(
            path="predictron_engine/example.py",
            symbol="digest",
            reason="   ",
        )
        monkeypatch.setattr(guard, "SUPPRESSIONS", (proposed,))

        assert main(["--root", str(REPO_ROOT)]) == 1
        err = capsys.readouterr().err
        assert "predictron_engine/example.py:digest" in err
        assert "Architecture Decision Record" in err
        assert "docs/adr/README.md" in err

    def test_finding_report_points_at_the_adr_policy(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Same for a plain violation: the escape hatch it suggests is
        itself gated, and the output should not hide that."""
        (tmp_path / "predictron_engine").mkdir()
        (tmp_path / "predictron_engine" / "offender.py").write_text(
            "import hashlib\n"
            "def _digest(payload):\n"
            "    return hashlib.sha256(str(payload).encode()).hexdigest()\n",
            encoding="utf-8",
        )

        assert main(["--root", str(tmp_path)]) == 1
        err = capsys.readouterr().err
        assert "docs/adr/README.md" in err
        assert "adr" in err.lower()


class TestCli:
    def test_exit_code_is_zero_when_clean(self) -> None:
        assert main(["--root", str(REPO_ROOT)]) == 0

    def test_list_mode_always_succeeds(self) -> None:
        assert main(["--root", str(REPO_ROOT), "--list"]) == 0

    def test_fails_on_an_injected_violation(self, tmp_path: Path) -> None:
        """End-to-end: a new direct digest breaks the build."""
        (tmp_path / "predictron_engine").mkdir()
        (tmp_path / "predictron_engine" / "offender.py").write_text(
            "import hashlib\n"
            "def _report_digest(payload):\n"
            "    return hashlib.sha256(str(payload).encode()).hexdigest()\n",
            encoding="utf-8",
        )
        assert main(["--root", str(tmp_path)]) == 1

    def test_module_runs_as_a_script(self) -> None:
        """CI invokes this via ``python -m``, so that path must work."""
        result = subprocess.run(
            [sys.executable, "-m", "tools.check_canonical_invariant"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert UNPARSEABLE_SENTINEL not in result.stdout


class TestGuardsAgainstBeingWeakened:
    def test_canonical_module_is_still_allowed(self) -> None:
        """The rule must not forbid the module that defines the encoding."""
        assert "predictron_engine/canonical.py" in ALLOWED_MODULES

    def test_canonical_module_actually_uses_the_primitives(self) -> None:
        """Otherwise the exemption is hiding an empty file."""
        source = (REPO_ROOT / "predictron_engine" / "canonical.py").read_text(
            encoding="utf-8-sig"
        )
        assert "hashlib.sha256" in source
