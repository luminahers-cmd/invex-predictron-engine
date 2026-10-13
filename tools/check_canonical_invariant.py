"""Enforce the canonical-hashing architectural invariant.

Invariant
---------
Every content-addressed identifier produced anywhere in this repository
must ultimately be computed through
:mod:`predictron_engine.canonical`.  No other module may implement
content hashing or canonical JSON serialization on its own.  A new
digest helper is permitted only as a *thin wrapper* — it must delegate to
``predictron_engine.canonical`` rather than reimplement the logic.

Why this is a build failure and not a review convention
------------------------------------------------------
The repository once had eleven compact-canonical call sites speaking
three dialects at three digest widths.  Nothing about that looked wrong
to a reader: each site was a reasonable-looking ``hashlib.sha256`` over
``json.dumps(..., sort_keys=True)``.  The failure was invisible because
two subsystems encoding the same logical fact computed two different
digests, and only a replay run comparing them would ever notice.

A digest that disagrees with its sibling is not a style problem, it is a
silent correctness bug.  So the guard is mechanical.

What it checks
--------------
``HASH_PRIMITIVE``
    A direct call into a ``hashlib`` construction function.
``SORTED_JSON``
    A ``json.dumps``/``json.dump`` call with ``sort_keys`` enabled, which
    is the canonical-serialization dialect.
``BUILTIN_HASH``
    A call to Python's built-in ``hash()``.  It is salted per process
    for ``str``, so it can never appear in a persisted identifier.
``HEXDIGEST``
    A ``.hexdigest()`` call outside the low-level module — the tail of a
    hand-rolled digest.
``UNWRAPPED_DIGEST_HELPER``
    A function whose name says it computes a digest, which performs
    hashing work, but which never delegates to
    ``predictron_engine.canonical``.  This is the rule that makes "thin
    wrapper" enforceable rather than aspirational.

    The rule deliberately requires *both* a digest-shaped name and
    actual hashing work in the body.  Either alone is too noisy:
    ``_recommendation_fingerprint`` is a name without hashing (it
    returns a normalized dedup key), and ``_normalise_text_for_hash`` is
    preprocessing that feeds a digest elsewhere.  Both are legitimate,
    and a guard that cries wolf gets disabled.

Suppressions
------------
Every suppression must state a reason.  A suppression without one is
itself a violation, because the reason is the part a reviewer actually
needs to read.

Adding a suppression relaxes an architectural invariant, which the
repository's evolution policy requires to be recorded in an ADR (see
``docs/adr/README.md``).  The checker does not parse ADRs — it cannot
tell you whether the motivation was sound — so the gate is review plus
that record, and this tool only makes sure the record has not been
skipped by accident.

Usage
-----
    python -m tools.check_canonical_invariant          # check, exit 1 on findings
    python -m tools.check_canonical_invariant --list   # list findings, always exit 0
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

#: Modules permitted to talk to ``hashlib`` and ``json.dumps`` directly.
#: This is the definition of the canonical encoding itself, so it is the
#: one place the primitive operations belong.
ALLOWED_MODULES: frozenset[str] = frozenset(
    {
        "predictron_engine/canonical.py",
        "tools/check_canonical_invariant.py",
    }
)

#: Path prefixes that are not call sites in the sense of the invariant.
#: Tests deliberately construct the legacy expressions in order to assert
#: the new helpers reproduce them byte-for-byte; that assertion is the
#: whole point, and production code cannot reach into ``tests``.
EXEMPT_PREFIXES: tuple[str, ...] = ("tests/",)

#: Dunder methods that implement the Python object protocol rather than
#: producing a persisted identifier.  ``__hash__`` returns a salted
#: ``int`` that CPython is free to collide and truncate; it is never
#: serialized, so the invariant does not apply to it.
EXEMPT_SYMBOLS: frozenset[str] = frozenset({"__hash__"})

#: Marker asserted absent from clean output by the guard's own tests.
UNPARSEABLE_SENTINEL = "UNPARSEABLE"

#: Splits ``camelCase`` into word segments without misreading a run of
#: capitals (``HTTPServer``) as one word.
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

#: Tree roots scanned by default.
DEFAULT_ROOTS: tuple[str, ...] = (
    "predictron_engine",
    "app",
    "benchmarks",
    "predictron_sdk",
    "scripts",
    "tools",
)

#: ``hashlib`` attributes that construct a message digest.
HASH_CONSTRUCTORS: frozenset[str] = frozenset(
    {
        "blake2b",
        "blake2s",
        "md5",
        "sha1",
        "sha224",
        "sha256",
        "sha384",
        "sha3_256",
        "sha3_512",
        "sha512",
    }
)

#: Functions exported by ``predictron_engine.canonical``.  A wrapper that
#: calls any of these is delegating, even under an import alias — the
#: research validators import ``content_digest`` as
#: ``canonical_content_digest``, and that is still a thin wrapper.
CANONICAL_SYMBOLS: frozenset[str] = frozenset(
    {"canonical_json", "content_digest", "text_digest"}
)

#: The module whose exports count as canonical, for alias resolution.
CANONICAL_MODULE = "predictron_engine.canonical"

#: Trailing terms that mark a function as digest-shaped.  See
#: :func:`_is_digest_name` for how a name is decomposed before matching.
#:
#: Deliberately broad.  ``_snapshot_id`` and ``_company_id`` are exactly
#: the thin wrappers the invariant wants to keep thin, and a narrower
#: list would exempt them from scrutiny.  The *hashing* condition in the
#: rule does the real filtering; name matching only nominates candidates.
_DIGEST_TERMS: frozenset[str] = frozenset(
    {"digest", "hash", "fingerprint", "checksum", "id"}
)

#: Words that turn a digest term into a description of an input rather
#: than a function that produces one.
_DIGEST_TARGET_WORDS: frozenset[str] = frozenset(
    {"for", "to", "from", "before", "after", "into"}
)


@dataclass(frozen=True)
class Suppression:
    """An approved exception, with the justification kept next to it."""

    path: str
    symbol: str
    reason: str


#: Approved exceptions.  Each one is a case where the primitive is not
#: producing a content-addressed identifier, or where the module *is* the
#: canonical definition.
SUPPRESSIONS: tuple[Suppression, ...] = (
    Suppression(
        path="predictron_engine/dataset/acquisition/state.py",
        symbol="compute_file_hash",
        reason=(
            "Digests raw file bytes for change detection, not the canonical "
            "encoding of a structure: there is no JSON step to canonicalize, "
            "and streaming avoids loading an arbitrarily large artifact into "
            "memory."
        ),
    ),
)


@dataclass(frozen=True)
class Finding:
    """One violation of the invariant.

    ``symbol`` names the enclosing function (or ``"<module>"``) so a
    :class:`Suppression` can target one function in a module without
    having to match on line numbers, which shift on every edit.
    """

    path: str
    line: int
    rule: str
    symbol: str
    detail: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.detail}"


class _ModuleChecker(ast.NodeVisitor):
    """Collects invariant violations from one parsed module."""

    def __init__(self, relpath: str) -> None:
        self.relpath = relpath
        self.findings: list[Finding] = []
        self._function_stack: list[str] = []
        #: name -> True when the body delegates to predictron_engine.canonical
        self._delegates: dict[str, bool] = {}
        #: name -> True when the body performs hashing work of its own
        self._hashes: dict[str, bool] = {}
        #: local name -> True when bound to a canonical export
        self._canonical_aliases: set[str] = set()

    # -- helpers ---------------------------------------------------------

    def _add(self, node: ast.AST, rule: str, detail: str) -> None:
        scope = self._scope()
        if scope in EXEMPT_SYMBOLS:
            return
        self.findings.append(
            Finding(
                path=self.relpath,
                line=getattr(node, "lineno", 0),
                rule=rule,
                symbol=scope,
                detail=detail,
            )
        )

    def _scope(self) -> str:
        return self._function_stack[-1] if self._function_stack else "<module>"

    def _mark_hashing(self) -> None:
        """Record that the enclosing function performs hashing work itself."""
        scope = self._scope()
        if scope in self._hashes:
            self._hashes[scope] = True

    # -- function bodies -------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_stack.append(node.name)
        self._delegates.setdefault(node.name, False)
        self._hashes.setdefault(node.name, False)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function_stack.append(node.name)
        self._delegates.setdefault(node.name, False)
        self._hashes.setdefault(node.name, False)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == CANONICAL_MODULE:
            for alias in node.names:
                self._canonical_aliases.add(alias.asname or alias.name)
        elif node.module == "hashlib" and any(
            a.name in HASH_CONSTRUCTORS for a in node.names
        ):
            self._add(
                node,
                "HASH_PRIMITIVE",
                "direct hashlib import; route through "
                "predictron_engine.canonical",
            )
            self._mark_hashing()
        self.generic_visit(node)

    # -- call sites ------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        # hashlib.<constructor>(...)
        if isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "hashlib"
                and node.func.attr in HASH_CONSTRUCTORS
            ):
                self._add(
                    node,
                    "HASH_PRIMITIVE",
                    f"hashlib.{node.func.attr}() called directly; use "
                    "predictron_engine.canonical.content_digest/text_digest",
                )
                self._mark_hashing()
            elif node.func.attr == "hexdigest":
                self._add(
                    node,
                    "HEXDIGEST",
                    ".hexdigest() outside the canonical module; "
                    "return the value from content_digest/text_digest",
                )
                self._mark_hashing()

        # json.dumps(..., sort_keys=True) / json.dump(..., sort_keys=True)
        #
        # Only the *compact canonical* dialect is a violation.  An
        # ``indent=2`` dump is a human-readable report, not an encoding
        # that feeds a digest, and demanding it route through
        # ``canonical_json`` would force pretty-printers to hand-assemble
        # their own whitespace.  The tell is ``separators`` being passed
        # explicitly, which only compact canonical output does.
        if _is_json_serializer(node.func):
            for keyword in node.keywords:
                if keyword.arg == "sort_keys" and not _is_falsy(keyword.value):
                    if any(kw.arg == "separators" for kw in node.keywords):
                        self._add(
                            node,
                            "SORTED_JSON",
                            "compact json.dumps(sort_keys=..., separators=...) "
                            "reimplements canonical serialization; use "
                            "predictron_engine.canonical.canonical_json",
                        )
                        self._mark_hashing()
                    break

        # built-in hash()
        if isinstance(node.func, ast.Name) and node.func.id == "hash":
            self._add(
                node,
                "BUILTIN_HASH",
                "built-in hash() is salted per process for str and cannot "
                "appear in a persisted identifier; use content_digest/text_digest",
            )
            self._mark_hashing()

        # Delegation detection, alias-aware.
        if isinstance(node.func, ast.Name) and (
            node.func.id in CANONICAL_SYMBOLS or node.func.id in self._canonical_aliases
        ):
            self._delegates[self._scope()] = True

        self.generic_visit(node)

    # -- reporting -------------------------------------------------------

    def finish(self) -> list[Finding]:
        """Add wrapper findings and return everything, sorted."""
        for name, hashes in self._hashes.items():
            if name in EXEMPT_SYMBOLS:
                continue
            if not hashes or not _is_digest_name(name):
                continue
            if self._delegates.get(name):
                continue
            self.findings.append(
                Finding(
                    path=self.relpath,
                    line=0,
                    rule="UNWRAPPED_DIGEST_HELPER",
                    symbol=name,
                    detail=(
                        f"{name}() is named like a digest helper and hashes "
                        "directly, but never calls predictron_engine.canonical; "
                        "a new digest helper must be a thin wrapper that "
                        "delegates"
                    ),
                )
            )
        return sorted(
            self.findings, key=lambda f: (f.rule, f.path, f.line, f.detail)
        )


def _is_json_serializer(func: ast.expr) -> bool:
    if not isinstance(func, ast.Attribute) or func.attr not in {"dump", "dumps"}:
        return False
    return isinstance(func.value, ast.Name) and func.value.id == "json"


def _is_falsy(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and not node.value


def _is_digest_name(name: str) -> bool:
    """Return whether ``name`` reads as computing a digest.

    Matched on underscore- or camel-delimited segments rather than a bare
    suffix, so ``_normalise_text_for_hash`` (preprocessing that feeds a
    digest) and ``_recommendation_fingerprint`` (a normalized dedup key)
    are not mistaken for digest helpers.  What is left is a function that
    claims to *be* a digest.
    """
    segments = _name_segments(name)
    if not segments:
        return False
    tail = segments[-1]
    if tail not in _DIGEST_TERMS:
        return False
    # ``..._for_hash`` / ``..._to_hash`` describe an input, not a result.
    return len(segments) < 2 or segments[-2] not in _DIGEST_TARGET_WORDS


def _name_segments(name: str) -> list[str]:
    """Split an identifier into lowercase word segments."""
    spaced = _CAMEL_BOUNDARY.sub("_", name.lstrip("_"))
    return [part.lower() for part in spaced.split("_") if part]


def _is_exempt(relpath: str) -> bool:
    return relpath in ALLOWED_MODULES or relpath.startswith(EXEMPT_PREFIXES)


def _is_suppressed(relpath: str, symbol: str) -> bool:
    return any(
        s.path == relpath and s.symbol == symbol and s.reason.strip()
        for s in SUPPRESSIONS
    )


def _suppressions_without_reasons() -> list[str]:
    return [
        f"{s.path}:{s.symbol}" for s in SUPPRESSIONS if not s.reason.strip()
    ]


def iter_python_files(roots: Sequence[str], base_dir: Path) -> Iterator[Path]:
    """Yield every Python file under ``roots``, resolved against ``base_dir``.

    ``base_dir`` is required rather than implied: resolving the roots
    against the process CWD would silently scan the wrong tree whenever
    the tool is invoked from anywhere but the repository root, which for
    a guard is the difference between passing and not running at all.
    """
    for root in roots:
        base = base_dir / root
        if not base.exists():
            continue
        if base.is_file() and base.suffix == ".py":
            yield base
            continue
        yield from sorted(base.rglob("*.py"))


def check_file(path: Path, repo_root: Path) -> list[Finding]:
    """Return invariant findings for one module."""
    try:
        relpath = path.relative_to(repo_root).as_posix()
    except ValueError:
        relpath = path.as_posix()
    if _is_exempt(relpath):
        return []
    # ``utf-8-sig`` rather than ``utf-8``: a few committed modules carry a
    # BOM, and a guard that cannot read them would silently skip them.
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source.lstrip("﻿"), filename=str(path))
    checker = _ModuleChecker(relpath)
    checker.visit(tree)
    findings = checker.finish()
    return [f for f in findings if not _is_suppressed(relpath, f.symbol)]


def check_repo(
    repo_root: Path, roots: Sequence[str] = DEFAULT_ROOTS
) -> list[Finding]:
    """Return every invariant finding in the repository."""
    findings: list[Finding] = []
    for path in iter_python_files(roots, repo_root):
        try:
            findings.extend(check_file(path, repo_root))
        except SyntaxError as exc:  # pragma: no cover - surfaced by ruff/mypy
                findings.append(
                    Finding(
                        path=path.as_posix(),
                        line=exc.lineno or 0,
                        rule="UNPARSEABLE",
                        symbol="<module>",
                        detail=f"could not parse module: {exc.msg}",
                    )
                )
    return sorted(findings, key=lambda f: (f.path, f.line, f.rule))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="print findings but always exit 0",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="repository root (default: current directory)",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.root).resolve()
    unreasoned = _suppressions_without_reasons()
    if unreasoned:
        for entry in unreasoned:
            print(
                f"canonical-invariant: suppression without a reason: {entry}",
                file=sys.stderr,
            )
        print(
            "\nA new suppression relaxes an architectural invariant and "
            "requires reviewer approval plus an Architecture Decision "
            "Record — see docs/adr/README.md.",
            file=sys.stderr,
        )
        return 1

    findings = check_repo(repo_root)

    if findings:
        print(
            "canonical-invariant: content hashing must go through "
            "predictron_engine.canonical",
            file=sys.stderr,
        )
        for finding in findings:
            print(f"  {finding.render()}", file=sys.stderr)
        print(
            f"\n{len(findings)} finding(s). Route new digests through "
            "predictron_engine.canonical, or add a reasoned Suppression in "
            "tools/check_canonical_invariant.py when the primitive is not "
            "producing a content-addressed identifier — a new suppression "
            "relaxing the invariant and needs an ADR (docs/adr/README.md).",
            file=sys.stderr,
        )
        return 0 if args.list else 1

    print("canonical-invariant: ok - all content hashing routes through canonical")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
