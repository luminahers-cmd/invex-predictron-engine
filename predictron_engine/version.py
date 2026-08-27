"""Canonical engine version.

Single source of truth for the Predictron Engine version.  All modules
that surface the engine version import from here instead of duplicating
the literal.
"""

ENGINE_VERSION = "0.12.1"

__all__ = ["ENGINE_VERSION"]
