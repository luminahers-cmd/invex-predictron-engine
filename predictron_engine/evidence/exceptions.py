"""Exception hierarchy for the Evidence Collection Layer.

All errors raised inside this subsystem derive from
:class:`EvidenceCollectionError` so callers can handle them uniformly
without coupling to implementation details.
"""


class EvidenceCollectionError(Exception):
    """Base class for every error raised by the evidence collection layer."""


class InvalidWebsiteError(EvidenceCollectionError):
    """Raised when the supplied website URL cannot be parsed or used."""


class DiscoveryError(EvidenceCollectionError):
    """Raised when page discovery fails irrecoverably."""


class FetchError(EvidenceCollectionError):
    """Reserved for fatal fetch-layer errors.

    Page-level failures are intentionally returned as ``FetchResult``
    records instead of being raised, so collection can continue with
    partial evidence.
    """


class CleanError(EvidenceCollectionError):
    """Raised when raw HTML cannot be parsed or cleaned."""
