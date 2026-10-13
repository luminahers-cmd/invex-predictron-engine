"""Live adapters that sit outside the pure research and engine layers.

These modules wire external subsystems (for example the async evidence
orchestrator) into the research layer's protocols.  They are kept out of
``predictron_engine.research`` so that package stays free of I/O.
"""

from __future__ import annotations

__all__: list[str] = []
