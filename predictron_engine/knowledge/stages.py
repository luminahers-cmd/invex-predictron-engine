"""Funding stage definitions and progression model.

This module defines the standard venture funding lifecycle. It provides
a canonical ordering, display labels, and typical context for each stage.

Usage:
  - The extractor maps raw description text to a FundingStage
  - The reasoning engine uses stage context to generate observations
  - The scoring engine uses stage ordering for comparative analysis
  - The knowledge module serves as the single source of truth for
    what each stage means within the Predictron system
"""

from enum import IntEnum


class FundingStage(IntEnum):
    """Venture funding stages in chronological order.

    Using IntEnum so stages are naturally ordered and comparable:
    FundingStage.PRE_SEED < FundingStage.SEED < ... etc.
    """

    PRE_SEED = 0
    SEED = 1
    SERIES_A = 2
    SERIES_B = 3
    SERIES_C = 4
    SERIES_D = 5
    SERIES_E_PLUS = 6
    GROWTH = 7
    IPO_READY = 8


# Stage metadata — descriptive context only, no scoring logic
STAGE_CONTEXT: dict[FundingStage, dict[str, str]] = {
    FundingStage.PRE_SEED: {
        "label": "Pre-Seed",
        "description": "Earliest stage, typically pre-product or early prototype.",
        "typical_use": (
            "Idea validation, founding team assembly, initial product development."
        ),
    },
    FundingStage.SEED: {
        "label": "Seed",
        "description": "Early product development and initial market testing.",
        "typical_use": (
            "Building MVP, finding product-market fit signals, early customers."
        ),
    },
    FundingStage.SERIES_A: {
        "label": "Series A",
        "description": "Product-market fit emerging, scaling initial traction.",
        "typical_use": (
            "Scaling go-to-market, optimizing unit economics, team growth."
        ),
    },
    FundingStage.SERIES_B: {
        "label": "Series B",
        "description": "Proven model, scaling operations and market expansion.",
        "typical_use": (
            "Market expansion, operational scaling, competitive positioning."
        ),
    },
    FundingStage.SERIES_C: {
        "label": "Series C",
        "description": "Established business accelerating growth.",
        "typical_use": "Aggressive growth, acquisitions, international expansion.",
    },
    FundingStage.SERIES_D: {
        "label": "Series D",
        "description": "Late-stage growth, often pre-IPO positioning.",
        "typical_use": "Final growth round, IPO preparation, market dominance.",
    },
    FundingStage.SERIES_E_PLUS: {
        "label": "Series E+",
        "description": "Very late-stage venture or private equity round.",
        "typical_use": "Late pre-IPO, bridge financing, large-scale expansion.",
    },
    FundingStage.GROWTH: {
        "label": "Growth / Late Stage",
        "description": "Growth equity or private equity stage.",
        "typical_use": "Profitability optimization, IPO readiness, M&A activity.",
    },
    FundingStage.IPO_READY: {
        "label": "IPO Ready",
        "description": "Company is preparing for or actively pursuing public listing.",
        "typical_use": "Public offering, direct listing, SPAC merger.",
    },
}

STAGE_KEYWORDS: dict[FundingStage, list[str]] = {
    FundingStage.PRE_SEED: ["pre-seed", "preseed", "bootstrapped", "idea stage"],
    FundingStage.SEED: ["seed", "angel", "friends and family", "mvp"],
    FundingStage.SERIES_A: ["series a", "series-a", "series a round"],
    FundingStage.SERIES_B: ["series b", "series-b", "series b round"],
    FundingStage.SERIES_C: ["series c", "series-c", "growth round"],
    FundingStage.SERIES_D: ["series d", "series-d"],
    FundingStage.SERIES_E_PLUS: ["series e", "series e+", "late stage"],
    FundingStage.GROWTH: ["growth", "growth equity", "private equity"],
    FundingStage.IPO_READY: ["ipo", "going public", "public offering", "direct listing"],
}
