"""Quantitative parsing — reusable numeric extraction from text.

Deterministic, rule-based parsers for extracting structured numeric
values from startup description text. Shared across all extractors
to avoid duplicated regex logic.

Supported primitives:
  - parse_dollar_amount: "$12M", "$45.5 million", "$1.2B" → float (USD)
  - parse_percentage: "120%", "3.5%" → float
  - parse_integer: "42 employees", "1,000 customers" → int
  - parse_year: "founded in 2019", "est. 2020" → int (year)

Supported domain values:
  - parse_funding_amount: "$12M raised", "$45M Series B" → float (USD)
  - parse_arr: "$4.2M ARR", "ARR of $1.2B" → float (USD)
  - parse_mrr: "$50K MRR", "$120,000 MRR" → float (USD)
  - parse_gmv: "$500M GMV", "$1.2B in GMV" → float (USD)
  - parse_customer_count: "500 enterprise customers" → int
  - parse_active_user_count: "10,000 MAU", "500K DAU" → int
  - parse_growth_rate: "150% MoM growth", "30% YoY growth" → float (0-100 scale)
  - parse_market_size: "$50 billion TAM", "$1T market" → float (USD)

Normalization rules:
  - All dollar amounts normalized to raw USD (e.g. "$12M" → 12_000_000.0)
  - Percentages returned as-is on 0-100 scale (e.g. "42%" → 42.0)
  - Years returned as 4-digit int (e.g. 2019)
  - Counts returned as raw int (e.g. "1,000" → 1000)
  - All functions return None when no match is found (never raise)

Limitations:
  - Only parses text that appears in the description string
  - Does not infer or calculate derived metrics
  - Multiplier suffixes limited to K, M, B, T
  - Assumes US-style number formatting (commas as thousands separator)
"""

from predictron_engine.extraction.quantitative.parsers import (
    parse_active_user_count,
    parse_arr,
    parse_customer_count,
    parse_dollar_amount,
    parse_funding_amount,
    parse_gmv,
    parse_growth_rate,
    parse_integer,
    parse_market_size,
    parse_mrr,
    parse_percentage,
    parse_year,
)

__all__ = [
    "parse_active_user_count",
    "parse_arr",
    "parse_customer_count",
    "parse_dollar_amount",
    "parse_funding_amount",
    "parse_gmv",
    "parse_growth_rate",
    "parse_integer",
    "parse_market_size",
    "parse_mrr",
    "parse_percentage",
    "parse_year",
]
