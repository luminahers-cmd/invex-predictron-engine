"""Comprehensive tests for quantitative parsers (Sprint 14).

Tests cover all parser functions with deterministic, traceable assertions.
"""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# parse_dollar_amount
# ---------------------------------------------------------------------------


class TestParseDollarAmount:
    def test_plain_number(self) -> None:
        assert parse_dollar_amount("$500") == 500.0

    def test_thousands_suffix_k(self) -> None:
        assert parse_dollar_amount("$50K") == 50_000.0

    def test_thousands_suffix_thousand(self) -> None:
        assert parse_dollar_amount("$50 thousand") == 50_000.0

    def test_millions_suffix_m(self) -> None:
        assert parse_dollar_amount("$12M") == 12_000_000.0

    def test_millions_suffix_million(self) -> None:
        assert parse_dollar_amount("$45.5 million") == 45_500_000.0

    def test_billions_suffix_b(self) -> None:
        assert parse_dollar_amount("$1.2B") == 1_200_000_000.0

    def test_billions_suffix_billion(self) -> None:
        assert parse_dollar_amount("$2.5 billion") == 2_500_000_000.0

    def test_trillions(self) -> None:
        assert parse_dollar_amount("$1T") == 1_000_000_000_000.0

    def test_trillions_suffix_trillion(self) -> None:
        assert parse_dollar_amount("$3 trillion") == 3_000_000_000_000.0

    def test_comma_formatted(self) -> None:
        assert parse_dollar_amount("$1,200,000") == 1_200_000.0

    def test_comma_with_suffix(self) -> None:
        assert parse_dollar_amount("$1,200K") == 1_200_000.0

    def test_decimal_value(self) -> None:
        assert parse_dollar_amount("$1.5M") == 1_500_000.0

    def test_no_dollar_sign_returns_none(self) -> None:
        assert parse_dollar_amount("12M") is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_dollar_amount("") is None

    def test_no_number_returns_none(self) -> None:
        assert parse_dollar_amount("no numbers here") is None

    def test_space_after_dollar(self) -> None:
        assert parse_dollar_amount("$ 12M") == 12_000_000.0

    def test_returns_first_match(self) -> None:
        result = parse_dollar_amount("raised $5M and then $10M")
        assert result == 5_000_000.0


# ---------------------------------------------------------------------------
# parse_percentage
# ---------------------------------------------------------------------------


class TestParsePercentage:
    def test_integer_percentage(self) -> None:
        assert parse_percentage("42%") == 42.0

    def test_decimal_percentage(self) -> None:
        assert parse_percentage("3.5%") == 3.5

    def test_percentage_with_text(self) -> None:
        assert parse_percentage("grew 120% MoM") == 120.0

    def test_comma_formatted_percentage(self) -> None:
        assert parse_percentage("1,200%") == 1200.0

    def test_no_percentage_returns_none(self) -> None:
        assert parse_percentage("no percent here") is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_percentage("") is None

    def test_returns_first_match(self) -> None:
        result = parse_percentage("42% growth and 10% churn")
        assert result == 42.0


# ---------------------------------------------------------------------------
# parse_integer
# ---------------------------------------------------------------------------


class TestParseInteger:
    def test_simple_integer(self) -> None:
        assert parse_integer("42") == 42

    def test_integer_in_text(self) -> None:
        assert parse_integer("team of 42 people") == 42

    def test_comma_formatted(self) -> None:
        assert parse_integer("1,000 customers") == 1000

    def test_large_comma_formatted(self) -> None:
        assert parse_integer("1,200,000 users") == 1_200_000

    def test_no_integer_returns_none(self) -> None:
        assert parse_integer("no numbers") is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_integer("") is None


# ---------------------------------------------------------------------------
# parse_year
# ---------------------------------------------------------------------------


class TestParseYear:
    def test_founded_in_year(self) -> None:
        assert parse_year("Founded in 2019") == 2019

    def test_established_year(self) -> None:
        assert parse_year("Established 2020") == 2020

    def test_started_year(self) -> None:
        assert parse_year("Started in 2015") == 2015

    def test_incorporated_year(self) -> None:
        assert parse_year("Incorporated 2018") == 2018

    def test_launched_year(self) -> None:
        assert parse_year("Launched in 2021") == 2021

    def test_est_dot(self) -> None:
        assert parse_year("est. 2017") == 2017

    def test_standalone_year(self) -> None:
        assert parse_year("The company was created in 2019") == 2019

    def test_invalid_year_returns_none(self) -> None:
        assert parse_year("founded in 1800") is None

    def test_future_year_returns_none(self) -> None:
        assert parse_year("founded in 2200") is None

    def test_no_year_returns_none(self) -> None:
        assert parse_year("no year here") is None


# ---------------------------------------------------------------------------
# parse_funding_amount
# ---------------------------------------------------------------------------


class TestParseFundingAmount:
    def test_raised_amount(self) -> None:
        assert parse_funding_amount("raised $12M") == 12_000_000.0

    def test_raised_with_suffix(self) -> None:
        assert parse_funding_amount("raised $45.5 million") == 45_500_000.0

    def test_series_round(self) -> None:
        assert parse_funding_amount("$20M Series A round") == 20_000_000.0

    def test_seed_round(self) -> None:
        assert parse_funding_amount("$5M seed round") == 5_000_000.0

    def test_total_funding(self) -> None:
        assert parse_funding_amount("total funding of $100M") == 100_000_000.0

    def test_funded_through(self) -> None:
        assert parse_funding_amount("$50M in total funding") == 50_000_000.0

    def test_no_funding_returns_none(self) -> None:
        assert parse_funding_amount("no funding mentioned") is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_funding_amount("") is None

    def test_billions(self) -> None:
        assert parse_funding_amount("raised $1.2B") == 1_200_000_000.0


# ---------------------------------------------------------------------------
# parse_arr
# ---------------------------------------------------------------------------


class TestParseArr:
    def test_arr_with_prefix(self) -> None:
        assert parse_arr("ARR of $4.2M") == 4_200_000.0

    def test_arr_with_suffix(self) -> None:
        assert parse_arr("$1.2B in ARR") == 1_200_000_000.0

    def test_arr_reaching(self) -> None:
        assert parse_arr("ARR reaching $500K") == 500_000.0

    def test_arr_exceeding(self) -> None:
        assert parse_arr("ARR exceeding $10M") == 10_000_000.0

    def test_arr_at(self) -> None:
        assert parse_arr("ARR at $2M") == 2_000_000.0

    def test_no_arr_returns_none(self) -> None:
        assert parse_arr("no arr mentioned") is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_arr("") is None


# ---------------------------------------------------------------------------
# parse_mrr
# ---------------------------------------------------------------------------


class TestParseMrr:
    def test_mrr_with_prefix(self) -> None:
        assert parse_mrr("MRR of $50K") == 50_000.0

    def test_mrr_with_suffix(self) -> None:
        assert parse_mrr("$120,000 MRR") == 120_000.0

    def test_mrr_reaching(self) -> None:
        assert parse_mrr("MRR reaching $25K") == 25_000.0

    def test_no_mrr_returns_none(self) -> None:
        assert parse_mrr("no mrr mentioned") is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_mrr("") is None


# ---------------------------------------------------------------------------
# parse_gmv
# ---------------------------------------------------------------------------


class TestParseGmv:
    def test_gmv_with_prefix(self) -> None:
        assert parse_gmv("$500M GMV") == 500_000_000.0

    def test_gmv_in_total(self) -> None:
        assert parse_gmv("$1.2B in GMV") == 1_200_000_000.0

    def test_gmv_of(self) -> None:
        assert parse_gmv("GMV of $200M") == 200_000_000.0

    def test_no_gmv_returns_none(self) -> None:
        assert parse_gmv("no gmv mentioned") is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_gmv("") is None


# ---------------------------------------------------------------------------
# parse_customer_count
# ---------------------------------------------------------------------------


class TestParseCustomerCount:
    def test_enterprise_customers(self) -> None:
        assert parse_customer_count("500 enterprise customers") == 500

    def test_clients(self) -> None:
        assert parse_customer_count("1,000 clients") == 1000

    def test_accounts(self) -> None:
        assert parse_customer_count("200 accounts") == 200

    def test_with_suffix_k(self) -> None:
        assert parse_customer_count("50K customers") == 50_000

    def test_with_suffix_m(self) -> None:
        assert parse_customer_count("2M customers") == 2_000_000

    def test_active_customers(self) -> None:
        assert parse_customer_count("100 active customers") == 100

    def test_no_customers_returns_none(self) -> None:
        assert parse_customer_count("no customers") is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_customer_count("") is None


# ---------------------------------------------------------------------------
# parse_active_user_count
# ---------------------------------------------------------------------------


class TestParseActiveUserCount:
    def test_mau_with_prefix(self) -> None:
        assert parse_active_user_count("10,000 monthly active users") == 10_000

    def test_mau_suffix_k(self) -> None:
        assert parse_active_user_count("500K monthly active users") == 500_000

    def test_dau(self) -> None:
        assert parse_active_user_count("50,000 daily active users") == 50_000

    def test_reversed_mau(self) -> None:
        assert parse_active_user_count("MAU of 25K") == 25_000

    def test_users_simple(self) -> None:
        assert parse_active_user_count("100,000 users") == 100_000

    def test_downloads(self) -> None:
        assert parse_active_user_count("1M downloads") == 1_000_000

    def test_no_users_returns_none(self) -> None:
        assert parse_active_user_count("no users") is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_active_user_count("") is None


# ---------------------------------------------------------------------------
# parse_growth_rate
# ---------------------------------------------------------------------------


class TestParseGrowthRate:
    def test_mom_growth(self) -> None:
        assert parse_growth_rate("150% MoM growth") == 150.0

    def test_month_over_month(self) -> None:
        assert parse_growth_rate("20% month-over-month growth") == 20.0

    def test_yoy_growth(self) -> None:
        assert parse_growth_rate("30% YoY growth") == 30.0

    def test_year_over_year(self) -> None:
        assert parse_growth_rate("50% year-over-year growth") == 50.0

    def test_growth_of(self) -> None:
        assert parse_growth_rate("growth of 25%") == 25.0

    def test_growing_at(self) -> None:
        assert parse_growth_rate("growing at 40%") == 40.0

    def test_grew_at(self) -> None:
        assert parse_growth_rate("grew at 35%") == 35.0

    def test_no_growth_returns_none(self) -> None:
        assert parse_growth_rate("no growth mentioned") is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_growth_rate("") is None


# ---------------------------------------------------------------------------
# parse_market_size
# ---------------------------------------------------------------------------


class TestParseMarketSize:
    def test_tam_with_prefix(self) -> None:
        assert parse_market_size("$50 billion TAM") == 50_000_000_000.0

    def test_sam_with_prefix(self) -> None:
        assert parse_market_size("$5B SAM") == 5_000_000_000.0

    def test_som_with_prefix(self) -> None:
        assert parse_market_size("$200M SOM") == 200_000_000.0

    def test_tam_reversed(self) -> None:
        assert parse_market_size("TAM of $10B") == 10_000_000_000.0

    def test_market_size_reversed(self) -> None:
        assert parse_market_size("market size of $1T") == 1_000_000_000_000.0

    def test_total_addressable_market(self) -> None:
        assert parse_market_size("$20B total addressable market") == 20_000_000_000.0

    def test_no_market_returns_none(self) -> None:
        assert parse_market_size("no market size") is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_market_size("") is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_dollar_amount_zero(self) -> None:
        assert parse_dollar_amount("$0") == 0.0

    def test_percentage_zero(self) -> None:
        assert parse_percentage("0%") == 0.0

    def test_large_numbers(self) -> None:
        assert parse_dollar_amount("$999,999,999,999") == 999_999_999_999.0

    def test_whitespace_handling(self) -> None:
        assert parse_dollar_amount("  $12M  ") == 12_000_000.0

    def test_case_insensitive_multipliers(self) -> None:
        assert parse_dollar_amount("$12m") == 12_000_000.0
        assert parse_dollar_amount("$12M") == 12_000_000.0

    def test_all_parsers_return_none_on_empty(self) -> None:
        assert parse_dollar_amount("") is None
        assert parse_percentage("") is None
        assert parse_integer("") is None
        assert parse_year("") is None
        assert parse_funding_amount("") is None
        assert parse_arr("") is None
        assert parse_mrr("") is None
        assert parse_gmv("") is None
        assert parse_customer_count("") is None
        assert parse_active_user_count("") is None
        assert parse_growth_rate("") is None
        assert parse_market_size("") is None
