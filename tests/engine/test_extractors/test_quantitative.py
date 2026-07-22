"""Comprehensive tests for quantitative parsers (Sprint 14).

Tests cover all parser functions with deterministic, traceable assertions.
"""

from __future__ import annotations

from predictron_engine.extraction.quantitative.parsers import (
    parse_active_user_count,
    parse_arr,
    parse_burn_rate,
    parse_cac,
    parse_churn,
    parse_customer_count,
    parse_dollar_amount,
    parse_funding_amount,
    parse_gmv,
    parse_growth_rate,
    parse_integer,
    parse_ltv,
    parse_market_size,
    parse_mrr,
    parse_nrr,
    parse_percentage,
    parse_runway,
    parse_team_size,
    parse_valuation,
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
        assert parse_valuation("") is None
        assert parse_nrr("") is None
        assert parse_churn("") is None
        assert parse_burn_rate("") is None
        assert parse_runway("") is None
        assert parse_cac("") is None
        assert parse_ltv("") is None
        assert parse_team_size("") is None


# ===========================================================================
# Sprint 14.1 — Valuation extraction (bug fix)
# ===========================================================================


class TestParseValuation:
    """Regression: valuation_usd must extract the valuation, not the first dollar amount."""

    def test_valuation_dollar_suffix(self) -> None:
        assert parse_valuation("$220M valuation") == 220_000_000.0

    def test_valued_at(self) -> None:
        assert parse_valuation("valued at $1.2B") == 1_200_000_000.0

    def test_valuation_of(self) -> None:
        assert parse_valuation("valuation of $500M") == 500_000_000.0

    def test_raised_at_valuation(self) -> None:
        result = parse_valuation("$42M raised at a $220M valuation")
        assert result == 220_000_000.0

    def test_raised_at_valuation_an(self) -> None:
        result = parse_valuation("$10M raised at an $80M valuation")
        assert result == 80_000_000.0

    def test_pre_money_valuation(self) -> None:
        result = parse_valuation("pre-money valuation of $500M")
        assert result == 500_000_000.0

    def test_post_money_valuation(self) -> None:
        result = parse_valuation("post-money valuation $300M")
        assert result == 300_000_000.0

    def test_valuation_small_amount(self) -> None:
        assert parse_valuation("$5M valuation") == 5_000_000.0

    def test_valuation_billions(self) -> None:
        assert parse_valuation("$2B valuation") == 2_000_000_000.0

    def test_no_valuation_returns_none(self) -> None:
        assert parse_valuation("no valuation mentioned") is None

    def test_empty_returns_none(self) -> None:
        assert parse_valuation("") is None

    def test_only_funding_no_valuation_returns_none(self) -> None:
        assert parse_valuation("raised $42M in seed round") is None


# ===========================================================================
# Sprint 14.1 — parse_arr improved coverage
# ===========================================================================


class TestParseArrImproved:
    def test_arr_has_reached(self) -> None:
        assert parse_arr("ARR has reached $5M") == 5_000_000.0

    def test_arr_stands_at(self) -> None:
        assert parse_arr("ARR stands at $3M") == 3_000_000.0

    def test_arr_surpassed(self) -> None:
        assert parse_arr("ARR surpassed $10M") == 10_000_000.0

    def test_arr_grown_to(self) -> None:
        assert parse_arr("ARR grown to $8M") == 8_000_000.0

    def test_annual_recurring_revenue_of(self) -> None:
        assert parse_arr("Annual recurring revenue of $10M") == 10_000_000.0

    def test_annual_recurring_revenue_at(self) -> None:
        assert parse_arr("annual recurring revenue at $2.5M") == 2_500_000.0

    def test_annual_recurring_revenue_reaching(self) -> None:
        assert parse_arr("annual recurring revenue reaching $1M") == 1_000_000.0


# ===========================================================================
# Sprint 14.1 — parse_nrr
# ===========================================================================


class TestParseNrr:
    def test_nrr_of_percent(self) -> None:
        assert parse_nrr("NRR of 120%") == 120.0

    def test_nrr_percent_net_retention(self) -> None:
        assert parse_nrr("130% net revenue retention") == 130.0

    def test_nrr_is_percent(self) -> None:
        assert parse_nrr("NRR is 115%") == 115.0

    def test_nrr_stands_at(self) -> None:
        assert parse_nrr("NRR stands at 125%") == 125.0

    def test_net_revenue_retention_is(self) -> None:
        assert parse_nrr("Net revenue retention is 118%") == 118.0

    def test_net_revenue_retention_stands_at(self) -> None:
        assert parse_nrr("Net revenue retention stands at 140%") == 140.0

    def test_net_revenue_retention_of(self) -> None:
        assert parse_nrr("net revenue retention of 105%") == 105.0

    def test_nrr_decimal(self) -> None:
        assert parse_nrr("NRR of 112.5%") == 112.5

    def test_nrr_no_percent_sign(self) -> None:
        assert parse_nrr("NRR of 135") == 135.0

    def test_no_nrr_returns_none(self) -> None:
        assert parse_nrr("no nrr mentioned") is None

    def test_empty_returns_none(self) -> None:
        assert parse_nrr("") is None


# ===========================================================================
# Sprint 14.1 — parse_churn improved coverage
# ===========================================================================


class TestParseChurnImproved:
    def test_churn_is_percent(self) -> None:
        assert parse_churn("churn is 5%") == 5.0

    def test_churn_is_below_percent(self) -> None:
        assert parse_churn("churn is below 4%") == 4.0

    def test_churn_is_under_percent(self) -> None:
        assert parse_churn("churn is under 3%") == 3.0

    def test_churn_rate_of(self) -> None:
        assert parse_churn("churn rate of 3.2%") == 3.2

    def test_monthly_churn_of(self) -> None:
        assert parse_churn("monthly churn of 2%") == 2.0

    def test_churned_at_rate(self) -> None:
        assert parse_churn("churned at a rate of 8%") == 8.0

    def test_percent_churn_rate(self) -> None:
        assert parse_churn("5% churn rate") == 5.0

    def test_percent_monthly_churn_rate(self) -> None:
        assert parse_churn("2% monthly churn rate") == 2.0

    def test_churn_decimal(self) -> None:
        assert parse_churn("churn is 3.5%") == 3.5

    def test_no_churn_returns_none(self) -> None:
        assert parse_churn("no churn mentioned") is None

    def test_empty_returns_none(self) -> None:
        assert parse_churn("") is None


# ===========================================================================
# Sprint 14.1 — parse_market_size improved coverage
# ===========================================================================


class TestParseMarketSizeImproved:
    def test_dollar_amount_market(self) -> None:
        assert parse_market_size("$5B market") == 5_000_000_000.0

    def test_dollar_amount_addressable_market(self) -> None:
        assert parse_market_size("$200M addressable market") == 200_000_000.0

    def test_dollar_amount_market_sam(self) -> None:
        assert parse_market_size("$50B SAM") == 50_000_000_000.0


# ===========================================================================
# Sprint 14.1 — parse_burn_rate
# ===========================================================================


class TestParseBurnRate:
    def test_monthly_burn_suffix(self) -> None:
        assert parse_burn_rate("$500K monthly burn") == 500_000.0

    def test_burn_rate_of(self) -> None:
        assert parse_burn_rate("burn rate of $1.2M") == 1_200_000.0

    def test_burning_per_month(self) -> None:
        assert parse_burn_rate("burning $200K per month") == 200_000.0

    def test_burning_monthly(self) -> None:
        assert parse_burn_rate("burning $300K monthly") == 300_000.0

    def test_burn_plain(self) -> None:
        assert parse_burn_rate("$1M burn") == 1_000_000.0

    def test_burn_rate_is(self) -> None:
        assert parse_burn_rate("burn rate is $750K") == 750_000.0

    def test_no_burn_returns_none(self) -> None:
        assert parse_burn_rate("no burn mentioned") is None

    def test_empty_returns_none(self) -> None:
        assert parse_burn_rate("") is None


# ===========================================================================
# Sprint 14.1 — parse_runway
# ===========================================================================


class TestParseRunway:
    def test_months_runway(self) -> None:
        assert parse_runway("18 months runway") == 18

    def test_runway_of_months(self) -> None:
        assert parse_runway("runway of 24 months") == 24

    def test_dash_months_runway(self) -> None:
        assert parse_runway("12-month runway") == 12

    def test_months_of_runway(self) -> None:
        assert parse_runway("36 months of runway") == 36

    def test_runway_at_months(self) -> None:
        assert parse_runway("runway at 10 months") == 10

    def test_runway_is_months(self) -> None:
        assert parse_runway("runway is 8 months") == 8

    def test_no_runway_returns_none(self) -> None:
        assert parse_runway("no runway mentioned") is None

    def test_empty_returns_none(self) -> None:
        assert parse_runway("") is None


# ===========================================================================
# Sprint 14.1 — parse_cac
# ===========================================================================


class TestParseCac:
    def test_cac_of_dollar(self) -> None:
        assert parse_cac("CAC of $500") == 500.0

    def test_dollar_cac(self) -> None:
        assert parse_cac("$1,200 CAC") == 1_200.0

    def test_cac_is(self) -> None:
        assert parse_cac("CAC is $800") == 800.0

    def test_cac_suffix_k(self) -> None:
        assert parse_cac("CAC of $2K") == 2_000.0

    def test_customer_acquisition_cost(self) -> None:
        assert parse_cac("customer acquisition cost is $600") == 600.0

    def test_cac_averaging(self) -> None:
        assert parse_cac("CAC averaging $1,500") == 1_500.0

    def test_no_cac_returns_none(self) -> None:
        assert parse_cac("no cac mentioned") is None

    def test_empty_returns_none(self) -> None:
        assert parse_cac("") is None


# ===========================================================================
# Sprint 14.1 — parse_ltv
# ===========================================================================


class TestParseLtv:
    def test_ltv_of_dollar(self) -> None:
        assert parse_ltv("LTV of $5,000") == 5_000.0

    def test_dollar_ltv(self) -> None:
        assert parse_ltv("$12K LTV") == 12_000.0

    def test_ltv_is(self) -> None:
        assert parse_ltv("LTV is $8,000") == 8_000.0

    def test_lifetime_value_of(self) -> None:
        assert parse_ltv("lifetime value of $3,500") == 3_500.0

    def test_lifetime_value_is(self) -> None:
        assert parse_ltv("lifetime value is $15,000") == 15_000.0

    def test_no_ltv_returns_none(self) -> None:
        assert parse_ltv("no ltv mentioned") is None

    def test_empty_returns_none(self) -> None:
        assert parse_ltv("") is None


# ===========================================================================
# Sprint 14.1 — parse_team_size
# ===========================================================================


class TestParseTeamSize:
    def test_team_of_number(self) -> None:
        assert parse_team_size("team of 42") == 42

    def test_person_team(self) -> None:
        assert parse_team_size("50-person team") == 50

    def test_employees(self) -> None:
        assert parse_team_size("200 employees") == 200

    def test_headcount_of(self) -> None:
        assert parse_team_size("headcount of 75") == 75

    def test_person_company(self) -> None:
        assert parse_team_size("30-person company") == 30

    def test_person_startup(self) -> None:
        assert parse_team_size("20 person startup") == 20

    def test_no_team_returns_none(self) -> None:
        assert parse_team_size("no team info") is None

    def test_empty_returns_none(self) -> None:
        assert parse_team_size("") is None


# ===========================================================================
# Sprint 14.1 — Valuation does NOT return the funding amount
# ===========================================================================


class TestValuationVsFundingSeparation:
    """Critical regression: valuation_usd must be distinct from funding_amount_usd."""

    def test_two_amounts_valuation_is_second(self) -> None:
        text = "$42M raised at a $220M valuation"
        assert parse_funding_amount(text) == 42_000_000.0
        assert parse_valuation(text) == 220_000_000.0

    def test_valued_at_not_funding(self) -> None:
        text = "The company raised a seed round and is now valued at $50M"
        assert parse_valuation(text) == 50_000_000.0

    def test_valuation_keyword_required(self) -> None:
        text = "$42M raised in Series A"
        assert parse_valuation(text) is None
        assert parse_funding_amount(text) == 42_000_000.0


# ===========================================================================
# Sprint 14.1 — parse_funding_amount coverage
# ===========================================================================


class TestParseFundingAmountImproved:
    def test_series_b_round(self) -> None:
        assert parse_funding_amount("$30M Series B round") == 30_000_000.0

    def test_pre_seed(self) -> None:
        assert parse_funding_amount("$2M pre-seed round") == 2_000_000.0

    def test_combined_funding(self) -> None:
        assert parse_funding_amount("combined funding of $75M") == 75_000_000.0
