"""
PRF (Pasture, Rangeland, Forage) Data Loaders — for Unified Optimizer.
All Snowflake data access for PRF insurance calculations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from prf_constants import (
    INTERVAL_ORDER_11, INTENDED_USE_CODE_MAP,
    MIN_ALLOCATION, MAX_ALLOCATION,
    extract_numeric_grid_id, get_practice_codes,
)


def _get_session():
    """Get the active Snowpark session."""
    from snowflake.snowpark.context import get_active_session
    return get_active_session()


@st.cache_data(ttl=86400, show_spinner=False)
def load_distinct_grids():
    """Load all distinct PRF grid IDs from county base values table."""
    session = _get_session()
    df = session.sql("""
        SELECT DISTINCT GRID_ID
        FROM PRF_COUNTY_BASE_VALUES
        ORDER BY GRID_ID
    """).to_pandas()
    return sorted(df['GRID_ID'].tolist())


@st.cache_data(ttl=3600, show_spinner=False)
def load_county_base_value(grid_id, intended_use='Grazing',
                            irrigation_practice='N/A',
                            organic_practice='No Organic Practice Specified'):
    """
    Load PRF County Base Value for a grid + practice combination.
    Returns a float (average CBV).
    """
    session = _get_session()
    codes = get_practice_codes(intended_use, irrigation_practice, organic_practice)

    if codes['irrigation_name_sql'] is None:
        irrigation_filter = "AND IRRIGATION_PRACTICE_CODE = 997"
    else:
        irrigation_filter = f"AND IRRIGATION_PRACTICE_CODE = {codes['irrigation_code']}"

    df = session.sql(f"""
        SELECT AVG(COUNTY_BASE_VALUE)
        FROM PRF_COUNTY_BASE_VALUES
        WHERE GRID_ID = '{grid_id}'
          AND INTENDED_USE = '{intended_use}'
          {irrigation_filter}
          AND ORGANIC_PRACTICE_CODE = {codes['organic_code']}
    """).to_pandas()

    if df.empty or df.iloc[0, 0] is None:
        return None
    return float(df.iloc[0, 0])


@st.cache_data(ttl=3600, show_spinner=False)
def load_all_indices(grid_id):
    """
    Load all historical rainfall indices for a grid across all 11 intervals.

    Returns:
        DataFrame with columns: YEAR, plus one column per interval name
    """
    numeric_grid_id = extract_numeric_grid_id(grid_id)
    session = _get_session()
    df = session.sql(f"""
        SELECT YEAR, INTERVAL_NAME, INDEX_VALUE
        FROM RAIN_INDEX_PLATINUM_ENHANCED
        WHERE GRID_ID = {numeric_grid_id}
        ORDER BY YEAR
    """).to_pandas()

    if df.empty:
        return pd.DataFrame()

    # Drop rows with missing index values (matches reference)
    df = df.dropna(subset=['INDEX_VALUE'])

    pivot = df.pivot_table(index='YEAR', columns='INTERVAL_NAME',
                           values='INDEX_VALUE', aggfunc='first')
    available = [iv for iv in INTERVAL_ORDER_11 if iv in pivot.columns]
    pivot = pivot[available]
    pivot = pivot.reset_index()
    return pivot


@st.cache_data(ttl=3600, show_spinner=False)
def load_premium_rates(grid_id, intended_use, coverage_levels,
                        rate_year, irrigation_practice='N/A',
                        organic_practice='No Organic Practice Specified'):
    """
    Load premium rates for a grid, returning a dict keyed by coverage level.

    Returns:
        dict: {coverage_level: {interval_name: rate, ...}, ...}
    """
    session = _get_session()
    numeric_grid_id = extract_numeric_grid_id(grid_id)
    codes = get_practice_codes(intended_use, irrigation_practice, organic_practice)

    all_premiums = {}
    for cov_level in coverage_levels:
        cov_string = f"{cov_level:.0%}"
        df = session.sql(f"""
            SELECT INDEX_INTERVAL_NAME, PREMIUMRATE
            FROM PRF_PREMIUM_RATES_ADM
            WHERE GRID_ID = {numeric_grid_id}
              AND INTENDED_USE = '{intended_use}'
              AND IRRIGATION_PRACTICE_CODE = {codes['irrigation_code']}
              AND ORGANIC_PRACTICE_CODE = {codes['organic_code']}
              AND COVERAGE_LEVEL = '{cov_string}'
              AND YEAR = {rate_year}
        """).to_pandas()
        df['PREMIUMRATE'] = pd.to_numeric(df['PREMIUMRATE'], errors='coerce')
        all_premiums[cov_level] = df.set_index('INDEX_INTERVAL_NAME')['PREMIUMRATE'].to_dict()
    return all_premiums


@st.cache_data(ttl=3600, show_spinner=False)
def load_subsidies(plan_code=13, coverage_levels=None):
    """
    Load federal subsidy percentages for PRF.

    Returns:
        dict: {coverage_level: subsidy_pct, ...}
    """
    session = _get_session()
    all_subsidies = {}
    for cov_level in coverage_levels:
        df = session.sql(f"""
            SELECT SUBSIDY_PERCENT
            FROM SUBSIDYPERCENT_YTD_PLATINUM
            WHERE INSURANCE_PLAN_CODE = {plan_code}
              AND COVERAGE_LEVEL_PERCENT = {cov_level}
            LIMIT 1
        """).to_pandas()
        all_subsidies[cov_level] = float(df.iloc[0, 0]) if not df.empty else 0.55
    return all_subsidies


@st.cache_data(ttl=3600, show_spinner=False)
def load_coverage_intervals():
    """
    Load PRF coverage interval availability and bounds.

    Returns:
        dict keyed by (sub_county_code, intended_use_code, irrigation_code, organic_code)
              -> (min_pct, max_pct)
    """
    session = _get_session()

    df = session.sql("""
        SELECT DISTINCT
            SUB_COUNTY_CODE::STRING AS SUB_COUNTY_CODE,
            INTENDED_USE_CODE::INT AS INTENDED_USE_CODE,
            IRRIGATION_PRACTICE_CODE::INT AS IRRIGATION_PRACTICE_CODE,
            ORGANIC_PRACTICE_CODE::INT AS ORGANIC_PRACTICE_CODE,
            MINIMUM_ACRE_PERCENT::FLOAT AS MIN_PCT,
            MAXIMUM_ACRE_PERCENT::FLOAT AS MAX_PCT
        FROM CAPITAL_MARKETS_SANDBOX.PUBLIC.PRF_COVERAGE_INTERVALS
    """).to_pandas()

    return {
        (str(r.SUB_COUNTY_CODE), int(r.INTENDED_USE_CODE),
         int(r.IRRIGATION_PRACTICE_CODE), int(r.ORGANIC_PRACTICE_CODE)):
        (float(r.MIN_PCT), float(r.MAX_PCT))
        for r in df.itertuples()
    }


def get_allocation_bounds(grid_id, intended_use, irrigation_practice,
                          organic_practice, intervals_cache):
    """
    Extract min/max allocation bounds from coverage interval info.

    Parameters:
        grid_id: grid ID string or label
        intended_use: 'Grazing' or 'Haying'
        irrigation_practice, organic_practice: practice strings
        intervals_cache: dict from load_coverage_intervals

    Returns:
        (min_alloc, max_alloc) tuple
    """
    sub_county_code = str(grid_id).split('(')[0].strip()
    use_code = INTENDED_USE_CODE_MAP.get(intended_use, 7)
    codes = get_practice_codes(intended_use, irrigation_practice, organic_practice)

    key = (sub_county_code, use_code, codes['irrigation_code'], codes['organic_code'])

    if key in intervals_cache:
        return intervals_cache[key]

    return (MIN_ALLOCATION, MAX_ALLOCATION)


@st.cache_data(ttl=3600, show_spinner=False)
def get_current_rate_year():
    """Query the maximum available rate year from Snowflake."""
    try:
        session = _get_session()
        df = session.sql("SELECT MAX(YEAR) FROM PRF_PREMIUM_RATES_ADM").to_pandas()
        if not df.empty and df.iloc[0, 0] is not None:
            return int(df.iloc[0, 0])
    except Exception:
        pass
    return 2026  # fallback
