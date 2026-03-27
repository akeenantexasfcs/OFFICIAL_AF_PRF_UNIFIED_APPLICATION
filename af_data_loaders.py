# ==============================================================================
# DATA LOADERS — Snowflake query functions for AF Decision Support Tool
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from snowflake.snowpark.context import get_active_session
from af_constants import AF_INTERVAL_MATRIX, get_buyup_intervals


def _get_session():
    return get_active_session()


@st.cache_data(ttl=3600, show_spinner=False)
def load_available_states():
    session = _get_session()
    df = session.sql("""
        SELECT DISTINCT STATE_CODE, STATE_NAME, STATE_ABBREVIATION
        FROM AF_COUNTY_BASE_VALUES
        WHERE STATE_NAME IS NOT NULL
        ORDER BY STATE_NAME
    """).to_pandas()
    labels = {}
    for _, row in df.iterrows():
        code = row['STATE_CODE']
        name = row['STATE_NAME']
        abbr = row['STATE_ABBREVIATION']
        labels[code] = f"{name} ({abbr})" if name and abbr else f"State {code}"
    return labels


@st.cache_data(ttl=3600, show_spinner=False)
def load_counties_for_state(state_code):
    session = _get_session()
    df = session.sql(f"""
        SELECT DISTINCT COUNTY_CODE, COUNTY_NAME
        FROM AF_COUNTY_BASE_VALUES
        WHERE STATE_CODE = '{state_code}'
          AND COUNTY_NAME IS NOT NULL
        ORDER BY COUNTY_NAME
    """).to_pandas()
    labels = {}
    for _, row in df.iterrows():
        code = row['COUNTY_CODE']
        name = row['COUNTY_NAME']
        labels[code] = f"{name} ({code})" if name else f"County {code}"
    return labels


@st.cache_data(ttl=3600, show_spinner=False)
def load_grids_for_county(state_code, county_code):
    session = _get_session()
    df = session.sql(f"""
        SELECT DISTINCT GRID_ID, GRID_LABEL
        FROM AF_COUNTY_BASE_VALUES
        WHERE STATE_CODE = '{state_code}'
          AND COUNTY_CODE = '{county_code}'
        ORDER BY GRID_ID
    """).to_pandas()
    return df['GRID_LABEL'].tolist()


def _grid_id_from_label(grid_label):
    """Extract numeric grid ID from label like '25318 (Lincoln - NE)' → 25318."""
    if grid_label is None:
        return None
    s = str(grid_label).strip()
    # If it's already numeric, return as-is
    try:
        return int(s)
    except (ValueError, TypeError):
        pass
    # Extract number before the first space or paren
    parts = s.split(' ', 1)
    try:
        return int(parts[0])
    except (ValueError, TypeError):
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_available_seasons(grid_id_or_label):
    grid_id = _grid_id_from_label(grid_id_or_label) or grid_id_or_label
    session = _get_session()
    df = session.sql(f"""
        SELECT DISTINCT GROWING_SEASON
        FROM AF_COUNTY_BASE_VALUES
        WHERE GRID_ID = {grid_id}
          AND COVERAGE_TYPE = 'Buy-Up'
        ORDER BY GROWING_SEASON
    """).to_pandas()
    return df['GROWING_SEASON'].astype(int).tolist()


@st.cache_data(ttl=3600, show_spinner=False)
def load_county_base_value(grid_id, growing_season):
    session = _get_session()
    df = session.sql(f"""
        SELECT
            COUNTY_BASE_VALUE,
            MAXIMUM_ACRE_PERCENT,
            MINIMUM_ACRE_PERCENT,
            MAXIMUM_PROTECTION_PER_ACRE
        FROM AF_COUNTY_BASE_VALUES
        WHERE GRID_ID = {grid_id}
          AND GROWING_SEASON = {growing_season}
          AND COVERAGE_TYPE = 'Buy-Up'
        LIMIT 1
    """).to_pandas()

    if df.empty:
        return {'county_base_value': 0.0, 'max_acre_percent': 0.0,
                'min_acre_percent': 0.0, 'maximum_protection_per_acre': 0.0}

    row = df.iloc[0]
    return {
        'county_base_value': float(row['COUNTY_BASE_VALUE']),
        'max_acre_percent': float(row['MAXIMUM_ACRE_PERCENT']),
        'min_acre_percent': float(row['MINIMUM_ACRE_PERCENT']),
        'maximum_protection_per_acre': float(row.get('MAXIMUM_PROTECTION_PER_ACRE', 0) or 0),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def load_premium_rates(grid_id, growing_season, coverage_level):
    session = _get_session()
    df = session.sql(f"""
        SELECT PRACTICE_CODE, BASE_RATE, PREMIUM_RATE_PER_100
        FROM AF_PREMIUM_RATES
        WHERE GRID_ID = {grid_id}
          AND GROWING_SEASON = {growing_season}
          AND COVERAGE_LEVEL_PERCENT = {coverage_level}
          AND COVERAGE_TYPE = 'Buy-Up'
        ORDER BY PRACTICE_CODE
    """).to_pandas()

    # Return raw BASE_RATE for calculation, display_rate for table display
    return {
        int(row['PRACTICE_CODE']): {
            'base_rate': float(row['BASE_RATE']),
            'display_rate': float(row['PREMIUM_RATE_PER_100']),
        }
        for _, row in df.iterrows()
    }


@st.cache_data(ttl=3600, show_spinner=False)
def load_subsidy_percent(coverage_level):
    session = _get_session()
    df = session.sql(f"""
        SELECT SUBSIDY_PERCENT
        FROM AF_SUBSIDY_VALUES
        WHERE COVERAGE_LEVEL_PERCENT = {coverage_level}
        LIMIT 1
    """).to_pandas()

    if df.empty:
        return 0.0
    return float(df.iloc[0]['SUBSIDY_PERCENT'])


@st.cache_data(ttl=3600, show_spinner=False)
def load_historical_indices(grid_id, growing_season):
    session = _get_session()
    intervals = AF_INTERVAL_MATRIX.get(growing_season, {})
    if not intervals:
        return pd.DataFrame()

    codes_str = ', '.join(f"'{c}'" for c in intervals.keys())

    df = session.sql(f"""
        SELECT YEAR, INTERVAL_CODE, INTERVAL_NAME, INDEX_VALUE
        FROM AF_RAIN_INDEX_PLATINUM
        WHERE GRID_ID = {grid_id}
          AND INTERVAL_CODE IN ({codes_str})
        ORDER BY YEAR, INTERVAL_CODE
    """).to_pandas()

    if df.empty:
        return pd.DataFrame()

    pivot = df.pivot_table(
        index='YEAR', columns='INTERVAL_NAME',
        values='INDEX_VALUE', aggfunc='first'
    ).reset_index()

    ordered_names = [intervals[c] for c in sorted(intervals.keys())]
    existing_cols = [n for n in ordered_names if n in pivot.columns]
    pivot = pivot[['YEAR'] + existing_cols]
    pivot = pivot.sort_values('YEAR', ascending=True).reset_index(drop=True)

    return pivot


def load_base_rates_array(grid_id, growing_season, coverage_level):
    """Load base rates as a (6,) NumPy array for buy-up intervals."""
    intervals = get_buyup_intervals(growing_season)
    codes = sorted(intervals.keys())
    rates = load_premium_rates(grid_id, growing_season, coverage_level)
    arr = np.zeros(6)
    for i, code in enumerate(codes):
        if code in rates:
            arr[i] = rates[code]['base_rate']
    return arr


def load_historical_matrix(grid_id, growing_season):
    """Load historical index values as (n_years, 6) matrix and year array."""
    intervals = get_buyup_intervals(growing_season)
    codes = sorted(intervals.keys())
    names = [intervals[c] for c in codes]

    hist_df = load_historical_indices(grid_id, growing_season)
    if hist_df.empty:
        return np.empty((0, 6)), np.array([])

    years = hist_df['YEAR'].values
    matrix = np.full((len(years), 6), 100.0)
    for i, name in enumerate(names):
        if name in hist_df.columns:
            vals = hist_df[name].values
            valid = pd.notna(vals)
            matrix[valid, i] = vals[valid].astype(float)

    # Ensure ascending year order (matches PRF convention)
    sort_order = np.argsort(years)
    years = years[sort_order]
    matrix = matrix[sort_order]

    return matrix, years
