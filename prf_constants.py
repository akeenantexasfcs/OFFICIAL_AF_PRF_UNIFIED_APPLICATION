"""
PRF (Pasture, Rangeland, Forage) Constants — for Unified Optimizer.
Aligned with reference PRF app constants (Snowflake-compatible interval names,
correct practice codes, and allocation bounds).
"""

import re

# ── 11 PRF Rainfall Intervals in standard order (hyphenated, matching Snowflake) ──
INTERVAL_ORDER_11 = [
    'Jan-Feb', 'Feb-Mar', 'Mar-Apr', 'Apr-May', 'May-Jun',
    'Jun-Jul', 'Jul-Aug', 'Aug-Sep', 'Sep-Oct', 'Oct-Nov', 'Nov-Dec',
]

# ── Intended Use Codes (numeric) ──
INTENDED_USE_CODE_MAP = {'Grazing': 7, 'Haying': 30}

# ── Default allocation bounds ──
MIN_ALLOCATION = 0.10  # 10%
MAX_ALLOCATION = 0.50  # 50% (matches reference; overridden per grid from PRF_COVERAGE_INTERVALS)

# ── Practice Combinations with code_map (matches reference exactly) ──
PRACTICE_COMBINATIONS = {
    'Grazing': {
        'irrigation_options': ['N/A'],
        'organic_options': {
            'N/A': ['No Organic Practice Specified']
        },
        'code_map': {
            ('N/A', 'No Organic Practice Specified'): {
                'irrigation_code': 997,
                'organic_code': 997,
                'type_code': 7,
                'irrigation_name_sql': None,
                'organic_name_sql': 'No Organic Practice Specified',
            },
        },
    },
    'Haying': {
        'irrigation_options': ['Irrigated', 'Non-Irrigated'],
        'organic_options': {
            'Irrigated': ['No Organic Practice Specified', 'Organic (Transitional Acreage)', 'Organic (100% Organic Acreage)'],
            'Non-Irrigated': ['No Organic Practice Specified', 'Organic (Transitional Acreage)', 'Organic (100% Organic Acreage)'],
        },
        'code_map': {
            ('Irrigated', 'No Organic Practice Specified'): {'irrigation_code': 2, 'organic_code': 997, 'type_code': 30, 'irrigation_name_sql': 'Irrigated', 'organic_name_sql': 'No Organic Practice Specified'},
            ('Irrigated', 'Organic (Transitional Acreage)'): {'irrigation_code': 2, 'organic_code': 2, 'type_code': 30, 'irrigation_name_sql': 'Irrigated', 'organic_name_sql': 'Organic (Transitional Acreage)'},
            ('Irrigated', 'Organic (100% Organic Acreage)'): {'irrigation_code': 2, 'organic_code': 1, 'type_code': 30, 'irrigation_name_sql': 'Irrigated', 'organic_name_sql': 'Organic (100% Organic Acreage)'},
            ('Non-Irrigated', 'No Organic Practice Specified'): {'irrigation_code': 3, 'organic_code': 997, 'type_code': 30, 'irrigation_name_sql': 'Non-Irrigated', 'organic_name_sql': 'No Organic Practice Specified'},
            ('Non-Irrigated', 'Organic (Transitional Acreage)'): {'irrigation_code': 3, 'organic_code': 2, 'type_code': 30, 'irrigation_name_sql': 'Non-Irrigated', 'organic_name_sql': 'Organic (Transitional Acreage)'},
            ('Non-Irrigated', 'Organic (100% Organic Acreage)'): {'irrigation_code': 3, 'organic_code': 1, 'type_code': 30, 'irrigation_name_sql': 'Non-Irrigated', 'organic_name_sql': 'Organic (100% Organic Acreage)'},
        },
    },
}

# ── Coverage levels (PRF) ──
PRF_COVERAGE_LEVELS = [0.90, 0.85, 0.80, 0.75, 0.70]

# ── Interval count range for optimization ──
PRF_MIN_INTERVALS = 2
PRF_MAX_INTERVALS = 6


def extract_numeric_grid_id(grid_id):
    """Extract numeric portion of a grid ID string."""
    if isinstance(grid_id, (int, float)):
        return int(grid_id)
    match = re.search(r'\d+', str(grid_id))
    return int(match.group()) if match else 0


def get_irrigation_options(intended_use):
    """Get available irrigation options for a given intended use."""
    combo = PRACTICE_COMBINATIONS.get(intended_use, {})
    return combo.get('irrigation_options', ['N/A'])


def get_organic_options(intended_use, irrigation):
    """Get available organic options for a given intended use and irrigation."""
    combo = PRACTICE_COMBINATIONS.get(intended_use, {})
    organic_opts = combo.get('organic_options', {})
    return organic_opts.get(irrigation, ['No Organic Practice Specified'])


def get_practice_codes(intended_use, irrigation_practice, organic_practice):
    """
    Convert practice string names to a dict of numeric codes for Snowflake queries.

    Returns:
        dict with keys: irrigation_code, organic_code, type_code,
                        irrigation_name_sql, organic_name_sql
    """
    combo = PRACTICE_COMBINATIONS.get(intended_use, {})
    code_map = combo.get('code_map', {})
    return code_map.get((irrigation_practice, organic_practice), {
        'irrigation_code': 997,
        'organic_code': 997,
        'type_code': 7,
        'irrigation_name_sql': None,
        'organic_name_sql': 'No Organic Practice Specified',
    })
