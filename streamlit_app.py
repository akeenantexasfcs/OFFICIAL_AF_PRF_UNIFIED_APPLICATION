"""
Unified PRF + AF Portfolio Optimizer — Proof of Concept
Streamlit-in-Snowflake Application

Combines PRF (Pasture, Rangeland, Forage) and AF (Annual Forage) insurance
optimization into a single portfolio view.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import itertools

# ── PRF imports ──
from prf_data_loaders import (
    load_county_base_value as prf_load_cbv,
    load_all_indices as prf_load_all_indices,
    load_premium_rates as prf_load_premium_rates,
    load_subsidies as prf_load_subsidies,
    load_coverage_intervals as prf_load_coverage_intervals,
    get_allocation_bounds as prf_get_allocation_bounds,
    get_current_rate_year as prf_get_current_rate_year,
    load_distinct_grids as prf_load_distinct_grids,
)
from prf_constants import (
    INTERVAL_ORDER_11,
    PRACTICE_COMBINATIONS as PRF_PRACTICE_COMBINATIONS,
    INTENDED_USE_CODE_MAP, MIN_ALLOCATION, MAX_ALLOCATION,
    PRF_COVERAGE_LEVELS, extract_numeric_grid_id,
    get_irrigation_options, get_organic_options,
)
from prf_optimization import (
    enumerate_prf_candidates,
    backtest_prf_candidates_vectorized,
)

# ── AF imports ──
from af_data_loaders import (
    _get_session,
    load_available_states, load_counties_for_state, load_grids_for_county,
    load_available_seasons, _grid_id_from_label,
    load_county_base_value as af_load_cbv,
    load_subsidy_percent as af_load_subsidy,
    load_base_rates_array as af_load_base_rates,
    load_historical_matrix as af_load_hist_matrix,
    load_historical_indices as af_load_hist_indices,
)
from af_constants import (
    FC_GREEN, FC_SLATE, RATE_YEAR,
    COVERAGE_LEVELS as AF_COVERAGE_LEVELS, CAT_COVERAGE_LEVEL,
    PRODUCTIVITY_FACTORS, SEASON_LABELS, OPTIMIZER_CONSTRAINTS,
    MONTH_NAMES_SHORT,
    get_buyup_intervals, get_cat_interval, interval_to_months,
    compute_shared_intervals, compute_next_eligible_season,
    AF_INTERVAL_MATRIX,
)
from af_calculations import _round_half_up, indemnity
from af_optimization import (
    enumerate_candidates as af_enumerate_candidates,
    backtest_candidates_vectorized as af_backtest_candidates,
    backtest_cat_unit as af_backtest_cat,
    run_joint_optimization, _score_independent, _score_portfolio,
    generate_insight_text,
)
from af_hrp import run_hrp
from unified_report_generator import generate_unified_optimizer_report_docx
import base64
import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _compute_all_metrics(returns, cost):
    """Compute all metrics using correct formulas matching the reference app."""
    std = float(np.std(returns))
    sharpe = float(np.mean(returns) / std) if std > 0 else 0.0
    cvar = float(np.percentile(returns, 5))
    winrate = float(np.mean(returns > 0))
    roi = float((np.mean(returns) + cost) / cost) if cost > 0 else 0.0
    mean_ret = float(np.mean(returns))
    return {'sharpe': sharpe, 'cvar': cvar, 'winrate': winrate,
            'roi': roi, 'mean_return': mean_ret, 'cost': cost,
            'producer_cost': cost, 'std_return': std}


def _extract_county(grid_label):
    """Extract county from grid label like '10020 (Dimmit - TX)' → 'Dimmit'
    or '25318 (Lincoln - NE)' → 'Lincoln'. Returns 'Unknown' if parsing fails."""
    if isinstance(grid_label, str) and '(' in grid_label:
        try:
            inner = grid_label.split('(')[1].split(')')[0]  # 'Dimmit - TX'
            county = inner.split('-')[0].strip()  # 'Dimmit'
            return county
        except (IndexError, AttributeError):
            pass
    return 'Unknown'


def _get_county_crop_key(cfg):
    """Build (county, crop_type) grouping key for a unit config."""
    crop_type = cfg['type']  # 'PRF' or 'AF'
    grid_label = cfg.get('grid_label') or cfg.get('grid', '')
    county = _extract_county(grid_label)
    return (county, crop_type)


COVERAGE_LEVELS_TO_TEST = [0.70, 0.75, 0.80, 0.85, 0.90]


def _enumerate_all_candidates(unit_configs, weight_step, prf_interval_range,
                               progress_callback=None):
    """Phase 1: Enumerate candidates for each unit ONCE.

    Candidate enumeration does not depend on coverage_level:
    - AF enumerate_candidates() depends only on growing_season
    - PRF enumerate_prf_candidates() uses allocation bounds keyed on
      intended_use/irrigation/organic — not coverage level
    - CAT units have a single hardcoded candidate

    Returns dict {uid: {'candidates': list, 'type': str, ...extra data}}.
    """
    cache = {}
    total_units = len(unit_configs)
    for enum_idx, (uid, cfg) in enumerate(unit_configs.items()):
        entry = {'type': cfg['type']}
        try:
            if cfg['type'] == 'PRF':
                candidates = enumerate_prf_candidates(
                    grid_id=cfg['grid'],
                    intended_use=cfg['intended_use'],
                    irrigation_practice=cfg['irrigation'],
                    organic_practice=cfg['organic'],
                    coverage_level=cfg['coverage_level'],
                    step=weight_step,
                    iterations=10000,
                    interval_count_range=prf_interval_range,
                )
                entry['candidates'] = candidates

            elif cfg['type'] == 'AF':
                grid_id = _grid_id_from_label(cfg['grid'])
                entry['grid_id'] = grid_id

                if cfg.get('is_cat', False):
                    entry['is_cat'] = True
                    # CAT: single hardcoded candidate — store CBV for backtest
                    cbv_d = af_load_cbv(grid_id, cfg['growing_season'])
                    entry['cbv'] = cbv_d['county_base_value']
                    entry['candidates'] = None  # backtested via af_backtest_cat
                else:
                    candidates = af_enumerate_candidates(
                        cfg['growing_season'], step=weight_step
                    )
                    entry['candidates'] = candidates
                    # Pre-load data that doesn't depend on coverage_level
                    cbv_d = af_load_cbv(grid_id, cfg['growing_season'])
                    entry['cbv'] = cbv_d['county_base_value']
                    hist_matrix, hist_years = af_load_hist_matrix(
                        grid_id, cfg['growing_season'],
                    )
                    entry['hist_matrix'] = hist_matrix
                    entry['hist_years'] = hist_years

        except Exception as ex:
            entry['error'] = str(ex)

        cache[uid] = entry
        if progress_callback:
            progress_callback((enum_idx + 1) / total_units)
    return cache


def _backtest_and_score(unit_configs, candidates_cache, opt_metric, opt_mode,
                        start_year, end_year, progress_callback=None, top_k=None,
                        calc_engine='python'):
    """Phase 2: Backtest pre-enumerated candidates and score.

    Coverage_level matters here (changes premium rates, trigger, DA).
    Takes pre-enumerated candidates from _enumerate_all_candidates().

    Returns (score, results_dict, units_data_for_joint).
    """
    results = {'units': [], 'mode': opt_mode, 'metric': opt_metric}
    units_data_for_joint = []
    total_units = len(unit_configs)

    for step_idx, (uid, cfg) in enumerate(unit_configs.items()):
        unit_result = {'config': cfg, 'uid': uid}
        cached = candidates_cache.get(uid, {})

        if 'error' in cached:
            unit_result['error'] = cached['error']
            results['units'].append(unit_result)
            continue

        try:
            if cfg['type'] == 'PRF':
                candidates = cached.get('candidates', [])
                unit_result['n_candidates'] = len(candidates)
                unit_result['candidates'] = candidates

                if not candidates:
                    results['units'].append(unit_result)
                    continue

                yearly_returns, producer_costs, years = \
                    backtest_prf_candidates_vectorized(
                        candidates=candidates,
                        grid_id=cfg['grid'],
                        start_year=start_year,
                        end_year=end_year,
                        intended_use=cfg['intended_use'],
                        irrigation_practice=cfg['irrigation'],
                        organic_practice=cfg['organic'],
                        coverage_level=cfg['coverage_level'],
                        productivity=cfg['productivity'],
                        insurable_interest=cfg['insurable_interest'],
                    )

                unit_result['yearly_returns'] = yearly_returns
                unit_result['producer_costs'] = producer_costs
                unit_result['years'] = years
                unit_result['interval_labels'] = list(INTERVAL_ORDER_11)

            elif cfg['type'] == 'AF':
                grid_id = cached.get('grid_id', _grid_id_from_label(cfg['grid']))

                if cfg.get('is_cat', False):
                    cbv = cached.get('cbv')
                    if cbv is None:
                        cbv_d = af_load_cbv(grid_id, cfg['growing_season'])
                        cbv = cbv_d['county_base_value']
                    candidates, yearly_returns, producer_costs, years = af_backtest_cat(
                        grid_id, cfg['growing_season'],
                        cbv,
                        cfg['productivity'], cfg['coverage_level'],
                        af_load_subsidy(cfg['coverage_level']),
                    )
                    unit_result['is_cat'] = True
                    unit_result['candidates'] = candidates
                    unit_result['n_candidates'] = len(candidates)
                    if len(candidates) > 0 and yearly_returns.size > 0:
                        yearly_returns *= cfg['insurable_interest']
                        producer_costs *= cfg['insurable_interest']
                        unit_result['yearly_returns'] = yearly_returns
                        unit_result['producer_costs'] = producer_costs
                        unit_result['years'] = years
                        cat_interval = get_cat_interval(cfg['growing_season'])
                        cat_name = list(cat_interval.values())[0] if cat_interval else 'CAT'
                        unit_result['interval_labels'] = [cat_name]
                    else:
                        results['units'].append(unit_result)
                        continue

                else:
                    candidates = cached.get('candidates', [])
                    unit_result['n_candidates'] = len(candidates)
                    unit_result['candidates'] = candidates

                    if not candidates:
                        results['units'].append(unit_result)
                        continue

                    cbv = cached.get('cbv')
                    if cbv is None:
                        cbv_d = af_load_cbv(grid_id, cfg['growing_season'])
                        cbv = cbv_d['county_base_value']
                    subsidy_pct = af_load_subsidy(cfg['coverage_level'])
                    base_rates = af_load_base_rates(
                        grid_id, cfg['growing_season'],
                        cfg['coverage_level'],
                    )
                    hist_matrix = cached.get('hist_matrix')
                    hist_years = cached.get('hist_years')
                    if hist_matrix is None or hist_years is None:
                        hist_matrix, hist_years = af_load_hist_matrix(
                            grid_id, cfg['growing_season'],
                        )

                    year_mask = (hist_years >= start_year) & (hist_years <= end_year)
                    hist_matrix_filtered = hist_matrix[year_mask]
                    years = hist_years[year_mask]

                    yearly_returns, producer_costs = af_backtest_candidates(
                        candidates, hist_matrix_filtered, cbv,
                        cfg['coverage_level'], cfg['productivity'],
                        base_rates, subsidy_pct,
                    )

                    yearly_returns *= cfg['insurable_interest']
                    producer_costs *= cfg['insurable_interest']

                    unit_result['yearly_returns'] = yearly_returns
                    unit_result['producer_costs'] = producer_costs
                    unit_result['years'] = years

                    intervals = get_buyup_intervals(cfg['growing_season'])
                    unit_result['interval_labels'] = [
                        intervals[k] for k in sorted(intervals.keys())
                    ]

        except Exception as ex:
            unit_result['error'] = str(ex)

        results['units'].append(unit_result)
        result_unit_idx = len(results['units']) - 1

        if 'yearly_returns' in unit_result:
            jd = {
                'yearly_returns': unit_result['yearly_returns'],
                'producer_costs': unit_result['producer_costs'],
                'acres': cfg['acres'],
                'candidates': unit_result.get('candidates', []),
                'label': f"Unit {step_idx+1} ({cfg['type']})",
                'unit_label': cfg.get('unit_label', f"Unit {step_idx+1} ({cfg['type']})"),
                'years': unit_result['years'],
                'is_cat': cfg.get('is_cat', False),
                'coverage_level': cfg['coverage_level'],
                'productivity': cfg['productivity'],
                'insurable_interest': cfg.get('insurable_interest', 1.0),
                'grid_id': cfg.get('grid_id') or cfg.get('grid'),
                'grid_label': cfg.get('grid_label') or cfg.get('grid', ''),
                'type': cfg['type'],
                '_result_idx': result_unit_idx,
            }
            if cfg['type'] == 'AF':
                jd['growing_season'] = cfg['growing_season']
                try:
                    cbv_d = af_load_cbv(jd['grid_id'], cfg['growing_season'])
                    jd['cbv'] = cbv_d['county_base_value']
                except Exception:
                    jd['cbv'] = 0.0
            if cfg['type'] == 'PRF':
                jd['intended_use'] = cfg.get('intended_use', 'Grazing')

            try:
                if cfg['type'] == 'AF':
                    _tc_cbv = jd.get('cbv', 0.0)
                else:
                    _tc_cbv = prf_load_cbv(
                        cfg['grid'],
                        cfg.get('intended_use', 'Grazing'),
                        cfg.get('irrigation', 'N/A'),
                        cfg.get('organic', 'No Organic Practice Specified'),
                    ) or 0.0
                    jd['cbv'] = _tc_cbv
                _tc_da = _round_half_up(
                    _tc_cbv * cfg['coverage_level'] * cfg['productivity'], 2
                )
                jd['total_coverage'] = (
                    _tc_da * cfg.get('insurable_interest', 1.0) * cfg['acres']
                )
            except Exception:
                jd['total_coverage'] = 0.0

            units_data_for_joint.append(jd)

        if progress_callback:
            progress_callback((step_idx + 1) / total_units)

    # Year alignment
    if len(units_data_for_joint) > 1:
        all_year_sets = [set(ud['years']) for ud in units_data_for_joint]
        common_years = all_year_sets[0]
        for ys in all_year_sets[1:]:
            common_years = common_years & ys
        common_years = sorted(common_years)

        if not common_years:
            return -np.inf, results, units_data_for_joint

        common_years_arr = np.array(common_years)
        for ud in units_data_for_joint:
            mask = np.isin(ud['years'], common_years_arr)
            ud['yearly_returns'] = ud['yearly_returns'][:, mask]
            ud['years'] = common_years_arr

    elif len(units_data_for_joint) == 1:
        common_years_arr = np.array(units_data_for_joint[0]['years'])

    # Score
    if not units_data_for_joint:
        return -np.inf, results, units_data_for_joint

    # Independent scoring
    for i, ud in enumerate(units_data_for_joint):
        best_idx, best_score = _score_independent(
            ud['yearly_returns'], ud['producer_costs'], opt_metric
        )
        ri = ud['_result_idx']
        results['units'][ri]['best_idx'] = best_idx
        best_returns = ud['yearly_returns'][best_idx]
        best_cost = ud['producer_costs'][best_idx]
        results['units'][ri]['metrics'] = _compute_all_metrics(best_returns, best_cost)
        if opt_mode == 'independent' and progress_callback:
            progress_callback((i + 1) / len(units_data_for_joint))

    if opt_mode == 'joint' and len(units_data_for_joint) >= 2:
        best_combo, best_score_j, top_combos = \
            run_joint_optimization(units_data_for_joint, opt_metric, progress_callback, top_k=top_k, calc_engine=calc_engine)

        for k in range(len(units_data_for_joint)):
            ri = units_data_for_joint[k]['_result_idx']
            results['units'][ri]['joint_idx'] = best_combo[k]

        total_acres_j = sum(ud['acres'] for ud in units_data_for_joint)
        joint_portfolio = sum(
            units_data_for_joint[k]['yearly_returns'][best_combo[k]:best_combo[k]+1, :] * units_data_for_joint[k]['acres']
            for k in range(len(units_data_for_joint))
        ) / total_acres_j
        joint_cost = sum(
            units_data_for_joint[k]['producer_costs'][best_combo[k]] * units_data_for_joint[k]['acres']
            for k in range(len(units_data_for_joint))
        ) / total_acres_j
        joint_metrics = _compute_all_metrics(joint_portfolio.flatten(), joint_cost)

        results['joint_metrics'] = joint_metrics
        results['best_combo'] = best_combo
        results['units_data_for_joint'] = units_data_for_joint

        indep_indices = []
        for i, ud in enumerate(units_data_for_joint):
            idx, _ = _score_independent(
                ud['yearly_returns'], ud['producer_costs'], opt_metric
            )
            indep_indices.append(idx)
        results['indep_indices'] = indep_indices
        results['top_combos'] = top_combos

        portfolio_score = _score_portfolio(joint_portfolio.flatten(), opt_metric,
                                           total_cost=joint_cost)
    else:
        # Independent mode or single unit: compute weighted portfolio score
        total_acres = sum(ud['acres'] for ud in units_data_for_joint)
        if total_acres > 0:
            port_ret = np.zeros_like(units_data_for_joint[0]['yearly_returns'][0])
            port_cost = 0.0
            for ud in units_data_for_joint:
                bi = results['units'][ud['_result_idx']]['best_idx']
                port_ret += ud['yearly_returns'][bi] * ud['acres']
                port_cost += ud['producer_costs'][bi] * ud['acres']
            port_ret /= total_acres
            port_cost /= total_acres
            portfolio_score = _score_portfolio(port_ret, opt_metric, total_cost=port_cost)
        else:
            portfolio_score = -np.inf

    return portfolio_score, results, units_data_for_joint


def _run_optimization_pipeline(unit_configs, opt_metric, opt_mode, weight_step,
                                prf_interval_range, start_year, end_year,
                                progress_callback=None, top_k=None, calc_engine='python'):
    """Run the full per-unit enumeration → backtest → score pipeline.

    Returns (score, results_dict, units_data_for_joint) where score is the
    portfolio-level metric value used for comparing coverage combos.
    """
    # Split progress: 0-40% for enumeration, 40-100% for backtest/scoring
    def _enum_progress(pct):
        if progress_callback:
            progress_callback(pct * 0.4)

    def _backtest_progress(pct, msg=''):
        if progress_callback:
            progress_callback(0.4 + pct * 0.6, msg)

    candidates_cache = _enumerate_all_candidates(
        unit_configs, weight_step, prf_interval_range,
        progress_callback=_enum_progress,
    )
    return _backtest_and_score(
        unit_configs, candidates_cache, opt_metric, opt_mode,
        start_year, end_year, _backtest_progress, top_k=top_k,
        calc_engine=calc_engine,
    )


def _override_coverage_all(unit_configs, cov_level):
    """Override coverage_level for ALL units (skip CAT units)."""
    modified = {}
    for uid, cfg in unit_configs.items():
        new_cfg = dict(cfg)
        if not cfg.get('is_cat', False):
            new_cfg['coverage_level'] = cov_level
        modified[uid] = new_cfg
    return modified


def _override_coverage_by_category(unit_configs, prf_cov, af_cov):
    """Override coverage by crop type: prf_cov for PRF, af_cov for AF (skip CAT)."""
    modified = {}
    for uid, cfg in unit_configs.items():
        new_cfg = dict(cfg)
        if cfg.get('is_cat', False):
            pass  # Leave CAT unchanged
        elif cfg['type'] == 'PRF':
            new_cfg['coverage_level'] = prf_cov
        elif cfg['type'] == 'AF':
            new_cfg['coverage_level'] = af_cov
        modified[uid] = new_cfg
    return modified


def _override_coverage_by_groups(unit_configs, groups, group_keys, combo):
    """Override coverage per (county, crop_type) group. combo is a tuple of coverage
    levels indexed by group_keys order. Skip CAT units."""
    group_cov = dict(zip(group_keys, combo))
    modified = {}
    for uid, cfg in unit_configs.items():
        new_cfg = dict(cfg)
        if not cfg.get('is_cat', False):
            key = _get_county_crop_key(cfg)
            if key in group_cov:
                new_cfg['coverage_level'] = group_cov[key]
        modified[uid] = new_cfg
    return modified


def _format_coverage_combo_label(mode, combo, group_keys=None):
    """Format a coverage combo for display."""
    if mode == 'uniform':
        return f"{int(combo * 100)}%"
    elif mode == 'per_category':
        return f"PRF {int(combo[0]*100)}% / AF {int(combo[1]*100)}%"
    elif mode == 'per_county_crop' and group_keys:
        parts = []
        for gk, cv in zip(group_keys, combo):
            county, crop = gk
            parts.append(f"{crop} {county}: {int(cv*100)}%")
        return ", ".join(parts)
    return str(combo)


# ═══════════════════════════════════════════════════════════════════════════════
# CSS & Page Config
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Unified Portfolio Optimizer", layout="wide")

CUSTOM_CSS = f"""
<style>
    .main-header {{
        font-size: 2rem;
        font-weight: 700;
        color: {FC_GREEN};
        margin-bottom: 0;
    }}
    .sub-header {{
        font-size: 1.1rem;
        color: {FC_SLATE};
        margin-top: 0;
        margin-bottom: 1.5rem;
    }}
    .metric-card {{
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }}
    .opt-metric-card {{
        background: #1a1a2e;
        border-radius: 10px;
        padding: 1.2rem 1rem;
        text-align: center;
        color: white;
        min-height: 100px;
    }}
    .opt-metric-label {{
        font-size: 0.8rem;
        color: #adb5bd;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }}
    .opt-metric-value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {FC_GREEN};
    }}
    .opt-comparison-box {{
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #e0e0e0;
    }}
    .opt-improve {{
        color: {FC_GREEN};
        font-weight: 600;
    }}
    .opt-worsen {{
        color: #dc3545;
        font-weight: 600;
    }}
    .unit-card {{
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-left: 4px solid {FC_GREEN};
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}
    .prf-badge {{
        display: inline-block;
        background: {FC_GREEN};
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 8px;
    }}
    .af-badge {{
        display: inline-block;
        background: {FC_SLATE};
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 8px;
    }}
    .sidebar-brand {{
        text-align: center;
        padding: 1rem 0;
    }}
    .sidebar-brand h2 {{
        color: {FC_GREEN};
        margin-bottom: 0;
    }}
    .sidebar-brand p {{
        color: {FC_SLATE};
        font-size: 0.9rem;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Session State Initialization
# ═══════════════════════════════════════════════════════════════════════════════

if 'units' not in st.session_state:
    st.session_state.units = []  # list of dicts: {'type': 'PRF'|'AF', 'id': int}
if 'unit_counter' not in st.session_state:
    st.session_state.unit_counter = 0
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None


def _add_unit(unit_type):
    st.session_state.unit_counter += 1
    st.session_state.units.append({
        'type': unit_type,
        'id': st.session_state.unit_counter,
    })
    st.session_state.optimization_results = None


def _remove_unit(unit_id):
    st.session_state.units = [u for u in st.session_state.units if u['id'] != unit_id]
    st.session_state.optimization_results = None


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_pct(val):
    """Format a decimal as percentage string."""
    return f"{val * 100:.0f}%"


def _fmt_dollar(val):
    """Format a dollar amount."""
    return f"${val:,.2f}"


def _fmt_delta(val, fmt='pct', higher_is_better=True):
    """Format a delta value with arrow and color."""
    if fmt == 'pct':
        text = f"{val * 100:+.1f}%"
    elif fmt == 'dollar':
        text = f"${val:+,.2f}"
    else:
        text = f"{val:+.4f}"

    if val > 0:
        css_class = 'opt-improve' if higher_is_better else 'opt-worsen'
        arrow = '▲'
    elif val < 0:
        css_class = 'opt-worsen' if higher_is_better else 'opt-improve'
        arrow = '▼'
    else:
        css_class = ''
        arrow = '─'

    return f'<span class="{css_class}">{arrow} {text}</span>'


def _render_metric_card(label, value):
    """Render a dark-themed metric card."""
    return f"""
    <div class="opt-metric-card">
        <div class="opt-metric-label">{label}</div>
        <div class="opt-metric-value">{value}</div>
    </div>
    """


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-brand">
        <h2>TFC Analytics</h2>
        <p>Unified Portfolio Optimizer</p>
        <hr style="border-color: {FC_GREEN}; margin: 0.5rem 0;">
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Resources**")
    st.link_button("USDA PRF Decision Support Tool", "https://prodwebnlb.rma.usda.gov/apps/prf")
    st.link_button("USDA AF Decision Support Tool", "http://af.agforceusa.com/ri")

    st.caption(f"{RATE_YEAR} Rates are used for this application.")


# ═══════════════════════════════════════════════════════════════════════════════
# Page Header
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="main-header">🌾 Unified Portfolio Optimizer</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">TFC Analytics Tool — PRF + Annual Forage</div>',
            unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Session Initialization
# ═══════════════════════════════════════════════════════════════════════════════

try:
    session = _get_session()
except Exception as e:
    session = None
    st.error(f"Snowflake connection error: {e}")
    st.stop()

# Pre-load state dictionary (cached)
@st.cache_data(ttl=3600, show_spinner=False)
def _cached_load_states():
    return load_available_states()

states_dict = _cached_load_states()
state_codes = list(states_dict.keys())
state_labels = list(states_dict.values())


@st.cache_data(ttl=3600, show_spinner=False)
def _load_all_af_grids():
    """Load all AF grids directly from AF_COUNTY_BASE_VALUES table."""
    sess = _get_session()
    df = sess.sql("""
        SELECT DISTINCT GRID_ID, GRID_LABEL
        FROM AF_COUNTY_BASE_VALUES
        ORDER BY GRID_ID
    """).to_pandas()
    return df['GRID_LABEL'].tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Optimization Period
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("### 📅 Optimization Period")

period_col1, period_col2, period_col3 = st.columns([1, 1, 2])
with period_col1:
    start_year = st.number_input("Start Year", min_value=1949, max_value=2026,
                                  value=1949, step=1, key="start_year")
with period_col2:
    end_year = st.number_input("End Year", min_value=1949, max_value=2026,
                                value=2026, step=1, key="end_year")
with period_col3:
    n_years_display = max(0, end_year - start_year + 1)
    st.markdown(f"<br><span style='color: #5B707F;'>{n_years_display} years of backtest data</span>",
                unsafe_allow_html=True)
    if n_years_display < 10:
        st.warning("Fewer than 10 years may produce unreliable results.")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Portfolio Units Section
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("### 🌾 Portfolio Units")
st.caption("Configure PRF and AF units for cross-program portfolio optimization.")

# Add unit buttons (outside form to trigger rerun)
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
with btn_col1:
    if st.button("➕ Add PRF Unit", key="add_prf"):
        _add_unit('PRF')
        st.rerun()
with btn_col2:
    if st.button("➕ Add AF Unit", key="add_af"):
        _add_unit('AF')
        st.rerun()
with btn_col3:
    if st.button("🐄 Nebraska Sandhill Demo", key="demo_sandhill"):
        st.session_state.units = []
        st.session_state.unit_counter = 0
        st.session_state.optimization_results = None
        if 's2_results' in st.session_state:
            del st.session_state['s2_results']

        demo_units = [
            {'type': 'PRF', 'grid': '25619 (Lincoln - NE)', 'intended_use': 'Grazing', 'irrigation': 'N/A', 'organic': 'No Organic Practice Specified', 'coverage': 0.90, 'productivity': 1.50, 'acres': 12640.0, 'ii': 100},
            {'type': 'PRF', 'grid': '25319 (Lincoln - NE)', 'intended_use': 'Grazing', 'irrigation': 'N/A', 'organic': 'No Organic Practice Specified', 'coverage': 0.90, 'productivity': 1.50, 'acres': 12640.0, 'ii': 100},
            {'type': 'PRF', 'grid': '25318 (Lincoln - NE)', 'intended_use': 'Grazing', 'irrigation': 'N/A', 'organic': 'No Organic Practice Specified', 'coverage': 0.90, 'productivity': 1.50, 'acres': 11553.0, 'ii': 100},
            {'type': 'PRF', 'grid': '25318 (Lincoln - NE)', 'intended_use': 'Haying', 'irrigation': 'Irrigated', 'organic': 'No Organic Practice Specified', 'coverage': 0.90, 'productivity': 1.50, 'acres': 636.0, 'ii': 100},
            {'type': 'AF', 'grid': '25318 (Lincoln - NE)', 'growing_season': 10, 'coverage': 0.90, 'productivity': 1.50, 'acres': 380.0, 'ii': 100},
            {'type': 'AF', 'grid': '25318 (Lincoln - NE)', 'growing_season': 2, 'coverage': 0.90, 'productivity': 1.50, 'acres': 380.0, 'ii': 100},
        ]

        for i, d in enumerate(demo_units):
            st.session_state.unit_counter += 1
            uid = st.session_state.unit_counter
            st.session_state.units.append({'type': d['type'], 'id': uid})

            if d['type'] == 'PRF':
                st.session_state[f'prf_grid_{uid}'] = d['grid']
                st.session_state[f'prf_use_{uid}'] = d['intended_use']
                st.session_state[f'prf_irr_{uid}'] = d['irrigation']
                st.session_state[f'prf_org_{uid}'] = d['organic']
                st.session_state[f'prf_cov_{uid}'] = f"{int(d['coverage']*100)}%"
                st.session_state[f'prf_pf_{uid}'] = f"{int(d['productivity']*100)}%"
                st.session_state[f'prf_acres_{uid}'] = d['acres']
                st.session_state[f'prf_ii_{uid}'] = d['ii']
            else:
                st.session_state[f'af_grid_{uid}'] = d['grid']
                st.session_state[f'af_season_{uid}'] = f"{d['growing_season']} - {SEASON_LABELS.get(d['growing_season'], '?')}"
                st.session_state[f'af_cov_{uid}'] = f"{int(d['coverage']*100)}%"
                st.session_state[f'af_pf_{uid}'] = f"{int(d['productivity']*100)}%"
                st.session_state[f'af_acres_{uid}'] = d['acres']
                st.session_state[f'af_ii_{uid}'] = d['ii']

        st.rerun()
    n_prf = sum(1 for u in st.session_state.units if u['type'] == 'PRF')
    n_af = sum(1 for u in st.session_state.units if u['type'] == 'AF')
    st.caption(f"{len(st.session_state.units)} units total "
               f"({n_prf} PRF, {n_af} AF)")

# Remove unit control
if st.session_state.units:
    rm_col1, rm_col2 = st.columns([2, 1])
    with rm_col1:
        remove_options = {
            f"Unit {i+1} ({u['type']}) — ID {u['id']}": u['id']
            for i, u in enumerate(st.session_state.units)
        }
        selected_remove = st.selectbox(
            "Select unit to remove",
            options=list(remove_options.keys()),
            key="remove_select",
            label_visibility="collapsed",
        )
    with rm_col2:
        if st.button("✕ Remove", key="remove_btn"):
            if selected_remove:
                _remove_unit(remove_options[selected_remove])
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Form: Unit Configuration + Optimization Controls
# ═══════════════════════════════════════════════════════════════════════════════

if not st.session_state.units:
    st.info("Add PRF or AF units above to begin configuring your portfolio.")
else:
    # ── Render each unit card ──
    unit_configs = {}

    for idx, unit in enumerate(st.session_state.units):
        uid = unit['id']
        utype = unit['type']
        unit_num = idx + 1

        badge = 'prf-badge' if utype == 'PRF' else 'af-badge'
        st.markdown(
            f'<div class="unit-card">'
            f'<strong>Unit {unit_num}</strong> '
            f'<span class="{badge}">{utype}</span></div>',
            unsafe_allow_html=True,
        )

        if utype == 'PRF':
            # Row 1: Grid selection (direct grid ID)
            loc_cols = st.columns([2, 2])
            with loc_cols[0]:
                prf_grids = prf_load_distinct_grids()
                grid = st.selectbox(
                    "Grid",
                    options=prf_grids if prf_grids else ["--"],
                    key=f"prf_grid_{uid}",
                )

            # Row 2: PRF-specific inputs
            cols = st.columns([1, 1, 1, 1, 1, 1, 1])

            with cols[0]:
                intended_use = st.selectbox(
                    "Intended Use", options=['Grazing', 'Haying'],
                    key=f"prf_use_{uid}",
                )
            with cols[1]:
                irr_opts = get_irrigation_options(intended_use)
                irrigation = st.selectbox(
                    "Irrigation", options=irr_opts,
                    key=f"prf_irr_{uid}",
                )
            with cols[2]:
                org_opts = get_organic_options(intended_use, irrigation)
                organic = st.selectbox(
                    "Organic", options=org_opts,
                    key=f"prf_org_{uid}",
                )
            with cols[3]:
                cov_labels = [f"{int(cl*100)}%" for cl in PRF_COVERAGE_LEVELS]
                cov_sel = st.selectbox(
                    "Coverage Level", options=cov_labels,
                    key=f"prf_cov_{uid}",
                )
                coverage_level = PRF_COVERAGE_LEVELS[cov_labels.index(cov_sel)]
            with cols[4]:
                pf_labels = [f"{int(pf*100)}%" for pf in PRODUCTIVITY_FACTORS]
                pf_sel = st.selectbox(
                    "Productivity Factor", options=pf_labels,
                    index=0,
                    key=f"prf_pf_{uid}",
                )
                prod_factor = PRODUCTIVITY_FACTORS[pf_labels.index(pf_sel)]
            with cols[5]:
                acres = st.number_input(
                    "Acres", min_value=1.0, value=1000.0, step=100.0,
                    key=f"prf_acres_{uid}",
                )
            with cols[6]:
                ins_interest_pct = st.number_input(
                    "Insurable Interest %", min_value=1, max_value=100,
                    value=100, step=1,
                    key=f"prf_ii_{uid}",
                )

            unit_configs[uid] = {
                'type': 'PRF',
                'grid': grid,
                'unit_label': f"PRF Grid {grid}",
                'intended_use': intended_use,
                'irrigation': irrigation,
                'organic': organic,
                'coverage_level': coverage_level,
                'productivity': prod_factor,
                'acres': acres,
                'insurable_interest': ins_interest_pct / 100.0,
            }

        elif utype == 'AF':
            # Row 1: Grid selector (all AF grids)
            all_af_grids = _load_all_af_grids()
            loc_cols = st.columns([2, 2])
            with loc_cols[0]:
                grid = st.selectbox(
                    "Grid",
                    options=all_af_grids if all_af_grids else ["--"],
                    key=f"af_grid_{uid}",
                )

            # Load seasons for grid
            af_seasons = []
            if grid and grid not in ("--", "N/A"):
                try:
                    af_seasons = load_available_seasons(grid)
                except Exception:
                    af_seasons = list(SEASON_LABELS.keys())

            # Row 2: AF-specific inputs
            cols = st.columns([1.2, 1, 1, 1, 1, 1])

            with cols[0]:
                season_options = [
                    f"{gs} - {SEASON_LABELS.get(gs, f'Season {gs}')}" for gs in af_seasons
                ] if af_seasons else ["N/A"]
                season_sel = st.selectbox(
                    "Growing Season", options=season_options,
                    key=f"af_season_{uid}",
                )
                if season_sel != "N/A":
                    try:
                        growing_season = int(season_sel.split(' - ')[0])
                    except (ValueError, TypeError):
                        growing_season = None
                else:
                    growing_season = None

            with cols[1]:
                af_cov_options = ['CAT (65%)'] + [f"{int(cl*100)}%" for cl in AF_COVERAGE_LEVELS]
                af_cov_sel = st.selectbox(
                    "Coverage", options=af_cov_options,
                    index=1,  # Default to 90%
                    key=f"af_cov_{uid}",
                )
                if af_cov_sel.startswith('CAT'):
                    af_coverage = CAT_COVERAGE_LEVEL
                    is_cat = True
                else:
                    af_coverage = AF_COVERAGE_LEVELS[
                        [f"{int(cl*100)}%" for cl in AF_COVERAGE_LEVELS].index(af_cov_sel)
                    ]
                    is_cat = False

            with cols[2]:
                pf_labels = [f"{int(pf*100)}%" for pf in PRODUCTIVITY_FACTORS]
                af_pf_sel = st.selectbox(
                    "Productivity Factor", options=pf_labels,
                    index=0,
                    key=f"af_pf_{uid}",
                )
                af_prod = PRODUCTIVITY_FACTORS[pf_labels.index(af_pf_sel)]

            with cols[3]:
                af_acres = st.number_input(
                    "Acres", min_value=1.0, value=1000.0, step=100.0,
                    key=f"af_acres_{uid}",
                )

            with cols[4]:
                af_ii_pct = st.number_input(
                    "Insurable Interest %", min_value=1, max_value=100,
                    value=100, step=1,
                    key=f"af_ii_{uid}",
                )

            grid_id = _grid_id_from_label(grid) if grid and grid not in ('--', 'N/A') else None
            unit_configs[uid] = {
                'type': 'AF',
                'grid': grid,
                'grid_id': grid_id,
                'grid_label': grid,
                'unit_label': f"AF GS{growing_season} Grid {grid}",
                'growing_season': growing_season,
                'coverage_level': af_coverage,
                'is_cat': is_cat,
                'productivity': af_prod,
                'acres': af_acres,
                'insurable_interest': af_ii_pct / 100.0,
            }

    # ── Optimization Controls ──
    st.markdown("### ⚙️ Optimization Controls")

    ctrl_col1, ctrl_col2 = st.columns(2)
    with ctrl_col1:
        opt_metric_label = st.radio(
            "Optimization Metric",
            options=["Risk-Adjusted (Sharpe)", "Tail Risk (CVaR 5%)",
                     "Max Return %", "Win Rate"],
            horizontal=True,
            key="opt_metric",
        )
        metric_map = {
            "Risk-Adjusted (Sharpe)": "sharpe",
            "Tail Risk (CVaR 5%)": "cvar",
            "Max Return %": "roi",
            "Win Rate": "winrate",
        }
        opt_metric = metric_map[opt_metric_label]

    with ctrl_col2:
        weight_step_label = st.radio(
            "Weight Increment",
            options=["5%", "10%"],
            horizontal=True,
            key="weight_step",
        )
        weight_step = 5 if weight_step_label == "5%" else 10

    n_prf = sum(1 for u in st.session_state.units if u['type'] == 'PRF')
    if n_prf > 0:
        prf_interval_range = st.slider(
            "PRF Active Intervals Range",
            min_value=2,
            max_value=6,
            value=(4, 6),
            help="Optimizer can choose ANY number of intervals within this range (PRF units only).",
            key="prf_interval_range",
        )
    else:
        prf_interval_range = (2, 6)

    opt_mode_label = st.radio(
        "Optimization Mode",
        options=["Independent", "Joint"],
        index=1,
        horizontal=True,
        key="opt_mode",
        help="Independent: each unit optimized in isolation. "
             "Joint: optimize all units together for best portfolio.",
    )
    opt_mode = opt_mode_label.lower()

    # Search Mode — only visible when Joint is selected
    if opt_mode == 'joint':
        search_col1, search_col2 = st.columns([1, 1])
        with search_col1:
            search_mode = st.radio(
                "Search Mode",
                options=["Standard", "Modified"],
                index=0,  # Default to Standard
                horizontal=True,
                key="search_mode",
                help=(
                    "**Standard:** TFC's proprietary application of Greedy Equivalence Search. "
                    "Evaluates the full combinatorial space sequentially for mathematically perfect global optimums.\n"
                    "**Modified:** Narrows each unit to its top candidates before joint optimization "
                    "for faster execution on very large portfolios."
                ),
            )
        with search_col2:
            if search_mode == "Modified":
                top_k_value = st.slider(
                    "Pre-Filter Top K per Unit",
                    min_value=100,
                    max_value=5000,
                    value=500,
                    step=100,
                    key="top_k_slider",
                    help=(
                        "Number of top independent candidates retained per unit before joint search. "
                        "500 is recommended. Higher values are slower but reduce the chance of missing "
                        "a rare cross-unit diversification benefit."
                    ),
                )
            else:
                top_k_value = None
        st.markdown("### Calculation Engine")
        calc_engine_label = st.radio(
            "Select the underlying computation engine:",
            options=["Python (Standard)", "Numba (JIT Compiled)"],
            index=0,
            help="Python uses standard NumPy vectorization. Numba uses Just-In-Time C-compilation to bypass Python's memory limits, computing billions of combinations in seconds."
        )
        calc_engine = 'numba' if 'Numba' in calc_engine_label else 'python'
    else:
        search_mode = "Standard"
        top_k_value = None
        calc_engine = 'python'

    cov_mode_label = st.radio(
        "Coverage Level Mode",
        options=[
            "No Coverage Level Optimization",
            "Uniform",
            "Per Category (PRF vs AF)",
            "Per County-Crop",
        ],
        horizontal=True,
        key="cov_mode",
        help=(
            "No Optimization: uses each unit's manually selected coverage level. "
            "Uniform: tests all 5 levels, applies the best single level to every unit. "
            "Per Category: best PRF level + best AF level independently (25 combos). "
            "Per County-Crop: optimizes per (county, crop_type) group subject to USDA rules."
        ),
    )
    coverage_mode_map = {
        "No Coverage Level Optimization": "none",
        "Uniform": "uniform",
        "Per Category (PRF vs AF)": "per_category",
        "Per County-Crop": "per_county_crop",
    }
    coverage_mode = coverage_mode_map[cov_mode_label]

    has_cat = any(cfg.get('is_cat', False) for cfg in unit_configs.values())
    if coverage_mode != 'none' and has_cat:
        st.caption("CAT units are excluded from coverage optimization (fixed at 65%).")

    submitted = st.button("⚡ Run Unified Optimizer")


    # ═══════════════════════════════════════════════════════════════════════════
    # Run Optimization
    # ═══════════════════════════════════════════════════════════════════════════

    if submitted:

        # ── Validation ──
        errors = []
        warnings = []

        # Check for duplicate PRF grid+use combos
        prf_keys = []
        af_keys = []
        grids_used = {}

        for uid, cfg in unit_configs.items():
            if cfg.get('grid') in (None, 'N/A', '--'):
                errors.append(f"Unit {uid}: No grid selected.")
                continue

            if cfg['type'] == 'PRF':
                key = (cfg['grid'], cfg['intended_use'])
                if key in prf_keys:
                    errors.append(
                        f"Duplicate PRF unit: grid {cfg['grid']} / {cfg['intended_use']}. "
                        "Same grid cannot be insured twice for the same intended use."
                    )
                prf_keys.append(key)

            elif cfg['type'] == 'AF':
                if cfg.get('growing_season') is None:
                    errors.append(f"Unit {uid}: No growing season selected.")
                    continue
                key = (cfg['grid'], cfg['growing_season'])
                if key in af_keys:
                    errors.append(
                        f"Duplicate AF unit: grid {cfg['grid']} / {cfg['growing_season']}."
                    )
                af_keys.append(key)

            # Track grids for cross-program info
            g = cfg['grid']
            if g not in grids_used:
                grids_used[g] = set()
            grids_used[g].add(cfg['type'])

        # Cross-program grid info
        for g, types in grids_used.items():
            if len(types) > 1:
                warnings.append(
                    f"Grid {g}: PRF and AF cover different land types on the same grid. "
                    "Ensure acreage doesn't overlap."
                )

        # AF cross-season overlap check (pairwise)
        af_by_grid = {}
        for uid, cfg in unit_configs.items():
            if cfg['type'] == 'AF' and cfg.get('growing_season'):
                g = cfg.get('grid_id') or _grid_id_from_label(cfg['grid'])
                if g not in af_by_grid:
                    af_by_grid[g] = []
                af_by_grid[g].append(cfg)

        for gid, group in af_by_grid.items():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                gs_i = group[i]['growing_season']
                for j in range(i + 1, len(group)):
                    gs_j = group[j]['growing_season']
                    overlap = compute_shared_intervals(gs_i, gs_j)
                    if overlap:
                        next_gs = compute_next_eligible_season(gs_i)
                        next_label = SEASON_LABELS.get(next_gs, '?')
                        warnings.append(
                            f"**{group[i].get('unit_label', f'Unit {i+1}')}** (GS-{gs_i}) and "
                            f"**{group[j].get('unit_label', f'Unit {j+1}')}** (GS-{gs_j}) on Grid {gid} "
                            f"share {len(overlap)} interval(s): {', '.join(sorted(overlap))}. "
                            f"Ensure these policies represent **different physical acreage**. "
                            f"If same dirt, the next eligible season after GS-{gs_i} is "
                            f"**GS-{next_gs} ({next_label})** — see Appendix for details."
                        )

        if errors:
            for e in errors:
                st.error(e)
        else:
            for w in warnings:
                st.warning(f"⚠️ {w}")

            # ── Run the optimizer ──
            progress_bar = st.progress(0, text="Preparing optimization...")

            metric_display_names = {
                'sharpe': 'Sharpe', 'cvar': 'CVaR (5th pctl)',
                'roi': 'Return %', 'winrate': 'Win Rate',
            }

            if coverage_mode == 'none':
                # ── Original flow: no coverage search ──
                progress_bar.progress(0, text="Running optimization...")
                def _none_progress(pct, msg=''):
                    progress_bar.progress(min(int(pct * 99), 99),
                                          text=msg or "Running optimization...")

                score, results, units_data_for_joint = _run_optimization_pipeline(
                    unit_configs, opt_metric, opt_mode, weight_step,
                    prf_interval_range, start_year, end_year,
                    progress_callback=_none_progress, top_k=top_k_value,
                    calc_engine=calc_engine,
                )
                results['coverage_mode'] = 'none'
                progress_bar.progress(100, text="Complete!")
                st.session_state.optimization_results = results
                if 's2_results' in st.session_state:
                    del st.session_state['s2_results']

            elif coverage_mode == 'uniform':
                # ── Uniform: test all 5 levels ──
                best_score = -np.inf
                best_results = None
                cov_comparison = []

                candidates_cache = _enumerate_all_candidates(
                    unit_configs, weight_step, prf_interval_range
                )
                n_cov = len(COVERAGE_LEVELS_TO_TEST)
                for c_idx, cov in enumerate(COVERAGE_LEVELS_TO_TEST):
                    base_pct = int(c_idx / n_cov * 100)
                    slice_size = int(100 / n_cov)
                    progress_bar.progress(
                        min(base_pct, 99),
                        text=f"Testing coverage: {int(cov*100)}% ({c_idx+1}/{n_cov})..."
                    )

                    def _uni_progress(pct, msg='', _base=base_pct, _slice=slice_size, _cov=cov, _cidx=c_idx, _n=n_cov, _state=[-1]):
                        inner_pct = min(_base + int(pct * _slice), 99)
                        if inner_pct > _state[0]:
                            _state[0] = inner_pct
                            progress_bar.progress(
                                inner_pct,
                                text=msg or f"Testing {int(_cov*100)}% ({_cidx+1}/{_n}) — optimizing portfolio..."
                            )

                    modified = _override_coverage_all(unit_configs, cov)
                    score, results_i, _ = _backtest_and_score(
                        modified, candidates_cache, opt_metric, opt_mode,
                        start_year, end_year,
                        progress_callback=_uni_progress, top_k=top_k_value,
                        calc_engine=calc_engine,
                    )
                    cov_comparison.append({
                        'combo': cov,
                        'label': f"{int(cov*100)}%",
                        'score': score,
                    })
                    if score > best_score:
                        best_score = score
                        best_results = results_i
                        best_cov = cov

                best_results['coverage_mode'] = 'uniform'
                best_results['coverage_best'] = best_cov
                best_results['coverage_comparison'] = cov_comparison
                best_results['coverage_metric_name'] = metric_display_names.get(opt_metric, opt_metric)
                progress_bar.progress(100, text="Complete!")
                st.session_state.optimization_results = best_results
                if 's2_results' in st.session_state:
                    del st.session_state['s2_results']

            elif coverage_mode == 'per_category':
                # ── Per Category: 5×5 = 25 combos ──
                best_score = -np.inf
                best_results = None
                cov_comparison = []
                total_combos = len(COVERAGE_LEVELS_TO_TEST) ** 2
                combo_num = 0

                candidates_cache = _enumerate_all_candidates(
                    unit_configs, weight_step, prf_interval_range
                )
                for prf_cov in COVERAGE_LEVELS_TO_TEST:
                    for af_cov in COVERAGE_LEVELS_TO_TEST:
                        combo_num += 1
                        base_pct = int((combo_num - 1) / total_combos * 100)
                        slice_size = max(1, int(100 / total_combos))
                        progress_bar.progress(
                            min(base_pct, 99),
                            text=f"Testing coverage: PRF {int(prf_cov*100)}% / AF {int(af_cov*100)}% "
                                 f"(combo {combo_num}/{total_combos})..."
                        )

                        def _cat_progress(pct, msg='', _base=base_pct, _slice=slice_size,
                                          _prf=prf_cov, _af=af_cov, _num=combo_num, _total=total_combos, _state=[-1]):
                            inner_pct = min(_base + int(pct * _slice), 99)
                            if inner_pct > _state[0]:
                                _state[0] = inner_pct
                                progress_bar.progress(
                                    inner_pct,
                                    text=msg or f"PRF {int(_prf*100)}% / AF {int(_af*100)}% "
                                         f"(combo {_num}/{_total}) — optimizing portfolio..."
                                )

                        modified = _override_coverage_by_category(
                            unit_configs, prf_cov, af_cov
                        )
                        score, results_i, _ = _backtest_and_score(
                            modified, candidates_cache, opt_metric, opt_mode,
                            start_year, end_year,
                            progress_callback=_cat_progress, top_k=top_k_value,
                            calc_engine=calc_engine,
                        )
                        cov_comparison.append({
                            'combo': (prf_cov, af_cov),
                            'label': f"PRF {int(prf_cov*100)}% / AF {int(af_cov*100)}%",
                            'score': score,
                        })
                        if score > best_score:
                            best_score = score
                            best_results = results_i
                            best_combo_cov = (prf_cov, af_cov)

                best_results['coverage_mode'] = 'per_category'
                best_results['coverage_best'] = best_combo_cov
                best_results['coverage_comparison'] = cov_comparison
                best_results['coverage_metric_name'] = metric_display_names.get(opt_metric, opt_metric)
                progress_bar.progress(100, text="Complete!")
                st.session_state.optimization_results = best_results
                if 's2_results' in st.session_state:
                    del st.session_state['s2_results']

            elif coverage_mode == 'per_county_crop':
                # ── Per County-Crop: group units, then exhaustive or greedy ──
                # Build groups (skip CAT units for grouping)
                groups = {}  # (county, crop_type) -> list of uids
                for uid, cfg in unit_configs.items():
                    if cfg.get('is_cat', False):
                        continue
                    key = _get_county_crop_key(cfg)
                    groups.setdefault(key, []).append(uid)

                group_keys = sorted(groups.keys())
                n_groups = len(group_keys)

                if n_groups > 0:
                    group_summary = []
                    for gk in group_keys:
                        county, crop = gk
                        n_units = len(groups[gk])
                        group_summary.append(f"{crop} {county} ({n_units} unit{'s' if n_units > 1 else ''})")
                    st.caption(f"County-Crop Groups: {', '.join(group_summary)} — units in the same group must share a coverage level (USDA Rule 1)")

                # Compute default coverage per group (from first unit in group)
                default_covs = {}
                for gk in group_keys:
                    first_uid = groups[gk][0]
                    default_covs[gk] = unit_configs[first_uid]['coverage_level']

                best_score = -np.inf
                best_results = None
                cov_comparison = []

                candidates_cache = _enumerate_all_candidates(
                    unit_configs, weight_step, prf_interval_range
                )

                if n_groups == 0:
                    # All CAT units — just run as-is
                    score, results_i, _ = _backtest_and_score(
                        unit_configs, candidates_cache, opt_metric, opt_mode,
                        start_year, end_year, top_k=top_k_value,
                        calc_engine=calc_engine,
                    )
                    best_results = results_i
                    best_combo_cov = ()

                elif n_groups <= 6:
                    # Exhaustive search
                    total_combos = len(COVERAGE_LEVELS_TO_TEST) ** n_groups
                    combo_num = 0
                    for combo in itertools.product(
                        COVERAGE_LEVELS_TO_TEST, repeat=n_groups
                    ):
                        combo_num += 1
                        label = _format_coverage_combo_label(
                            'per_county_crop', combo, group_keys
                        )
                        base_pct = int((combo_num - 1) / total_combos * 100)
                        slice_size = max(1, int(100 / total_combos))
                        progress_bar.progress(
                            min(base_pct, 99),
                            text=f"Testing: {label} "
                                 f"(combo {combo_num}/{total_combos})..."
                        )

                        def _cc_progress(pct, msg='', _num=combo_num, _total=total_combos, _label=label, _state=[-1]):
                            base = int((_num - 1) / _total * 100)
                            slice_s = max(1, int(100 / _total))
                            inner_pct = min(base + int(pct * slice_s), 99)
                            if inner_pct > _state[0]:
                                _state[0] = inner_pct
                                progress_bar.progress(
                                    inner_pct,
                                    text=msg or f"{_label} (combo {_num}/{_total}) — optimizing portfolio..."
                                )

                        modified = _override_coverage_by_groups(
                            unit_configs, groups, group_keys, combo
                        )
                        score, results_i, _ = _backtest_and_score(
                            modified, candidates_cache, opt_metric, opt_mode,
                            start_year, end_year,
                            progress_callback=_cc_progress, top_k=top_k_value,
                            calc_engine=calc_engine,
                        )
                        cov_comparison.append({
                            'combo': combo,
                            'label': label,
                            'score': score,
                        })
                        if score > best_score:
                            best_score = score
                            best_results = results_i
                            best_combo_cov = combo

                else:
                    # Greedy: sort groups by total coverage descending
                    group_total_cov = {}
                    for gk in group_keys:
                        total = 0.0
                        for uid in groups[gk]:
                            cfg = unit_configs[uid]
                            total += cfg['acres'] * cfg.get('coverage_level', 0.9)
                        group_total_cov[gk] = total

                    sorted_group_keys = sorted(
                        group_keys,
                        key=lambda gk: (-group_total_cov[gk], gk)
                    )

                    locked = {}
                    best_level_results = None
                    for g_idx, gk in enumerate(sorted_group_keys):
                        best_level_score = -np.inf
                        best_level = default_covs[gk]
                        county, crop = gk
                        for cov in COVERAGE_LEVELS_TO_TEST:
                            progress_bar.progress(
                                min(int((g_idx) / n_groups * 99), 99),
                                text=f"Optimizing group {crop}-{county} "
                                     f"({g_idx+1}/{n_groups}): "
                                     f"testing {int(cov*100)}%..."
                            )

                            def _cc_greedy_progress(pct, msg='', _gidx=g_idx, _n=n_groups,
                                                    _crop=crop, _county=county, _cov=cov, _state=[-1]):
                                base = int((_gidx) / _n * 100)
                                slice_s = max(1, int(100 / _n / len(COVERAGE_LEVELS_TO_TEST)))
                                inner_pct = min(base + int(pct * slice_s), 99)
                                if inner_pct > _state[0]:
                                    _state[0] = inner_pct
                                    progress_bar.progress(
                                        inner_pct,
                                        text=msg or f"Optimizing {_crop}-{_county} at {int(_cov*100)}% "
                                             f"({_gidx+1}/{_n})..."
                                    )

                            # Build combo: locked groups + current test + defaults for rest
                            combo_dict = dict(locked)
                            combo_dict[gk] = cov
                            for other_gk in group_keys:
                                if other_gk not in combo_dict:
                                    combo_dict[other_gk] = default_covs[other_gk]
                            combo = tuple(combo_dict[k] for k in group_keys)
                            modified = _override_coverage_by_groups(
                                unit_configs, groups, group_keys, combo
                            )
                            score, results_i, _ = _backtest_and_score(
                                modified, candidates_cache, opt_metric, opt_mode,
                                start_year, end_year,
                                progress_callback=_cc_greedy_progress, top_k=top_k_value,
                                calc_engine=calc_engine,
                            )
                            cov_comparison.append({
                                'combo': cov,
                                'label': f"{crop} {county}: {int(cov*100)}%",
                                'score': score,
                                'group': f"{crop} {county}",
                                'is_locked': False,
                            })
                            if score > best_level_score:
                                best_level_score = score
                                best_level = cov
                                best_level_results = results_i

                        locked[gk] = best_level
                        # Mark the locked level for this group
                        for entry in cov_comparison:
                            if entry.get('group') == f"{crop} {county}" and entry['combo'] == best_level:
                                entry['is_locked'] = True

                    # Final combo
                    best_combo_cov = tuple(locked.get(k, default_covs[k]) for k in group_keys)
                    if best_level_results is not None:
                        best_results = best_level_results
                        best_score = best_level_score
                    else:
                        # Fallback: run with defaults
                        best_score, best_results, _ = _backtest_and_score(
                            unit_configs, candidates_cache, opt_metric, opt_mode,
                            start_year, end_year, top_k=top_k_value,
                            calc_engine=calc_engine,
                        )

                best_results['coverage_mode'] = 'per_county_crop'
                best_results['coverage_best'] = best_combo_cov
                best_results['coverage_group_keys'] = group_keys
                best_results['coverage_comparison'] = cov_comparison
                best_results['coverage_metric_name'] = metric_display_names.get(opt_metric, opt_metric)
                progress_bar.progress(100, text="Complete!")
                st.session_state.optimization_results = best_results
                if 's2_results' in st.session_state:
                    del st.session_state['s2_results']


    # ═══════════════════════════════════════════════════════════════════════════
    # Results Display
    # ═══════════════════════════════════════════════════════════════════════════

    res = st.session_state.optimization_results

    if res is not None and res.get('units'):
        st.markdown("---")

        # ── Coverage Level Optimization Results ──
        cov_mode_res = res.get('coverage_mode', 'none')
        if cov_mode_res != 'none':
            cov_best = res.get('coverage_best')
            cov_comparison = res.get('coverage_comparison', [])
            cov_metric_name = res.get('coverage_metric_name', 'Score')
            cov_group_keys = res.get('coverage_group_keys', [])

            # Summary banner
            if cov_mode_res == 'uniform' and cov_best is not None:
                banner_text = f"Optimal Coverage: {int(cov_best*100)}%"
            elif cov_mode_res == 'per_category' and cov_best is not None:
                banner_text = (f"Optimal Coverage: PRF {int(cov_best[0]*100)}% / "
                               f"AF {int(cov_best[1]*100)}%")
            elif cov_mode_res == 'per_county_crop' and cov_best is not None and cov_group_keys:
                parts = []
                for gk, cv in zip(cov_group_keys, cov_best):
                    county, crop = gk
                    parts.append(f"{crop} {county}: {int(cv*100)}%")
                banner_text = "Optimal Coverage: " + ", ".join(parts)
            else:
                banner_text = "Coverage Level Optimization Complete"

            st.markdown(f"""
            <div style="background:linear-gradient(135deg, {FC_GREEN}22, {FC_GREEN}11);
                        border:2px solid {FC_GREEN}; border-radius:10px; padding:16px 20px;
                        margin-bottom:16px;">
                <div style="font-size:1.2em; font-weight:700; color:{FC_GREEN};">
                    🎯 {banner_text}
                </div>
            </div>""", unsafe_allow_html=True)

            # Comparison table
            if cov_comparison:
                st.markdown("#### Coverage Level Comparison")

                # Find the best score
                best_score_val = max(c['score'] for c in cov_comparison
                                     if np.isfinite(c['score'])) if cov_comparison else -np.inf

                has_groups = any('group' in c for c in cov_comparison)

                table_rows = []
                for entry in cov_comparison:
                    is_locked = entry.get('is_locked', False)
                    is_best = (abs(entry['score'] - best_score_val) < 1e-10
                               and np.isfinite(entry['score']))
                    marker = '\u2b50' if (is_locked if has_groups else is_best) else ''
                    score_fmt = (f"{entry['score']:.4f}" if np.isfinite(entry['score'])
                                 else "N/A")
                    row = {
                        '': marker,
                        'Coverage': entry['label'],
                        cov_metric_name: score_fmt,
                    }
                    if has_groups:
                        row['Group'] = entry.get('group', '')
                    table_rows.append(row)

                cov_df = pd.DataFrame(table_rows)
                if has_groups:
                    cov_df = cov_df[['', 'Group', 'Coverage', cov_metric_name]]
                st.dataframe(cov_df, use_container_width=True, hide_index=True)

        mode = res['mode']
        metric_key = res['metric']
        valid_units = [u for u in res['units']
                       if 'metrics' in u and 'error' not in u]

        # Determine which index to use for each unit
        use_key = 'joint_idx' if (mode == 'joint' and 'joint_metrics' in res) else 'best_idx'

        # ── Compute display portfolio metrics ──
        if mode == 'joint' and 'joint_metrics' in res:
            display_metrics = res['joint_metrics']
            best_combo = res.get('best_combo')
            units_data = res.get('units_data_for_joint', [])
            indep_indices = res.get('indep_indices', [])
        else:
            display_metrics = None
            best_combo = tuple(
                res['units'][ud['_result_idx']]['best_idx']
                for ud in res.get('units_data_for_joint', [])
            )
            units_data = res.get('units_data_for_joint', [])
            indep_indices = res.get('indep_indices', [])

        # ── 1. Headline Metrics (Portfolio Results) ──
        st.markdown("### 📊 Portfolio Results")
        port_metrics = None  # will be set in independent mode

        if display_metrics:
            joint_m = display_metrics
            total_acres = sum(u['config']['acres'] for u in valid_units)
            headline_cols = st.columns(5)
            with headline_cols[0]:
                st.markdown(
                    _render_metric_card("Sharpe Ratio", f"{joint_m['sharpe']:.3f}"),
                    unsafe_allow_html=True,
                )
            with headline_cols[1]:
                st.markdown(
                    _render_metric_card("CVaR (5th pctl)", _fmt_dollar(joint_m['cvar'])),
                    unsafe_allow_html=True,
                )
            with headline_cols[2]:
                st.markdown(
                    _render_metric_card("Win Rate", f"{joint_m['winrate']*100:.1f}%"),
                    unsafe_allow_html=True,
                )
            with headline_cols[3]:
                st.markdown(
                    _render_metric_card("Return %", f"{joint_m['roi']*100:.0f}%"),
                    unsafe_allow_html=True,
                )
            with headline_cols[4]:
                st.markdown(
                    _render_metric_card("Producer Premium/ac",
                                        _fmt_dollar(joint_m['producer_cost'])),
                    unsafe_allow_html=True,
                )
        else:
            # Independent mode — show weighted portfolio metrics
            total_acres = sum(u['config']['acres'] for u in valid_units)
            if total_acres > 0 and len(valid_units) > 0:
                # Compute actual portfolio returns
                first_u = valid_units[0]
                n_years_ret = len(first_u['yearly_returns'][first_u['best_idx']])
                port_ret = np.zeros(n_years_ret)
                port_cost = 0.0
                for u in valid_units:
                    bi = u['best_idx']
                    port_ret += u['yearly_returns'][bi] * u['config']['acres']
                    port_cost += u['producer_costs'][bi] * u['config']['acres']
                port_ret /= total_acres
                port_cost /= total_acres
                port_metrics = _compute_all_metrics(port_ret, port_cost)

                headline_cols = st.columns(5)
                with headline_cols[0]:
                    st.markdown(
                        _render_metric_card("Sharpe Ratio", f"{port_metrics['sharpe']:.3f}"),
                        unsafe_allow_html=True,
                    )
                with headline_cols[1]:
                    st.markdown(
                        _render_metric_card("CVaR (5th pctl)", _fmt_dollar(port_metrics['cvar'])),
                        unsafe_allow_html=True,
                    )
                with headline_cols[2]:
                    st.markdown(
                        _render_metric_card("Win Rate", f"{port_metrics['winrate']*100:.1f}%"),
                        unsafe_allow_html=True,
                    )
                with headline_cols[3]:
                    st.markdown(
                        _render_metric_card("Return %", f"{port_metrics['roi']*100:.0f}%"),
                        unsafe_allow_html=True,
                    )
                with headline_cols[4]:
                    st.markdown(
                        _render_metric_card("Producer Premium/ac",
                                            _fmt_dollar(port_metrics['producer_cost'])),
                        unsafe_allow_html=True,
                    )

        # Unified Stage 1 metrics for downstream (Stage 2, reporting)
        if mode == 'joint' and 'joint_metrics' in res:
            stage1_metrics = res['joint_metrics']
        else:
            stage1_metrics = port_metrics

        # ── 2. Total Premium + Total Coverage side-by-side ──
        if len(valid_units) >= 1:
            prem_cov_col1, prem_cov_col2 = st.columns(2)

            with prem_cov_col1:
                st.markdown(f"""
                <div style="background:#f8f9fa; border:1px solid #dee2e6; border-radius:8px; padding:16px;">
                    <div style="font-weight:700; color:{FC_SLATE}; font-size:1.05em; margin-bottom:10px;">
                        💰 Total Premium
                    </div>""", unsafe_allow_html=True)

                prem_total = 0.0
                has_cat_units = False
                for u in valid_units:
                    cfg = u['config']
                    use_idx_u = u.get(use_key, u.get('best_idx', 0))
                    unit_cost_per_ac = u['producer_costs'][use_idx_u]
                    unit_prem_total = unit_cost_per_ac * cfg['acres']
                    prem_total += unit_prem_total
                    is_cat_unit = cfg.get('is_cat', False)
                    if is_cat_unit:
                        has_cat_units = True
                    cat_tag = (" <span style='background:#FF9800; color:white; font-size:0.75em; "
                              "padding:1px 6px; border-radius:3px;'>CAT</span>") if is_cat_unit else ""
                    unit_desc = cfg.get('unit_label', f"{cfg['type']} Grid {cfg['grid']}")
                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; padding:4px 0;
                                border-bottom:1px solid #eee; font-size:0.9em;">
                        <span>{unit_desc}{cat_tag}</span>
                        <span><strong>${unit_prem_total:,.0f}</strong>
                              <span style="color:#888;">({unit_cost_per_ac:,.2f}/ac)</span></span>
                    </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; padding:8px 0 0;
                                font-size:1em; font-weight:700; color:{FC_GREEN};">
                        <span>Portfolio Total</span>
                        <span>${prem_total:,.0f}
                              <span style="font-weight:400; color:#888;">({prem_total/max(total_acres,1):,.2f}/ac)</span></span>
                    </div>""", unsafe_allow_html=True)

                if has_cat_units:
                    st.markdown("""
                    <div style="margin-top:8px; padding:8px 10px; background:#fff3e0;
                                border-left:3px solid #FF9800; border-radius:0 4px 4px 0; font-size:0.82em;">
                        <strong>CAT Note:</strong> CAT premium is 100% subsidized ($0/ac rate-based).
                        Producer pays a <strong>$655 administrative fee</strong> per crop per county,
                        not reflected in per-acre calculations.
                    </div>""", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

            with prem_cov_col2:
                st.markdown(f"""
                <div style="background:#f8f9fa; border:1px solid #dee2e6; border-radius:8px; padding:16px;">
                    <div style="font-weight:700; color:{FC_SLATE}; font-size:1.05em; margin-bottom:10px;">
                        🛡️ Total Coverage
                    </div>""", unsafe_allow_html=True)

                cov_total = 0.0
                for u in valid_units:
                    cfg = u['config']
                    ii = cfg.get('insurable_interest', 1.0)
                    # Compute dollar amount of protection
                    try:
                        if cfg['type'] == 'AF' and cfg.get('grid_id'):
                            cbv_d = af_load_cbv(cfg['grid_id'], cfg['growing_season'])
                            cbv = cbv_d['county_base_value']
                        elif cfg['type'] == 'PRF':
                            cbv = prf_load_cbv(
                                cfg['grid'], cfg.get('intended_use', 'Grazing'),
                                cfg.get('irrigation', 'N/A'),
                                cfg.get('organic', 'No Organic Practice Specified'),
                            ) or 0.0
                        else:
                            cbv = 0.0
                    except Exception:
                        cbv = 0.0
                    da_display = _round_half_up(cbv * cfg['coverage_level'] * cfg['productivity'], 2)
                    unit_coverage = da_display * ii * cfg['acres']
                    cov_per_ac = da_display * ii
                    cov_total += unit_coverage
                    is_cat_unit = cfg.get('is_cat', False)
                    cat_tag = (" <span style='background:#FF9800; color:white; font-size:0.75em; "
                              "padding:1px 6px; border-radius:3px;'>CAT</span>") if is_cat_unit else ""
                    unit_desc = cfg.get('unit_label', f"{cfg['type']} Grid {cfg['grid']}")
                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; padding:4px 0;
                                border-bottom:1px solid #eee; font-size:0.9em;">
                        <span>{unit_desc}{cat_tag}</span>
                        <span><strong>${unit_coverage:,.0f}</strong>
                              <span style="color:#888;">({cov_per_ac:,.2f}/ac)</span></span>
                    </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; padding:8px 0 0;
                                font-size:1em; font-weight:700; color:{FC_GREEN};">
                        <span>Portfolio Total</span>
                        <span>${cov_total:,.0f}
                              <span style="font-weight:400; color:#888;">({cov_total/max(total_acres,1):,.2f}/ac)</span></span>
                    </div>
                </div>""", unsafe_allow_html=True)

        # ── 3. Why Joint Matters ──
        if mode == 'joint' and 'joint_metrics' in res and len(valid_units) >= 2 and units_data and indep_indices:
            st.markdown("### 🔄 Why Joint Matters")

            # Compute independent portfolio metrics
            total_ac = sum(ud['acres'] for ud in units_data)
            indep_portfolio_ret = sum(
                units_data[k]['yearly_returns'][indep_indices[k]:indep_indices[k]+1, :] * units_data[k]['acres']
                for k in range(len(units_data))
            ).flatten() / total_ac
            indep_cost = sum(
                units_data[k]['producer_costs'][indep_indices[k]] * units_data[k]['acres']
                for k in range(len(units_data))
            ) / total_ac
            indep_metrics = _compute_all_metrics(indep_portfolio_ret, indep_cost)

            joint_m = res['joint_metrics']

            comp_col1, comp_col2 = st.columns(2)

            with comp_col1:
                st.markdown(f"""
                <div class="opt-comparison-box">
                    <div style="font-weight:700; margin-bottom:8px;">Independent Optimization</div>
                    <p>Sharpe: <strong>{indep_metrics['sharpe']:.3f}</strong></p>
                    <p>CVaR: <strong>${indep_metrics['cvar']:,.2f}</strong></p>
                    <p>Win Rate: <strong>{indep_metrics['winrate']*100:.1f}%</strong></p>
                    <p>Return: <strong>{indep_metrics['roi']*100:.0f}%</strong></p>
                </div>""", unsafe_allow_html=True)

            with comp_col2:
                st.markdown(f"""
                <div class="opt-comparison-box" style="border-color: {FC_GREEN};">
                    <div style="font-weight:700; color:{FC_GREEN}; margin-bottom:8px;">Joint Optimization</div>
                    <p>Sharpe: <strong>{joint_m['sharpe']:.3f}</strong> {_fmt_delta(joint_m['sharpe'] - indep_metrics['sharpe'], fmt='raw')}</p>
                    <p>CVaR: <strong>${joint_m['cvar']:,.2f}</strong> {_fmt_delta(joint_m['cvar'] - indep_metrics['cvar'], fmt='dollar', higher_is_better=True)}</p>
                    <p>Win Rate: <strong>{joint_m['winrate']*100:.1f}%</strong> {_fmt_delta(joint_m['winrate'] - indep_metrics['winrate'], fmt='pct')}</p>
                    <p>Return: <strong>{joint_m['roi']*100:.0f}%</strong> {_fmt_delta(joint_m['roi'] - indep_metrics['roi'], fmt='pct')}</p>
                </div>""", unsafe_allow_html=True)

            # Insight text using generate_insight_text for AF-only portfolios
            all_af = all(ud.get('type') == 'AF' or ud.get('growing_season') for ud in units_data)
            if all_af and best_combo and indep_indices:
                try:
                    insight = generate_insight_text(best_combo, indep_indices, units_data)
                    st.info(insight)
                except Exception:
                    pass  # Fallback: no insight text

        # ── 4. Per-Unit Allocation Table ──
        if valid_units:
            st.markdown("### 📋 Per-Unit Allocation")

            alloc_rows = []
            for u_idx, ur in enumerate(valid_units):
                cfg = ur['config']
                best = ur.get(use_key, ur.get('best_idx', 0))

                if 'candidates' in ur and ur['candidates']:
                    cand = ur['candidates'][best]

                    if cfg.get('is_cat'):
                        cat_iv = get_cat_interval(cfg.get('growing_season'))
                        cat_name = list(cat_iv.values())[0] if cat_iv else 'Full Season'
                        selected_names = [f"{cat_name} (CAT)"]
                        selected_weights = ['100%']
                    elif cfg['type'] == 'AF' and cfg.get('growing_season'):
                        intervals_k = get_buyup_intervals(cfg['growing_season'])
                        codes_k = sorted(intervals_k.keys())
                        selected_names = []
                        selected_weights = []
                        for idx in range(len(cand[1])):
                            if cand[1][idx] > 0:
                                if idx < len(codes_k):
                                    selected_names.append(intervals_k[codes_k[idx]])
                                selected_weights.append(f"{cand[1][idx]*100:.0f}%")
                    else:
                        # PRF
                        labels = ur.get('interval_labels', [])
                        selected_names = []
                        selected_weights = []
                        for idx, w in enumerate(cand[1]):
                            if w > 0.005:
                                lbl = labels[idx] if idx < len(labels) else f"Interval {idx+1}"
                                selected_names.append(lbl)
                                selected_weights.append(f"{w*100:.0f}%")

                    # Compute unit-level metrics
                    unit_returns = ur['yearly_returns'][best]
                    unit_cost = ur['producer_costs'][best]
                    unit_roi = float((np.mean(unit_returns) + unit_cost) / unit_cost) if unit_cost > 0 else 0
                    unit_winrate = float(np.mean(unit_returns > 0))

                    gs_label = ""
                    if cfg['type'] == 'AF' and cfg.get('growing_season'):
                        gs = cfg['growing_season']
                        gs_label = f"GS-{gs} ({SEASON_LABELS.get(gs, '')})"

                    alloc_rows.append({
                        'Unit': cfg.get('unit_label', f"Unit {u_idx+1}"),
                        'Grid': cfg.get('grid', ''),
                        'Season': gs_label if cfg['type'] == 'AF' else 'PRF',
                        'Coverage': f"{int(cfg['coverage_level']*100)}%",
                        'Acres': f"{cfg['acres']:,.0f}",
                        'Selected Intervals': ', '.join(selected_names),
                        'Weights': ', '.join(selected_weights),
                        'Unit Return %': f"{unit_roi*100:.0f}%",
                        'Unit Win Rate': f"{unit_winrate*100:.1f}%",
                    })

            if alloc_rows:
                st.dataframe(
                    pd.DataFrame(alloc_rows),
                    use_container_width=True,
                    hide_index=True,
                )

        # ── 5. Month Coverage Timeline ──
        all_timeline_units = [(u_idx, ur) for u_idx, ur in enumerate(valid_units)]
        if all_timeline_units:
            st.markdown("### 📅 Month Coverage Timeline")

            # Build coverage map: month -> set of unit indices
            all_months_covered = {}
            for u_idx, ur in all_timeline_units:
                cfg = ur['config']
                best = ur.get(use_key, ur.get('best_idx', 0))
                cand = ur['candidates'][best]

                if cfg['type'] == 'AF':
                    if cfg.get('is_cat'):
                        for name in get_buyup_intervals(cfg['growing_season']).values():
                            months = interval_to_months(name)
                            if months:
                                for m in months:
                                    all_months_covered.setdefault(m, set()).add(u_idx)
                    else:
                        intervals_k = get_buyup_intervals(cfg['growing_season'])
                        codes_k = sorted(intervals_k.keys())
                        for idx in range(len(cand[1])):
                            if cand[1][idx] > 0 and idx < len(codes_k):
                                name = intervals_k[codes_k[idx]]
                                months = interval_to_months(name)
                                if months:
                                    for m in months:
                                        all_months_covered.setdefault(m, set()).add(u_idx)
                else:
                    # PRF unit: use INTERVAL_ORDER_11 and interval_to_months
                    weights = cand[1]
                    for idx, interval_name in enumerate(INTERVAL_ORDER_11):
                        if idx < len(weights) and weights[idx] > 0.005:
                            months = interval_to_months(interval_name)
                            if months:
                                for m in months:
                                    all_months_covered.setdefault(m, set()).add(u_idx)

            # Fiscal year order: Oct-Sep
            month_order = [9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8]
            month_labels = [MONTH_NAMES_SHORT[m] for m in month_order]

            # Build HTML table
            html_rows = ""
            for u_idx, ur in all_timeline_units:
                cfg = ur['config']
                unit_label = cfg.get('unit_label', f"Unit {u_idx+1}")
                cells = f"<td style='font-weight:600;'>{unit_label}</td>"
                for m_idx in month_order:
                    units_at_month = all_months_covered.get(m_idx, set())
                    if u_idx in units_at_month:
                        count = len(units_at_month)
                        circles = '●' * count
                        if count > 1:
                            cells += f"<td style='background:#FF6B35; color:white; text-align:center;'>{circles}</td>"
                        else:
                            cells += f"<td style='background:{FC_GREEN}; color:white; text-align:center;'>{circles}</td>"
                    else:
                        cells += "<td style='background:#f0f0f0; text-align:center;'>—</td>"
                html_rows += f"<tr>{cells}</tr>"

            header_cells = "<th>Unit</th>" + "".join(
                f"<th style='text-align:center;'>{m}</th>" for m in month_labels
            )

            st.markdown(f"""
            <table style="width:100%; border-collapse:collapse; font-size:0.9em;">
                <tr style="background:{FC_SLATE}; color:white;">{header_cells}</tr>
                {html_rows}
            </table>
            <p style="font-size:0.8em; color:#888; margin-top:4px;">
                <span style="color:{FC_GREEN};">●</span> = one unit &nbsp;
                <span style="color:#FF6B35;">●●</span> = two units &nbsp;
                <span style="color:#FF6B35;">●●●</span> = three units, etc.
            </p>
            """, unsafe_allow_html=True)

        # ── 6. Yearly Returns Chart ──
        if valid_units and 'yearly_returns' in valid_units[0]:
            st.markdown("---")
            st.markdown("### 📈 Historical Performance")

            # Find common years across all valid units to avoid shape mismatch
            year_sets = []
            for u in valid_units:
                u_years = u.get('years', np.array([]))
                if len(u_years) > 0 and 'yearly_returns' in u:
                    year_sets.append(set(u_years.tolist() if hasattr(u_years, 'tolist') else list(u_years)))

            if year_sets:
                common_years_set = year_sets[0]
                for ys in year_sets[1:]:
                    common_years_set = common_years_set & ys
                common_years = sorted(common_years_set)
            else:
                common_years = []

            if not common_years:
                st.warning("No overlapping historical years across units. Cannot display combined chart.")
            else:
                total_unit_years = max(len(ys) for ys in year_sets) if year_sets else 0
                if len(common_years) < total_unit_years:
                    st.info(f"Displaying {len(common_years)} common years across all units (some units have different year ranges).")

                common_years_arr = np.array(common_years)
                portfolio_ret = np.zeros(len(common_years))
                total_acres_chart = 0.0

                for u in valid_units:
                    idx = u.get(use_key, u.get('best_idx', 0))
                    if 'yearly_returns' in u:
                        u_years = u.get('years', np.array([]))
                        u_returns = u['yearly_returns'][idx]
                        # Filter to common years only
                        mask = np.isin(u_years, common_years_arr)
                        filtered_returns = u_returns[mask]
                        portfolio_ret += filtered_returns * u['config']['acres']
                        total_acres_chart += u['config']['acres']

                if total_acres_chart > 0:
                    portfolio_ret_per_acre = portfolio_ret / total_acres_chart
                else:
                    portfolio_ret_per_acre = portfolio_ret

                years_arr = common_years_arr

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=years_arr,
                    y=portfolio_ret_per_acre,
                    marker_color=[FC_GREEN if v >= 0 else '#dc3545' for v in portfolio_ret_per_acre],
                    name='Net Return/Acre',
                ))
                fig.update_layout(
                    title="Joint Portfolio — Net Return per Acre by Year",
                    xaxis_title="Year",
                    yaxis_title="Net Return per Acre ($)",
                    template="plotly_white",
                    height=400,
                    showlegend=False,
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red",
                              annotation_text="Break-even",
                              annotation_position="top left")
                st.plotly_chart(fig, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════════
        # Stage 2: HRP Acre Rebalancing
        # ══════════════════════════════════════════════════════════════════════
        if res is not None and len(units_data) >= 2:

            st.markdown("---")
            # 3a — Section header
            st.markdown(f"""
            <div style="height:4px; background:linear-gradient(90deg, {FC_GREEN}, {FC_SLATE});
                        border-radius:2px; margin-bottom:12px;"></div>
            <div style="background:linear-gradient(135deg, {FC_GREEN}18, {FC_GREEN}08);
                        border:1px solid {FC_GREEN}44; border-radius:10px; padding:16px 20px;
                        margin-bottom:16px;">
                <div style="font-size:1.15em; font-weight:700; color:{FC_GREEN};">
                    ⚖️ Stage 2: Acre Rebalancing (HRP)
                </div>
                <div style="font-size:0.92em; color:{FC_SLATE}; margin-top:4px;">
                    Optionally rebalance acres across units using Hierarchical Risk Parity
                    and apply budget constraints. Stage 1 interval weights are preserved.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 3b — Controls row 1
            s2_c1, s2_c2 = st.columns(2)
            with s2_c1:
                st.toggle("Enable HRP Acre Rebalancing", key="s2_enable_hrp")
            with s2_c2:
                if st.session_state.get('s2_enable_hrp', False):
                    st.slider("Max Turnover (%)", 0, 100, value=10, step=5, key="s2_turnover")

            # 3c — Controls row 2: budget
            s2_b1, s2_b2, s2_b3 = st.columns(3)
            with s2_b1:
                st.checkbox("Set Annual Premium Budget", key="s2_budget_enabled")
            with s2_b2:
                if st.session_state.get('s2_budget_enabled', False):
                    st.number_input("Max Annual Premium ($)", min_value=1000,
                                    value=50000, step=1000, key="s2_budget_amount")
            with s2_b3:
                if st.session_state.get('s2_budget_enabled', False):
                    st.checkbox("Auto-fill (scale up if under)", key="s2_budget_autofill")

            # 3d — Run button
            s2_any_active = (st.session_state.get('s2_enable_hrp', False)
                             or st.session_state.get('s2_budget_enabled', False))
            if s2_any_active:
                s2_run = st.button("⚖️ Run Acre Rebalancing", type="primary", key="s2_run")
            else:
                s2_run = False

            # ── Stage 2 execution logic ──
            if s2_run:
                n_units = len(units_data)
                total_acres = sum(ud['acres'] for ud in units_data)
                initial_weights = [ud['acres'] / total_acres for ud in units_data]
                rebalanced_acres = [ud['acres'] for ud in units_data]
                optimal_weights = list(initial_weights)
                budget_scale_factor = 1.0

                # HRP rebalancing
                if st.session_state.get('s2_enable_hrp', False):
                    returns_df = pd.DataFrame({
                        f"Unit_{k}": units_data[k]['yearly_returns'][best_combo[k]]
                        for k in range(n_units)
                    })
                    raw_weights = run_hrp(returns_df)

                    # Map back to ordered list
                    hrp_w = [raw_weights.get(f"Unit_{k}", 1.0 / n_units)
                             for k in range(n_units)]
                    hrp_sum = sum(hrp_w)
                    if hrp_sum > 0:
                        hrp_w = [w / hrp_sum for w in hrp_w]

                    # Apply turnover constraint
                    turnover = st.session_state.get('s2_turnover', 10) / 100.0
                    clipped = []
                    for k in range(n_units):
                        lo = initial_weights[k] * (1 - turnover)
                        hi = initial_weights[k] * (1 + turnover)
                        clipped.append(max(lo, min(hi, hrp_w[k])))

                    clip_sum = sum(clipped)
                    if clip_sum > 0:
                        optimal_weights = [c / clip_sum for c in clipped]
                    else:
                        optimal_weights = list(initial_weights)

                    rebalanced_acres = [total_acres * w for w in optimal_weights]

                # Budget constraint
                if st.session_state.get('s2_budget_enabled', False):
                    budget = float(st.session_state.get('s2_budget_amount', 50000))
                    total_cost = sum(
                        rebalanced_acres[k] * units_data[k]['producer_costs'][best_combo[k]]
                        for k in range(n_units)
                    )
                    if total_cost > budget:
                        budget_scale_factor = (budget * 0.9995) / total_cost
                        rebalanced_acres = [a * budget_scale_factor for a in rebalanced_acres]
                    elif (total_cost < budget
                          and st.session_state.get('s2_budget_autofill', False)):
                        budget_scale_factor = (budget * 0.9995) / total_cost
                        rebalanced_acres = [a * budget_scale_factor for a in rebalanced_acres]

                # Round to whole acres
                rebalanced_acres = [round(a) for a in rebalanced_acres]
                # Reconcile rounding error on the largest unit
                target_total = round(total_acres * budget_scale_factor)
                acre_diff = target_total - sum(rebalanced_acres)
                if acre_diff != 0:
                    largest_idx = max(range(n_units),
                                     key=lambda i: rebalanced_acres[i])
                    rebalanced_acres[largest_idx] += acre_diff

                # Compute new portfolio metrics
                new_total_acres = sum(rebalanced_acres)
                if new_total_acres > 0:
                    new_portfolio_ret = sum(
                        units_data[k]['yearly_returns'][best_combo[k]] * rebalanced_acres[k]
                        for k in range(n_units)
                    ) / new_total_acres
                    new_cost = sum(
                        units_data[k]['producer_costs'][best_combo[k]] * rebalanced_acres[k]
                        for k in range(n_units)
                    ) / new_total_acres
                else:
                    new_portfolio_ret = np.zeros(
                        len(units_data[0]['yearly_returns'][best_combo[0]])
                    )
                    new_cost = 0.0
                new_metrics = _compute_all_metrics(new_portfolio_ret, new_cost)

                st.session_state.s2_results = {
                    'original_acres': [ud['acres'] for ud in units_data],
                    'rebalanced_acres': rebalanced_acres,
                    'hrp_enabled': st.session_state.get('s2_enable_hrp', False),
                    'turnover_pct': st.session_state.get('s2_turnover', 10),
                    'budget_enabled': st.session_state.get('s2_budget_enabled', False),
                    'budget_amount': (float(st.session_state.get('s2_budget_amount', 50000))
                                      if st.session_state.get('s2_budget_enabled', False)
                                      else None),
                    'budget_scale_factor': budget_scale_factor,
                    'original_metrics': stage1_metrics,
                    'rebalanced_metrics': new_metrics,
                }

            # ── Stage 2 results display ──
            if 's2_results' in st.session_state:
                s2r = st.session_state.s2_results
                st.markdown("---")

                # --- Helper for delta formatting ---
                def _s2_delta(new_val, old_val, fmt=".3f", pct=False):
                    """Format delta between new and old values with arrow."""
                    delta = new_val - old_val
                    if pct:
                        text = f"{delta * 100:+.1f}%"
                    else:
                        text = f"{delta:+{fmt}}"
                    if delta > 0:
                        return f'<span class="opt-improve">▲ {text}</span>'
                    elif delta < 0:
                        return f'<span class="opt-worsen">▼ {text}</span>'
                    else:
                        return f'<span>─ {text}</span>'

                # 1. Before/After Acre Comparison Table
                st.markdown("#### Before / After Acre Allocation")
                table_rows = []
                for k, ud in enumerate(units_data):
                    orig = s2r['original_acres'][k]
                    rebal = s2r['rebalanced_acres'][k]
                    orig_total = sum(s2r['original_acres'])
                    rebal_total = sum(s2r['rebalanced_acres'])
                    orig_pct = (orig / orig_total * 100) if orig_total > 0 else 0
                    rebal_pct = (rebal / rebal_total * 100) if rebal_total > 0 else 0

                    unit_type = ud.get('type', 'AF')
                    if unit_type == 'AF':
                        gs = ud.get('growing_season', '')
                        season_str = f"GS-{gs} ({SEASON_LABELS.get(gs, '?')})"
                    else:
                        season_str = 'PRF'

                    grid_id = ud.get('grid_label') or ud.get('grid_id', '?')

                    table_rows.append({
                        'Unit': f"Unit {k+1}",
                        'Grid': str(grid_id),
                        'Season': season_str,
                        'Original Acres': orig,
                        'Original %': f"{orig_pct:.1f}%",
                        'Rebalanced Acres': rebal,
                        'Rebalanced %': f"{rebal_pct:.1f}%",
                        'Change': rebal - orig,
                    })
                st.dataframe(pd.DataFrame(table_rows),
                             use_container_width=True, hide_index=True)

                # 2. Before/After Metrics Comparison
                st.markdown("#### Before / After Portfolio Metrics")
                orig_m = s2r['original_metrics']
                new_m = s2r['rebalanced_metrics']

                s2_mc1, s2_mc2 = st.columns(2)
                with s2_mc1:
                    st.markdown(f"""
                    <div class="opt-comparison-box">
                        <div style="font-weight:700; margin-bottom:8px;">Stage 1 (Original)</div>
                        <p>Sharpe: <strong>{orig_m['sharpe']:.3f}</strong></p>
                        <p>CVaR: <strong>${orig_m['cvar']:,.2f}</strong></p>
                        <p>Win Rate: <strong>{orig_m['winrate']*100:.1f}%</strong></p>
                        <p>Return: <strong>{orig_m['mean_return']:+.2f}</strong> $/ac</p>
                        <p>Premium: <strong>${orig_m.get('producer_cost', orig_m.get('cost', 0)):.2f}</strong>/ac</p>
                    </div>""", unsafe_allow_html=True)
                with s2_mc2:
                    st.markdown(f"""
                    <div class="opt-comparison-box" style="border-color: {FC_GREEN};">
                        <div style="font-weight:700; color:{FC_GREEN}; margin-bottom:8px;">Stage 2 (Rebalanced)</div>
                        <p>Sharpe: <strong>{new_m['sharpe']:.3f}</strong> {_s2_delta(new_m['sharpe'], orig_m['sharpe'])}</p>
                        <p>CVaR: <strong>${new_m['cvar']:,.2f}</strong> {_s2_delta(new_m['cvar'], orig_m['cvar'], fmt=".2f")}</p>
                        <p>Win Rate: <strong>{new_m['winrate']*100:.1f}%</strong> {_s2_delta(new_m['winrate'], orig_m['winrate'], pct=True)}</p>
                        <p>Return: <strong>{new_m['mean_return']:+.2f}</strong> $/ac {_s2_delta(new_m['mean_return'], orig_m['mean_return'], fmt=".2f")}</p>
                        <p>Premium: <strong>${new_m.get('producer_cost', new_m.get('cost', 0)):.2f}</strong>/ac {_s2_delta(new_m.get('producer_cost', new_m.get('cost', 0)), orig_m.get('producer_cost', orig_m.get('cost', 0)), fmt=".2f")}</p>
                    </div>""", unsafe_allow_html=True)

                # 3. Budget Info callout
                if s2r['budget_enabled'] and abs(s2r['budget_scale_factor'] - 1.0) > 1e-6:
                    orig_premium = sum(
                        s2r['original_acres'][k] * units_data[k]['producer_costs'][best_combo[k]]
                        for k in range(len(units_data))
                    )
                    adj_premium = sum(
                        s2r['rebalanced_acres'][k] * units_data[k]['producer_costs'][best_combo[k]]
                        for k in range(len(units_data))
                    )
                    direction = "scaled down" if s2r['budget_scale_factor'] < 1.0 else "scaled up"
                    st.info(
                        f"Budget constraint applied — acres {direction}. "
                        f"Original total premium: ${orig_premium:,.0f} → "
                        f"Adjusted: ${adj_premium:,.0f} "
                        f"(budget: ${s2r['budget_amount']:,.0f}, "
                        f"scale factor: {s2r['budget_scale_factor']:.4f})"
                    )

                # 4. HRP Dendrogram (3+ units, HRP enabled)
                if (s2r['hrp_enabled']
                        and len(units_data) >= 3):
                    from scipy.cluster.hierarchy import linkage as _linkage, dendrogram as _dendrogram
                    from scipy.spatial.distance import squareform as _squareform

                    st.markdown("#### HRP Dendrogram")
                    returns_df = pd.DataFrame({
                        f"Unit_{k}": units_data[k]['yearly_returns'][best_combo[k]]
                        for k in range(len(units_data))
                    })
                    corr = returns_df.corr().values
                    dist = np.sqrt(0.5 * (1 - corr))
                    np.fill_diagonal(dist, 0)
                    condensed = _squareform(dist, checks=False)
                    link = _linkage(condensed, method='single')

                    # Build labels
                    labels = []
                    for k, ud in enumerate(units_data):
                        unit_type = ud.get('type', 'AF')
                        if unit_type == 'AF':
                            gs = ud.get('growing_season', '?')
                            gl = ud.get('grid_label') or ud.get('grid_id', '?')
                            labels.append(
                                f"U{k+1}: {SEASON_LABELS.get(gs, '?')} · {gl}"
                            )
                        else:
                            gl = ud.get('grid_label') or ud.get('grid_id', '?')
                            labels.append(f"U{k+1}: PRF · {gl}")

                    fig_dend, ax = plt.subplots(figsize=(8, 4))
                    _dendrogram(
                        link,
                        labels=labels,
                        ax=ax,
                        leaf_rotation=30,
                        leaf_font_size=9,
                        above_threshold_color=FC_GREEN,
                        color_threshold=0,
                    )
                    ax.set_title("HRP Clustering Dendrogram",
                                 fontsize=12, color=FC_SLATE)
                    ax.set_ylabel("Correlation Distance", fontsize=9, color=FC_SLATE)
                    ax.tick_params(axis='x', colors=FC_SLATE)
                    ax.tick_params(axis='y', colors=FC_SLATE)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    st.pyplot(fig_dend)
                    plt.close(fig_dend)

        # ══════════════════════════════════════════════════════════════════════
        # Audit & Reporting
        # ══════════════════════════════════════════════════════════════════════
        st.markdown(f"""
<div style="background: linear-gradient(90deg, {FC_GREEN}, {FC_SLATE});
        height: 3px; border-radius: 2px; margin: 32px 0 24px 0;"></div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div style="background:#f8f9fa; border-left:4px solid {FC_SLATE}; padding:16px 20px;
        border-radius:0 8px 8px 0; margin-bottom:16px;">
<span style="font-weight:700; color:{FC_SLATE}; font-size:1.1em;">📋 Audit & Reporting</span>
<br><span style="color:#666; font-size:0.9em;">
Export strategy reports and review optimization details.</span>
</div>
""", unsafe_allow_html=True)

        if best_combo and units_data:
            # Report version selector (only if Stage 2 was run)
            report_use_stage2 = False
            if 's2_results' in st.session_state:
                report_version = st.radio(
                    "Export report using:",
                    options=["Stage 1 — Original Acres", "Stage 2 — Rebalanced Acres"],
                    horizontal=True,
                    key="report_version_selector"
                )
                report_use_stage2 = (report_version == "Stage 2 — Rebalanced Acres")

            if st.button("📄 Generate Strategy Report (Word)", key="unified_generate_report"):
                try:
                    if report_use_stage2 and 's2_results' in st.session_state:
                        s2 = st.session_state.s2_results
                        report_units_data = copy.deepcopy(units_data)
                        for k, new_acres in enumerate(s2['rebalanced_acres']):
                            report_units_data[k]['acres'] = float(new_acres)
                        report_label = "Stage 2 (Rebalanced)"
                    else:
                        report_units_data = units_data
                        report_label = "Stage 1 (Original)"

                    with st.spinner("Generating Word document... This may take a few seconds."):
                        report_buf = generate_unified_optimizer_report_docx(
                            units_data=report_units_data,
                            best_combo=best_combo,
                            indep_results=list(zip(indep_indices, [0]*len(indep_indices))),
                            metric_key=metric_key,
                            start_year=int(min(units_data[0]['years'])),
                            end_year=int(max(units_data[0]['years'])),
                            coverage_mode=res.get('coverage_mode', 'none'),
                            coverage_best=res.get('coverage_best'),
                            coverage_comparison=res.get('coverage_comparison'),
                            coverage_group_keys=res.get('coverage_group_keys'),
                            coverage_metric_name=res.get('coverage_metric_name'),
                            get_buyup_intervals_fn=get_buyup_intervals,
                            get_cat_interval_fn=get_cat_interval,
                            rate_year=RATE_YEAR,
                            stage2_results=st.session_state.get('s2_results'),
                            report_stage=2 if report_use_stage2 else 1,
                        )

                    docx_bytes = report_buf.getvalue()
                    report_filename = f"unified_optimizer_report_{'s2' if report_use_stage2 else 's1'}.docx"
                    st.caption(f"Report generated using **{report_label}** acres.")
                    st.caption(f"Report size: {len(docx_bytes):,} bytes")

                    b64 = base64.b64encode(docx_bytes).decode()
                    href = (
                        f'<a href="data:application/vnd.openxmlformats-officedocument'
                        f'.wordprocessingml.document;base64,{b64}" '
                        f'download="{report_filename}">'
                        f'📄 Download Strategy Report (Word)</a>'
                    )
                    st.markdown(href, unsafe_allow_html=True)

                except Exception as e:
                    st.warning(f"Report generation failed: {e}")

            # -----------------------------------------------------------
            # Year-by-Year Audit Table
            # -----------------------------------------------------------
            with st.expander("🔍 Year-by-Year Audit"):

                # Configuration summary
                config_parts = []
                for k in range(len(units_data)):
                    ud = units_data[k]
                    cand = ud['candidates'][best_combo[k]]

                    if ud.get('type') == 'PRF':
                        # PRF unit
                        labels = INTERVAL_ORDER_11
                        sel = [f"{labels[i]} ({cand[1][i]*100:.0f}%)"
                               for i in range(len(cand[1])) if cand[1][i] > 0.005]
                        sel_str = ', '.join(sel)
                        season_str = f"PRF — {ud.get('intended_use', 'N/A')}"
                    elif ud.get('is_cat'):
                        cat_iv = get_cat_interval(ud['growing_season'])
                        cat_name = list(cat_iv.values())[0] if cat_iv else 'Full Season'
                        sel_str = f"{cat_name} (CAT 100%)"
                        season_str = f"GS-{ud['growing_season']} ({SEASON_LABELS.get(ud['growing_season'], '')})"
                    else:
                        intervals_k = get_buyup_intervals(ud['growing_season'])
                        codes_k = sorted(intervals_k.keys())
                        sel = [f"{intervals_k[codes_k[i]]} ({cand[1][i]*100:.0f}%)"
                               for i in range(6) if cand[1][i] > 0]
                        sel_str = ', '.join(sel)
                        season_str = f"GS-{ud['growing_season']} ({SEASON_LABELS.get(ud['growing_season'], '')})"

                    cov_label = "CAT (65%)" if ud.get('is_cat') else f"{int(ud['coverage_level']*100)}%"
                    config_parts.append(
                        f"**{ud['unit_label']}** — Grid {ud.get('grid_label', ud.get('grid_id', ''))}, "
                        f"{season_str}, "
                        f"Cov {cov_label}, "
                        f"PF {int(ud['productivity']*100)}%, "
                        f"{ud['acres']:,.0f} ac · {sel_str}"
                    )
                st.markdown("  \n".join(config_parts))
                st.markdown("---")

                # Pre-load per-unit audit data
                audit_unit_data = []
                for k in range(len(units_data)):
                    ud = units_data[k]
                    cand = ud['candidates'][best_combo[k]]

                    if ud.get('type') == 'PRF':
                        # PRF unit: load index data
                        indices_df = prf_load_all_indices(ud['grid_id'])
                        prf_years = indices_df['YEAR'].values.astype(int)
                        prf_matrix = np.zeros((len(prf_years), 11))
                        for i, iv in enumerate(INTERVAL_ORDER_11):
                            if iv in indices_df.columns:
                                prf_matrix[:, i] = indices_df[iv].values.astype(float)
                            else:
                                prf_matrix[:, i] = 100.0
                        audit_unit_data.append({
                            'is_cat': False,
                            'is_prf': True,
                            'weight_arr': cand[1],
                            'hist_matrix': prf_matrix,
                            'full_years': prf_years,
                            'interval_labels': INTERVAL_ORDER_11,
                            'da_full': ud.get('cbv', 0) * ud['coverage_level'] * ud['productivity'],
                            'trigger': 100.0 * ud['coverage_level'],
                            'producer_cost': ud['producer_costs'][best_combo[k]],
                            'acres': ud['acres'],
                        })
                    elif ud.get('is_cat'):
                        # AF CAT unit
                        hist_df = af_load_hist_indices(ud['grid_id'], ud['growing_season'])
                        cat_iv = get_cat_interval(ud['growing_season'])
                        cat_name = list(cat_iv.values())[0] if cat_iv else None
                        cat_years = hist_df['YEAR'].values if not hist_df.empty else np.array([])
                        cat_vals = hist_df[cat_name].values.astype(float) if (not hist_df.empty and cat_name in hist_df.columns) else np.array([])
                        audit_unit_data.append({
                            'is_cat': True,
                            'is_prf': False,
                            'cat_name': cat_name,
                            'cat_years': cat_years,
                            'cat_vals': cat_vals,
                            'da_full': ud['cbv'] * ud['coverage_level'] * ud['productivity'],
                            'trigger': 100.0 * ud['coverage_level'],
                            'producer_cost': ud['producer_costs'][best_combo[k]],
                            'acres': ud['acres'],
                        })
                    else:
                        # AF Buy-Up unit
                        hist_matrix_full, full_years = af_load_hist_matrix(ud['grid_id'], ud['growing_season'])
                        intervals_k = get_buyup_intervals(ud['growing_season'])
                        codes_k = sorted(intervals_k.keys())
                        audit_unit_data.append({
                            'is_cat': False,
                            'is_prf': False,
                            'weight_arr': cand[1],
                            'hist_matrix': hist_matrix_full,
                            'full_years': full_years,
                            'intervals_k': intervals_k,
                            'codes_k': codes_k,
                            'da_full': ud['cbv'] * ud['coverage_level'] * ud['productivity'],
                            'trigger': 100.0 * ud['coverage_level'],
                            'producer_cost': ud['producer_costs'][best_combo[k]],
                            'acres': ud['acres'],
                        })

                # Build year-by-year audit rows
                audit_rows = []
                plot_years_list = list(units_data[0]['years'])

                for yr_idx, year in enumerate(plot_years_list):
                    row = {'Year': int(year)}
                    portfolio_indemnity = 0.0
                    portfolio_premium = 0.0

                    for k in range(len(units_data)):
                        aud = audit_unit_data[k]
                        unit_indemnity = 0.0
                        unit_interval_details = []

                        if aud['is_cat']:
                            yr_pos = np.where(aud['cat_years'] == year)[0]
                            if len(yr_pos) > 0 and yr_pos[0] < len(aud['cat_vals']):
                                idx_val = float(aud['cat_vals'][yr_pos[0]])
                                if pd.notna(idx_val):
                                    protection_raw = aud['da_full'] * 1.0
                                    payout_i = indemnity(idx_val, aud['trigger'] / 100.0, protection_raw)
                                    unit_indemnity = payout_i
                                    unit_interval_details.append(f"{aud['cat_name']}={idx_val:.1f}")
                        elif aud.get('is_prf'):
                            yr_pos = np.where(aud['full_years'] == year)[0]
                            if len(yr_pos) > 0:
                                yr_row = aud['hist_matrix'][yr_pos[0]]
                                for i in range(11):
                                    if aud['weight_arr'][i] > 0.005:
                                        idx_val = yr_row[i]
                                        protection_raw = aud['da_full'] * aud['weight_arr'][i]
                                        payout_i = indemnity(idx_val, aud['trigger'] / 100.0, protection_raw)
                                        unit_indemnity += payout_i
                                        unit_interval_details.append(
                                            f"{INTERVAL_ORDER_11[i]}={idx_val:.1f}"
                                        )
                        else:
                            yr_pos = np.where(aud['full_years'] == year)[0]
                            if len(yr_pos) > 0:
                                yr_row = aud['hist_matrix'][yr_pos[0]]
                                for i in range(6):
                                    if aud['weight_arr'][i] > 0:
                                        idx_val = yr_row[i]
                                        protection_raw = aud['da_full'] * aud['weight_arr'][i]
                                        payout_i = indemnity(idx_val, aud['trigger'] / 100.0, protection_raw)
                                        unit_indemnity += payout_i
                                        unit_interval_details.append(
                                            f"{aud['intervals_k'][aud['codes_k'][i]]}={idx_val:.1f}"
                                        )

                        unit_cost = aud['producer_cost']
                        unit_net = unit_indemnity - unit_cost

                        row[f'U{k+1} Indices'] = ', '.join(unit_interval_details) if unit_interval_details else '—'
                        row[f'U{k+1} Indemnity/ac'] = f"${unit_indemnity:,.2f}"
                        row[f'U{k+1} Premium/ac'] = f"${unit_cost:,.2f}"
                        row[f'U{k+1} Net/ac'] = f"${unit_net:,.2f}"

                        portfolio_indemnity += unit_indemnity * aud['acres']
                        portfolio_premium += unit_cost * aud['acres']

                    total_ac_audit = sum(aud_u['acres'] for aud_u in audit_unit_data)
                    port_indem_ac = portfolio_indemnity / total_ac_audit
                    port_prem_ac = portfolio_premium / total_ac_audit
                    port_net_ac = port_indem_ac - port_prem_ac

                    row['Portfolio Indemnity/ac'] = f"${port_indem_ac:,.2f}"
                    row['Portfolio Premium/ac'] = f"${port_prem_ac:,.2f}"
                    row['Portfolio Net/ac'] = f"${port_net_ac:,.2f}"

                    audit_rows.append(row)

                audit_df = pd.DataFrame(audit_rows)
                audit_df = audit_df.sort_values('Year', ascending=False).reset_index(drop=True)

                # Compute joint portfolio for summary stats
                total_ac_s = sum(ud['acres'] for ud in units_data)
                joint_portfolio_audit = sum(
                    units_data[kk]['yearly_returns'][best_combo[kk]:best_combo[kk]+1, :] * units_data[kk]['acres']
                    for kk in range(len(units_data))
                ).flatten() / total_ac_s
                joint_cost_audit = sum(
                    units_data[kk]['producer_costs'][best_combo[kk]] * units_data[kk]['acres']
                    for kk in range(len(units_data))
                ) / total_ac_s
                joint_net_audit = joint_portfolio_audit - joint_cost_audit

                st.markdown(f"""
                <div style="display:flex; gap:16px; margin-bottom:12px; font-size:0.88em;">
                    <span><strong>Years:</strong> {len(plot_years_list)}</span>
                    <span><strong>Avg Net/ac:</strong> ${np.mean(joint_net_audit):,.2f}</span>
                    <span><strong>Win Rate:</strong> {np.mean(joint_net_audit > 0)*100:.1f}%</span>
                    <span><strong>Best Year:</strong> ${np.max(joint_net_audit):,.2f}</span>
                    <span><strong>Worst Year:</strong> ${np.min(joint_net_audit):,.2f}</span>
                </div>
                """, unsafe_allow_html=True)

                st.dataframe(
                    audit_df,
                    use_container_width=True,
                    hide_index=True,
                    height=500,
                )

            # -----------------------------------------------------------
            # Top Tested Combinations
            # -----------------------------------------------------------
            with st.expander("📋 View Top Tested Combinations"):
                top_combos = res.get('top_combos', [])
                if top_combos:
                    combo_rows = []
                    total_ac_tc = sum(ud['acres'] for ud in units_data)
                    for combo_pair, score in top_combos[:50]:
                        row_data = {}
                        for k in range(len(units_data)):
                            if k < len(combo_pair):
                                ud = units_data[k]
                                cidx = combo_pair[k]
                                cand = ud['candidates'][cidx]

                                if ud.get('type') == 'PRF':
                                    labels = INTERVAL_ORDER_11
                                    names_k = []
                                    weights_k = []
                                    for idx, w in enumerate(cand[1]):
                                        if w > 0.005:
                                            lbl = labels[idx] if idx < len(labels) else f"Interval {idx+1}"
                                            names_k.append(lbl)
                                            weights_k.append(f"{w*100:.0f}%")
                                    row_data[f'U{k+1} Intervals'] = ', '.join(names_k)
                                    row_data[f'U{k+1} Weights'] = ', '.join(weights_k)
                                elif ud.get('is_cat'):
                                    cat_iv = get_cat_interval(ud['growing_season'])
                                    cat_name = list(cat_iv.values())[0] if cat_iv else 'CAT'
                                    row_data[f'U{k+1} Intervals'] = f"{cat_name} (CAT)"
                                    row_data[f'U{k+1} Weights'] = '100%'
                                else:
                                    intervals_k = get_buyup_intervals(ud['growing_season'])
                                    codes_k = sorted(intervals_k.keys())
                                    names_k = []
                                    weights_k = []
                                    for idx in range(6):
                                        if cand[1][idx] > 0:
                                            names_k.append(intervals_k[codes_k[idx]])
                                            weights_k.append(f"{cand[1][idx]*100:.0f}%")
                                    row_data[f'U{k+1} Intervals'] = ', '.join(names_k)
                                    row_data[f'U{k+1} Weights'] = ', '.join(weights_k)

                        # Compute portfolio metrics for this combo
                        port_ret = sum(
                            units_data[k]['yearly_returns'][combo_pair[k]:combo_pair[k]+1, :] * units_data[k]['acres']
                            for k in range(min(len(combo_pair), len(units_data)))
                        ).flatten() / total_ac_tc

                        port_cost = sum(
                            units_data[k]['producer_costs'][combo_pair[k]] * units_data[k]['acres']
                            for k in range(min(len(combo_pair), len(units_data)))
                        ) / total_ac_tc

                        p_std = np.std(port_ret)
                        row_data['Sharpe'] = f"{np.mean(port_ret)/p_std:.3f}" if p_std > 0 else "N/A"
                        row_data['CVaR'] = f"${np.percentile(port_ret, 5):,.2f}"
                        row_data['Return %'] = f"{(np.mean(port_ret)+port_cost)/port_cost*100:.0f}%" if port_cost > 0 else "N/A"
                        row_data['Win Rate'] = f"{np.mean(port_ret > 0)*100:.1f}%"

                        combo_rows.append(row_data)

                    st.dataframe(
                        pd.DataFrame(combo_rows),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No detailed combo data available (greedy mode does not generate alternatives).")

        else:
            st.caption("Strategy report is available for Joint optimization mode only.")


# ═══════════════════════════════════════════════════════════════════════════════
# Disclaimer Footer
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.caption(
    "Past Performance is not a guarantee of Future Returns. "
    "This is a Risk Management Decision Making Tool only. "
    f"{RATE_YEAR} Rates are used for this application."
)
