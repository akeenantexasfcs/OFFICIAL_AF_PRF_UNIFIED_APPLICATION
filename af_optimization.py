import copy
import numpy as np
import pandas as pd
import itertools
from af_constants import (
    OPTIMIZER_CONSTRAINTS, SEASON_LABELS,
    get_buyup_intervals, get_cat_interval
)
from af_data_loaders import (
    _get_session, load_historical_indices,
    load_base_rates_array, load_historical_matrix
)
from af_calculations import _round_half_up


def prefilter_top_k(yearly_returns, producer_costs, metric, k=500):
    """
    Pre-filter candidates to top-k by independent score.

    Returns:
        filtered_indices: 1D numpy array of original indices for the top-k candidates

    The caller is responsible for slicing yearly_returns, producer_costs,
    and candidates using these indices, and for remapping joint optimization
    results back to the original indices.
    """
    n = len(yearly_returns)
    if n <= k:
        return np.arange(n)

    # Score each candidate independently (same logic as _score_independent)
    if metric == 'sharpe':
        means = yearly_returns.mean(axis=1)
        stds = yearly_returns.std(axis=1)
        scores = np.full(n, -np.inf)
        mask = stds > 0
        scores[mask] = means[mask] / stds[mask]
    elif metric == 'cvar':
        scores = np.percentile(yearly_returns, 5, axis=1)
    elif metric == 'roi':
        means = yearly_returns.mean(axis=1)
        scores = np.full(n, -np.inf)
        mask = producer_costs > 0
        scores[mask] = (means[mask] + producer_costs[mask]) / producer_costs[mask]
    elif metric == 'winrate':
        scores = (yearly_returns > 0).mean(axis=1)
    else:
        scores = np.zeros(n)

    # Get indices of top-k scores
    top_k_indices = np.argsort(scores)[-k:]
    # Sort them to maintain stable ordering
    top_k_indices.sort()
    return top_k_indices


def generate_weight_sets(max_weight_pct, min_weight_pct, num_intervals, step=5):
    """Generate all valid weight combinations (as integer percentages) summing to 100."""
    if num_intervals == 2:
        results = set()
        for w1 in range(max_weight_pct, min_weight_pct - 1, -step):
            w2 = 100 - w1
            if min_weight_pct <= w2 <= max_weight_pct:
                results.add(tuple(sorted((w1, w2), reverse=True)))
        return [r for r in results]

    # 3 intervals
    results = set()
    for w1 in range(max_weight_pct, min_weight_pct - 1, -step):
        for w2 in range(min(w1, 100 - w1 - min_weight_pct), min_weight_pct - 1, -step):
            w3 = 100 - w1 - w2
            if min_weight_pct <= w3 <= max_weight_pct and w3 <= w2:
                results.add((w1, w2, w3))
    return [r for r in results]


def enumerate_candidates(growing_season, step=5):
    """
    Enumerate all valid (interval_combo, weight_array_6) candidates.
    ENFORCES USDA RULE 24-RI-AF Section 2(d): Intervals within the same
    growing season cannot contain the same months. Adjacent buy-up indices
    always share a month (e.g., Oct-Nov and Nov-Dec share November), so
    any combo with adjacent indices is excluded.
    Returns list of (selected_indices_tuple, weights_6_array).
    """
    constraints = OPTIMIZER_CONSTRAINTS[growing_season]
    max_w = int(constraints['max_weight'] * 100)
    min_w = int(constraints['min_weight'] * 100)
    allow_fewer = constraints.get('allow_fewer', False)

    candidates = []
    buyup_indices = list(range(6))

    # 3-interval combinations
    for combo in itertools.combinations(buyup_indices, 3):
        # USDA NO-OVERLAP: skip if any two selected indices are adjacent
        if any(combo[i+1] - combo[i] == 1 for i in range(len(combo)-1)):
            continue
        for ws in generate_weight_sets(max_w, min_w, 3, step):
            for perm in set(itertools.permutations(ws)):
                w6 = np.zeros(6)
                for idx, c in enumerate(combo):
                    w6[c] = perm[idx] / 100.0
                candidates.append((combo, w6))

    # For GS 10-12: also 2-interval combinations
    if allow_fewer:
        for combo in itertools.combinations(buyup_indices, 2):
            # USDA NO-OVERLAP: skip if the two selected indices are adjacent
            if combo[1] - combo[0] == 1:
                continue
            for ws in generate_weight_sets(max_w, min_w, 2, step):
                for perm in set(itertools.permutations(ws)):
                    w6 = np.zeros(6)
                    for idx, c in enumerate(combo):
                        w6[c] = perm[idx] / 100.0
                    candidates.append((combo, w6))

    return candidates


def backtest_candidates_vectorized(candidates, hist_matrix, cbv, coverage_level,
                                   productivity, base_rates_arr, subsidy_pct):
    """
    Vectorized backtest of all candidates.

    Returns:
        yearly_returns: (n_candidates, n_years) net return per acre per year
        producer_costs: (n_candidates,) producer cost per acre
    """
    # Calculate both unrounded and rounded Dollar Amounts
    da_full = cbv * coverage_level * productivity
    da_display = _round_half_up(da_full, 2)
    trigger = 100.0 * coverage_level

    weight_matrix = np.array([c[1] for c in candidates])  # (n_cand, 6)

    # Protection uses da_full (unrounded) to match DST payout logic
    protection_raw = da_full * weight_matrix  # (n_cand, 6) — unrounded

    # Premium uses da_display (rounded)
    premium = da_display * weight_matrix * base_rates_arr[np.newaxis, :]
    subsidy = premium * subsidy_pct
    producer = premium - subsidy
    producer_costs = producer.sum(axis=1)  # (n_cand,)

    # Payout calculations
    payout = np.maximum(0, (1 - hist_matrix[np.newaxis, :, :] / trigger)) * protection_raw[:, np.newaxis, :]
    yearly_indemnity = payout.sum(axis=2)  # (n_cand, n_years)
    yearly_returns = yearly_indemnity - producer_costs[:, np.newaxis]

    return yearly_returns, producer_costs


def backtest_cat_unit(grid_id, growing_season, cbv, productivity, coverage_level, subsidy_pct):
    """
    Build backtest data for a CAT unit.
    CAT has a single candidate: the full-season 7th interval at 100% weight.

    Returns:
        candidates: list with one dummy candidate
        yearly_returns: (1, n_years) net return per acre per year
        producer_costs: (1,) producer cost per acre
        years: array of years
    """
    cat_interval = get_cat_interval(growing_season)
    if not cat_interval:
        return [], np.empty((0, 0)), np.array([0.0]), np.array([])

    cat_code = list(cat_interval.keys())[0]
    cat_name = cat_interval[cat_code]

    # Load historical indices (full set includes the CAT interval)
    hist_df = load_historical_indices(grid_id, growing_season)
    if hist_df.empty or cat_name not in hist_df.columns:
        return [], np.empty((0, 0)), np.array([0.0]), np.array([])

    years = hist_df['YEAR'].values
    cat_indices = hist_df[cat_name].values.astype(float)
    cat_indices = np.where(pd.notna(cat_indices), cat_indices, 100.0)

    # Load CAT premium rate
    session = _get_session()
    rate_df = session.sql(f"""
        SELECT BASE_RATE
        FROM AF_PREMIUM_RATES
        WHERE GRID_ID = {grid_id}
          AND GROWING_SEASON = {growing_season}
          AND COVERAGE_LEVEL_PERCENT = {coverage_level}
          AND COVERAGE_TYPE = 'CAT'
        LIMIT 1
    """).to_pandas()
    cat_base_rate = float(rate_df.iloc[0]['BASE_RATE']) if not rate_df.empty else 0.0

    # Calculations
    da_full = cbv * coverage_level * productivity
    da_display = _round_half_up(da_full, 2)
    trigger = 100.0 * coverage_level

    # Protection uses da_full, premium uses da_display
    protection_raw = da_full * 1.0
    premium = da_display * 1.0 * cat_base_rate
    subsidy = premium * subsidy_pct
    producer_cost = premium - subsidy

    # Payout per year
    payouts = np.maximum(0, (1 - cat_indices / trigger)) * protection_raw
    yearly_returns = (payouts - producer_cost).reshape(1, -1)  # (1, n_years)
    producer_costs = np.array([producer_cost])

    candidates = [((0,), np.zeros(6))]

    return candidates, yearly_returns, producer_costs, years


def _score_portfolio(portfolio_returns, metric, total_cost=None):
    """Score a portfolio returns array."""
    if metric == 'sharpe':
        std = np.std(portfolio_returns)
        if std == 0:
            return -np.inf
        return float(np.mean(portfolio_returns) / std)
    elif metric == 'cvar':
        return float(np.percentile(portfolio_returns, 5))
    elif metric == 'roi':
        if total_cost is None or total_cost == 0:
            return -np.inf
        return float((np.mean(portfolio_returns) + total_cost) / total_cost)
    elif metric == 'winrate':
        return float(np.mean(portfolio_returns > 0))
    return -np.inf


def _score_independent(yearly_returns, producer_costs, metric):
    """Find the best independent candidate for a single unit."""
    yr = yearly_returns
    pc = producer_costs
    n = len(yr)
    if metric == 'sharpe':
        means = yr.mean(axis=1)
        stds = yr.std(axis=1)
        scores = np.full(n, -np.inf)
        mask = stds > 0
        scores[mask] = means[mask] / stds[mask]
    elif metric == 'cvar':
        scores = np.percentile(yr, 5, axis=1)
    elif metric == 'roi':
        means = yr.mean(axis=1)
        scores = np.full(n, -np.inf)
        mask = pc > 0
        scores[mask] = (means[mask] + pc[mask]) / pc[mask]
    elif metric == 'winrate':
        scores = (yr > 0).mean(axis=1)
    else:
        scores = np.zeros(n)
    best_idx = int(np.argmax(scores))
    return best_idx, float(scores[best_idx])


def _optimize_2_units(units_data, metric, progress_callback=None):
    """Optimized 2-unit joint optimization with vectorized inner loop."""
    u1, u2 = units_data
    r1, r2 = u1['yearly_returns'], u2['yearly_returns']
    a1, a2 = u1['acres'], u2['acres']
    c1, c2 = u1['producer_costs'], u2['producer_costs']
    total_acres = a1 + a2
    n1, n2 = len(r1), len(r2)
    # Pre-compute weighted returns (avoids repeated multiplication in inner loop)
    r1_weighted = r1 * (a1 / total_acres)  # (n1, n_years)
    r2_weighted = r2 * (a2 / total_acres)  # (n2, n_years)

    best_score = -np.inf
    best_combo = (0, 0)
    top_combos = []

    for i in range(n1):
        if progress_callback and i % max(1, n1 // 20) == 0:
            progress_callback(i / n1)
        portfolio = r1_weighted[i:i+1, :] + r2_weighted  # (n2, n_years)

        if metric == 'sharpe':
            means = portfolio.mean(axis=1)
            stds = portfolio.std(axis=1)
            scores = np.full(n2, -np.inf)
            mask = stds > 0
            scores[mask] = means[mask] / stds[mask]
        elif metric == 'cvar':
            scores = np.percentile(portfolio, 5, axis=1)
        elif metric == 'roi':
            total_cost = (c1[i] * a1 + c2 * a2) / total_acres
            means = portfolio.mean(axis=1)
            scores = np.full(n2, -np.inf)
            mask = total_cost > 0
            scores[mask] = (means[mask] + total_cost[mask]) / total_cost[mask]
        elif metric == 'winrate':
            scores = (portfolio > 0).mean(axis=1)
        else:
            scores = np.zeros(n2)

        best_j = int(np.argmax(scores))
        if scores[best_j] > best_score:
            best_score = float(scores[best_j])
            best_combo = (i, best_j)

        top_j_indices = np.argsort(scores)[-3:]
        for j in top_j_indices:
            if np.isfinite(scores[j]):
                top_combos.append(((i, int(j)), float(scores[j])))

    if progress_callback:
        progress_callback(1.0)

    top_combos.sort(key=lambda x: x[1], reverse=True)
    top_combos = top_combos[:50]

    return best_combo, best_score, top_combos


def _optimize_3_units(units_data, metric, progress_callback=None):
    """3-unit joint optimization."""
    u1, u2, u3 = units_data
    r1, r2, r3 = u1['yearly_returns'], u2['yearly_returns'], u3['yearly_returns']
    a1, a2, a3 = u1['acres'], u2['acres'], u3['acres']
    c1, c2, c3 = u1['producer_costs'], u2['producer_costs'], u3['producer_costs']
    total_acres = a1 + a2 + a3
    n1, n2, n3 = len(r1), len(r2), len(r3)
    # Pre-compute weighted returns
    r1_weighted = r1 * (a1 / total_acres)
    r2_weighted = r2 * (a2 / total_acres)
    r3_weighted = r3 * (a3 / total_acres)

    best_score = -np.inf
    best_combo = (0, 0, 0)
    top_combos = []
    total_iters = n1 * n2
    current = 0

    for i in range(n1):
        if progress_callback and i % max(1, n1 // 20) == 0:
            progress_callback(i / n1)
        for j in range(n2):
            current += 1

            portfolio = r1_weighted[i:i+1, :] + r2_weighted[j:j+1, :] + r3_weighted

            if metric == 'sharpe':
                means = portfolio.mean(axis=1)
                stds = portfolio.std(axis=1)
                scores = np.full(n3, -np.inf)
                mask = stds > 0
                scores[mask] = means[mask] / stds[mask]
            elif metric == 'cvar':
                scores = np.percentile(portfolio, 5, axis=1)
            elif metric == 'roi':
                total_cost = (c1[i] * a1 + c2[j] * a2 + c3 * a3) / total_acres
                means = portfolio.mean(axis=1)
                scores = np.full(n3, -np.inf)
                mask = total_cost > 0
                scores[mask] = (means[mask] + total_cost[mask]) / total_cost[mask]
            elif metric == 'winrate':
                scores = (portfolio > 0).mean(axis=1)
            else:
                scores = np.zeros(n3)

            best_k = int(np.argmax(scores))
            if scores[best_k] > best_score:
                best_score = float(scores[best_k])
                best_combo = (i, j, best_k)

            top_k = np.argsort(scores)[-2:]
            for k in top_k:
                if np.isfinite(scores[k]):
                    top_combos.append(((i, j, int(k)), float(scores[k])))

    if progress_callback:
        progress_callback(1.0)

    top_combos.sort(key=lambda x: x[1], reverse=True)
    top_combos = top_combos[:50]

    return best_combo, best_score, top_combos


def run_joint_optimization(units_data, metric, progress_callback=None, top_k=None):
    """Run joint optimization across multiple units."""
    # Pre-filter candidates if top_k is set
    index_maps = []  # Per-unit mapping from filtered index -> original index
    if top_k is not None:
        units_data = copy.deepcopy(units_data)
        for i, ud in enumerate(units_data):
            filtered_indices = prefilter_top_k(
                ud['yearly_returns'], ud['producer_costs'], metric, k=top_k
            )
            index_maps.append(filtered_indices)
            ud['yearly_returns'] = ud['yearly_returns'][filtered_indices]
            ud['producer_costs'] = ud['producer_costs'][filtered_indices]
            if ud.get('candidates') is not None:
                ud['candidates'] = [ud['candidates'][j] for j in filtered_indices]
    else:
        index_maps = [np.arange(len(ud['yearly_returns'])) for ud in units_data]

    n_units = len(units_data)
    if n_units == 2:
        best_combo, best_score, top_combos = _optimize_2_units(units_data, metric, progress_callback)
    elif n_units == 3:
        best_combo, best_score, top_combos = _optimize_3_units(units_data, metric, progress_callback)
    else:
        # 4+ units: greedy sequential pairing, sorted by total_coverage
        # Tag each unit with its original index
        for i, ud in enumerate(units_data):
            ud['_original_idx'] = i

        # Sort: highest total_coverage first, then unit_label ascending as tiebreaker
        sorted_units = sorted(
            units_data,
            key=lambda ud: (-ud.get('total_coverage', 0), ud.get('unit_label', ''))
        )

        # Pair the two highest-coverage units first
        best_combo_01, _, _ = _optimize_2_units(sorted_units[:2], metric)
        u0, u1 = sorted_units[0], sorted_units[1]
        merged_returns = (
            u0['yearly_returns'][best_combo_01[0]:best_combo_01[0]+1, :] * u0['acres'] +
            u1['yearly_returns'][best_combo_01[1]:best_combo_01[1]+1, :] * u1['acres']
        ) / (u0['acres'] + u1['acres'])

        # Track accumulated per-acre cost of the merged portfolio
        merged_cost = (
            u0['producer_costs'][best_combo_01[0]] * u0['acres'] +
            u1['producer_costs'][best_combo_01[1]] * u1['acres']
        ) / (u0['acres'] + u1['acres'])

        # Map original index -> chosen candidate index
        result_map = {
            sorted_units[0]['_original_idx']: best_combo_01[0],
            sorted_units[1]['_original_idx']: best_combo_01[1],
        }

        for k in range(2, len(sorted_units)):
            uk = sorted_units[k]
            merged_unit = {
                'yearly_returns': merged_returns,
                'producer_costs': np.array([merged_cost]),
                'acres': sum(sorted_units[j]['acres'] for j in range(k)),
            }
            pair_data = [merged_unit, uk]
            best_pair, _, _ = _optimize_2_units(pair_data, metric)
            result_map[uk['_original_idx']] = best_pair[1]

            new_merged_acres = merged_unit['acres'] + uk['acres']
            merged_returns = (
                merged_returns * merged_unit['acres'] +
                uk['yearly_returns'][best_pair[1]:best_pair[1]+1, :] * uk['acres']
            ) / new_merged_acres
            merged_cost = (
                merged_cost * merged_unit['acres'] +
                uk['producer_costs'][best_pair[1]] * uk['acres']
            ) / new_merged_acres

        if progress_callback:
            progress_callback(1.0)

        # Reconstruct result in original unit order
        result_indices = tuple(result_map[i] for i in range(len(units_data)))

        # Final portfolio scoring uses original units_data order
        total_acres = sum(u['acres'] for u in units_data)
        portfolio = sum(
            units_data[k]['yearly_returns'][result_indices[k]:result_indices[k]+1, :] * units_data[k]['acres']
            for k in range(len(units_data))
        ) / total_acres
        total_cost = sum(
            units_data[k]['producer_costs'][result_indices[k]] * units_data[k]['acres']
            for k in range(len(units_data))
        ) / total_acres
        score = _score_portfolio(portfolio.flatten(), metric, total_cost=total_cost)

        best_combo = result_indices
        best_score = score
        top_combos = []

    # Remap filtered indices back to original candidate indices
    best_combo = tuple(int(index_maps[k][best_combo[k]]) for k in range(len(best_combo)))

    # Also remap top_combos if present
    if top_combos:
        remapped_top = []
        for combo_tuple, combo_score in top_combos:
            remapped = tuple(int(index_maps[k][combo_tuple[k]]) for k in range(min(len(combo_tuple), len(index_maps))))
            remapped_top.append((remapped, combo_score))
        top_combos = remapped_top

    return best_combo, best_score, top_combos


def generate_insight_text(joint_combo, indep_indices, units_data):
    """Generate human-readable insight about why joint optimization differs."""
    differences = []
    for u_idx in range(len(units_data)):
        joint_idx = joint_combo[u_idx]
        indep_idx = indep_indices[u_idx]
        if joint_idx != indep_idx:
            ud = units_data[u_idx]
            intervals = get_buyup_intervals(ud['growing_season'])
            codes = sorted(intervals.keys())

            joint_cand = ud['candidates'][joint_idx]
            indep_cand = ud['candidates'][indep_idx]

            joint_intervals = [intervals[codes[i]] for i in joint_cand[0]]
            indep_intervals = [intervals[codes[i]] for i in indep_cand[0]]

            added = set(joint_intervals) - set(indep_intervals)
            removed = set(indep_intervals) - set(joint_intervals)

            if added or removed:
                parts = []
                if removed:
                    parts.append(f"away from {', '.join(sorted(removed))}")
                if added:
                    parts.append(f"toward {', '.join(sorted(added))}")
                differences.append(
                    f"{ud['unit_label']} shifted {' and '.join(parts)}"
                )

    if differences:
        return ("The joint optimizer " + "; ".join(differences) +
                " to reduce portfolio correlation and improve risk-adjusted returns.")
    return "The joint and independent optimizations converged on the same allocation."
