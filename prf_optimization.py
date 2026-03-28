"""
PRF (Pasture, Rangeland, Forage) Optimization — for Unified Optimizer.

Key functions:
    - enumerate_prf_candidates: Exhaustive enumeration of valid PRF interval weight combos
    - backtest_prf_candidates_vectorized: Vectorized backtesting of PRF candidates
"""

import itertools
import numpy as np
from prf_constants import (
    INTERVAL_ORDER_11,
    MIN_ALLOCATION, MAX_ALLOCATION,
    PRF_MIN_INTERVALS, PRF_MAX_INTERVALS,
)
from prf_data_loaders import (
    load_county_base_value, load_all_indices,
    load_premium_rates, load_subsidies,
    load_coverage_intervals, get_allocation_bounds,
    get_current_rate_year,
)
from af_calculations import _round_half_up


def _generate_weight_sets(n_slots, total, min_w, max_w, step):
    """
    Generate all weight tuples of length n_slots that sum to total,
    with each weight in [min_w, max_w] at step increments.
    """
    if n_slots == 0:
        return []
    if n_slots == 1:
        if min_w <= total <= max_w:
            return [(total,)]
        return []

    results = []
    w = min_w
    while w <= min(max_w, total - (n_slots - 1) * min_w):
        remainder = total - w
        sub = _generate_weight_sets(n_slots - 1, remainder, min_w, max_w, step)
        for s in sub:
            results.append((w,) + s)
        w += step
    return results


def enumerate_prf_candidates(grid_id, intended_use, irrigation_practice,
                              organic_practice, coverage_level,
                              intervals_cache=None, step=5,
                              iterations=10000,  # ignored — kept for backward compat
                              interval_count_range=(2, 6),
                              **kwargs):
    """
    Generate PRF candidate allocations using exhaustive combinatorial enumeration.

    Enumerates all valid (non-adjacent interval combo, weight permutation) pairs.
    The actual candidate count is ~22,000-30,000 at step=5, which is completely
    feasible. This guarantees the global optimum per unit with reproducible results.

    Note: the `iterations` parameter is ignored (kept for backward compatibility).

    Parameters:
        grid_id: NOAA grid ID
        intended_use: 'Grazing' or 'Haying'
        irrigation_practice: 'Irrigated' or 'Non-Irrigated'
        organic_practice: organic practice string
        coverage_level: e.g. 0.90
        intervals_cache: optional pre-loaded coverage interval bounds
        step: weight step increment in percentage points (default 5)
        iterations: ignored — kept for backward compatibility
        interval_count_range: tuple (min, max) active intervals (default (2, 6))

    Returns:
        list of (selected_indices_tuple, weights_11_array) candidates
    """
    # Load allocation bounds
    if intervals_cache is None:
        intervals_cache = load_coverage_intervals()

    min_alloc_frac, max_alloc_frac = get_allocation_bounds(
        grid_id, intended_use, irrigation_practice, organic_practice,
        intervals_cache,
    )

    min_w_pct = int(round(min_alloc_frac * 100))
    max_w_pct = int(round(max_alloc_frac * 100))
    min_count, max_count = interval_count_range

    candidates = []

    for k in range(min_count, max_count + 1):
        # Step 1: Generate all non-adjacent interval combinations of size k
        for combo in itertools.combinations(range(11), k):
            # Adjacency check: reject if any two indices differ by exactly 1
            adjacent = False
            for i in range(len(combo) - 1):
                if combo[i + 1] - combo[i] == 1:
                    adjacent = True
                    break
            if adjacent:
                continue

            # Step 2: Generate all weight assignments for this combo size.
            # _generate_weight_sets already produces all ordered tuples (all
            # permutations of each partition), so each tuple directly maps to
            # a unique weight-to-interval assignment — no extra permutation needed.
            weight_sets = _generate_weight_sets(k, 100, min_w_pct, max_w_pct, step)

            for wt in weight_sets:
                weights_11 = np.zeros(11)
                for i, idx in enumerate(combo):
                    weights_11[idx] = wt[i] / 100.0
                candidates.append((combo, weights_11))

    print(f"PRF exhaustive enumeration: {len(candidates)} candidates "
          f"(K={min_count}-{max_count}, step={step}, bounds={min_w_pct}-{max_w_pct}%)")

    return candidates


def backtest_prf_candidates_vectorized(candidates, grid_id,
                                        start_year, end_year,
                                        intended_use, irrigation_practice,
                                        organic_practice, coverage_level,
                                        productivity, insurable_interest,
                                        **kwargs):
    """
    Vectorized backtesting of PRF candidates.

    Parameters:
        candidates: list of (selected_indices, weights_11_array)
        grid_id: NOAA grid ID
        start_year, end_year: backtest period
        intended_use, irrigation_practice, organic_practice: practice params
        coverage_level: e.g. 0.90
        productivity: productivity factor (e.g. 1.50)
        insurable_interest: fraction (0-1)

    Returns:
        (yearly_returns, producer_costs, years):
            yearly_returns: (n_candidates, n_years) net $/acre
            producer_costs: (n_candidates,) producer premium per acre
            years: 1D array of year values
    """
    # 1. Load data using PRF data loaders
    cbv = load_county_base_value(grid_id, intended_use,
                                  irrigation_practice, organic_practice)
    if cbv is None:
        raise ValueError(f"No County Base Value found for grid {grid_id} "
                         f"with {intended_use}/{irrigation_practice}/{organic_practice}")

    indices_df = load_all_indices(grid_id)
    if indices_df.empty:
        raise ValueError(f"No historical index data found for grid {grid_id}")

    rate_year = get_current_rate_year()
    rates_dict = load_premium_rates(grid_id, intended_use,
                                     [coverage_level], rate_year,
                                     irrigation_practice, organic_practice)
    subsidies = load_subsidies(coverage_levels=[coverage_level])
    subsidy_pct = subsidies.get(coverage_level, 0.55)

    # 2. Build index_matrix (n_years, 11) and premium_rates_array (11,)
    # Filter to year range
    indices_df = indices_df[
        (indices_df['YEAR'] >= start_year) & (indices_df['YEAR'] <= end_year)
    ].copy()
    indices_df = indices_df.sort_values('YEAR')
    years = indices_df['YEAR'].values.astype(int)

    # Build index matrix — columns in INTERVAL_ORDER_11
    index_matrix = np.zeros((len(years), 11))
    for i, iv in enumerate(INTERVAL_ORDER_11):
        if iv in indices_df.columns:
            index_matrix[:, i] = indices_df[iv].values.astype(float)
        else:
            index_matrix[:, i] = 100.0  # Default to 100 (no loss) if missing

    # Premium rates array
    cl_rates = rates_dict.get(coverage_level, {})
    premium_rates_array = np.zeros(11)
    for i, iv in enumerate(INTERVAL_ORDER_11):
        premium_rates_array[i] = cl_rates.get(iv, 0.0)

    # 3. Compute yearly returns
    da_full = cbv * coverage_level * productivity
    da_display = _round_half_up(da_full, 2)
    trigger = 100.0 * coverage_level

    n_cand = len(candidates)
    weight_matrix = np.array([c[1] for c in candidates])  # (n_cand, 11)

    # USDA Cascading Rounding Sequence (matches Decision Support Tool exactly)
    # Each step uses np.floor(x + 0.5) to simulate ROUND_HALF_UP across arrays.

    # Step 2: Policy Protection (total dollars) — DA * II * acres * weight
    # Note: insurable_interest is applied after in Step 4, but for PRF the
    # protection per interval uses da_display directly
    pp_total = np.floor(da_display * weight_matrix + 0.5)  # (n_cand, 11) per-acre protection

    # Step 3: Total Premium — rounded from pp_total
    premium = np.floor(pp_total * premium_rates_array[np.newaxis, :] + 0.5)

    # Step 4: Subsidy — rounded from premium
    subsidy = np.floor(premium * subsidy_pct + 0.5)

    # Step 5: Producer Premium (derived, no rounding)
    producer = premium - subsidy
    producer_costs = producer.sum(axis=1)  # (n_cand,)

    # Step 6: Indemnity — rounded from pp_total
    shortfall = np.maximum(0, 1.0 - index_matrix / trigger)  # (n_years, 11)
    payout = np.floor(shortfall[np.newaxis, :, :] * pp_total[:, np.newaxis, :] + 0.5)
    yearly_indemnity = payout.sum(axis=2)  # (n_cand, n_years)

    yearly_returns = yearly_indemnity - producer_costs[:, np.newaxis]

    # Step 7: Apply insurable interest
    yearly_returns *= insurable_interest
    producer_costs *= insurable_interest

    return yearly_returns, producer_costs, years
