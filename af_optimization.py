import copy
import numpy as np
from numba import njit, prange
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
            # Flatten any nested numpy arrays into standard hashable Python floats
            ws_clean = tuple(float(x) for x in np.ravel(ws))
            for perm in set(itertools.permutations(ws_clean)):
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
                # Flatten any nested numpy arrays into standard hashable Python floats
                ws_clean = tuple(float(x) for x in np.ravel(ws))
                for perm in set(itertools.permutations(ws_clean)):
                    w6 = np.zeros(6)
                    for idx, c in enumerate(combo):
                        w6[c] = perm[idx] / 100.0
                    candidates.append((combo, w6))

    return candidates


def backtest_candidates_vectorized(candidates, hist_matrix, cbv, coverage_level,
                                   productivity, base_rates_arr, subsidy_pct,
                                   acres=1.0, insurable_interest=1.0):
    """
    Vectorized backtest of all candidates using the full USDA DST cascade.

    Insurable interest and acres are baked into Policy Protection before
    computing premium and indemnity, matching the USDA Decision Support Tool
    dollar-for-dollar.

    Returns:
        yearly_returns: (n_candidates, n_years) net return per acre per year
        producer_costs: (n_candidates,) producer cost per acre
    """
    # USDA Cascading Rounding Sequence (matches Decision Support Tool exactly)
    # Each step uses np.floor(x + 0.5) to simulate ROUND_HALF_UP across arrays.

    # Step 1: Dollar Amount of Protection
    da_display = _round_half_up(cbv * coverage_level * productivity, 2)
    trigger = 100.0 * coverage_level

    weight_matrix = np.array([c[1] for c in candidates])  # (n_cand, 6)

    # Step 2: Policy Protection (total dollars) — DA * II * acres * weight
    pp_total = np.floor(da_display * insurable_interest * acres * weight_matrix + 0.5)  # (n_cand, 6)

    # Step 3: Total Premium — rounded from pp_total
    premium = np.floor(pp_total * base_rates_arr[np.newaxis, :] + 0.5)

    # Step 4: Subsidy — rounded from premium
    subsidy = np.floor(premium * subsidy_pct + 0.5)

    # Step 5: Producer Premium (derived, no rounding) — total dollars
    producer = premium - subsidy
    producer_total = producer.sum(axis=1)  # (n_cand,)

    # Step 6: Indemnity (total dollars) — rounded from pp_total
    shortfall = np.maximum(0, 1.0 - hist_matrix[np.newaxis, :, :] / trigger)
    payout = np.floor(shortfall * pp_total[:, np.newaxis, :] + 0.5)
    yearly_indemnity_total = payout.sum(axis=2)  # (n_cand, n_years)

    # Convert back to per-acre for return values (optimizer works in per-acre space)
    yearly_returns = yearly_indemnity_total / acres - producer_total[:, np.newaxis] / acres
    producer_costs = producer_total / acres

    return yearly_returns, producer_costs


def backtest_cat_unit(grid_id, growing_season, cbv, productivity, coverage_level, subsidy_pct,
                      acres=1.0, insurable_interest=1.0):
    """
    Build backtest data for a CAT unit.
    CAT has a single candidate: the full-season 7th interval at 100% weight.

    Insurable interest and acres are baked into Policy Protection before
    computing premium and indemnity, matching the USDA Decision Support Tool.

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

    # USDA Cascading Rounding Sequence (matches Decision Support Tool exactly)
    # Step 1: Dollar Amount of Protection
    da_display = _round_half_up(cbv * coverage_level * productivity, 2)
    trigger = 100.0 * coverage_level

    # Step 2: Policy Protection (total dollars) — CAT is single interval at 100% weight
    pp = np.floor(da_display * insurable_interest * acres * 1.0 + 0.5)

    # Step 3: Total Premium — rounded from pp
    premium = np.floor(pp * cat_base_rate + 0.5)

    # Step 4: Subsidy — rounded from premium
    subsidy = np.floor(premium * subsidy_pct + 0.5)

    # Step 5: Producer Premium (derived, no rounding) — total dollars
    producer_total = premium - subsidy

    # Step 6: Indemnity (total dollars) — rounded from pp
    shortfall = np.maximum(0, 1.0 - cat_indices / trigger)
    payouts = np.floor(shortfall * pp + 0.5)

    # Convert back to per-acre for return values (optimizer works in per-acre space)
    yearly_returns = (payouts / acres - producer_total / acres).reshape(1, -1)  # (1, n_years)
    producer_costs = np.array([producer_total / acres])

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


@njit(parallel=True, fastmath=True, cache=True)
def _numba_pairwise_sharpe(r0, r1, a0, a1):
    N = r0.shape[0]
    M = r1.shape[0]
    T = r0.shape[1]
    total_a = a0 + a1
    best_scores = np.full(N, -np.inf)
    best_js = np.zeros(N, dtype=np.int32)
    for i in prange(N):
        loc_best = -np.inf
        loc_j = 0
        for j in range(M):
            sum_r = 0.0
            sum_sq = 0.0
            for t in range(T):
                val = (r0[i, t] * a0 + r1[j, t] * a1) / total_a
                sum_r += val
                sum_sq += val * val
            mean = sum_r / T
            var = (sum_sq / T) - (mean * mean)
            if var > 0:
                score = mean / np.sqrt(var)
            else:
                score = -np.inf
            if score > loc_best:
                loc_best = score
                loc_j = j
        best_scores[i] = loc_best
        best_js[i] = loc_j
    return best_scores, best_js

@njit(parallel=True, fastmath=True, cache=True)
def _numba_pairwise_roi(r0, r1, c0, c1, a0, a1):
    N = r0.shape[0]
    M = r1.shape[0]
    T = r0.shape[1]
    total_a = a0 + a1
    best_scores = np.full(N, -np.inf)
    best_js = np.zeros(N, dtype=np.int32)
    for i in prange(N):
        loc_best = -np.inf
        loc_j = 0
        for j in range(M):
            sum_r = 0.0
            for t in range(T):
                sum_r += (r0[i, t] * a0 + r1[j, t] * a1) / total_a
            mean = sum_r / T
            cost = (c0[i] * a0 + c1[j] * a1) / total_a
            if cost > 0:
                score = (mean + cost) / cost
            else:
                score = -np.inf
            if score > loc_best:
                loc_best = score
                loc_j = j
        best_scores[i] = loc_best
        best_js[i] = loc_j
    return best_scores, best_js

@njit(parallel=True, fastmath=True, cache=True)
def _numba_pairwise_cvar(r0, r1, a0, a1):
    N = r0.shape[0]
    M = r1.shape[0]
    T = r0.shape[1]
    total_a = a0 + a1
    best_scores = np.full(N, -np.inf)
    best_js = np.zeros(N, dtype=np.int32)
    for i in prange(N):
        loc_best = -np.inf
        loc_j = 0
        arr = np.zeros(T)
        for j in range(M):
            for t in range(T):
                arr[t] = (r0[i, t] * a0 + r1[j, t] * a1) / total_a

            # Inline percentile calculation for 5th percentile
            # Sort the tiny array (T elements)
            sorted_arr = np.sort(arr)
            k = (T - 1) * 0.05
            f = int(np.floor(k))
            c = int(np.ceil(k))
            if f == c:
                score = sorted_arr[int(k)]
            else:
                d0 = sorted_arr[f] * (c - k)
                d1 = sorted_arr[c] * (k - f)
                score = d0 + d1

            if score > loc_best:
                loc_best = score
                loc_j = j
        best_scores[i] = loc_best
        best_js[i] = loc_j
    return best_scores, best_js


@njit(parallel=True, fastmath=True, cache=True)
def _numba_pairwise_winrate(r0, r1, a0, a1):
    N = r0.shape[0]
    M = r1.shape[0]
    T = r1.shape[1]
    total_a = a0 + a1
    best_scores = np.full(N, -np.inf)
    best_js = np.zeros(N, dtype=np.int32)
    for i in prange(N):
        loc_best = -np.inf
        loc_j = 0
        for j in range(M):
            count = 0
            for t in range(T):
                val = (r0[i, t] * a0 + r1[j, t] * a1) / total_a
                if val > 0.0:
                    count += 1
            score = count / T
            if score > loc_best:
                loc_best = score
                loc_j = j
        best_scores[i] = loc_best
        best_js[i] = loc_j
    return best_scores, best_js


def run_joint_optimization(units_data, metric, progress_callback=None, top_k=None, calc_engine='python', calc_precision='float64'):
    """
    Pairwise-Exhaustive + Sequential Greedy Optimization.
    Round 1: exhaustive pairwise search over the top two units (by TIV) —
    every candidate of Unit 0 × every candidate of Unit 1.
    Rounds 2+: each remaining unit is greedily folded against the locked synthetic.
    """
    # Pre-filter candidates if top_k is set
    index_maps = []
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

    # 1. Tag original index and sort by TIV (total_coverage)
    for i, ud in enumerate(units_data):
        ud['_original_idx'] = i

    sorted_units = sorted(
        units_data,
        key=lambda ud: (
            -ud.get('total_coverage', 0),    # 1. Highest TIV first
            -len(ud.get('candidates', [])),  # 2. More candidates = larger feasible set = benefits more from exhaustive
            ud.get('unit_label', '')         # 3. Alphabetical fallback for determinism
        )
    )

    # 2. Round 1: Exhaustive Pairwise Search of Top Two Units
    #    This is the core of the algorithm. The two highest-TIV units are searched
    #    exhaustively — every candidate of Unit 0 × every candidate of Unit 1.
    #    No approximation, no shortcuts. This is where the money is.

    if len(sorted_units) == 1:
        # Single unit — just pick the best independent candidate
        u0 = sorted_units[0]
        best_u0_idx, _ = _score_independent(
            u0['yearly_returns'], u0['producer_costs'], metric
        )
        result_map = {u0['_original_idx']: best_u0_idx}
        locked_returns = u0['yearly_returns'][best_u0_idx:best_u0_idx+1, :] * u0['acres']
        locked_cost = u0['producer_costs'][best_u0_idx] * u0['acres']
        locked_acres = u0['acres']

    else:
        # Two or more units — exhaustive pairwise for the top two
        u0 = sorted_units[0]
        u1 = sorted_units[1]
        r0, r1 = u0['yearly_returns'], u1['yearly_returns']
        c0, c1 = u0['producer_costs'], u1['producer_costs']
        a0, a1 = u0['acres'], u1['acres']
        n0, n1 = len(r0), len(r1)
        total_acres_01 = a0 + a1

        # Matrix Multiplication & Safe Chunking Fix
        best_score = -np.inf
        best_i, best_j = 0, 0

        if calc_engine == 'numba':
            # Determine target dtype
            numba_dtype = np.float32 if calc_precision == 'float32' else np.float64

            # Force arrays into contiguous C-memory blocks with the chosen precision
            r0_arr = np.ascontiguousarray(r0, dtype=numba_dtype)
            r1_arr = np.ascontiguousarray(r1, dtype=numba_dtype)
            c0_arr = np.ascontiguousarray(c0, dtype=numba_dtype)
            c1_arr = np.ascontiguousarray(c1, dtype=numba_dtype)

            if metric == 'sharpe':
                best_scores, best_js = _numba_pairwise_sharpe(r0_arr, r1_arr, float(a0), float(a1))
            elif metric == 'roi':
                best_scores, best_js = _numba_pairwise_roi(r0_arr, r1_arr, c0_arr, c1_arr, float(a0), float(a1))
            elif metric == 'cvar':
                best_scores, best_js = _numba_pairwise_cvar(r0_arr, r1_arr, float(a0), float(a1))
            elif metric == 'winrate':
                best_scores, best_js = _numba_pairwise_winrate(r0_arr, r1_arr, float(a0), float(a1))
            else:
                # Fallback to python if metric is unsupported
                calc_engine = 'python'

            if calc_engine == 'numba':
                best_i = int(np.argmax(best_scores))
                best_j = int(best_js[best_i])

                # Re-calculate the actual best score accurately for the return payload
                best_returns = (r0[best_i] * a0 + r1[best_j] * a1) / total_acres_01
                best_cost = (c0[best_i] * a0 + c1[best_j] * a1) / total_acres_01
                best_score = _score_portfolio(best_returns, metric, total_cost=best_cost)

        if calc_engine == 'python':
            if metric in ['sharpe', 'roi']:
                # Fast Vectorized Linear Math (No 3D Tensor Required)
                mu0 = r0.mean(axis=1)
                mu1 = r1.mean(axis=1)

                if metric == 'sharpe':
                    var0 = r0.var(axis=1)
                    var1 = r1.var(axis=1)
                    r0_c = r0 - mu0[:, None]
                    r1_c = r1 - mu1[:, None]
                    n_years = r0.shape[1]

                # Chunking to keep the 2D output matrices memory safe (~350MB per chunk)
                chunk_size = 2000
                n_chunks = (n0 + chunk_size - 1) // chunk_size

                for chunk_idx in range(n_chunks):
                    if progress_callback and chunk_idx % max(1, n_chunks // 10) == 0:
                        progress_callback(chunk_idx / n_chunks * 0.5)

                    start = chunk_idx * chunk_size
                    end = min(start + chunk_size, n0)

                    means = (mu0[start:end, None] * a0 + mu1[None, :] * a1) / total_acres_01

                    if metric == 'roi':
                        total_costs = (c0[start:end, None] * a0 + c1[None, :] * a1) / total_acres_01
                        scores = np.full(means.shape, -np.inf)
                        mask = total_costs > 0
                        scores[mask] = (means[mask] + total_costs[mask]) / total_costs[mask]
                    else:
                        # Sharpe: Compute covariance using highly-optimized BLAS matrix multiplication
                        # (chunk, T) @ (T, n1) -> (chunk, n1)
                        cov_01 = (r0_c[start:end] @ r1_c.T) / n_years

                        vars_comb = (
                            (a0**2) * var0[start:end, None] +
                            (a1**2) * var1[None, :] +
                            2 * a0 * a1 * cov_01
                        ) / (total_acres_01**2)

                        stds = np.sqrt(np.maximum(vars_comb, 0))
                        scores = np.full(stds.shape, -np.inf)
                        mask = stds > 0
                        scores[mask] = means[mask] / stds[mask]

                    chunk_best_flat = int(np.argmax(scores))
                    chunk_best_i_local = chunk_best_flat // n1
                    chunk_best_j = chunk_best_flat % n1
                    chunk_best_score = float(scores[chunk_best_i_local, chunk_best_j])

                    if chunk_best_score > best_score:
                        best_score = chunk_best_score
                        best_i = start + chunk_best_i_local
                        best_j = chunk_best_j

            else:
                # For non-linear metrics (CVaR, Winrate), we MUST use the 3D tensor.
                # Safe memory limit: 25 × 22,000 × 76 × 8 bytes = ~334 MB per chunk
                n_years = r0.shape[1]
                chunk_size = 25
                n_chunks = (n0 + chunk_size - 1) // chunk_size

                for chunk_idx in range(n_chunks):
                    if progress_callback and chunk_idx % max(1, n_chunks // 20) == 0:
                        progress_callback(chunk_idx / n_chunks * 0.5)

                    start = chunk_idx * chunk_size
                    end = min(start + chunk_size, n0)

                    # 3D broadcast: (chunk, 1, n_years) * a0 + (1, n1, n_years) * a1 -> (chunk, n1, n_years)
                    portfolio = (r0[start:end, np.newaxis, :] * a0 + r1[np.newaxis, :, :] * a1) / total_acres_01

                    if metric == 'cvar':
                        scores = np.percentile(portfolio, 5, axis=2)
                    elif metric == 'winrate':
                        scores = (portfolio > 0).mean(axis=2)
                    else:
                        scores = np.zeros((end - start, n1))

                    chunk_best_flat = int(np.argmax(scores))
                    chunk_best_i_local = chunk_best_flat // n1
                    chunk_best_j = chunk_best_flat % n1
                    chunk_best_score = float(scores[chunk_best_i_local, chunk_best_j])

                    if chunk_best_score > best_score:
                        best_score = chunk_best_score
                        best_i = start + chunk_best_i_local
                        best_j = chunk_best_j

        # Lock both units
        result_map = {
            u0['_original_idx']: best_i,
            u1['_original_idx']: best_j,
        }

        # Build the locked synthetic from the Round 1 winner
        locked_returns = (
            r0[best_i:best_i+1, :] * a0 +
            r1[best_j:best_j+1, :] * a1
        )
        locked_cost = c0[best_i] * a0 + c1[best_j] * a1
        locked_acres = total_acres_01

    # 3. Rounds 2+: Sequential Folding
    #    Each remaining unit is scored against the locked synthetic.
    #    Only one evaluation per candidate — computationally trivial.
    start_k = 2 if len(sorted_units) >= 2 else 1
    for k in range(start_k, len(sorted_units)):
        if progress_callback:
            # Progress for Rounds 2+: 50% to 100% of total
            progress_callback(0.5 + (k - start_k + 1) / max(1, len(sorted_units) - start_k) * 0.5)

        uk = sorted_units[k]
        rk = uk['yearly_returns']
        ck = uk['producer_costs']
        ak = uk['acres']
        nk = len(rk)

        new_total_acres = locked_acres + ak
        portfolio_returns = (locked_returns + rk * ak) / new_total_acres
        portfolio_costs = (locked_cost + ck * ak) / new_total_acres

        if metric == 'sharpe':
            means = portfolio_returns.mean(axis=1)
            stds = portfolio_returns.std(axis=1)
            scores = np.full(nk, -np.inf)
            mask = stds > 0
            scores[mask] = means[mask] / stds[mask]
        elif metric == 'cvar':
            scores = np.percentile(portfolio_returns, 5, axis=1)
        elif metric == 'roi':
            means = portfolio_returns.mean(axis=1)
            scores = np.full(nk, -np.inf)
            mask = portfolio_costs > 0
            scores[mask] = (means[mask] + portfolio_costs[mask]) / portfolio_costs[mask]
        elif metric == 'winrate':
            scores = (portfolio_returns > 0).mean(axis=1)
        else:
            scores = np.zeros(nk)

        best_uk_idx = int(np.argmax(scores))
        result_map[uk['_original_idx']] = best_uk_idx

        locked_returns = locked_returns + rk[best_uk_idx:best_uk_idx+1, :] * ak
        locked_cost = locked_cost + ck[best_uk_idx] * ak
        locked_acres = new_total_acres

    if progress_callback:
        progress_callback(1.0)

    # Reconstruct result in original unit order
    best_combo = tuple(result_map[i] for i in range(len(units_data)))

    # Final portfolio scoring
    final_portfolio = locked_returns / locked_acres
    final_cost = locked_cost / locked_acres
    best_score = _score_portfolio(final_portfolio.flatten(), metric, total_cost=final_cost)

    top_combos = [] # Greedy sequential explicitly abandons N-dimensional alternatives

    # Remap filtered indices back to original candidate indices
    best_combo = tuple(int(index_maps[k][best_combo[k]]) for k in range(len(best_combo)))

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


def _warmup_numba():
    """Pre-compile Numba kernels at import time so JIT cost never hits the user."""
    _dummy_r = np.zeros((2, 2), dtype=np.float64)
    _dummy_c = np.zeros(2, dtype=np.float64)
    _numba_pairwise_sharpe(_dummy_r, _dummy_r, 1.0, 1.0)
    _numba_pairwise_roi(_dummy_r, _dummy_r, _dummy_c, _dummy_c, 1.0, 1.0)
    _numba_pairwise_cvar(_dummy_r, _dummy_r, 1.0, 1.0)
    _numba_pairwise_winrate(_dummy_r, _dummy_r, 1.0, 1.0)

_warmup_numba()
