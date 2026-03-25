"""
Lightweight Hierarchical Risk Parity (HRP) implementation.

Uses only scipy + numpy (no pypfopt dependency).
Exposes a single function ``run_hrp(returns_df)`` that returns
a dict of {column_name: weight}.
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


def _tree_clustering(cov, corr):
    """Perform hierarchical clustering on correlation distance matrix."""
    dist = np.sqrt(0.5 * (1 - corr))
    # Convert to condensed distance matrix for linkage
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method='single')
    sort_ix = leaves_list(link).tolist()
    return sort_ix, link


def _get_rec_bipart(cov, sort_ix):
    """Recursive bisection to compute HRP weights."""
    w = np.ones(len(sort_ix))
    c_items = [sort_ix]

    while c_items:
        # bisect each cluster
        next_items = []
        for subset in c_items:
            if len(subset) <= 1:
                continue
            mid = len(subset) // 2
            left = subset[:mid]
            right = subset[mid:]

            # Inverse-variance weight for each half
            left_var = _get_cluster_var(cov, left)
            right_var = _get_cluster_var(cov, right)

            alloc_factor = 1.0 - left_var / (left_var + right_var)

            w[left] *= alloc_factor
            w[right] *= (1.0 - alloc_factor)

            next_items.append(left)
            next_items.append(right)

        c_items = next_items

    return w


def _get_cluster_var(cov, indices):
    """Compute variance of a cluster using inverse-variance portfolio."""
    cov_slice = cov[np.ix_(indices, indices)]
    # Inverse-variance weights within cluster
    ivp = 1.0 / np.diag(cov_slice)
    ivp /= ivp.sum()
    return float(ivp @ cov_slice @ ivp)


def run_hrp(returns_df):
    """
    Run Hierarchical Risk Parity on a DataFrame of asset returns.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Each column is an asset, each row is a time period.
        Returns should be in the same units (e.g., $/acre/year).

    Returns
    -------
    dict
        {column_name: weight} where weights sum to 1.0.
    """
    cov = returns_df.cov().values
    corr = returns_df.corr().values

    sort_ix, _link = _tree_clustering(cov, corr)
    weights = _get_rec_bipart(cov, sort_ix)

    # Normalize (should already sum to ~1, but ensure precision)
    weights /= weights.sum()

    columns = returns_df.columns.tolist()
    return {columns[sort_ix[i]]: float(weights[i]) for i in range(len(sort_ix))}
