"""
Statistical analysis methods for AgentReliabilityBench.

Implements:
  - Stratified bootstrap CIs (task family as resampling unit)
  - Paired permutation tests within matched families
  - Benjamini-Hochberg FDR correction
  - Cliff's delta (non-parametric effect size)
  - ICC (intra-class correlation) for cross-domain analysis
  - Davies-like cliff detection for degradation curves
"""

import numpy as np
from scipy import stats
from typing import Optional


def bootstrap_ci(
    values: np.ndarray,
    group_ids: Optional[np.ndarray] = None,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    statistic: str = "mean",
    seed: int = 42,
) -> dict:
    """
    Stratified bootstrap confidence interval.

    If group_ids is provided, resamples at the group level
    (task family as the resampling unit) to account for
    within-family correlation.

    Returns: {"estimate": float, "ci_lower": float, "ci_upper": float}
    """
    rng = np.random.RandomState(seed)
    stat_fn = np.mean if statistic == "mean" else np.median

    if group_ids is not None:
        unique_groups = np.unique(group_ids)
        boot_stats = []
        for _ in range(n_bootstrap):
            sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            sampled_values = np.concatenate([values[group_ids == g] for g in sampled_groups])
            boot_stats.append(stat_fn(sampled_values))
    else:
        boot_stats = []
        for _ in range(n_bootstrap):
            sample = rng.choice(values, size=len(values), replace=True)
            boot_stats.append(stat_fn(sample))

    boot_stats = np.array(boot_stats)
    alpha = 1 - confidence
    ci_lower = np.percentile(boot_stats, 100 * alpha / 2)
    ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    return {
        "estimate": float(stat_fn(values)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "se": float(np.std(boot_stats)),
    }


def paired_permutation_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Paired permutation test within matched task families.

    Tests H0: no difference between conditions A and B.
    Exchangeability justified by controlled design (same family,
    different perturbation).

    Returns: {"observed_diff": float, "p_value": float}
    """
    assert len(values_a) == len(values_b), "Arrays must be paired (same families)"
    rng = np.random.RandomState(seed)

    observed_diff = np.mean(values_a) - np.mean(values_b)
    diffs = values_a - values_b

    count_extreme = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_diff = np.mean(diffs * signs)
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)  # +1 to avoid p=0

    return {
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
    }


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> dict:
    """
    Benjamini-Hochberg procedure for FDR control.

    Returns: {
        "adjusted_p_values": list[float],
        "rejected": list[bool],
        "num_rejected": int,
    }
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    adjusted = [0.0] * n
    rejected = [False] * n

    # Compute adjusted p-values (step-up)
    prev = 1.0
    for rank_minus_1 in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_minus_1]
        rank = rank_minus_1 + 1
        adjusted_p = min(prev, p * n / rank)
        adjusted[orig_idx] = adjusted_p
        prev = adjusted_p

    for i in range(n):
        rejected[i] = adjusted[i] < alpha

    return {
        "adjusted_p_values": adjusted,
        "rejected": rejected,
        "num_rejected": sum(rejected),
    }


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Cliff's delta — non-parametric effect size.

    delta = (#{x_i > y_j} - #{x_i < y_j}) / (n_x * n_y)

    Interpretation:
        |d| < 0.147  → negligible
        |d| < 0.33   → small
        |d| < 0.474  → medium
        |d| >= 0.474 → large
    """
    n_x, n_y = len(x), len(y)
    more = 0
    less = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                more += 1
            elif xi < yj:
                less += 1

    delta = (more - less) / (n_x * n_y)

    abs_d = abs(delta)
    if abs_d < 0.147:
        magnitude = "negligible"
    elif abs_d < 0.33:
        magnitude = "small"
    elif abs_d < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"

    return {"delta": float(delta), "magnitude": magnitude}


def compute_icc(
    data: np.ndarray,
    domain_labels: np.ndarray,
    model_labels: np.ndarray,
) -> dict:
    """
    Intra-class correlation coefficient (ICC, two-way random, single
    measures — ICC(2,1)) for cross-domain reliability analysis.

    High ICC → model reliability is intrinsic (generalizes across domains).
    Low ICC  → reliability is domain-dependent.

    Args:
        data:          1D array of metric values
        domain_labels: 1D array of domain names (same length)
        model_labels:  1D array of model names (same length)

    Returns: {"icc": float, "interpretation": str}
    """
    unique_domains = np.unique(domain_labels)
    unique_models = np.unique(model_labels)
    k = len(unique_domains)
    n = len(unique_models)

    if k < 2 or n < 2:
        return {"icc": np.nan, "interpretation": "insufficient data"}

    # Build model × domain matrix
    matrix = np.full((n, k), np.nan)
    for i, m in enumerate(unique_models):
        for j, d in enumerate(unique_domains):
            mask = (model_labels == m) & (domain_labels == d)
            vals = data[mask]
            if len(vals) > 0:
                matrix[i, j] = np.mean(vals)

    # Drop rows/cols with NaN
    valid_rows = ~np.any(np.isnan(matrix), axis=1)
    matrix = matrix[valid_rows]
    n = matrix.shape[0]
    if n < 2:
        return {"icc": np.nan, "interpretation": "insufficient data"}

    grand_mean = np.mean(matrix)
    row_means = np.mean(matrix, axis=1)
    col_means = np.mean(matrix, axis=0)

    ss_total = np.sum((matrix - grand_mean) ** 2)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    ms_cols = ss_cols / (k - 1)

    # ICC(2,1) formula
    icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n)
    icc = float(np.clip(icc, -1.0, 1.0))

    if icc < 0.50:
        interp = "poor — reliability is domain-dependent"
    elif icc < 0.75:
        interp = "moderate — some generalization across domains"
    elif icc < 0.90:
        interp = "good — reliability mostly intrinsic to model"
    else:
        interp = "excellent — reliability fully generalizes"

    return {"icc": icc, "interpretation": interp}


def spearman_rank_correlation(rankings_a: list, rankings_b: list) -> dict:
    """
    Spearman rank correlation between model rankings across
    different conflict types.

    Low correlation → conflict resolution is type-dependent, not general.
    """
    rho, p_value = stats.spearmanr(rankings_a, rankings_b)
    return {"rho": float(rho), "p_value": float(p_value)}
