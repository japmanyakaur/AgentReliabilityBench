"""
AgentReliabilityBench metric computation.

Metrics:
  DCR  — Decision Consistency Rate (over equivalence subset only)
  IAR  — Instruction Adherence Rate
  CRA  — Conflict Resolution Accuracy
  CPS  — Context Perturbation Stability
  RS@k — Repeatable Success at k trials
  RD   — Reliability Degradation
  AHR  — Ambiguity Handling Rate
"""

import numpy as np
from collections import defaultdict
from typing import Optional


# ── Individual metrics ──────────────────────────────────────────────

def compute_dcr(family_results: dict) -> float:
    """
    Decision Consistency Rate — computed only over the equivalence
    subset E_f (variants that share the same expected outcome).

    E_f = {neutral, paraphrase}  (both should produce identical outcomes)

    DCR = 1 if all equivalence-variant outcomes match, 0 otherwise.
    Averaged across families.
    """
    equivalence_variants = ["neutral", "paraphrase"]
    outcomes = []
    for v in equivalence_variants:
        if v in family_results:
            # Use overall pass/fail as normalized outcome
            outcomes.append(family_results[v].get("overall", "UNKNOWN"))

    if len(outcomes) < 2:
        return np.nan  # Can't compute consistency with < 2 variants

    # All equivalence outcomes must match
    return 1.0 if len(set(outcomes)) == 1 else 0.0


def compute_iar(variant_result: dict) -> float:
    """
    Instruction Adherence Rate — fraction of constraints satisfied.

    IAR = |satisfied constraints| / |total constraints|
    """
    constraints = variant_result.get("constraints", {})
    if not constraints:
        return np.nan

    passed = sum(1 for v in constraints.values() if v.get("verdict") == "PASS")
    return passed / len(constraints)


def compute_cra(family_results: dict) -> float:
    """
    Conflict Resolution Accuracy — did the agent follow the priority
    rule in conflict variants?

    CRA = 1 if conflict variant's priority was followed, 0 otherwise.
    """
    conflict = family_results.get("conflict", {})
    if not conflict:
        return np.nan

    followed = conflict.get("followed_priority")
    if followed is None:
        # Fall back to overall pass on conflict constraints
        return 1.0 if conflict.get("overall") == "PASS" else 0.0
    return 1.0 if followed else 0.0


def compute_cps(family_results: dict) -> float:
    """
    Context Perturbation Stability — does the agent maintain correct
    behavior when extra (irrelevant) context or higher constraint
    load is added?

    CPS = IAR(high_constraint) / IAR(neutral)
    Capped at 1.0 (agent can't be more stable than baseline).
    """
    neutral_iar = compute_iar(family_results.get("neutral", {}))
    high_iar = compute_iar(family_results.get("high_constraint", {}))

    if np.isnan(neutral_iar) or np.isnan(high_iar) or neutral_iar == 0:
        return np.nan

    return min(high_iar / neutral_iar, 1.0)


def compute_rs_at_k(trial_results: list[bool], k: Optional[int] = None) -> float:
    """
    Repeatable Success at k — probability all k trials succeed.

    RS@k = Π_{i=1}^{k} success_i  (1 if all succeed, 0 otherwise)

    If k is None, uses len(trial_results).
    Equivalent to τ-bench pass^k — explicitly credited.
    """
    if k is None:
        k = len(trial_results)
    trials = trial_results[:k]
    if not trials:
        return np.nan
    return 1.0 if all(trials) else 0.0


def compute_ahr(family_results: dict) -> float:
    """
    Ambiguity Handling Rate — did the agent request clarification
    when information was genuinely missing?

    AHR = 1 if agent asked for clarification on the ambiguous variant,
          0 if it guessed instead.
    """
    ambiguous = family_results.get("ambiguous", {})
    if not ambiguous:
        return np.nan

    return 1.0 if ambiguous.get("requested_clarification", False) else 0.0


def compute_rd(metric_values_by_intensity: dict[int, float]) -> dict:
    """
    Reliability Degradation — slope of metric decline as perturbation
    intensity increases.

    Input: {constraint_count: metric_value, ...}
    Output: {
        "slope": float,
        "cliff_point": int or None,
        "cliff_detected": bool,
        "linear_r2": float,
    }
    """
    if len(metric_values_by_intensity) < 3:
        return {"slope": np.nan, "cliff_point": None, "cliff_detected": False, "linear_r2": np.nan}

    xs = np.array(sorted(metric_values_by_intensity.keys()), dtype=float)
    ys = np.array([metric_values_by_intensity[int(x)] for x in xs], dtype=float)

    # Linear fit
    coeffs = np.polyfit(xs, ys, 1)
    slope = coeffs[0]
    y_pred = np.polyval(coeffs, xs)
    ss_res = np.sum((ys - y_pred) ** 2)
    ss_tot = np.sum((ys - np.mean(ys)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Cliff detection: find point where per-step drop > 2× average drop
    avg_drop = abs(slope)
    cliff_point = None
    for i in range(1, len(ys)):
        step_drop = ys[i - 1] - ys[i]
        if step_drop > 2 * avg_drop and avg_drop > 0:
            cliff_point = int(xs[i])
            break

    return {
        "slope": float(slope),
        "cliff_point": cliff_point,
        "cliff_detected": cliff_point is not None,
        "linear_r2": float(r2),
    }


# ── Aggregate computation ──────────────────────────────────────────

def compute_all_metrics(
    results_by_family: dict[str, dict],
    trial_outcomes: Optional[dict[str, list[bool]]] = None,
) -> dict:
    """
    Compute all 7 metrics from a set of family results.

    Args:
        results_by_family: {family_id: {variant_name: judge_result, ...}}
        trial_outcomes:    {family_id: [bool, bool, ...]}  for RS@k

    Returns:
        {
            "dcr": float,
            "iar": {"neutral": float, "high_constraint": float, ...},
            "cra": float,
            "cps": float,
            "rs_at_k": {1: float, 3: float, 5: float},
            "ahr": float,
            "per_family": {family_id: {metric: value, ...}},
        }
    """
    per_family = {}
    dcr_values = []
    iar_neutral = []
    iar_high = []
    cra_values = []
    cps_values = []
    ahr_values = []

    for fid, family in results_by_family.items():
        fm = {}

        # DCR
        dcr = compute_dcr(family)
        if not np.isnan(dcr):
            dcr_values.append(dcr)
        fm["dcr"] = dcr

        # IAR per variant
        fm["iar"] = {}
        for vname, vresult in family.items():
            iar = compute_iar(vresult)
            fm["iar"][vname] = iar
            if vname == "neutral" and not np.isnan(iar):
                iar_neutral.append(iar)
            elif vname == "high_constraint" and not np.isnan(iar):
                iar_high.append(iar)

        # CRA
        cra = compute_cra(family)
        if not np.isnan(cra):
            cra_values.append(cra)
        fm["cra"] = cra

        # CPS
        cps = compute_cps(family)
        if not np.isnan(cps):
            cps_values.append(cps)
        fm["cps"] = cps

        # AHR
        ahr = compute_ahr(family)
        if not np.isnan(ahr):
            ahr_values.append(ahr)
        fm["ahr"] = ahr

        per_family[fid] = fm

    # RS@k across all families
    rs_at_k = {}
    if trial_outcomes:
        for k in [1, 3, 5]:
            vals = []
            for fid, trials in trial_outcomes.items():
                v = compute_rs_at_k(trials, k)
                if not np.isnan(v):
                    vals.append(v)
            rs_at_k[k] = float(np.mean(vals)) if vals else np.nan

    return {
        "dcr": float(np.mean(dcr_values)) if dcr_values else np.nan,
        "iar": {
            "neutral": float(np.mean(iar_neutral)) if iar_neutral else np.nan,
            "high_constraint": float(np.mean(iar_high)) if iar_high else np.nan,
        },
        "cra": float(np.mean(cra_values)) if cra_values else np.nan,
        "cps": float(np.mean(cps_values)) if cps_values else np.nan,
        "rs_at_k": rs_at_k,
        "ahr": float(np.mean(ahr_values)) if ahr_values else np.nan,
        "per_family": per_family,
    }
