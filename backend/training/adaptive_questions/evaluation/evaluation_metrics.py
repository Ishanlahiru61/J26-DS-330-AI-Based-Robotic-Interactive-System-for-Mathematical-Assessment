"""Shared, dependency-light metric functions for the C2 evaluation package."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def equal_frequency_ece(y_true, probability, n_bins: int = 10) -> tuple[float, list[dict]]:
    """Expected calibration error using equal-frequency bins.

    ECE = sum_b (n_b / n) * |mean(y_b) - mean(p_b)|.
    Duplicate probability values may make bin sizes differ slightly.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    order = np.argsort(p, kind="mergesort")
    bins = np.array_split(order, n_bins)
    rows, value = [], 0.0
    for index, loc in enumerate(bins, start=1):
        if len(loc) == 0:
            continue
        observed = float(y[loc].mean())
        predicted = float(p[loc].mean())
        weight = len(loc) / len(y)
        value += weight * abs(observed - predicted)
        rows.append({
            "bin": index,
            "n": int(len(loc)),
            "mean_predicted": predicted,
            "observed_fraction": observed,
            "absolute_gap": abs(observed - predicted),
        })
    return float(value), rows


def binary_prediction_metrics(y_true, probability, threshold: float = 0.5) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    predicted = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    prevalence = float(y.mean())
    baseline_p = np.full(len(y), prevalence, dtype=float)
    ece, bins = equal_frequency_ece(y, p, 10)
    return {
        "n_interactions": int(len(y)),
        "positive_prevalence": prevalence,
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall_sensitivity": float(recall_score(y, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
        "ece_10_equal_frequency": ece,
        "calibration_bins": bins,
        "majority_class_accuracy": max(prevalence, 1.0 - prevalence),
        "prevalence_probability_log_loss": float(log_loss(y, baseline_p, labels=[0, 1])),
        "prevalence_probability_brier_score": float(brier_score_loss(y, baseline_p)),
        "threshold": threshold,
    }


def clustered_bootstrap_intervals(
    y_true,
    probability,
    cluster_ids,
    replicates: int = 200,
    seed: int = 20261021,
) -> dict:
    """Percentile CIs from a learner-cluster bootstrap.

    Learners, rather than interactions, are sampled with replacement. A sampled
    learner's complete interaction history receives the learner multiplicity as
    a sample weight. This preserves within-learner dependence without expanding
    arrays in memory.
    """
    if replicates <= 0:
        return {}
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    clusters = np.asarray(cluster_ids)
    unique, inverse = np.unique(clusters, return_inverse=True)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    pred = (p >= 0.5).astype(int)
    for _ in range(replicates):
        multiplicity = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weights = multiplicity[inverse].astype(float)
        if weights[y == 0].sum() == 0 or weights[y == 1].sum() == 0:
            continue
        values["roc_auc"].append(roc_auc_score(y, p, sample_weight=weights))
        values["average_precision"].append(average_precision_score(y, p, sample_weight=weights))
        values["accuracy"].append(accuracy_score(y, pred, sample_weight=weights))
        values["log_loss"].append(log_loss(y, p, labels=[0, 1], sample_weight=weights))
        values["brier_score"].append(float(np.average((p - y) ** 2, weights=weights)))
    result = {}
    for name, samples in values.items():
        lo, hi = np.percentile(samples, [2.5, 97.5])
        result[name] = {
            "lower_95": float(lo),
            "upper_95": float(hi),
            "replicates_requested": replicates,
            "replicates_valid": len(samples),
            "cluster_unit": "learner/user_id",
            "seed": seed,
        }
    return result


def safe_rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def policy_compliance_metrics(rows: list[dict]) -> dict:
    total = len(rows)
    curriculum = sum(bool(row.get("curriculum_valid")) for row in rows)
    tier = sum(bool(row.get("tier_policy_valid")) for row in rows)
    selected = [row.get("activity_id") for row in rows if row.get("activity_id")]
    return {
        "recommendations": total,
        "curriculum_valid": curriculum,
        "curriculum_compliance_rate": safe_rate(curriculum, total),
        "curriculum_violations": total - curriculum,
        "tier_policy_valid": tier,
        "tier_policy_compliance_rate": safe_rate(tier, total),
        "tier_policy_violations": total - tier,
        "distinct_activities": len(set(selected)),
    }


def paired_change_metrics(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("pair_id"):
            groups[row["pair_id"]].append(row)
    comparable, changed, tier_changed, score_changed, rationale_changed, expected_agreement = 0, 0, 0, 0, 0, 0
    invariant_pairs, invariant_changes = 0, 0
    score_comparable = 0
    details = []
    for pair_id, items in sorted(groups.items()):
        if len(items) != 2:
            continue
        left, right = items
        recommendation_changed = left.get("activity_id") != right.get("activity_id")
        tier_change = left.get("tier") != right.get("tier")
        left_score, right_score = left.get("probability"), right.get("probability")
        scores_available = left_score is not None and right_score is not None
        probability_changed = scores_available and abs(float(left_score) - float(right_score)) > 1e-12
        reason_changed = left.get("reason") != right.get("reason")
        expected = bool(left.get("expect_change") or right.get("expect_change"))
        observable_change = recommendation_changed or tier_change or probability_changed or reason_changed
        if expected:
            comparable += 1
            changed += recommendation_changed
            tier_changed += tier_change
            rationale_changed += reason_changed
            if scores_available:
                score_comparable += 1
                score_changed += probability_changed
        else:
            invariant_pairs += 1
            invariant_changes += observable_change
        agrees = observable_change if expected else not observable_change
        expected_agreement += agrees
        details.append({
            "pair_id": pair_id,
            "left": left.get("scenario_id"),
            "right": right.get("scenario_id"),
            "recommendation_changed": recommendation_changed,
            "tier_changed": tier_change,
            "probability_left": left_score,
            "probability_right": right_score,
            "probability_difference": (float(left_score) - float(right_score)) if scores_available else None,
            "probability_changed": probability_changed,
            "rationale_changed": reason_changed,
            "expect_change": expected,
            "expectation_agreement": agrees,
        })
    return {
        "matched_pairs": comparable,
        "recommendation_changes": changed,
        "recommendation_change_rate": safe_rate(changed, comparable),
        "tier_changes": tier_changed,
        "tier_change_rate": safe_rate(tier_changed, comparable),
        "score_comparable_pairs": score_comparable,
        "score_changes": score_changed,
        "score_change_rate": safe_rate(score_changed, score_comparable),
        "rationale_changes": rationale_changed,
        "rationale_change_rate": safe_rate(rationale_changed, comparable),
        "observable_history_response_rate": safe_rate(sum(bool(row["recommendation_changed"] or row["tier_changed"] or row["probability_changed"] or row["rationale_changed"]) for row in details if row["expect_change"]), comparable),
        "invariant_pairs": invariant_pairs,
        "invariant_changes": invariant_changes,
        "invariant_stability_rate": safe_rate(invariant_pairs - invariant_changes, invariant_pairs),
        "policy_expectation_agreement": safe_rate(expected_agreement, comparable + invariant_pairs),
        "details": details,
    }
