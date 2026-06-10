"""
Binary classification metrics for Normal (0) / Abnormal (1).

All functions accept plain Python lists or numpy arrays.

Metrics computed
----------------
  accuracy        — (TP+TN) / total
  sensitivity     — TP / (TP+FN)   (recall / TPR)
  specificity     — TN / (TN+FP)   (TNR)
  precision       — TP / (TP+FP)   (PPV)
  npv             — TN / (TN+FN)   (negative predictive value)
  f1              — harmonic mean of precision + sensitivity
  youden_j        — sensitivity + specificity - 1
  auroc           — area under ROC curve  (requires scores)
  auprc           — area under precision-recall curve  (requires scores)
  confusion_matrix — [[TN, FP], [FN, TP]]
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_score: Optional[List[float]] = None,   # probability of class 1
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute full binary classification metric suite.

    Parameters
    ----------
    y_true   : ground-truth labels (0 / 1)
    y_pred   : predicted labels (0 / 1)
    y_score  : predicted probability for the positive (abnormal) class.
               Required for AUROC / AUPRC; pass None to skip those.
    threshold: decision threshold used when y_pred was derived from y_score.
               Stored in output for traceability.

    Returns
    -------
    dict with all metric names as keys and float values.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    total = len(y_true)

    def safe_div(num, denom):
        return float(num) / float(denom) if denom > 0 else float("nan")

    sensitivity = safe_div(tp, tp + fn)          # recall / TPR
    specificity = safe_div(tn, tn + fp)          # TNR
    precision   = safe_div(tp, tp + fp)          # PPV
    npv         = safe_div(tn, tn + fn)

    metrics: Dict[str, float] = {
        "accuracy":     float(accuracy_score(y_true, y_pred)),
        "sensitivity":  sensitivity,
        "specificity":  specificity,
        "precision":    precision,
        "npv":          npv,
        "f1":           float(f1_score(y_true, y_pred, zero_division=0)),
        "youden_j":     sensitivity + specificity - 1,
        "tp":           int(tp),
        "tn":           int(tn),
        "fp":           int(fp),
        "fn":           int(fn),
        "total":        int(total),
        "threshold":    threshold,
    }

    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        # AUROC — requires at least one sample per class
        if len(np.unique(y_true)) == 2:
            metrics["auroc"] = float(roc_auc_score(y_true, y_score))
            metrics["auprc"] = float(average_precision_score(y_true, y_score))
        else:
            metrics["auroc"] = float("nan")
            metrics["auprc"] = float("nan")
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")

    return metrics


def combine_scores(
    val_test_records: List[Tuple[int, float]],   # (y_true, y_score)
    golden_records:   List[Tuple[int, float]],
) -> float:
    """
    Compute the combined AUROC across val_test + golden.

    Parameters
    ----------
    val_test_records / golden_records :
        list of (true_label, predicted_score) tuples

    Returns
    -------
    float — combined AUROC
    """
    all_true  = [r[0] for r in val_test_records + golden_records]
    all_score = [r[1] for r in val_test_records + golden_records]

    if len(np.unique(all_true)) < 2:
        return float("nan")
    return float(roc_auc_score(all_true, all_score))
