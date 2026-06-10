import pytest
from tasks.binary_metrics import compute_metrics, combine_scores


def test_perfect_classifier():
    m = compute_metrics([0, 0, 1, 1], [0, 0, 1, 1], [0.1, 0.1, 0.9, 0.9])
    assert m["accuracy"]    == 1.0
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["auroc"]       == 1.0


def test_all_wrong():
    m = compute_metrics([0, 0, 1, 1], [1, 1, 0, 0], [0.9, 0.9, 0.1, 0.1])
    assert m["accuracy"]    == 0.0
    assert m["sensitivity"] == 0.0
    assert m["auroc"]       == 0.0


def test_sensitivity_specificity():
    # TP=2, FN=1, TN=3, FP=0
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 0, 0, 0]
    m = compute_metrics(y_true, y_pred)
    assert abs(m["sensitivity"] - 2/3) < 1e-6
    assert m["specificity"]    == 1.0
    assert m["fp"]             == 0
    assert m["fn"]             == 1


def test_youden_j():
    m = compute_metrics([0, 1, 0, 1], [0, 1, 1, 0])
    assert abs(m["youden_j"] - (m["sensitivity"] + m["specificity"] - 1)) < 1e-9


def test_combine_scores():
    vt = [(1, 0.9), (0, 0.1), (1, 0.8)]
    g  = [(0, 0.2), (1, 0.7), (0, 0.3)]
    auroc = combine_scores(vt, g)
    assert 0.0 <= auroc <= 1.0


def test_no_score_gives_nan_auroc():
    m = compute_metrics([0, 1], [0, 1], y_score=None)
    import math
    assert math.isnan(m["auroc"])


def test_counts():
    m = compute_metrics([0, 0, 1, 1], [1, 0, 1, 0])
    assert m["tp"] == 1
    assert m["tn"] == 1
    assert m["fp"] == 1
    assert m["fn"] == 1
