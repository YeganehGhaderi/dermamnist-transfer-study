import numpy as np
import pytest

from derma.metrics import compute_metrics, softmax


def one_hot_logits(preds, k=3):
    logits = np.zeros((len(preds), k))
    logits[np.arange(len(preds)), preds] = 10.0
    return logits


def test_softmax_rows_sum_to_one_and_is_stable():
    p = softmax(np.array([[1000.0, 1000.0, 0.0], [1.0, 2.0, 3.0]]))
    assert np.allclose(p.sum(1), 1) and np.isfinite(p).all()
    assert np.allclose(p[0], [0.5, 0.5, 0.0])


def test_metrics_on_hand_computed_example():
    # true:  0 0 0 0 1 2      pred: 0 0 0 0 0 2
    # recall: class0 4/4=1, class1 0/1=0, class2 1/1=1 -> balanced acc = 2/3
    # F1: class0 P=4/5 R=1 -> 8/9, class1 0, class2 1 -> macro F1 = (8/9 + 0 + 1)/3 = 17/27
    labels = np.array([0, 0, 0, 0, 1, 2])
    m = compute_metrics(one_hot_logits([0, 0, 0, 0, 0, 2]), labels, num_classes=3)
    assert m["accuracy"] == pytest.approx(5 / 6)
    assert m["balanced_acc"] == pytest.approx(2 / 3)
    assert m["macro_f1"] == pytest.approx(17 / 27)


def test_majority_class_trap():
    # Always predicting the majority class: high accuracy, poor macro-F1. This is why we use macro-F1.
    labels = np.array([0] * 90 + [1] * 5 + [2] * 5)
    m = compute_metrics(one_hot_logits([0] * 100), labels, num_classes=3)
    assert m["accuracy"] == pytest.approx(0.9)
    assert m["balanced_acc"] == pytest.approx(1 / 3)
    assert m["macro_f1"] < 0.33


def test_perfect_ranking_gives_auc_one():
    labels = np.array([0, 1, 2, 0, 1, 2])
    assert compute_metrics(one_hot_logits(labels), labels, num_classes=3)["auc_macro"] == pytest.approx(1.0)
