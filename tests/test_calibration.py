import numpy as np
import pytest

from derma.calibration import TemperatureScaler, calibration_bins, expected_calibration_error, nll
from derma.metrics import softmax


def test_ece_hand_example():
    # 4 predictions with confidence 0.9 (3 right, 1 wrong) and 2 with confidence 0.6 (both right)
    probs = np.array([[0.9, 0.1]] * 4 + [[0.6, 0.4]] * 2)
    labels = np.array([0, 0, 0, 1, 0, 0])
    # bin of 0.9: accuracy 0.75 -> gap 0.15, weight 4/6;  bin of 0.6: accuracy 1.0 -> gap 0.4, weight 2/6
    assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(4 / 6 * 0.15 + 2 / 6 * 0.4)


def test_ece_zero_when_confidence_matches_accuracy():
    probs = np.array([[0.75, 0.25]] * 4)
    labels = np.array([0, 0, 0, 1])                 # 75% right at 75% confidence
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0)


def test_bins_count_every_prediction_once():
    rng = np.random.default_rng(0)
    probs = softmax(rng.normal(size=(500, 7)) * 3)
    _, _, count = calibration_bins(probs, rng.integers(0, 7, 500), n_bins=15)
    assert count.sum() == 500


def make_overconfident(n=2000, k=7, acc=0.7, seed=0):
    """Logits that always put ~all probability on the prediction, but are right only 70% of the time."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, n)
    preds = np.where(rng.random(n) < acc, labels, (labels + rng.integers(1, k, n)) % k)
    logits = rng.normal(0, 0.5, (n, k))
    logits[np.arange(n), preds] += 8.0
    return logits, labels


def test_temperature_softens_overconfident_model_and_keeps_accuracy():
    logits, labels = make_overconfident()
    scaler = TemperatureScaler()
    t = scaler.fit(logits, labels)
    scaled = scaler.transform(logits)
    assert t > 1.0
    assert (scaled.argmax(1) == logits.argmax(1)).all()          # same predictions -> same accuracy
    assert nll(scaled, labels) < nll(logits, labels)
    assert expected_calibration_error(softmax(scaled), labels) < expected_calibration_error(softmax(logits), labels)
