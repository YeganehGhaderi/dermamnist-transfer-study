import numpy as np
import pytest
import torch
from PIL import Image

from derma.data import build_transforms, compute_class_weights


def test_class_weights_formula():
    labels = np.array([0] * 6 + [1] * 3 + [2] * 1)          # N=10, K=3
    w = compute_class_weights(labels, num_classes=3)
    assert w == pytest.approx([10 / 18, 10 / 9, 10 / 3])
    # weighted mean over samples is exactly 1, so the loss scale does not change
    assert w[labels].mean() == pytest.approx(1.0)


def test_eval_transform_is_deterministic_and_normalized():
    img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8))
    t = build_transforms(train=False)
    a, b = t(img), t(img)
    assert a.shape == (3, 64, 64) and a.dtype == torch.float32
    assert torch.equal(a, b)
