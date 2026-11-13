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


def test_oversampling_draws_classes_equally_often():
    from torch.utils.data import WeightedRandomSampler

    from derma.data import oversampling_weights
    labels = np.array([0] * 900 + [1] * 90 + [2] * 10)
    w = torch.as_tensor(oversampling_weights(labels, num_classes=3), dtype=torch.double)
    g = torch.Generator().manual_seed(0)
    drawn = labels[list(WeightedRandomSampler(w, num_samples=30000, replacement=True, generator=g))]
    freq = np.bincount(drawn, minlength=3) / len(drawn)
    assert np.allclose(freq, 1 / 3, atol=0.02)


def test_stratified_subset_keeps_proportions():
    from derma.data import stratified_subset
    labels = np.array([0] * 1000 + [1] * 100 + [2] * 5)
    idx = stratified_subset(labels, 0.1, seed=0)
    assert np.bincount(labels[idx]).tolist() == [100, 10, 1]   # every class keeps at least one image
    assert len(np.unique(idx)) == len(idx)


@pytest.mark.parametrize("level", ["none", "hflip", "flips", "flips_rot", "flips_rot_color", "full"])
def test_every_augmentation_level_keeps_shape(level):
    img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8))
    assert build_transforms(train=True, augment=level, size=64)(img).shape == (3, 64, 64)


def test_unknown_augmentation_level_fails_loudly():
    with pytest.raises(ValueError):
        build_transforms(train=True, augment="flipz")
