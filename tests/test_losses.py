import pytest
import torch
import torch.nn.functional as F

from derma.losses import FocalLoss, build_criterion


def test_focal_with_gamma_zero_equals_cross_entropy():
    torch.manual_seed(0)
    logits, y = torch.randn(16, 7), torch.randint(0, 7, (16,))
    assert torch.allclose(FocalLoss(gamma=0.0)(logits, y), F.cross_entropy(logits, y), atol=1e-6)


def test_focal_down_weights_easy_examples():
    y = torch.tensor([0])
    easy = torch.tensor([[5.0, 0, 0, 0, 0, 0, 0]])     # confident and right
    hard = torch.tensor([[0.0, 0, 0, 0, 0, 0, 0]])     # unsure
    ce_easy, ce_hard = F.cross_entropy(easy, y), F.cross_entropy(hard, y)
    fl_easy, fl_hard = FocalLoss(gamma=2.0)(easy, y), FocalLoss(gamma=2.0)(hard, y)
    # focal loss shrinks both, but the easy one much more (relative to cross-entropy)
    assert fl_easy / ce_easy < fl_hard / ce_hard < 1.0


def test_focal_loss_decreases_with_gamma():
    torch.manual_seed(0)
    logits, y = torch.randn(32, 7), torch.randint(0, 7, (32,))
    values = [FocalLoss(gamma=g)(logits, y).item() for g in [0.0, 1.0, 2.0, 5.0]]
    assert values == sorted(values, reverse=True)


def test_focal_alpha_scales_each_class():
    logits, y = torch.zeros(2, 3), torch.tensor([0, 1])
    alpha = torch.tensor([2.0, 0.0, 1.0])
    # only the first sample counts (alpha=2), the second one has alpha=0
    expected = 2.0 * FocalLoss(gamma=2.0)(logits[:1], y[:1]) / 2
    assert torch.allclose(FocalLoss(gamma=2.0, alpha=alpha)(logits, y), expected)


def test_build_criterion_variants():
    labels = torch.tensor([0, 0, 0, 1, 2, 3, 4, 5, 6]).numpy()
    cpu = torch.device("cpu")
    assert isinstance(build_criterion({"loss": "ce"}, labels, cpu), torch.nn.CrossEntropyLoss)
    weighted = build_criterion({"loss": "weighted_ce"}, labels, cpu)
    assert weighted.weight[0] < weighted.weight[1]          # class 0 is the most frequent
    assert isinstance(build_criterion({"loss": "focal"}, labels, cpu), FocalLoss)
    with pytest.raises(ValueError):
        build_criterion({"loss": "weighted_ce", "sampler": "oversample"}, labels, cpu)
