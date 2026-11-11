import pytest
import torch

from derma.models import build_model, count_parameters, param_groups

NAMES = ["smallcnn", "resnet18_scratch", "resnet18_pretrained"]


@pytest.mark.parametrize("name", NAMES)
def test_output_shape(name):
    model = build_model(name, num_classes=7, load_weights=False).eval()
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 7)


def test_smallcnn_accepts_other_sizes():
    # Global Average Pooling makes the classifier independent of the input size.
    model = build_model("smallcnn").eval()
    assert model(torch.randn(2, 3, 28, 28)).shape == (2, 7)


def test_param_groups_cover_every_parameter_once():
    model = build_model("resnet18_pretrained", load_weights=False)
    groups = param_groups(model, "resnet18_pretrained", lr=1e-3, lr_backbone=1e-4)
    grouped = [id(p) for g in groups for p in g["params"]]
    assert len(grouped) == len(set(grouped)) == len(list(model.parameters()))
    assert groups[0]["lr"] == 1e-4 and groups[1]["lr"] == 1e-3


def test_resnet_head_size():
    model = build_model("resnet18_scratch")
    assert count_parameters(model.fc) == 512 * 7 + 7
