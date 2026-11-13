import pytest
import torch
import torch.nn.functional as F

from derma.gradcam import GradCAM
from derma.models import build_model


@pytest.mark.parametrize("name", ["smallcnn", "resnet18_scratch"])
def test_shape_range_and_no_leftover_hooks(name):
    torch.manual_seed(0)
    model = build_model(name, load_weights=False)
    cam_fn = GradCAM(model)
    cams, logits, cls = cam_fn(torch.randn(3, 3, 64, 64))
    assert cams.shape == (3, 64, 64) and logits.shape == (3, 7) and cls.shape == (3,)
    assert cams.min() >= 0 and cams.max() <= 1 + 1e-6
    assert len(cam_fn.target_layer._forward_hooks) == 0
    assert all(p.grad is None for p in model.parameters())     # we cleaned up after ourselves


def test_matches_manual_computation_on_smallcnn():
    """Compute Grad-CAM by hand with torch.autograd.grad and compare it with the hook version."""
    torch.manual_seed(0)
    model = build_model("smallcnn").eval()
    x = torch.randn(2, 3, 64, 64)
    target = torch.tensor([1, 4])

    acts = model.features[:-1](x)                   # output of the last Conv-BN-ReLU block
    out = model.fc(model.dropout(model.pool(model.features[-1](acts)).flatten(1)))
    grads = torch.autograd.grad(out.gather(1, target[:, None]).sum(), acts)[0]
    manual = F.relu((grads.mean((2, 3), keepdim=True) * acts).sum(1, keepdim=True))
    manual = F.interpolate(manual, size=(64, 64), mode="bilinear", align_corners=False).squeeze(1)
    lo = manual.flatten(1).min(1).values[:, None, None]
    hi = manual.flatten(1).max(1).values[:, None, None]
    manual = (manual - lo) / (hi - lo).clamp_min(1e-8)

    cams, _, _ = GradCAM(model)(x, class_idx=target)
    assert torch.allclose(cams, manual.detach(), atol=1e-5)


def test_works_inside_no_grad_and_with_int_class():
    model = build_model("smallcnn")
    with torch.no_grad():
        cams, _, cls = GradCAM(model)(torch.randn(2, 3, 64, 64), class_idx=3)
    assert cams.shape == (2, 64, 64) and (cls == 3).all()
