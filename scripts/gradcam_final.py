"""Final Grad-CAM figures (layer3, test set) and a randomised-weights sanity check.

    python scripts/gradcam_final.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from derma.data import CLASS_NAMES, DermaDataset, build_transforms
from derma.gradcam import GradCAM
from derma.models import build_model

R = "results/runs"
ds = DermaDataset("test", 64, "data", build_transforms(False))
X = torch.stack([ds[i][0] for i in range(len(ds))])
y = np.asarray(ds.labels).reshape(-1)


def make_model(name):
    try:
        return build_model(name, load_weights=False)
    except TypeError:
        return build_model(name)


def load_run(run):
    ck = torch.load(f"{R}/{run}/best.pt", map_location="cpu")
    name = ck["config"]["model"]
    model = make_model(name)
    model.load_state_dict(ck["model"])
    return model.eval(), name


def explain(model, x):
    cams, _, pred = GradCAM(model, model.layer3)(x)
    cams = np.stack([np.asarray(torch.as_tensor(c).detach().cpu()).squeeze() for c in cams])
    return cams, np.asarray(torch.as_tensor(pred).cpu()).reshape(-1)


pre, pre_name = load_run("expD_resnet18_pretrained_seed0")
scr, _ = load_run("expD_resnet18_scratch_seed0")
pred_pre = np.load(f"{R}/expD_resnet18_pretrained_seed0/test_logits.npz")["logits"].argmax(1)

# ---- Figures: first test image of each class that the pretrained model gets right / wrong ----
for kind, ok in (("correct", pred_pre == y), ("wrong", pred_pre != y)):
    idx = [int(np.where((y == c) & ok)[0][0]) for c in range(7) if ((y == c) & ok).any()]
    cam_p, p_p = explain(pre, X[idx])
    cam_s, p_s = explain(scr, X[idx])
    n = len(idx)
    fig, axes = plt.subplots(3, n, figsize=(2.1 * n, 6.8))
    for j, i in enumerate(idx):
        axes[0, j].imshow(ds.images[i])
        axes[0, j].set_title(f"true: {CLASS_NAMES[y[i]]}", fontsize=9)
        for row, cam, p, tag in ((1, cam_p, p_p, "pretrained"), (2, cam_s, p_s, "scratch")):
            axes[row, j].imshow(ds.images[i])
            axes[row, j].imshow(cam[j], cmap="jet", alpha=0.45)
            axes[row, j].set_title(f"{tag}: {CLASS_NAMES[p[j]]}", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"Grad-CAM (layer3), test images the pretrained model gets {kind}")
    plt.tight_layout()
    plt.savefig(f"figures/gradcam_{kind}.png", dpi=150)
    plt.close()
    print("saved", f"figures/gradcam_{kind}.png")

# ---- Sanity check: trained model vs a model with random weights ----
torch.manual_seed(0)
rand = make_model(pre_name).eval()                       # random weights, nothing learned
sel = torch.arange(200)
cam_t, _ = explain(pre, X[sel])
cam_r, _ = explain(rand, X[sel])
rho = [spearmanr(a.ravel(), b.ravel())[0] for a, b in zip(cam_t, cam_r)]
print("undefined correlations:", int(np.isnan(rho).sum()), "of", len(rho))
print(f"\nSanity check on 200 test images: mean Spearman correlation between the maps of the "
      f"trained and the random model = {np.nanmean(rho):.3f} (median {np.nanmedian(rho):.3f})")

fig, axes = plt.subplots(3, 6, figsize=(12, 6.5))
for j in range(6):
    for row, (cam, tag) in enumerate(((None, "image"), (cam_t, "trained"), (cam_r, "random weights"))):
        axes[row, j].imshow(ds.images[j])
        if cam is not None:
            axes[row, j].imshow(cam[j], cmap="jet", alpha=0.45)
        axes[row, j].set_title(tag, fontsize=9)
        axes[row, j].axis("off")
fig.suptitle("Grad-CAM sanity check (layer3)")
plt.tight_layout()
plt.savefig("figures/gradcam_sanity_check.png", dpi=150)
print("saved figures/gradcam_sanity_check.png")