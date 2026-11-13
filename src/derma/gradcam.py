"""Grad-CAM (Selvaraju et al., 2017), written with plain PyTorch hooks.

Idea: take the feature maps A_k of a late convolutional layer (each k is one channel, e.g. 512 maps
of 2x2 in ResNet-18's layer4). Ask how much the score of the target class changes when each map
changes: that is the gradient dScore/dA_k. Average it over the spatial positions to get one weight
per channel, w_k. The heat map is ReLU(sum_k w_k * A_k): the places whose features push the score up.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def default_target_layer(model: nn.Module) -> nn.Module:
    """Last convolutional stage: layer4 for ResNet-18, the last Conv-BN-ReLU block for SmallCNN."""
    if hasattr(model, "layer4"):
        return model.layer4
    return model.features[-2]   # SmallCNN: [..., conv_bn_relu, MaxPool] -> the block before the pool


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else default_target_layer(model)

    def __call__(self, x: torch.Tensor, class_idx=None):
        """x: (B, 3, H, W). class_idx: None (= predicted class), an int, or a (B,) tensor.

        Returns (cams (B, H, W) in [0, 1], logits (B, K), class_idx (B,)).
        """
        store = {}

        def forward_hook(module, inputs, output):
            store["acts"] = output                                     # A: (B, C, h, w)
            output.register_hook(lambda grad: store.__setitem__("grads", grad))   # dScore/dA

        handle = self.target_layer.register_forward_hook(forward_hook)
        try:
            self.model.eval()                    # BatchNorm/Dropout in inference mode
            with torch.enable_grad():            # works even if the caller is inside torch.no_grad()
                logits = self.model(x)
                if class_idx is None:
                    class_idx = logits.argmax(dim=1)
                elif isinstance(class_idx, int):
                    class_idx = torch.full((x.size(0),), class_idx, device=x.device, dtype=torch.long)
                # sum over the batch: each image's score only depends on its own activations
                score = logits.gather(1, class_idx[:, None]).sum()
                self.model.zero_grad(set_to_none=True)
                score.backward()
        finally:
            handle.remove()                      # never leave a hook behind on the model

        acts, grads = store["acts"].detach(), store["grads"].detach()
        weights = grads.mean(dim=(2, 3), keepdim=True)                 # w_k: (B, C, 1, 1)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))        # (B, 1, h, w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        flat = cam.flatten(1)                                          # min-max normalise per image
        lo, hi = flat.min(1).values[:, None, None], flat.max(1).values[:, None, None]
        cam = (cam - lo) / (hi - lo).clamp_min(1e-8)
        self.model.zero_grad(set_to_none=True)
        return cam, logits.detach(), class_idx
