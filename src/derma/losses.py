import torch
import torch.nn as nn
import torch.nn.functional as F

from derma.data import compute_class_weights


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017): FL = -alpha_y * (1 - p_y)^gamma * log(p_y).

    p_y is the probability the model gives to the correct class. When the model is already
    confident and right (p_y close to 1), (1 - p_y)^gamma is close to 0, so easy examples barely
    count and training focuses on the hard ones. gamma = 0 and alpha = None give plain cross-entropy.
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        # a buffer moves to the GPU with model.to(device) / criterion.to(device), but is not trained
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits.float(), dim=1)          # float32 even under mixed precision
        logp_y = logp.gather(1, y[:, None]).squeeze(1)       # log p of the correct class, shape (B,)
        p_y = logp_y.exp()
        loss = -((1.0 - p_y) ** self.gamma) * logp_y
        if self.alpha is not None:
            loss = loss * self.alpha[y]
        return loss.mean()


def build_criterion(cfg: dict, train_labels, device) -> nn.Module:
    """cfg['loss']: 'ce' | 'weighted_ce' | 'focal' (with cfg['focal_gamma'], default 2)."""
    loss = cfg.get("loss", "ce")
    if loss == "weighted_ce" and cfg.get("sampler") == "oversample":
        raise ValueError("weighted loss + oversampling would correct the imbalance twice")
    if loss == "ce":
        criterion = nn.CrossEntropyLoss()
    elif loss == "weighted_ce":
        w = torch.as_tensor(compute_class_weights(train_labels), dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=w)
    elif loss == "focal":
        criterion = FocalLoss(gamma=cfg.get("focal_gamma", 2.0))
    else:
        raise ValueError(f"unknown loss: {loss}")
    return criterion.to(device)
