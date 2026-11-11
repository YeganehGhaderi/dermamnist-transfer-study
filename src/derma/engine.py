import numpy as np
import torch


def train_one_epoch(model, loader, criterion, optimizer, scaler, device) -> float:
    """One pass over the training data. Returns the mean loss per sample."""
    model.train()                                  # Dropout on, BatchNorm uses batch statistics
    use_amp = device.type == "cuda"
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)      # 1) clear old gradients
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)                      # 2) forward pass (in float16 on GPU)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()              # 3) backward pass on the scaled loss
        scaler.step(optimizer)                     # 4) unscale gradients, then update weights
        scaler.update()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


@torch.no_grad()                                   # no gradients: faster, less memory
def evaluate(model, loader, criterion, device):
    """Returns (mean loss, logits as numpy (N, K), labels as numpy (N,))."""
    model.eval()                                   # Dropout off, BatchNorm uses running statistics
    all_logits, all_labels, total_loss, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x).float()
        total_loss += criterion(logits, y).item() * x.size(0)
        n += x.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    return total_loss / n, logits, labels


class EarlyStopping:
    """Stop when the monitored value has not improved for `patience` epochs.

    step(value) returns True if `value` is a new best (caller then saves a checkpoint).
    """

    def __init__(self, patience: int = 8, mode: str = "max", min_delta: float = 0.0):
        assert mode in ("max", "min")
        self.patience, self.mode, self.min_delta = patience, mode, min_delta
        self.best = -np.inf if mode == "max" else np.inf
        self.bad_epochs = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        improved = (value > self.best + self.min_delta) if self.mode == "max" \
            else (value < self.best - self.min_delta)
        if improved:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
            self.should_stop = self.bad_epochs >= self.patience
        return improved
