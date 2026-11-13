"""Is the model's confidence honest?  ECE, reliability diagrams and temperature scaling."""
import numpy as np
import torch
import torch.nn.functional as F

from derma.metrics import softmax


def calibration_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """Group predictions by confidence (= highest probability) into equal-width bins on [0, 1].

    Returns, per bin: mean confidence, accuracy and number of predictions.
    """
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # bin i holds confidences in (edges[i], edges[i+1]]
    which = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    mean_conf, acc, count = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = which == b
        count[b] = m.sum()
        if count[b]:
            mean_conf[b], acc[b] = conf[m].mean(), correct[m].mean()
    return mean_conf, acc, count


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE = sum over bins of (bin size / N) * |accuracy - confidence|.  0 = perfectly honest."""
    mean_conf, acc, count = calibration_bins(probs, labels, n_bins)
    return float(np.sum(count / count.sum() * np.abs(acc - mean_conf)))


def nll(logits: np.ndarray, labels: np.ndarray) -> float:
    """Average negative log-likelihood (= cross-entropy) of the true labels."""
    p = softmax(logits)[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(p, 1e-12, None)).mean())


def plot_reliability(ax, probs, labels, n_bins: int = 15, title: str = ""):
    mean_conf, acc, count = calibration_bins(probs, labels, n_bins)
    centers = (np.arange(n_bins) + 0.5) / n_bins
    has = count > 0
    ax.bar(centers[has], acc[has], width=1 / n_bins, edgecolor="black", alpha=0.8, label="accuracy")
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="confidence", ylabel="accuracy",
           title=f"{title}  ECE={expected_calibration_error(probs, labels, n_bins):.3f}")
    ax.legend(loc="upper left", fontsize=8)


class TemperatureScaler:
    """Divide all logits by one number T > 0, fitted on the validation set (Guo et al., 2017).

    T > 1 softens over-confident probabilities, T < 1 sharpens under-confident ones.
    Dividing by T never changes which class has the largest logit, so accuracy stays the same.
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        z = torch.as_tensor(logits, dtype=torch.float64)
        y = torch.as_tensor(labels, dtype=torch.long)
        log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)   # T = exp(log_t) is always > 0
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(z / log_t.exp(), y)
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature = float(log_t.exp().item())
        return self.temperature

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return logits / self.temperature
