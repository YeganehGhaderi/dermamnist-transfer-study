import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)   # subtract the max for numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def compute_metrics(logits: np.ndarray, labels: np.ndarray, num_classes: int = 7) -> dict:
    probs = softmax(logits)
    preds = probs.argmax(axis=1)
    out = {
        "accuracy": accuracy_score(labels, preds),
        "balanced_acc": balanced_accuracy_score(labels, preds),   # mean recall over classes
        "macro_f1": f1_score(labels, preds, average="macro",
                             labels=list(range(num_classes)), zero_division=0),
    }
    try:  # one-vs-rest AUC per class, then the plain mean over classes
        out["auc_macro"] = roc_auc_score(labels, probs, multi_class="ovr", average="macro",
                                         labels=list(range(num_classes)))
    except ValueError:  # e.g. a class missing from `labels`
        out["auc_macro"] = float("nan")
    return {k: float(v) for k, v in out.items()}
