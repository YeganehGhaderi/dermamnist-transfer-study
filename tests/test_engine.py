import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from derma.engine import EarlyStopping, evaluate, train_one_epoch
from derma.models import build_model
from derma.utils import set_seed


def test_overfit_one_batch():
    """A correct training loop must be able to memorise 32 examples (even with random labels)."""
    set_seed(0)
    x = torch.randn(32, 3, 32, 32)
    y = torch.randint(0, 7, (32,))
    loader = DataLoader(TensorDataset(x, y), batch_size=32)
    model = build_model("smallcnn")
    model.dropout.p = 0.0                      # dropout would fight memorisation
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    device = torch.device("cpu")

    first = train_one_epoch(model, loader, criterion, optimizer, scaler, device)
    for _ in range(300):
        last = train_one_epoch(model, loader, criterion, optimizer, scaler, device)
    _, logits, labels = evaluate(model, loader, criterion, device)

    assert last < 0.1 < first
    assert (logits.argmax(1) == labels).all()


def test_evaluate_returns_logits_and_labels_in_order():
    model = build_model("smallcnn")
    x, y = torch.randn(10, 3, 32, 32), torch.arange(10) % 7
    loss, logits, labels = evaluate(model, DataLoader(TensorDataset(x, y), batch_size=4),
                                    nn.CrossEntropyLoss(), torch.device("cpu"))
    assert logits.shape == (10, 7) and (labels == y.numpy()).all() and loss > 0


def test_early_stopping():
    stopper = EarlyStopping(patience=2, mode="max")
    assert stopper.step(0.5) is True          # first value is always a new best
    assert stopper.step(0.6) is True
    assert stopper.step(0.6) is False         # equal is not an improvement
    assert not stopper.should_stop
    assert stopper.step(0.55) is False
    assert stopper.should_stop                # 2 epochs without improvement
    assert stopper.best == 0.6
