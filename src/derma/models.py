import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    # bias=False because BatchNorm right after it has its own shift (beta); a bias would be redundant.
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SmallCNN(nn.Module):
    """3 blocks of [Conv-BN-ReLU] x2 + MaxPool, then Global Average Pooling and a linear classifier.

    64x64 input -> 32x32 -> 16x16 -> 8x8 feature maps with 32 -> 64 -> 128 channels.
    """

    def __init__(self, num_classes: int = 7, channels=(32, 64, 128), dropout: float = 0.3):
        super().__init__()
        layers, in_ch = [], 3
        for out_ch in channels:
            layers += [conv_bn_relu(in_ch, out_ch), conv_bn_relu(out_ch, out_ch), nn.MaxPool2d(2)]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)          # (B, C, H, W) -> (B, C, 1, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)                  # (B, C)
        return self.fc(self.dropout(x))              # logits, (B, num_classes)


def build_model(name: str, num_classes: int = 7, load_weights: bool = True) -> nn.Module:
    """name: 'smallcnn' | 'resnet18_scratch' | 'resnet18_pretrained'.

    load_weights=False builds the pretrained architecture without downloading weights (for tests).
    """
    if name == "smallcnn":
        return SmallCNN(num_classes)
    if name in ("resnet18_scratch", "resnet18_pretrained"):
        weights = ResNet18_Weights.IMAGENET1K_V1 if (name == "resnet18_pretrained" and load_weights) else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)   # 1000 ImageNet classes -> 7
        return model
    raise ValueError(f"unknown model: {name}")


def param_groups(model: nn.Module, name: str, lr: float, lr_backbone: float):
    """Pretrained ResNet: small lr for the backbone (keep what it learned), larger lr for the new head."""
    if name != "resnet18_pretrained":
        return [{"params": model.parameters(), "lr": lr}]
    head = list(model.fc.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return [{"params": backbone, "lr": lr_backbone}, {"params": head, "lr": lr}]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
