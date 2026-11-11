import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from derma.utils import seed_worker

# DermaMNIST label index -> short name (from the HAM10000 dataset)
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_DESCRIPTIONS = [
    "Actinic keratoses / intraepithelial carcinoma",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions",
]
NUM_CLASSES = len(CLASS_NAMES)

# Mean / std of ImageNet, used for all three models (the pretrained ResNet expects them).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DermaDataset(Dataset):
    """DermaMNIST split as uint8 images (N, H, W, 3) and integer labels (N,)."""

    def __init__(self, split: str, size: int = 64, root: str = "data", transform=None):
        from medmnist import DermaMNIST  # imported here so tests without medmnist still run

        os.makedirs(root, exist_ok=True)
        raw = DermaMNIST(split=split, size=size, root=root, download=True)
        self.images = raw.imgs                          # numpy uint8, shape (N, size, size, 3)
        self.labels = raw.labels.squeeze(1).astype(np.int64)  # (N, 1) -> (N,)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])   # torchvision transforms work on PIL images
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


def build_transforms(train: bool, augment: str = "none"):
    """augment: 'none' or 'hflip' for now (more options will be added on day 2)."""
    steps = []
    if train and augment == "hflip":
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    steps += [
        transforms.ToTensor(),                               # uint8 [0,255] HWC -> float [0,1] CHW
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),   # (x - mean) / std per channel
    ]
    return transforms.Compose(steps)


def class_counts(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    return np.bincount(labels, minlength=num_classes)


def compute_class_weights(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """w_c = N / (K * n_c): rare classes get large weights, the mean weight over samples is 1."""
    counts = class_counts(labels, num_classes).astype(np.float64)
    return len(labels) / (num_classes * counts)


def make_loaders(cfg: dict, seed: int):
    """Train loader (shuffled, augmented) and val loader (fixed order, no augmentation)."""
    root, size = cfg["data_root"], cfg["img_size"]
    train_ds = DermaDataset("train", size, root, build_transforms(True, cfg["augment"]))
    val_ds = DermaDataset("val", size, root, build_transforms(False))

    g = torch.Generator()
    g.manual_seed(seed)
    common = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
                  pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker)
    train_loader = DataLoader(train_ds, shuffle=True, generator=g, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader
