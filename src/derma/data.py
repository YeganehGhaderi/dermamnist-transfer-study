import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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


# Augmentation levels for experiment C. Each level adds one technique on top of the previous one
# (a "cumulative ablation"). 'hflip' is the setting used in experiments A and B.
AUG_LEVELS = ["none", "flips", "flips_rot", "flips_rot_color", "full"]


def build_transforms(train: bool, augment: str = "none", size: int = 64):
    """Training images get random augmentation, validation/test images never do."""
    steps = []
    if train and augment == "hflip":
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    elif train and augment != "none":
        if augment not in AUG_LEVELS:
            raise ValueError(f"unknown augment level: {augment}")
        level = AUG_LEVELS.index(augment)
        if level >= 4:   # random zoom: keep 80-100% of the area, then resize back to size x size
            steps.append(transforms.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1)))
        if level >= 1:   # a lesion has no natural "up" or "left": mirror both ways
            steps += [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
        if level >= 2:   # any angle between -180 and +180 degrees
            steps.append(transforms.RandomRotation(degrees=180))
        if level >= 3:   # mild changes in lighting and skin tone between cameras
            steps.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02))
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


def oversampling_weights(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Weight of each *sample* = 1 / (size of its class): every class is then drawn equally often."""
    counts = class_counts(labels, num_classes).astype(np.float64)
    return 1.0 / counts[labels]


def stratified_subset(labels: np.ndarray, fraction: float, seed: int = 0) -> np.ndarray:
    """Indices of a random `fraction` of the data that keeps the class proportions (at least 1 per class)."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        n = max(1, int(round(fraction * len(idx))))
        keep.append(rng.choice(idx, n, replace=False))
    return np.sort(np.concatenate(keep))


def make_loaders(cfg: dict, seed: int):
    """Train loader (augmented; shuffled or oversampled) and val loader (fixed order, no augmentation)."""
    root, size = cfg["data_root"], cfg["img_size"]
    train_ds = DermaDataset("train", size, root, build_transforms(True, cfg["augment"], size))
    val_ds = DermaDataset("val", size, root, build_transforms(False, size=size))

    fraction = cfg.get("train_fraction", 1.0)
    if fraction < 1.0:   # optional experiment E: train on less data (same subset for every model)
        idx = stratified_subset(train_ds.labels, fraction, seed=0)
        train_ds.images, train_ds.labels = train_ds.images[idx], train_ds.labels[idx]

    g = torch.Generator()
    g.manual_seed(seed)
    common = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
                  pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker)
    if cfg.get("sampler", "shuffle") == "oversample":
        weights = torch.as_tensor(oversampling_weights(train_ds.labels), dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True, generator=g)
        train_loader = DataLoader(train_ds, sampler=sampler, **common)   # sampler replaces shuffle
    else:
        train_loader = DataLoader(train_ds, shuffle=True, generator=g, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader
