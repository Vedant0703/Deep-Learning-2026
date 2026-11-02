"""Data loading for the competition's CIFAR-10 files, plus augmentation.

Expected layout (the unzipped competition data):
    data/train_labels.csv    id,filename,label
    data/train_images/*.png  50,000 32x32 RGB training images
    data/test.csv            id,filename
    data/test_images/*.png   2,000 unlabeled evaluation images
    data/sample_submission.csv

PNGs are decoded once and cached as data/cache/<split>.npz.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Per-channel statistics of the 50k training images (they match the standard CIFAR-10 values).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def load_split(data_dir, split):
    """Returns (images uint8 [N,32,32,3], labels int64 [N] or None, ids int64 [N])."""
    data_dir = Path(data_dir)
    csv, folder = {"train": ("train_labels.csv", "train_images"), "test": ("test.csv", "test_images")}[split]
    cache = data_dir / "cache" / f"{split}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["images"], (z["labels"] if "labels" in z else None), z["ids"]

    df = pd.read_csv(data_dir / csv)
    images = np.stack([np.asarray(Image.open(data_dir / folder / f).convert("RGB")) for f in df.filename])
    ids = df.id.to_numpy(np.int64)
    labels = df.label.to_numpy(np.int64) if "label" in df else None

    cache.parent.mkdir(exist_ok=True)
    arrays = {"images": images, "ids": ids, **({"labels": labels} if labels is not None else {})}
    tmp = cache.with_name(f"{split}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, cache)  # atomic, so concurrent runs never read a half-written cache
    return images, labels, ids


class ArrayDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = self.transform(Image.fromarray(self.images[i]))
        return (x, int(self.labels[i])) if self.labels is not None else x


def train_transform(aug="trivial", erasing=0.25):
    """Random crop + flip, plus an optional policy-based augmentation and random erasing (Cutout)."""
    ops = [transforms.RandomCrop(32, padding=4, padding_mode="reflect"), transforms.RandomHorizontalFlip()]
    if aug == "trivial":
        ops.append(transforms.TrivialAugmentWide())
    elif aug == "autoaugment":
        ops.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
    elif aug != "basic":
        raise ValueError(f"unknown aug '{aug}'")
    ops += [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    if erasing > 0:
        ops.append(transforms.RandomErasing(p=erasing, value=0))
    return transforms.Compose(ops)


def eval_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])


def _loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=shuffle,
                      persistent_workers=num_workers > 0)


def stratified_split(labels, val_size, seed):
    """Indices for a class-balanced train/val split of the training set."""
    rng = np.random.default_rng(seed)
    per_class = val_size // 10
    val = np.concatenate([rng.permutation(np.flatnonzero(labels == c))[:per_class] for c in range(10)])
    train = np.setdiff1d(np.arange(len(labels)), val)
    return train, np.sort(val)


def build_loaders(data_dir, batch_size=256, val_size=5000, aug="trivial", erasing=0.25,
                  num_workers=2, seed=0, subset=None):
    """Returns (train_loader, val_loader). val_loader is None when val_size == 0 (train on all 50k)."""
    images, labels, _ = load_split(data_dir, "train")
    if val_size > 0:
        tr_idx, va_idx = stratified_split(labels, val_size, seed)
    else:
        tr_idx, va_idx = np.arange(len(labels)), np.array([], dtype=np.int64)
    if subset:  # tiny runs for debugging
        tr_idx, va_idx = tr_idx[:subset], va_idx[:subset]

    train_ds = ArrayDataset(images[tr_idx], labels[tr_idx], train_transform(aug, erasing))
    train_loader = _loader(train_ds, batch_size, True, num_workers)
    val_loader = None
    if len(va_idx):
        val_ds = ArrayDataset(images[va_idx], labels[va_idx], eval_transform())
        val_loader = _loader(val_ds, batch_size, False, num_workers)
    return train_loader, val_loader


def build_test_loader(data_dir, batch_size=500, num_workers=2):
    """Returns (loader over the unlabeled evaluation images, ids in test.csv order)."""
    images, _, ids = load_split(data_dir, "test")
    return _loader(ArrayDataset(images, None, eval_transform()), batch_size, False, num_workers), ids
