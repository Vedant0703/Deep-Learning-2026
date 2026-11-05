"""Data loading for the competition's CIFAR-10 files, plus augmentation.

Expected layout (the unzipped competition data):
    data/train_labels.csv    id,filename,label
    data/train_images/*.png  50,000 32x32 RGB training images
    data/test.csv            id,filename
    data/test_images/*.png   2,000 unlabeled evaluation images
    data/sample_submission.csv

PNGs are decoded once and cached as data/cache/<split>.npz.
"""
import io
import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

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


class GaussianNoise:
    """Adds N(0, s^2) pixel noise, s ~ U(0, max_std), to a fraction p of images ([0, 1] tensors, before Normalize)."""

    def __init__(self, max_std, p=0.5):
        self.max_std, self.p = max_std, p

    def __call__(self, x):
        if torch.rand(()) < self.p:
            x = (x + torch.randn_like(x) * torch.rand(()) * self.max_std).clamp(0, 1)
        return x


def train_transform(aug="trivial", erasing=0.25, jitter=0.0, noise=0.0):
    """Random crop + flip, optional brightness/contrast jitter, a policy-based augmentation, optional Gaussian noise,
    and random erasing. aug="augmix" is AugMix (Hendrycks et al., ICLR 2020), aimed at corruption robustness."""
    ops = [transforms.RandomCrop(32, padding=4, padding_mode="reflect"), transforms.RandomHorizontalFlip()]
    if jitter > 0:  # robustness to global lighting changes, applied to every image (TrivialAugment does so rarely)
        ops.append(transforms.ColorJitter(brightness=jitter, contrast=jitter))
    if aug == "trivial":
        ops.append(transforms.TrivialAugmentWide())
    elif aug == "autoaugment":
        ops.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
    elif aug == "augmix":
        ops.append(transforms.AugMix(severity=3, mixture_width=3, all_ops=True))
    elif aug != "basic":
        raise ValueError(f"unknown aug '{aug}'")
    ops.append(transforms.ToTensor())
    if noise > 0:
        ops.append(GaussianNoise(noise))
    ops.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    if erasing > 0:
        ops.append(transforms.RandomErasing(p=erasing, value=0))
    return transforms.Compose(ops)


def _gaussian_noise(img, std):
    a = np.asarray(img, np.float32) / 255
    rng = np.random.default_rng(zlib.crc32(a.tobytes()))  # same noise for the same image, so runs are comparable
    return Image.fromarray((np.clip(a + rng.normal(0, std, a.shape), 0, 1) * 255).round().astype(np.uint8))


def _jpeg(img, quality):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(quality))
    return Image.open(buf).convert("RGB")


# Generic corruptions (CIFAR-10-C families, Hendrycks & Dietterich, ICLR 2019) for robustness checks on the val split.
CORRUPTIONS = {
    "brightness": TF.adjust_brightness,
    "contrast": TF.adjust_contrast,
    "saturation": TF.adjust_saturation,
    "sharpness": TF.adjust_sharpness,
    "blur": lambda img, sigma: img.filter(ImageFilter.GaussianBlur(sigma)),
    "noise": _gaussian_noise,
    "jpeg": _jpeg,
}


class Corrupt:
    """Applies a fixed sequence of (corruption name, level) pairs to a PIL image."""

    def __init__(self, ops):
        self.ops = tuple(ops)

    def __call__(self, img):
        for name, level in self.ops:
            img = CORRUPTIONS[name](img, level)
        return img


def eval_transform(corruption=()):
    ops = [Corrupt(corruption)] if corruption else []
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])


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
                  num_workers=2, seed=0, subset=None, jitter=0.0, noise=0.0, extra_train=None, extra_repeat=1):
    """Returns (train_loader, val_loader). val_loader is None when val_size == 0 (train on all 50k).
    extra_train: an .npz of extra labelled images made from our own training images (scripts/make_hard.py),
    appended extra_repeat times to the training set."""
    images, labels, _ = load_split(data_dir, "train")
    if val_size > 0:
        tr_idx, va_idx = stratified_split(labels, val_size, seed)
    else:
        tr_idx, va_idx = np.arange(len(labels)), np.array([], dtype=np.int64)
    if subset:  # tiny runs for debugging
        tr_idx, va_idx = tr_idx[:subset], va_idx[:subset]

    train_images, train_labels = images[tr_idx], labels[tr_idx]
    if extra_train:
        z = np.load(extra_train)
        # Never train on images derived from the held-out split.
        assert not np.isin(z["sources"], va_idx).any(), f"{extra_train} uses held-out images"
        train_images = np.concatenate([train_images] + [z["images"]] * extra_repeat)
        train_labels = np.concatenate([train_labels] + [z["labels"]] * extra_repeat)
    train_ds = ArrayDataset(train_images, train_labels, train_transform(aug, erasing, jitter, noise))
    train_loader = _loader(train_ds, batch_size, True, num_workers)
    val_loader = None
    if len(va_idx):
        val_ds = ArrayDataset(images[va_idx], labels[va_idx], eval_transform())
        val_loader = _loader(val_ds, batch_size, False, num_workers)
    return train_loader, val_loader


def build_val_loader(data_dir, val_size=5000, seed=0, corruption=(), batch_size=500, num_workers=2, part="val"):
    """The same held-out split as build_loaders, optionally corrupted ((name, level) pairs, see CORRUPTIONS).
    part="train" gives the other side of the split (the 45k training images, un-augmented) instead."""
    images, labels, _ = load_split(data_dir, "train")
    tr_idx, va_idx = stratified_split(labels, val_size, seed)
    idx = va_idx if part == "val" else tr_idx
    ds = ArrayDataset(images[idx], labels[idx], eval_transform(corruption))
    return _loader(ds, batch_size, False, num_workers)


def build_test_loader(data_dir, batch_size=500, num_workers=2):
    """Returns (loader over the unlabeled evaluation images, ids in test.csv order)."""
    images, _, ids = load_split(data_dir, "test")
    return _loader(ArrayDataset(images, None, eval_transform()), batch_size, False, num_workers), ids
