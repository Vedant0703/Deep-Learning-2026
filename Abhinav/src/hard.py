"""Hard-factor versions of our own labelled images: low resolution, scale, orientation, viewpoint, background
clutter and class similarity. Each output keeps its source image's label (the majority source for blends).

Used two ways, always on images from the permitted training set:
- the hard suite (src.evaluate --hard-suite): every factor at a fixed level on the 5k held-out split, to examine
  where a model fails;
- a training set (scripts/make_hard.py): 2,000 images from the 45k training side with 1-2 random factors each;
- on-the-fly training augmentation (hard_batch, `src.train --hard-aug`), the version that worked (Runs L and M).

All ops take float images in [0, 1], shape [N, 3, H, W], and per-image parameters, so results are reproducible.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

# Pairs CIFAR-10 models confuse most (class index -> partner classes): airplane/ship/bird, automobile/truck,
# cat/dog, deer/horse, frog/cat/bird.
CONFUSABLE = {0: [8, 2], 1: [9], 2: [0, 6], 3: [5], 4: [7, 2], 5: [3], 6: [3, 2], 7: [4, 5], 8: [0], 9: [1]}


def to_float(images):
    """uint8 [N, H, W, 3] -> float [N, 3, H, W] in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float() / 255


def to_uint8(x):
    """float [N, 3, H, W] in [0, 1] -> uint8 [N, H, W, 3], rounded like a saved PNG."""
    return (x.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()


def low_res(x, size):
    """Downsample to size x size (antialiased), then back up to the original size."""
    h = x.shape[-1]
    small = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    return F.interpolate(small, size=(h, h), mode="bilinear", align_corners=False)


def zoom_in(x, factor):
    """Centre crop of 1/factor of the side, resized back: the object fills more of the frame."""
    h = x.shape[-1]
    c = max(4, round(h / factor))
    o = (h - c) // 2
    return F.interpolate(x[..., o:o + c, o:o + c], size=(h, h), mode="bilinear", antialias=True,
                         align_corners=False)


def _shrink(x, factor):
    h = x.shape[-1]
    s = max(4, round(h * factor))
    return F.interpolate(x, size=(s, s), mode="bilinear", antialias=True, align_corners=False), h, s


def zoom_out(x, factor):
    """Shrink to factor of the side and pad back by repeating the edge pixels: the object is smaller in the frame
    (reflect padding would mirror the object into a second copy)."""
    small, h, s = _shrink(x, factor)
    lo = (h - s) // 2
    return F.pad(small, (lo, h - s - lo, lo, h - s - lo), mode="replicate")


def background(x, other, factor):
    """Shrink to factor of the side and paste into the centre of another image: a cluttered background."""
    small, h, s = _shrink(x, factor)
    lo = (h - s) // 2
    out = other.clone()
    out[..., lo:lo + s, lo:lo + s] = small
    return out


def blend(x, other, lam):
    """Pixel blend lam * x + (1 - lam) * other, with lam >= 0.5 so x's label stays the majority."""
    lam = torch.as_tensor(lam, dtype=x.dtype).view(-1, 1, 1, 1)
    return lam * x + (1 - lam) * other


def rotate(x, degrees):
    """Rotate each image by its angle (degrees, per image), reflecting at the borders."""
    a = torch.as_tensor(degrees, dtype=x.dtype, device=x.device).view(-1) * math.pi / 180
    cos, sin, zero = torch.cos(a), torch.sin(a), torch.zeros_like(a)
    theta = torch.stack([torch.stack([cos, -sin, zero], 1), torch.stack([sin, cos, zero], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def perspective(x, offsets):
    """Perspective warp: offsets [N, 4, 2] move the four corners, in normalized units (the side is 2)."""
    n = x.shape[0]
    src = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=torch.float32).expand(n, 4, 2)
    dst = src + torch.as_tensor(offsets, dtype=torch.float32).cpu()
    # Homography mapping output (dst) coordinates to input (src) coordinates, solved per image (8 unknowns) on the
    # CPU, where torch.linalg.solve works for every backend; only the sampling grid goes to x's device.
    rows, rhs = [], []
    for d, c in zip(dst.unbind(1), src.unbind(1)):
        (u, v), (xs, ys) = d.unbind(-1), c.unbind(-1)
        one, zero = torch.ones_like(u), torch.zeros_like(u)
        rows += [torch.stack([u, v, one, zero, zero, zero, -u * xs, -v * xs], 1),
                 torch.stack([zero, zero, zero, u, v, one, -u * ys, -v * ys], 1)]
        rhs += [xs, ys]
    h = torch.linalg.solve(torch.stack(rows, 1), torch.stack(rhs, 1))
    hm = torch.cat([h, torch.ones(n, 1)], 1).view(n, 3, 3)
    size = x.shape[-1]
    lin = (torch.arange(size, dtype=torch.float32) * 2 + 1) / size - 1
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    pts = torch.stack([gx, gy, torch.ones_like(gx)], -1).view(1, -1, 3)
    mapped = pts @ hm.transpose(1, 2)
    grid = (mapped[..., :2] / mapped[..., 2:]).view(n, size, size, 2).to(x.device, x.dtype)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def pick_partners(labels, rng, confusable):
    """For each image, the index of a partner image: from a confusable class, or from any other class."""
    by_class = {c: np.flatnonzero(labels == c) for c in range(10)}
    out = np.empty(len(labels), dtype=np.int64)
    for i, c in enumerate(labels):
        pc = rng.choice(CONFUSABLE[int(c)]) if confusable else rng.choice([k for k in range(10) if k != c])
        out[i] = rng.choice(by_class[int(pc)])
    return out


def apply_factor(name, x, idx, images, labels, rng, level=None):
    """Applies one factor to x (float [n, 3, H, W], the images at positions idx of `images` / `labels`).
    level=None draws a level per image from the training range; a number fixes it (the hard suite)."""
    n = len(x)
    ranges = {"lowres": [12, 16, 20, 24], "zoom_in": (1.2, 1.5), "zoom_out": (0.6, 0.85), "rotate": (15, 30),
              "perspective": (0.3, 0.5), "background": (0.6, 0.85), "confuse": (0.65, 0.8)}
    if level is not None:
        levels = np.full(n, float(level))
    elif name == "lowres":
        levels = rng.choice(ranges[name], n).astype(float)
    else:
        levels = rng.uniform(*ranges[name], n)
    if name in ("zoom_in", "zoom_out", "background"):
        levels = np.round(levels * 32) / 32  # a few distinct sizes, resized in groups
    if name in ("lowres", "zoom_in", "zoom_out", "background"):
        op = {"lowres": lambda v, f, o: low_res(v, int(f)), "zoom_in": lambda v, f, o: zoom_in(v, f),
              "zoom_out": lambda v, f, o: zoom_out(v, f), "background": lambda v, f, o: background(v, o, f)}[name]
        other = to_float(images[pick_partners(labels[idx], rng, False)]) if name == "background" else None
        out = x.clone()
        for f in np.unique(levels):
            m = torch.from_numpy(levels == f)
            out[m] = op(x[m], float(f), other[m] if other is not None else None)
        return out
    lv = torch.from_numpy(levels).float()
    if name == "rotate":
        sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0) if level is not None else rng.choice([-1.0, 1.0], n)
        return rotate(x, torch.from_numpy(sign).float() * lv)
    if name == "perspective":
        return perspective(x, torch.from_numpy(rng.uniform(-1, 1, (n, 4, 2))).float() * lv.view(-1, 1, 1))
    if name == "confuse":
        return blend(x, to_float(images[pick_partners(labels[idx], rng, True)]), lv)
    raise ValueError(f"unknown factor '{name}'")


# The hard suite: one fixed level per row, applied to the 5k held-out images.
HARD_SUITE = {"lowres16": ("lowres", 16), "lowres24": ("lowres", 24), "zoom_in1.3": ("zoom_in", 1.3),
              "zoom_out0.7": ("zoom_out", 0.7), "rot20": ("rotate", 20), "persp0.4": ("perspective", 0.4),
              "bg0.7": ("background", 0.7), "confuse0.7": ("confuse", 0.7)}


def hard_suite_images(images, labels, seed=0):
    """{row name: uint8 images} for the hard suite on (images, labels); the same seed gives the same images."""
    x, idx = to_float(images), np.arange(len(images))
    return {row: to_uint8(apply_factor(name, x, idx, images, labels, np.random.default_rng([seed, i]), level))
            for i, (row, (name, level)) in enumerate(HARD_SUITE.items())}


FACTORS = ("lowres", "zoom_in", "zoom_out", "rotate", "perspective", "background", "confuse")
BATCH_FACTORS = ("lowres", "zoom_in", "zoom_out", "background")  # hard_batch's default factors (Runs L and M)
ALL_BATCH_FACTORS = BATCH_FACTORS + ("rotate", "perspective")  # every factor hard_batch can apply on the fly

# Level ranges per factor: "default" is the training range of Runs L and M; "wide" reaches smaller, blurrier and
# more zoomed objects. Each entry maps u ~ U(0, 1) (one draw per factor per batch) to a level.
HARD_LEVELS = {
    "default": {"lowres": lambda u: (12, 16, 20, 24)[int(u * 4)], "zoom_in": lambda u: 1.2 + 0.3 * u,
                "zoom_out": lambda u: 0.6 + 0.25 * u, "background": lambda u: 0.6 + 0.25 * u,
                "rotate": lambda u: 15 + 15 * u, "perspective": lambda u: 0.3 + 0.2 * u},
    "wide": {"lowres": lambda u: (8, 12, 16, 20, 24)[int(u * 5)], "zoom_in": lambda u: 1.2 + 0.6 * u,
             "zoom_out": lambda u: 0.5 + 0.35 * u, "background": lambda u: 0.5 + 0.35 * u,
             "rotate": lambda u: 15 + 15 * u, "perspective": lambda u: 0.3 + 0.2 * u},
}


def hard_batch(x, p, factors, mean, std, levels="default"):
    """On-the-fly hard augmentation of a normalized training batch, on its device: a random fraction p of the images
    gets one random factor from `factors` (ALL_BATCH_FACTORS), with one level per factor per batch drawn from
    HARD_LEVELS[levels]. Background partners are other images of the same batch; rotations get a random sign and
    perspective warps random corner offsets per image. Returns (batch, bool mask of changed images)."""
    n = x.shape[0]
    mask = torch.rand(n, device=x.device) < p
    if not mask.any():
        return x, mask
    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    img = (x * s + m).clamp(0, 1)
    out = img.clone()
    which = torch.randint(len(factors), (n,), device=x.device)
    partner = img[torch.randperm(n, device=x.device)]
    for k, name in enumerate(factors):
        sel = mask & (which == k)
        if not sel.any():
            continue
        level = HARD_LEVELS[levels][name](torch.rand(()).item())
        if name == "lowres":
            out[sel] = low_res(img[sel], level)
        elif name == "zoom_in":
            out[sel] = zoom_in(img[sel], level)
        elif name == "zoom_out":
            out[sel] = zoom_out(img[sel], level)
        elif name == "background":
            out[sel] = background(img[sel], partner[sel], level)
        elif name == "rotate":
            k_sel = int(sel.sum())
            sign = torch.randint(2, (k_sel,), device=x.device).float() * 2 - 1
            out[sel] = rotate(img[sel], sign * level)
        elif name == "perspective":
            out[sel] = perspective(img[sel], (torch.rand(int(sel.sum()), 4, 2) * 2 - 1) * level)
        else:
            raise ValueError(f"hard_batch does not support '{name}'")
    return torch.where(mask.view(-1, 1, 1, 1), (out - m) / s, x), mask  # others untouched (not even clamped)
