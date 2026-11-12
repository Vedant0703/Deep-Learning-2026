"""Builds a labelled hard-factor training set from our own training images (src/hard.py): 200 images per class from
the 45k training side of the split, each with 1-2 random factors, labelled with its source image's label.

    python scripts/make_hard.py --factors lowres zoom_in zoom_out background confuse --out data/cache/hard_train2k.npz

Partner images (backgrounds, confusable-class blends) also come from the 45k side, so nothing touches the 5k held-out
split. The same arguments give byte-identical output (the MD5 is printed).
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import CLASSES, load_split, stratified_split  # noqa: E402
from src.hard import FACTORS, apply_factor, to_float, to_uint8  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--factors", nargs="+", default=list(FACTORS), choices=FACTORS)
    p.add_argument("--per-class", type=int, default=200)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--val-size", type=int, default=5000, help="held-out size of the split to stay out of")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/cache/hard_train2k.npz")
    p.add_argument("--preview", help="optional PNG with 40 of the images, enlarged and labelled")
    args = p.parse_args(argv)

    images, labels, _ = load_split(args.data_dir, "train")
    pool = stratified_split(labels, args.val_size, args.split_seed)[0]  # the 45k training side
    pool_images, pool_labels = images[pool], labels[pool]
    rng = np.random.default_rng(args.seed)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(pool_labels == c), args.per_class, replace=False)
                                  for c in range(10)]))
    n, k = len(idx), len(args.factors)
    steps = np.where(rng.random(n) < 0.5, 1, 2)  # 1 or 2 factors per image
    plan = np.array([rng.choice(k, 2, replace=False) for _ in range(n)])
    x = to_float(pool_images[idx])
    for s in range(2):
        for f in range(k):
            m = (plan[:, s] == f) & (steps > s)
            if m.any():
                x[torch.from_numpy(m)] = apply_factor(args.factors[f], x[torch.from_numpy(m)], idx[m],
                                                      pool_images, pool_labels, rng)
    out = to_uint8(x)
    names = np.array(["+".join(args.factors[plan[i, s]] for s in range(steps[i])) for i in range(n)])
    np.savez(args.out, images=out, labels=pool_labels[idx], sources=pool[idx], factors=names)
    md5 = hashlib.md5(out.tobytes() + pool_labels[idx].tobytes()).hexdigest()
    print(f"wrote {args.out}: {n} images, {np.bincount(pool_labels[idx], minlength=10).tolist()} per class, "
          f"factors {args.factors}, images+labels MD5 {md5}")
    print("factor combinations:", dict(zip(*np.unique(names, return_counts=True))))

    if args.preview:
        from PIL import Image, ImageDraw
        S, cols = 96, 10
        sheet = Image.new("RGB", (cols * S, 4 * (S + 14)), "white")
        draw = ImageDraw.Draw(sheet)
        for j, i in enumerate(rng.choice(n, 40, replace=False)):
            r, c = divmod(j, cols)
            sheet.paste(Image.fromarray(out[i]).resize((S, S), Image.NEAREST), (c * S, r * (S + 14) + 14))
            draw.text((c * S + 2, r * (S + 14)), f"{CLASSES[pool_labels[idx[i]]][:5]} {names[i][:12]}", fill="black")
        sheet.save(args.preview)
        print("preview:", args.preview)


if __name__ == "__main__":
    main()
