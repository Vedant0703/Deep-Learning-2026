"""Saves an example grid of the hard-factor suite (src/hard.py): one held-out image per class, each factor at its
suite level, labelled by row and class. Used for Figure 1 of the Round 9 write-up.

    python scripts/hard_suite_figure.py figures/hard_suite_examples.png
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import CLASSES, load_split, stratified_split  # noqa: E402
from src.hard import hard_suite_images  # noqa: E402


def main(out, data_dir="data"):
    images, labels, _ = load_split(data_dir, "train")
    va = stratified_split(labels, 5000, 0)[1]  # the 5k held-out side of every twin
    vi, vl = images[va], labels[va]
    rows = [("clean", vi)] + list(hard_suite_images(vi, vl, seed=0).items())
    pick = [int(np.flatnonzero(vl == c)[3]) for c in range(10)]
    S, L, T = 84, 112, 24
    font = ImageFont.load_default(size=15)
    sheet = Image.new("RGB", (L + 10 * S, T + len(rows) * S), "white")
    d = ImageDraw.Draw(sheet)
    for c in range(10):
        d.text((L + c * S + 2, 3), CLASSES[c], fill="black", font=font)
    for r, (name, imgs) in enumerate(rows):
        d.text((4, T + r * S + S // 2 - 9), name, fill="black", font=font)
        for c, i in enumerate(pick):
            sheet.paste(Image.fromarray(imgs[i]).resize((S - 2, S - 2), Image.NEAREST), (L + c * S, T + r * S))
    sheet.save(out)
    print(out, sheet.size)


if __name__ == "__main__":
    main(*sys.argv[1:])
