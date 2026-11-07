"""Paired PreAct-vs-M analysis on the 5k held-out split with the submission decoding (flip_shift TTA + Sinkhorn
balancing): exact McNemar, the both / A-only / B-only error groups by class and confusion pair, and how often A
beats B on random 1,000-image subsets (the size of the public half). Usage: paired.py RUN_A RUN_B (run names)."""
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, ".")  # run from the project folder
from src.data import CLASSES, eval_transform, load_split, stratified_split
from src.evaluate import mcnemar_p
from src.predict import balance_probs
from src.train import get_device, load_model, tta_probs

if __name__ == "__main__":
    a_run, b_run = sys.argv[1:3]
    dev = get_device()
    images, labels, _ = load_split("data", "train")
    va = stratified_split(labels, 5000, 0)[1]
    t = eval_transform()
    x_all = torch.stack([t(img) for img in images[va]])
    y = labels[va]
    pred = {}
    for r in (a_run, b_run):
        model = load_model(f"runs/{r}/final.pt", dev)
        probs = []
        with torch.no_grad():
            for i in range(0, len(x_all), 500):
                probs.append(tta_probs(model, x_all[i:i + 500].to(dev), "flip_shift").float().cpu())
        p = torch.cat(probs).numpy()
        pred[r] = (p.argmax(1), balance_probs(p).argmax(1))
        print(f"{r:<20} fs {100 * (pred[r][0] == y).mean():.2f}  fs+bal {100 * (pred[r][1] == y).mean():.2f}")

    for k, name in ((1, "fs+bal"), (0, "fs")):
        wa, wb = pred[a_run][k] != y, pred[b_run][k] != y
        only_a_right, only_b_right = int((~wa & wb).sum()), int((wa & ~wb).sum())
        print(f"\n{name}: both wrong {int((wa & wb).sum())} | only {a_run} right {only_a_right} | only {b_run} right "
              f"{only_b_right} | McNemar p = {mcnemar_p(only_a_right, only_b_right):.3f} | "
              f"Jaccard {(wa & wb).sum() / (wa | wb).sum():.2f}")

    wa, wb = pred[a_run][1] != y, pred[b_run][1] != y
    groups = {"both wrong": wa & wb, f"only {b_run} wrong": ~wa & wb, f"only {a_run} wrong": wa & ~wb}
    print("\nerrors by true class (fs+bal):", " ".join(f"{c[:5]:>6}" for c in CLASSES))
    for g, m in groups.items():
        print(f"{g:<28}", " ".join(f"{int((y[m] == c).sum()):>6}" for c in range(10)), f"| {int(m.sum())}")
    print("\ntop confusions (true -> predicted, fs+bal):")
    for g, m in groups.items():
        src = pred[b_run][1] if g == f"only {b_run} wrong" else pred[a_run][1]
        top = Counter((CLASSES[t], CLASSES[p]) for t, p in zip(y[m], src[m])).most_common(5)
        print(f"  {g:<26}", ", ".join(f"{t}->{p} {n}" for (t, p), n in top))

    rng = np.random.default_rng(0)
    ca, cb = (pred[a_run][1] == y).astype(float), (pred[b_run][1] == y).astype(float)
    gaps = np.array([100 * (ca[s].mean() - cb[s].mean())
                     for s in (rng.choice(len(y), 1000, replace=False) for _ in range(10000))])
    print(f"\n10,000 random 1,000-image subsets, fs+bal gap {a_run} - {b_run} (points): mean {gaps.mean():+.2f}, "
          f"5-95% [{np.percentile(gaps, 5):+.1f}, {np.percentile(gaps, 95):+.1f}]\n"
          f"  P(gap > 0) {(gaps > 0).mean():.2f} | P(gap = 0) {(gaps == 0).mean():.2f} | P(gap < 0) {(gaps < 0).mean():.2f}"
          f" | P(gap >= +0.4) {(gaps >= 0.4 - 1e-9).mean():.2f} | P(gap <= -0.4) {(gaps <= -0.4 + 1e-9).mean():.2f}")
