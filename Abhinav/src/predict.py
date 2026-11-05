"""Predict labels for the competition's evaluation images and write submission.csv.

Example:
    python -m src.predict --ckpt runs/final_a/final.pt --out submission.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import CLASSES, build_test_loader
from .models import PARAM_BUDGET, count_params
from .train import TTA_CHOICES, get_device, load_model, tta_probs


@torch.no_grad()
def predict_probs(model, loader, device, tta="flip"):
    """Softmax probabilities for every image, averaged over the test-time views selected by `tta`."""
    return torch.cat([tta_probs(model, x.to(device), tta).cpu() for x in loader]).numpy()


def _logsumexp(x, axis):
    m = x.max(axis, keepdims=True)
    return m + np.log(np.exp(x - m).sum(axis, keepdims=True))


def balance_probs(probs, iters=100, temp=1.0):
    """Sinkhorn-rescale the class columns so every class gets the same total probability mass (a class-prior
    adjustment that assumes a class-balanced test set). Uses only the unlabeled predictions; no labels, no training.

    temp < 1 sharpens the probabilities first (p ** (1 / temp)); as temp -> 0 this approaches the exact
    optimal-transport assignment, N / K predictions per class with the highest total log-probability."""
    logq = np.log(np.clip(probs.astype(np.float64), 1e-30, None)) / temp
    log_target = np.log(probs.shape[0] / probs.shape[1])
    for _ in range(max(iters, int(iters / temp))):  # sharper inputs need more iterations to converge
        logq += log_target - _logsumexp(logq, 0)
        logq -= _logsumexp(logq, 1)
    return np.exp(logq)


def em_prior_probs(probs, iters=100):
    """Saerens et al. (2002): EM estimate of the test-set class prior (the training prior is uniform), returning the
    posteriors reweighted by it. Unlike balance_probs it does not assume a balanced test set."""
    p = probs.astype(np.float64)
    prior = np.full(p.shape[1], 1 / p.shape[1])
    for _ in range(iters):
        q = p * prior
        q /= q.sum(1, keepdims=True)
        prior = q.mean(0)
    return q


def write_submission(ids, preds, out_path, sample_path):
    sub = pd.DataFrame({"id": ids, "label": preds.astype(int)})
    sample = pd.read_csv(sample_path)
    assert list(sub.columns) == list(sample.columns), f"columns {list(sub.columns)} != {list(sample.columns)}"
    assert (sub.id.to_numpy() == sample.id.to_numpy()).all(), "ids/order differ from sample_submission.csv"
    assert sub.label.between(0, 9).all()
    sub.to_csv(out_path, index=False)
    return sub


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--tta", default="flip", choices=TTA_CHOICES,
                   help="test-time views averaged per image (see src.train.tta_probs)")
    p.add_argument("--balance", action="store_true",
                   help="class-balanced decoding (balance_probs); approved by the professor on Nov 4")
    p.add_argument("--balance-temp", type=float, default=1.0,
                   help="temperature for --balance; small values approach exactly N/10 predictions per class")
    args = p.parse_args(argv)

    device = get_device()
    model = load_model(args.ckpt, device)
    # Distillation teachers can be loaded too; make sure one is never submitted.
    assert count_params(model) <= PARAM_BUDGET, f"{count_params(model):,} params is over the {PARAM_BUDGET:,} budget"
    loader, ids = build_test_loader(args.data_dir)
    probs = predict_probs(model, loader, device, tta=args.tta)
    preds = probs.argmax(1)
    if args.balance:
        balanced = balance_probs(probs, temp=args.balance_temp).argmax(1)
        print(f"Class-balanced decoding changed {(balanced != preds).sum()} of {len(preds)} predictions")
        preds = balanced
    sub = write_submission(ids, preds, args.out, Path(args.data_dir) / "sample_submission.csv")

    print(f"Model params: {count_params(model):,} | wrote {len(sub)} predictions to {args.out}")
    counts = np.bincount(sub.label, minlength=10)
    print("Predicted class counts:", ", ".join(f"{c}={n}" for c, n in zip(CLASSES, counts)))


if __name__ == "__main__":
    main()
