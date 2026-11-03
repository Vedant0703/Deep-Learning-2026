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
from .train import get_device, load_model, tta_probs


@torch.no_grad()
def predict_probs(model, loader, device, tta="flip"):
    """Softmax probabilities for every image, averaged over the test-time views selected by `tta`."""
    return torch.cat([tta_probs(model, x.to(device), tta).cpu() for x in loader]).numpy()


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
    p.add_argument("--tta", default="flip", choices=["none", "flip", "flip_shift"],
                   help="test-time views averaged per image (see src.train.tta_probs)")
    args = p.parse_args(argv)

    device = get_device()
    model = load_model(args.ckpt, device)
    # Distillation teachers can be loaded too; make sure one is never submitted.
    assert count_params(model) <= PARAM_BUDGET, f"{count_params(model):,} params is over the {PARAM_BUDGET:,} budget"
    loader, ids = build_test_loader(args.data_dir)
    probs = predict_probs(model, loader, device, tta=args.tta)
    sub = write_submission(ids, probs.argmax(1), args.out, Path(args.data_dir) / "sample_submission.csv")

    print(f"Model params: {count_params(model):,} | wrote {len(sub)} predictions to {args.out}")
    counts = np.bincount(sub.label, minlength=10)
    print("Predicted class counts:", ", ".join(f"{c}={n}" for c, n in zip(CLASSES, counts)))


if __name__ == "__main__":
    main()
