"""Compare trained models on the held-out validation split, clean and under generic image corruptions.

The hidden evaluation images come from a shifted distribution: a model with 97% validation accuracy scored about
87.5% on them. Only a few submissions are possible, so candidate models are compared by accuracy on the clean
validation split and under a fixed suite of mild, generic corruptions of it. The suite uses CIFAR-10-C families
(Hendrycks & Dietterich, ICLR 2019) at generic levels; it is not fitted to the evaluation images.

Only meaningful for models trained with this same held-out split (--val-size 5000, --split-seed 0).

Examples:
    python -m src.evaluate runs/base_c/final.pt runs/kd_c/final.pt --tta flip
    python -m src.evaluate runs/teacher_wrn/final.pt+runs/teacher_s1/final.pt     an ensemble (averaged softmax)
    # paired McNemar test against a baseline, and agreement with the teacher
    python -m src.evaluate runs/r1_fm/final.pt --baseline runs/twin_strong/final.pt \
        --reference runs/teacher_wrn300/final.pt --tta none flip
    # how closely the student follows the teacher on the images it was distilled on (no corruption suite)
    python -m src.evaluate runs/twin_strong/final.pt --reference runs/teacher_wrn300/final.pt --split train --no-suite
"""
import argparse
import math

import numpy as np
import torch

from .data import build_val_loader
from .train import TeacherEnsemble, evaluate, get_device, load_model, tta_probs

SUITE = {
    "bright+": [("brightness", 1.2)],
    "bright-": [("brightness", 0.8)],
    "contr+": [("contrast", 1.2)],
    "contr-": [("contrast", 0.8)],
    "sat0.5": [("saturation", 0.5)],
    "sat1.5": [("saturation", 1.5)],
    "sharp": [("sharpness", 2.5)],
    "blur": [("blur", 0.6)],
    "noise.02": [("noise", 0.02)],
    "noise.04": [("noise", 0.04)],
    "jpeg50": [("jpeg", 50)],
}


def load(ckpt, device):
    """A model, or the averaged-softmax ensemble of several checkpoints joined with +."""
    paths = ckpt.split("+")
    model = load_model(paths[0], device) if len(paths) == 1 else \
        TeacherEnsemble([load_model(path, device) for path in paths])
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    name = ckpt if len(paths) == 1 else "+".join(p.split("/")[-2] for p in paths)
    return model, name


@torch.no_grad()
def predict(model, loader, device, use_amp, tta):
    """(softmax averaged over the TTA views, labels) for every image in the loader."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if device.type == "cuda":
            x = x.to(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            probs.append(tta_probs(model, x, tta).float().cpu())
        labels.append(y)
    return torch.cat(probs), torch.cat(labels)


def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value for b images only one model gets right and c only the other gets right."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpts", nargs="+", help="final.pt / best.pt files; join several with + to evaluate their ensemble")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0, help="split seed (the training runs' --split-seed)")
    p.add_argument("--tta", nargs="+", default=["flip"], choices=["none", "flip", "flip_shift"])
    p.add_argument("--baseline", help="checkpoint to compare against with a paired McNemar test on the clean images")
    p.add_argument("--reference", help="checkpoint (usually the teacher) to measure top-1 agreement and KL against")
    p.add_argument("--split", default="val", choices=["val", "train"],
                   help="clean images to use: the 5k val split, or the 45k training side of it (no augmentation)")
    p.add_argument("--no-suite", action="store_true", help="skip the corruption suite")
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args(argv)

    device = get_device()
    use_amp = device.type == "cuda"
    clean = build_val_loader(args.data_dir, args.val_size, args.seed, num_workers=args.num_workers, part=args.split)
    suite = {} if args.no_suite else {
        name: build_val_loader(args.data_dir, args.val_size, args.seed, ops, num_workers=args.num_workers)
        for name, ops in SUITE.items()}
    # Baseline and reference predictions are computed once per TTA mode and reused for every checkpoint.
    others = {}
    for role in ("baseline", "reference"):
        if getattr(args, role):
            model, name = load(getattr(args, role), device)
            others[role] = (name, {tta: predict(model, clean, device, use_amp, tta)[0] for tta in args.tta})
            del model

    print(f"{'model':<34} {'tta':<10} {args.split:>6} {'mean':>6} {'worst':>6} | "
          + " ".join(f"{name:>8}" for name in suite))
    for ckpt in args.ckpts:
        model, name = load(ckpt, device)
        for tta in args.tta:
            probs, labels = predict(model, clean, device, use_amp, tta)
            right = probs.argmax(1) == labels
            acc = [evaluate(model, loader, device, use_amp, tta)[1] for loader in suite.values()]
            line = f"{name[-34:]:<34} {tta:<10} {100 * right.float().mean():6.2f} "
            line += (f"{np.mean(acc):6.2f} {min(acc):6.2f} | " + " ".join(f"{a:8.2f}" for a in acc)) if acc else ""
            print(line, flush=True)
            if "baseline" in others:
                base_name, base = others["baseline"]
                base_right = base[tta].argmax(1) == labels
                b, c = int((right & ~base_right).sum()), int((~right & base_right).sum())
                print(f"{'':<34} {'':<10} vs {base_name}: only this right {b}, only baseline right {c}, "
                      f"McNemar p={mcnemar_p(b, c):.3f}")
            if "reference" in others:
                ref_name, ref = others["reference"]
                agree = (probs.argmax(1) == ref[tta].argmax(1)).float().mean()
                kl = (ref[tta] * (ref[tta].clamp_min(1e-12).log() - probs.clamp_min(1e-12).log())).sum(1).mean()
                print(f"{'':<34} {'':<10} vs {ref_name}: top-1 agreement {100 * agree:.2f}%, "
                      f"KL(reference || model) {kl:.4f}")


if __name__ == "__main__":
    main()
