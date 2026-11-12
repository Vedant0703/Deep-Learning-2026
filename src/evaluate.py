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
    # where a model fails: low resolution, scale, rotation, viewpoint, background and confusable classes (src.hard)
    python -m src.evaluate runs/h_fm800_div/final.pt --no-suite --hard-suite --confusion
"""
import argparse
import math

import numpy as np
import torch

from .data import CLASSES, ArrayDataset, build_val_loader, eval_transform, load_split, stratified_split, _loader
from .hard import HARD_SUITE, hard_suite_images
from .predict import balance_probs, em_prior_probs
from .train import TTA_CHOICES, TeacherEnsemble, evaluate, get_device, load_model, tta_probs

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


def hard_loaders(data_dir, val_size, seed, num_workers, ambiguous_ref, device):
    """Loaders for the hard suite on the held-out split, plus "ambiguous": the 10% of held-out images the reference
    model (a teacher trained without them) is least sure of, unmodified."""
    images, labels, _ = load_split(data_dir, "train")
    va = stratified_split(labels, val_size, seed)[1]
    sets = {name: (x, labels[va]) for name, x in hard_suite_images(images[va], labels[va], seed).items()}
    if ambiguous_ref:
        clean = _loader(ArrayDataset(images[va], labels[va], eval_transform()), 500, False, num_workers)
        conf = predict(load(ambiguous_ref, device)[0], clean, device, device.type == "cuda", "flip")[0].max(1).values
        keep = np.sort(conf.numpy().argsort()[:len(va) // 10])
        sets["ambiguous"] = (images[va][keep], labels[va][keep])
    return {name: _loader(ArrayDataset(x, y, eval_transform()), 500, False, num_workers) for name, (x, y) in sets.items()}


def top_confusions(probs, labels, k=3):
    """The k most frequent (true -> predicted) mistakes, as text."""
    pred = probs.argmax(1).numpy()
    y = labels.numpy()
    pairs = np.bincount(y[pred != y] * 10 + pred[pred != y], minlength=100)
    return ", ".join(f"{CLASSES[i // 10]}->{CLASSES[i % 10]} {pairs[i]}" for i in pairs.argsort()[::-1][:k] if pairs[i])


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
    p.add_argument("--tta", nargs="+", default=["flip"], choices=TTA_CHOICES)
    p.add_argument("--baseline", help="checkpoint to compare against with a paired McNemar test on the clean images")
    p.add_argument("--reference", help="checkpoint (usually the teacher) to measure top-1 agreement and KL against")
    p.add_argument("--split", default="val", choices=["val", "train"],
                   help="clean images to use: the 5k val split, or the 45k training side of it (no augmentation)")
    p.add_argument("--no-suite", action="store_true", help="skip the corruption suite")
    p.add_argument("--balance", action="store_true",
                   help="also report class-balanced decoding (src.predict --balance); the val split is class-balanced")
    p.add_argument("--balance-temps", type=float, nargs="+", default=[1.0],
                   help="temperatures for --balance, one row each (src.predict --balance-temp)")
    p.add_argument("--em-prior", action="store_true",
                   help="also report EM prior estimation (em_prior_probs), which does not assume balanced classes")
    p.add_argument("--hard-suite", action="store_true",
                   help="also report the hard-factor suite (src.hard) and the ambiguous 10%% of the val split")
    p.add_argument("--ambiguous-ref", default="runs/teacher_wrn300/final.pt",
                   help="model whose least-confident val images form the 'ambiguous' row ('' to skip)")
    p.add_argument("--confusion", action="store_true", help="with --hard-suite: the top confusions per row")
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args(argv)

    device = get_device()
    use_amp = device.type == "cuda"
    clean = build_val_loader(args.data_dir, args.val_size, args.seed, num_workers=args.num_workers, part=args.split)
    suite = {} if args.no_suite else {
        name: build_val_loader(args.data_dir, args.val_size, args.seed, ops, num_workers=args.num_workers)
        for name, ops in SUITE.items()}
    hard = hard_loaders(args.data_dir, args.val_size, args.seed, args.num_workers, args.ambiguous_ref, device) \
        if args.hard_suite else {}
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
            decoders = [(f"  + balanced T={t:g}", lambda q, t=t: balance_probs(q, temp=t))
                        for t in args.balance_temps] if args.balance else []
            if args.em_prior:
                decoders.append(("  + EM prior", em_prior_probs))
            if decoders:  # keep the suite's probabilities so they can be decoded several ways
                suite_out = [predict(model, loader, device, use_amp, tta) for loader in suite.values()]
                acc = [100 * (q.argmax(1) == y).float().mean().item() for q, y in suite_out]
            else:
                acc = [evaluate(model, loader, device, use_amp, tta)[1] for loader in suite.values()]
            line = f"{name[-34:]:<34} {tta:<10} {100 * right.float().mean():6.2f} "
            line += (f"{np.mean(acc):6.2f} {min(acc):6.2f} | " + " ".join(f"{a:8.2f}" for a in acc)) if acc else ""
            print(line, flush=True)
            for label, decode in decoders:
                dacc = lambda q, y: 100 * (decode(q.numpy()).argmax(1) == y.numpy()).mean()  # noqa: E731
                bacc = [dacc(q, y) for q, y in suite_out]
                line = f"{label:<34} {tta:<10} {dacc(probs, labels):6.2f} "
                line += (f"{np.mean(bacc):6.2f} {min(bacc):6.2f} | " + " ".join(f"{a:8.2f}" for a in bacc)) if bacc else ""
                print(line, flush=True)
            if hard:
                hard_out = [predict(model, loader, device, use_amp, tta) for loader in hard.values()]
                for label, decode in [("  hard suite", None)] + [(f"  hard suite{d[0][1:]}", d[1]) for d in decoders]:
                    acc = [100 * ((decode(q.numpy()) if decode else q.numpy()).argmax(1) == y.numpy()).mean()
                           for q, y in hard_out]
                    print(f"{label:<34} {tta:<10} {'':>6} {np.mean(acc):6.2f} {min(acc):6.2f} | "
                          + " ".join(f"{name}={a:.2f}" for name, a in zip(hard, acc)), flush=True)
                if args.confusion:
                    for name, (q, y) in zip(["clean", *hard], [(probs, labels), *hard_out]):
                        print(f"{'':<34} {tta:<10} {name}: {top_confusions(q, y)}")
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
