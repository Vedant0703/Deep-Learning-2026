"""Model soup (Wortsman et al., ICML 2022): the uniform average of the weights of several fine-tunes of one model,
which keeps a single network (same size as each ingredient) while getting some of an ensemble's gain.

    python scripts/soup.py runs/h_fm600_haug/final.pt runs/s_twin_a/final.pt runs/s_twin_b/final.pt \
        --val-size 5000 --out runs/soup_twin/final.pt          # twin soup: BN recalibrated on the 45k side
    python scripts/soup.py <final ingredients> --val-size 0 --out runs/soup_final/final.pt

Every floating-point tensor of the ingredients' (EMA) weights is averaged; integer buffers come from the first
ingredient. The ingredients must share one architecture and, for averaging to work, one starting point (fine-tunes
of the same model). BatchNorm statistics are then recomputed for the averaged weights with forward passes over
training batches (the normal training transform, no mixing, no gradients), since the average of the ingredients'
running statistics does not match the averaged network. The result is a final.pt that src.train.load_model reads.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import build_loaders  # noqa: E402
from src.models import PARAM_BUDGET, build_model, count_params  # noqa: E402
from src.train import get_device  # noqa: E402


def average(ckpts):
    names = {c["model_name"] for c in ckpts}
    assert len(names) == 1, f"ingredients have different architectures: {names}"
    keys = list(ckpts[0]["model"])
    assert all(list(c["model"]) == keys for c in ckpts), "ingredients have different parameter names"
    avg = {}
    for k in keys:
        t = ckpts[0]["model"][k]
        if t.is_floating_point():
            avg[k] = (sum(c["model"][k].double() for c in ckpts) / len(ckpts)).to(t.dtype)
        else:
            avg[k] = t.clone()
    return avg


@torch.no_grad()
def recalibrate_bn(model, loader, device, batches):
    """Recompute every BatchNorm layer's running mean / var as the plain average over `batches` training batches."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative average instead of an exponential one
    model.train()
    for i, (x, _) in enumerate(loader):
        if i >= batches:
            break
        model(x.to(device))
    for bn in bns:
        bn.momentum = 0.1
    return model.eval()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpts", nargs="+", help="final.pt of each ingredient")
    p.add_argument("--out", required=True)
    p.add_argument("--val-size", type=int, required=True,
                   help="5000 for a twin soup (recalibrate on the 45k side, as its ingredients trained), 0 for a final")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--bn-batches", type=int, default=50, help="training batches of 256 for BN recalibration")
    p.add_argument("--no-recalibrate", action="store_true", help="keep the averaged running statistics")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    ckpts = [torch.load(c, map_location="cpu") for c in args.ckpts]
    state = average(ckpts)
    device = get_device()
    model = build_model(ckpts[0]["model_name"], **ckpts[0]["model_kwargs"])
    model.load_state_dict(state)
    assert count_params(model) <= PARAM_BUDGET
    model = model.to(device)
    if not args.no_recalibrate:
        loader, _ = build_loaders(args.data_dir, 256, args.val_size, num_workers=2, seed=0)
        model = recalibrate_bn(model, loader, device, args.bn_batches)
    soup = {**ckpts[0], "model": {k: v.cpu() for k, v in model.state_dict().items()},
            "soup_of": list(args.ckpts), "bn_recalibrated": not args.no_recalibrate}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(soup, args.out)
    print(f"soup of {len(ckpts)} ({ckpts[0]['model_name']}, {count_params(model):,} params) -> {args.out}"
          + (f", BN recalibrated on {args.bn_batches} training batches" if not args.no_recalibrate else ""))


if __name__ == "__main__":
    main()
