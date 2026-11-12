"""What class-balanced decoding (Sinkhorn T=1) changes on the 5k held-out split, per twin: wrong->right and
right->wrong transitions by true class and prediction move, and the M/N disagreement after balancing (cat/dog block
vs the rest). Usage (from the project folder): sinkhorn_transitions.py RUN_A RUN_B (run names, flip_shift TTA)."""
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, ".")  # run from the project folder
from src.data import CLASSES, eval_transform, load_split, stratified_split
from src.predict import balance_probs
from src.train import get_device, load_model, tta_probs

CATDOG = [CLASSES.index("cat"), CLASSES.index("dog")]

if __name__ == "__main__":
    runs = sys.argv[1:]
    dev = get_device()
    images, labels, _ = load_split("data", "train")
    va = stratified_split(labels, 5000, 0)[1]
    t = eval_transform()
    x_all = torch.stack([t(img) for img in images[va]])
    y = labels[va]
    cd = np.isin(y, CATDOG)
    bal_pred = {}
    for r in runs:
        model = load_model(f"runs/{r}/final.pt", dev)
        probs = []
        with torch.no_grad():
            for i in range(0, len(x_all), 500):
                probs.append(tta_probs(model, x_all[i:i + 500].to(dev), "flip_shift").float().cpu())
        p = torch.cat(probs).numpy()
        fs, bal = p.argmax(1), balance_probs(p).argmax(1)
        bal_pred[r] = bal
        ok_fs, ok_bal, changed = fs == y, bal == y, fs != bal
        print(f"\n== {r}: fs {100 * ok_fs.mean():.2f} -> fs+bal {100 * ok_bal.mean():.2f} | balancing changed "
              f"{changed.sum()} predictions: wrong->right {(~ok_fs & ok_bal).sum()}, right->wrong "
              f"{(ok_fs & ~ok_bal).sum()}, wrong->wrong {(~ok_fs & ~ok_bal & changed).sum()}")
        print(f"   {'true class':<11}{'w->r':>6}{'r->w':>6}{'net':>6}{'#pred fs':>10}{'#pred bal':>10}")
        for c, name in enumerate(CLASSES):
            m = y == c
            wr, rw = (~ok_fs & ok_bal & m).sum(), (ok_fs & ~ok_bal & m).sum()
            print(f"   {name:<11}{wr:>6}{rw:>6}{wr - rw:>+6}{(fs == c).sum():>10}{(bal == c).sum():>10}")
        for tag, mask in (("wrong->right", ~ok_fs & ok_bal), ("right->wrong", ok_fs & ~ok_bal)):
            moves = Counter(f"{CLASSES[a]}->{CLASSES[b]}" for a, b in zip(fs[mask], bal[mask]))
            print(f"   {tag} moves (fs pred -> bal pred): {', '.join(f'{k} {v}' for k, v in moves.most_common(6))}")
    if len(runs) == 2:
        a, b = (bal_pred[r] for r in runs)
        dis = a != b
        print(f"\n== {runs[0]} vs {runs[1]} after balancing: they disagree on {dis.sum()} images "
              f"(true cat/dog {(dis & cd).sum()}, other {(dis & ~cd).sum()})")
        for name, pr in zip(runs, (a, b)):
            ok = pr == y
            print(f"   {name}: right on {(dis & ok).sum()} of them (cat/dog {(dis & ok & cd).sum()}, "
                  f"other {(dis & ok & ~cd).sum()})")
        both = (a != y) & (b != y)
        print(f"   both wrong {both.sum()}, of which true cat/dog {(both & cd).sum()}")
