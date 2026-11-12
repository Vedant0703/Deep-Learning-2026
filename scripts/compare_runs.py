"""Compare training curves of runs at the same fraction of their schedules, and flag problems in each run.

Runs of different lengths have different learning rates at the same epoch (cosine decay over the whole run), so
they are compared at e.g. 50% of training rather than at epoch 100.

    python scripts/compare_runs.py runs/base_c runs/twin_kdn          local runs/RUN/history.json files
    python scripts/compare_runs.py --remote twin_kdn twin_kdn500      download the histories from the Colab VM first

With --remote, each run's config.json and history.json are saved to runs/RUN/, so the latest copy survives a lost VM.
Notes on the logged numbers:
- train loss/acc are measured on augmented (and, with --student-noise, noisy) training batches. With distillation
  the loss is (1 - kd_alpha) * CE + kd_alpha * T^2 * KL, so it is not comparable to a run without distillation.
- val numbers come from the EMA model when --ema is set. The EMA lags the raw weights by about 1 / (1 - decay)
  steps, so early val accuracy is low.
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def load(run, remote):
    """(name, history, config) for a run name or directory."""
    path = Path(run) if Path(run).is_dir() else ROOT / "runs" / run
    if remote:
        from colab import REMOTE, read_remote
        path.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "history.json"):
            text = read_remote(f"{REMOTE}/runs/{path.name}/{name}")
            if text is None:
                sys.exit(f"no {name} for {path.name} on the VM")
            (path / name).write_text(text)
    hist = json.loads((path / "history.json").read_text())
    config = json.loads((path / "config.json").read_text()) if (path / "config.json").exists() else {}
    return path.name, hist, config


def at_fraction(hist, total, frac):
    """The history row at frac of the schedule, or None if the run hasn't got there (or the row is missing)."""
    epoch = round(frac * total)
    return next((h for h in hist if h["epoch"] == epoch), None)


def expected_lr(epoch, total, peak, warmup):
    """LR at the start of `epoch` (1-based) for linear warmup then cosine decay, as logged by src/train.py."""
    if epoch <= warmup:
        return None  # per-step warmup; the logged value depends on the step count, so it is not checked
    return 0.5 * peak * (1 + math.cos(math.pi * (epoch - 1 - warmup) / (total - warmup)))


def diagnose(name, hist, total, config):
    """Plain-language flags about the run so far."""
    notes = []
    last = hist[-1]
    times = [h["time"] for h in hist[-10:]]
    s_per_epoch = statistics.median(times)
    notes.append(f"epoch {last['epoch']}/{total} ({last['epoch'] / total:.0%}), {s_per_epoch:.1f}s/epoch, "
                 f"about {(total - last['epoch']) * s_per_epoch / 60:.0f} min left at this speed")

    bad = [h["epoch"] for h in hist if not math.isfinite(h["train_loss"])
           or (h.get("val_loss") is not None and not math.isfinite(h["val_loss"]))]
    if bad:
        notes.append(f"NON-FINITE LOSS at epochs {bad[:5]}")

    peak, warmup = config.get("lr", 0.1), config.get("warmup_epochs", 3)
    off = [(h["epoch"], h["lr"], exp) for h in hist
           if (exp := expected_lr(h["epoch"], total, peak, warmup)) is not None and abs(h["lr"] - exp) > 0.01 * peak]
    notes.append("lr follows warmup + cosine" if not off else
                 f"LR OFF SCHEDULE at {len(off)} epochs, first: epoch {off[0][0]} lr {off[0][1]:.4f} "
                 f"vs {off[0][2]:.4f} expected")

    spikes = [h["epoch"] for prev, h in zip(hist, hist[1:])
              if h["epoch"] > warmup + 2 and h["train_loss"] > 1.05 * prev["train_loss"]]
    if spikes:
        notes.append(f"train loss jumped >5% epoch-to-epoch at {len(spikes)} epochs (latest {spikes[-3:]})")

    if len(hist) >= 11:
        d_loss = (hist[-1]["train_loss"] - hist[-11]["train_loss"]) / 10
        notes.append(f"train loss {last['train_loss']:.3f}, {d_loss:+.4f}/epoch over the last 10 epochs; "
                     f"train acc (augmented) {last['train_acc']:.2f}")

    if last.get("val_acc") is not None:
        best = max(hist, key=lambda h: h["val_acc"])
        low = min(hist, key=lambda h: h["val_loss"])
        notes.append(f"val acc {last['val_acc']:.2f} (best {best['val_acc']:.2f} at epoch {best['epoch']}), "
                     f"val loss {last['val_loss']:.4f} (lowest {low['val_loss']:.4f} at epoch {low['epoch']})")
        if last["val_loss"] > low["val_loss"] * 1.03 and last["epoch"] - low["epoch"] >= 10:
            notes.append("VAL LOSS RISING: up >3% from its minimum for 10+ epochs (overconfidence or overfitting)")
        if len(hist) >= 2 * max(1, total // 10):
            k = total // 10
            notes.append(f"val acc change over the last 10% of the schedule: "
                         f"{last['val_acc'] - hist[-1 - k]['val_acc']:+.2f}")
        notes.append(f"val acc - train acc (augmented) = {last['val_acc'] - last['train_acc']:+.2f} "
                     "(positive: the augmented training data is still harder than val, i.e. not overfitting)")
    return notes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run names (runs/RUN) or run directories")
    p.add_argument("--remote", action="store_true", help="download history.json from the Colab VM first")
    args = p.parse_args()

    runs = [load(r, args.remote) for r in args.runs]
    print(f"{'run':<14} {'frac':>5} {'epoch':>5} {'lr':>7} {'tr loss':>8} {'tr acc':>7} {'val loss':>8} {'val acc':>7}")
    for name, hist, config in runs:
        total = config.get("epochs", len(hist))
        for frac in FRACTIONS:
            h = at_fraction(hist, total, frac)
            if h is None:
                continue
            val =(f"{h['val_loss']:8.4f} {h['val_acc']:7.2f}" if h.get("val_acc") is not None
                   else f"{'-':>8} {'-':>7}")
            print(f"{name:<14} {frac:5.0%} {h['epoch']:5d} {h['lr']:7.4f} {h['train_loss']:8.3f} "
                  f"{h['train_acc']:7.2f} {val}")
        print()

    for name, hist, config in runs:
        print(f"== {name}")
        for note in diagnose(name, hist, config.get("epochs", len(hist)), config):
            print(f"   - {note}")

    # Matched-fraction comparison of val accuracy against the first run with val numbers.
    with_val = [(n, h, c.get("epochs", len(h))) for n, h, c in runs if h[-1].get("val_acc") is not None]
    if len(with_val) > 1:
        ref_name, ref_hist, ref_total = with_val[0]
        print(f"\nval acc relative to {ref_name} at the same fraction of the schedule:")
        for name, hist, total in with_val[1:]:
            cells = []
            for frac in FRACTIONS:
                a, b = at_fraction(hist, total, frac), at_fraction(ref_hist, ref_total, frac)
                if a is None or b is None:
                    continue
                cells.append(f"{frac:.0%} {a['val_acc'] - b['val_acc']:+.2f}")
            print(f"   {name:<14} " + ("  ".join(cells) if cells else "(no common fraction yet)"))


if __name__ == "__main__":
    main()
