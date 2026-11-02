"""Train a ResNet on CIFAR-10.

Examples:
    # screening run on a stratified 45k/5k train/val split
    python -m src.train --model resnet_a --epochs 40 --out runs/screen_a
    # final run on all 50k training images (no held-out set)
    python -m src.train --model resnet_a --epochs 200 --val-size 0 --out runs/final_a
    # resume after a Colab disconnect
    python -m src.train --model resnet_a --epochs 200 --val-size 0 --out runs/final_a --resume
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2

from .data import build_loaders
from .models import CONFIGS, build_model, count_params


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="resnet_a", choices=list(CONFIGS))
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1, help="peak learning rate")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--wd", type=float, default=5e-4, help="weight decay (conv/linear weights only)")
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--aug", default="trivial", choices=["basic", "autoaugment", "trivial"])
    p.add_argument("--erasing", type=float, default=0.25, help="RandomErasing probability (0 = off)")
    p.add_argument("--mix-alpha", type=float, default=0.0, help="MixUp/CutMix alpha (0 = off)")
    p.add_argument("--mix-prob", type=float, default=0.5, help="fraction of batches that get MixUp/CutMix")
    p.add_argument("--val-size", type=int, default=5000, help="held-out images (0 = train on all 50k)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", required=True, help="run directory for checkpoints and logs")
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--resume", action="store_true", help="continue from <out>/last.pt if present")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 2))
    p.add_argument("--subset", type=int, default=None, help="use only N images (debugging)")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (CUDA only)")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args(argv)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_optimizer(model, args):
    # No weight decay on BatchNorm parameters and biases.
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    return torch.optim.SGD([{"params": decay, "weight_decay": args.wd},
                            {"params": no_decay, "weight_decay": 0.0}],
                           lr=args.lr, momentum=args.momentum, nesterov=True)


def make_scheduler(optimizer, steps_per_epoch, args):
    """Per-step schedule: linear warmup, then cosine decay to zero."""
    warmup = args.warmup_epochs * steps_per_epoch
    total = args.epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, use_amp, mixer, mix_prob):
    model.train()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if device.type == "cuda":
            x = x.to(memory_format=torch.channels_last)
        target = y
        if mixer is not None and random.random() < mix_prob:
            x, target = mixer(x, y)

        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            out = model(x)
            loss = criterion(out, target)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_sum += loss.detach() * len(y)
        correct += (out.argmax(1) == y).sum()
        n += len(y)
    return loss_sum.item() / n, 100.0 * correct.item() / n


@torch.no_grad()
def evaluate(model, loader, device, use_amp, tta=False):
    """Returns (loss, accuracy %). With tta=True, averages softmax over the image and its horizontal flip."""
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if device.type == "cuda":
            x = x.to(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            probs = model(x).float().softmax(1)
            if tta:
                probs = (probs + model(torch.flip(x, dims=[3])).float().softmax(1)) / 2
        loss_sum += nn.functional.nll_loss(probs.clamp_min(1e-12).log(), y, reduction="sum").item()
        correct += (probs.argmax(1) == y).sum().item()
        n += len(y)
    return loss_sum / n, 100.0 * correct / n


def save_checkpoint(path, **state):
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic, so a disconnect mid-save can't corrupt the last good checkpoint


def main(argv=None):
    args = get_args(argv)
    os.makedirs(args.out, exist_ok=True)
    seed_everything(args.seed)
    device = get_device()
    use_amp = device.type == "cuda" and not args.no_amp
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader = build_loaders(
        args.data_dir, args.batch_size, args.val_size, args.aug, args.erasing,
        args.num_workers, args.seed, args.subset)

    model_kwargs = {"dropout": args.dropout}
    raw_model = build_model(args.model, **model_kwargs).to(device)
    if device.type == "cuda":
        raw_model = raw_model.to(memory_format=torch.channels_last)
    n_params = count_params(raw_model)
    print(f"Model {args.model}: {n_params:,} trainable parameters | device {device} | AMP {use_amp}")
    print(f"Train images: {len(train_loader.dataset):,} | "
          f"val images: {len(val_loader.dataset) if val_loader else 0:,}")

    model = torch.compile(raw_model) if (args.compile and device.type == "cuda") else raw_model
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = make_optimizer(raw_model, args)
    scheduler = make_scheduler(optimizer, len(train_loader), args)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    mixer = None
    if args.mix_alpha > 0:
        mixer = v2.RandomChoice([v2.MixUp(alpha=args.mix_alpha, num_classes=10),
                                 v2.CutMix(alpha=args.mix_alpha, num_classes=10)])

    start_epoch, history, best_acc = 0, [], 0.0
    last_path = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, history, best_acc = ckpt["epoch"], ckpt["history"], ckpt["best_acc"]
        print(f"Resumed from {last_path} at epoch {start_epoch}")

    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({**vars(args), "n_params": n_params}, f, indent=2)

    def checkpoint_state(epoch):
        return dict(model=raw_model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), epoch=epoch,
                    history=history, best_acc=best_acc, model_name=args.model,
                    model_kwargs=model_kwargs, args=vars(args))

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler,
                                                scaler, device, use_amp, mixer, args.mix_prob)
        val_loss, val_acc = evaluate(model, val_loader, device, use_amp) if val_loader else (None, None)
        dt = time.time() - t0
        history.append(dict(epoch=epoch + 1, lr=lr, train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc, time=dt))
        val_msg = f" | val loss {val_loss:.4f} acc {val_acc:5.2f}" if val_loader else ""
        print(f"epoch {epoch + 1:3d}/{args.epochs} | lr {lr:.4f} | train loss {train_loss:.4f} "
              f"acc {train_acc:5.2f}{val_msg} | {dt:.1f}s", flush=True)

        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump(history, f, indent=1)
        if val_loader and val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(os.path.join(args.out, "best.pt"), **checkpoint_state(epoch + 1))
        if (epoch + 1) % args.ckpt_every == 0 or epoch + 1 == args.epochs:
            save_checkpoint(last_path, **checkpoint_state(epoch + 1))

    save_checkpoint(os.path.join(args.out, "final.pt"), model=raw_model.state_dict(),
                    model_name=args.model, model_kwargs=model_kwargs, n_params=n_params)
    print(f"Trainable parameters: {n_params:,}")
    if val_loader:
        _, final_acc = evaluate(model, val_loader, device, use_amp)
        _, final_tta = evaluate(model, val_loader, device, use_amp, tta=True)
        print(f"Final val accuracy: {final_acc:.2f}% (flip TTA: {final_tta:.2f}%) | best: {best_acc:.2f}%")
    else:
        print(f"Trained on all {len(train_loader.dataset):,} images; final train accuracy {train_acc:.2f}%")


if __name__ == "__main__":
    main()
