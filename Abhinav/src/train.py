"""Train a ResNet on CIFAR-10.

Examples:
    # screening run on a stratified 45k/5k train/val split
    python -m src.train --model resnet_a --epochs 40 --out runs/screen_a
    # final run on all 50k training images (no held-out set)
    python -m src.train --model resnet_a --epochs 200 --val-size 0 --out runs/final_a
    # resume after a Colab disconnect
    python -m src.train --model resnet_a --epochs 200 --val-size 0 --out runs/final_a --resume
    # distillation: train a teacher from scratch, then a student that also matches its predictions
    python -m src.train --model wrn28_10 --epochs 200 --ema 0.999 --out runs/teacher
    python -m src.train --model resnet_a --epochs 300 --ema 0.999 --teacher runs/teacher/final.pt --out runs/kd_a
    # distillation from an ensemble: teachers that differ only in --seed share the 45k/5k split (--split-seed)
    python -m src.train --model wrn28_10 --epochs 200 --ema 0.999 --seed 1 --out runs/teacher_s1
    python -m src.train --model resnet_a --epochs 300 --ema 0.999 --out runs/kd_ens \
        --teacher runs/teacher/final.pt runs/teacher_s1/final.pt
    # function matching (Beyer et al., CVPR 2022): the teacher labels the same mixup/CutMix images the student sees
    python -m src.train --model resnet_c --epochs 300 --ema 0.999 --teacher runs/teacher/final.pt \
        --mix-alpha 1.0 --mix-prob 0.5 --out runs/fm_c
"""
import argparse
import contextlib
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2

from .data import CIFAR10_MEAN, CIFAR10_STD, build_loaders
from .hard import BATCH_FACTORS, hard_batch
from .models import CONFIGS, PREACT_CONFIGS, TEACHER_CONFIGS, build_model, count_params


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="resnet_a", choices=list(CONFIGS) + list(PREACT_CONFIGS) + list(TEACHER_CONFIGS))
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1, help="peak learning rate")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--wd", type=float, default=5e-4, help="weight decay (conv/linear weights only)")
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--aug", default="trivial", choices=["basic", "autoaugment", "trivial", "augmix"])
    p.add_argument("--erasing", type=float, default=0.25, help="RandomErasing probability (0 = off)")
    p.add_argument("--jitter", type=float, default=0.0,
                   help="brightness/contrast ColorJitter strength applied to every image, e.g. 0.2 (0 = off)")
    p.add_argument("--noise", type=float, default=0.0,
                   help="max std of Gaussian pixel noise added to half the images, e.g. 0.04 (0 = off)")
    p.add_argument("--mix-alpha", type=float, default=0.0, help="MixUp/CutMix alpha (0 = off)")
    p.add_argument("--mix-prob", type=float, default=0.5, help="fraction of batches that get MixUp/CutMix")
    p.add_argument("--mix-off-epochs", type=int, default=0, help="turn MixUp/CutMix off for the last N epochs")
    p.add_argument("--ema", type=float, default=0.0,
                   help="per-step EMA decay of the weights, e.g. 0.999; the EMA model is evaluated and saved (0 = off)")
    p.add_argument("--teacher", nargs="+", default=None,
                   help="final.pt of a trained teacher for knowledge distillation; several files = an ensemble whose "
                        "averaged temperature-softened prediction is the target")
    p.add_argument("--kd-alpha", type=float, default=0.9, help="weight of the distillation loss")
    p.add_argument("--kd-temp", type=float, default=4.0, help="softmax temperature for distillation")
    p.add_argument("--kd-loss", default="kl", choices=["kl", "dist"],
                   help="kl: Hinton et al. (2015); dist: DIST correlation loss (Huang et al., NeurIPS 2022)")
    p.add_argument("--sam-rho", type=float, default=0.0,
                   help="SAM neighbourhood size (Foret et al., ICLR 2021), e.g. 0.05 (0 = plain SGD)")
    p.add_argument("--student-noise", type=float, default=0.0,
                   help="max std of Gaussian pixel noise added on the GPU to half the batch, after the teacher has "
                        "seen it clean, so the student learns the teacher's clean prediction on noisy input (0 = off)")
    p.add_argument("--hard-aug", type=float, default=0.0,
                   help="fraction of each batch given a random hard factor (src.hard.hard_batch) on the GPU, after the "
                        "teacher has seen the clean image (0 = off)")
    p.add_argument("--hard-factors", nargs="+", default=list(BATCH_FACTORS), choices=BATCH_FACTORS)
    p.add_argument("--hard-label-weight", type=float, default=None,
                   help="weight of the true-label loss on hard-augmented images (the rest is distillation); "
                        "default: the same 1 - kd_alpha as other images")
    p.add_argument("--val-size", type=int, default=5000, help="held-out images (0 = train on all 50k)")
    p.add_argument("--extra-train", default=None,
                   help=".npz of extra labelled images made from our training images (scripts/make_hard.py)")
    p.add_argument("--extra-repeat", type=int, default=1, help="times the --extra-train images are repeated per epoch")
    p.add_argument("--init", default=None,
                   help="start from the weights of a final.pt from an earlier run of this code (a fine-tune)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", required=True, help="run directory for checkpoints and logs")
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--resume", action="store_true", help="continue from <out>/last.pt if present")
    p.add_argument("--wait-for", nargs="+", default=[],
                   help="files to wait for before starting, e.g. the final.pt of teachers that are still training")
    p.add_argument("--seed", type=int, default=0, help="initialization, data order and augmentation randomness")
    p.add_argument("--split-seed", type=int, default=0,
                   help="seed of the 45k/5k split; keep it at 0 so every run is validated on the same held-out images")
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


def load_model(ckpt_path, device):
    """Rebuild a model from a final.pt / best.pt weights file, in eval mode."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt["model_name"], **ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


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


def distillation_loss(student_logits, teacher_logits, temp):
    """Hinton et al. (2015): KL(teacher || student) on temperature-softened outputs, scaled by T^2."""
    return nn.functional.kl_div(nn.functional.log_softmax(student_logits.float() / temp, 1),
                                nn.functional.log_softmax(teacher_logits.float() / temp, 1),
                                reduction="batchmean", log_target=True) * temp * temp


def distillation_loss_per_image(student_logits, teacher_logits, temp):
    """distillation_loss for each image separately (its mean over the batch is distillation_loss)."""
    return nn.functional.kl_div(nn.functional.log_softmax(student_logits.float() / temp, 1),
                                nn.functional.log_softmax(teacher_logits.float() / temp, 1),
                                reduction="none", log_target=True).sum(1) * temp * temp


def pearson_rows(a, b, eps=1e-8):
    """Mean over rows of the Pearson correlation between corresponding rows of a and b."""
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)).mean()


def dist_loss(student_logits, teacher_logits, temp):
    """DIST (Huang et al., NeurIPS 2022): 1 - Pearson correlation of the softened predictions, across classes for each
    image (inter-class) and across the batch for each class (intra-class), scaled by T^2. It matches the teacher's
    relative preferences rather than its exact probabilities, which helps when the teacher is much stronger."""
    ps = (student_logits.float() / temp).softmax(1)
    pt = (teacher_logits.float() / temp).softmax(1)
    return temp * temp * ((1 - pearson_rows(ps, pt)) + (1 - pearson_rows(ps.t(), pt.t())))


@torch.no_grad()
def sam_ascend(params, rho):
    """SAM's ascent step: move the weights by rho * g / ||g|| and return the offsets, or None if ||g|| isn't finite.
    The gradients may be GradScaler-scaled; the direction doesn't depend on the scale."""
    grads = [p.grad for p in params if p.grad is not None]
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.float()) for g in grads]))
    if not torch.isfinite(norm) or norm == 0:
        return None
    eps = [None if p.grad is None else p.grad * (rho / norm) for p in params]
    for p, e in zip(params, eps):
        if e is not None:
            p.add_(e)
    return eps


@contextlib.contextmanager
def frozen_bn_stats(model):
    """BatchNorm still normalizes with batch statistics but leaves its running statistics unchanged."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    saved = [m.momentum for m in bns]
    for m in bns:
        m.momentum = 0.0
    try:
        yield
    finally:
        for m, momentum in zip(bns, saved):
            m.momentum = momentum


class TeacherEnsemble(nn.Module):
    """Frozen teacher(s). forward(x, temp) returns temp * log(mean_i softmax(z_i / temp)), the logits of the members'
    averaged temperature-softened prediction, so distillation_loss treats an ensemble like one teacher. With a single
    member this equals the member's logits up to a per-image constant, which softmax ignores."""

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x, temp=1.0):
        logp = torch.stack([nn.functional.log_softmax(m(x).float() / temp, 1) for m in self.members])
        return temp * (torch.logsumexp(logp, 0) - math.log(len(self.members)))


def add_pixel_noise(x, max_std, p=0.5):
    """Gaussian noise on a normalized batch: for a fraction p of images, std ~ U(0, max_std) in [0, 1] pixel units."""
    sigma = torch.rand(x.size(0), 1, 1, 1, device=x.device) * max_std
    sigma = sigma * (torch.rand_like(sigma) < p)
    pixel_std = torch.tensor(CIFAR10_STD, device=x.device).view(1, 3, 1, 1)
    return x + torch.randn_like(x) * sigma / pixel_std


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, use_amp, mixer, mix_prob,
                    ema=None, teacher=None, kd_alpha=0.0, kd_temp=1.0, student_noise=0.0,
                    kd_loss=distillation_loss, sam_rho=0.0, hard_aug=0.0, hard_factors=BATCH_FACTORS,
                    hard_label_weight=None):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
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
            if device.type == "cuda":  # CutMix may return a contiguous tensor
                x = x.to(memory_format=torch.channels_last)

        teacher_out = None
        if teacher is not None:
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                teacher_out = teacher(x, kd_temp)  # same augmented (and mixed) batch, before any student noise
        xs = add_pixel_noise(x, student_noise) if student_noise > 0 else x
        hard = None
        if hard_aug > 0:  # after the teacher, so it labels the clean image
            xs, hard = hard_batch(xs, hard_aug, hard_factors, CIFAR10_MEAN, CIFAR10_STD)
            if device.type == "cuda":
                xs = xs.contiguous(memory_format=torch.channels_last)
        per_image = teacher_out is not None and hard is not None and hard_label_weight is not None

        def forward_loss():
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                out = model(xs)
                if per_image:  # a larger true-label share on the hard-augmented images
                    ce = nn.functional.cross_entropy(out.float(), target, reduction="none",
                                                     label_smoothing=criterion.label_smoothing)
                    alpha = torch.where(hard, 1 - hard_label_weight, kd_alpha)
                    loss = ((1 - alpha) * ce
                            + alpha * distillation_loss_per_image(out, teacher_out, kd_temp)).mean()
                else:
                    loss = criterion(out, target)
                    if teacher_out is not None:
                        loss = (1 - kd_alpha) * loss + kd_alpha * kd_loss(out, teacher_out, kd_temp)
            return out, loss

        out, loss = forward_loss()
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if sam_rho > 0 and (eps := sam_ascend(params, sam_rho)) is not None:
            # SAM: take the gradient at the worst-case nearby weights, then step from the original weights.
            optimizer.zero_grad(set_to_none=True)
            with frozen_bn_stats(model):  # BN running statistics are updated once per step
                _, sam_loss = forward_loss()
            scaler.scale(sam_loss).backward()
            with torch.no_grad():
                for p, e in zip(params, eps):
                    if e is not None:
                        p.sub_(e)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update_parameters(getattr(model, "_orig_mod", model))  # unwrap torch.compile

        loss_sum += loss.detach() * len(y)
        correct += (out.argmax(1) == y).sum()
        n += len(y)
    return loss_sum.item() / n, 100.0 * correct.item() / n


TTA_SHIFTS = ((0, 0), (2, 0), (-2, 0), (0, 2), (0, -2))


def tta_probs(model, x, tta="flip"):
    """Softmax averaged over test-time views.

    "none": the image only; "flip" (or True): image + horizontal flip;
    "flip_shift": both of those, each also shifted by 2 pixels in four directions (10 views).
    """
    if tta is True:
        tta = "flip"
    views = [x]
    if tta in ("flip", "flip_shift"):
        views.append(torch.flip(x, dims=[3]))
    if tta == "flip_shift":
        padded = [nn.functional.pad(v, (2, 2, 2, 2), mode="reflect") for v in views]
        views = [p[:, :, 2 + dy:34 + dy, 2 + dx:34 + dx] for p in padded for dy, dx in TTA_SHIFTS]
    return torch.stack([model(v).float().softmax(1) for v in views]).mean(0)


@torch.no_grad()
def evaluate(model, loader, device, use_amp, tta="none"):
    """Returns (loss, accuracy %), averaging softmax over the views selected by `tta` (see tta_probs)."""
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if device.type == "cuda":
            x = x.to(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            probs = tta_probs(model, x, tta or "none")
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
    if missing := [f for f in args.wait_for if not os.path.exists(f)]:
        print(f"waiting for {', '.join(missing)}", flush=True)
        while any(not os.path.exists(f) for f in args.wait_for):  # final.pt is written atomically
            time.sleep(60)
    seed_everything(args.seed)
    device = get_device()
    use_amp = device.type == "cuda" and not args.no_amp
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader = build_loaders(
        args.data_dir, args.batch_size, args.val_size, args.aug, args.erasing,
        args.num_workers, args.split_seed, args.subset, args.jitter, args.noise, args.extra_train, args.extra_repeat)

    model_kwargs = {"dropout": args.dropout}
    raw_model = build_model(args.model, **model_kwargs).to(device)
    if device.type == "cuda":
        raw_model = raw_model.to(memory_format=torch.channels_last)
    if args.init:  # our own earlier run's weights (EMA), so the schedule continues it as a fine-tune
        init = torch.load(args.init, map_location=device)
        assert init["model_name"] == args.model, f"{args.init} is a {init['model_name']}, not a {args.model}"
        raw_model.load_state_dict(init["model"])
        print(f"Initialized from {args.init} (epoch {init.get('epoch')})")
    n_params = count_params(raw_model)
    print(f"Model {args.model}: {n_params:,} trainable parameters | device {device} | AMP {use_amp}")
    print(f"Train images: {len(train_loader.dataset):,} | "
          f"val images: {len(val_loader.dataset) if val_loader else 0:,}")

    model = torch.compile(raw_model) if (args.compile and device.type == "cuda") else raw_model
    # EMA of weights and BN statistics; when enabled it is the model that gets evaluated and saved.
    ema = None
    if args.ema > 0:
        ema = torch.optim.swa_utils.AveragedModel(
            raw_model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema), use_buffers=True)
    eval_model = ema.module if ema is not None else model

    teacher = None
    if args.teacher:
        teacher = TeacherEnsemble([load_model(path, device) for path in args.teacher])
        if device.type == "cuda":
            teacher = teacher.to(memory_format=torch.channels_last)
        _, teacher_acc = evaluate(teacher, val_loader, device, use_amp) if val_loader else (None, None)
        print(f"Teacher {' + '.join(args.teacher)}: {count_params(teacher):,} params"
              + (f" | val acc {teacher_acc:.2f}" if val_loader else ""))
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
        if ema is not None:
            ema.load_state_dict(ckpt["ema"])
        start_epoch, history, best_acc = ckpt["epoch"], ckpt["history"], ckpt["best_acc"]
        print(f"Resumed from {last_path} at epoch {start_epoch}")

    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({**vars(args), "n_params": n_params}, f, indent=2)

    def checkpoint_state(epoch):
        """Full training state for --resume."""
        return dict(model=raw_model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                    ema=ema.state_dict() if ema is not None else None, epoch=epoch,
                    history=history, best_acc=best_acc, model_name=args.model,
                    model_kwargs=model_kwargs, args=vars(args))

    def weights_state(epoch):
        """Weights of the evaluated model (the EMA model if enabled), as read by load_model."""
        weights = ema.module if ema is not None else raw_model  # not the torch.compile wrapper
        return dict(model=weights.state_dict(), model_name=args.model, model_kwargs=model_kwargs,
                    n_params=n_params, epoch=epoch)

    kd_loss = dist_loss if args.kd_loss == "dist" else distillation_loss
    assert args.hard_label_weight is None or args.kd_loss == "kl", "--hard-label-weight needs --kd-loss kl"
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        mix_prob = args.mix_prob if epoch < args.epochs - args.mix_off_epochs else 0.0
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler,
                                                scaler, device, use_amp, mixer, mix_prob,
                                                ema, teacher, args.kd_alpha, args.kd_temp, args.student_noise,
                                                kd_loss, args.sam_rho, args.hard_aug, args.hard_factors,
                                                args.hard_label_weight)
        val_loss, val_acc = evaluate(eval_model, val_loader, device, use_amp) if val_loader else (None, None)
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
            save_checkpoint(os.path.join(args.out, "best.pt"), **weights_state(epoch + 1))
        if (epoch + 1) % args.ckpt_every == 0 or epoch + 1 == args.epochs:
            save_checkpoint(last_path, **checkpoint_state(epoch + 1))

    save_checkpoint(os.path.join(args.out, "final.pt"), **weights_state(args.epochs))
    print(f"Trainable parameters: {n_params:,}")
    if val_loader:
        _, final_acc = evaluate(eval_model, val_loader, device, use_amp)
        _, final_tta = evaluate(eval_model, val_loader, device, use_amp, tta=True)
        print(f"Final val accuracy: {final_acc:.2f}% (flip TTA: {final_tta:.2f}%) | best: {best_acc:.2f}%")
    else:
        print(f"Trained on all {len(train_loader.dataset):,} images; final train accuracy {train_acc:.2f}%")


if __name__ == "__main__":
    main()
