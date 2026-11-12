"""Configurable ResNet for CIFAR-10 under a 5M trainable-parameter budget.

Design variables follow the project brief:
    N   number of residual layers (stages)       -> len(channels)
    B_i residual blocks in layer i               -> blocks[i]
    C_i channels in layer i                      -> channels[i]
    F_i conv kernel size                         -> kernel_size
    K_i skip-connection kernel size              -> skip_kernel_size
    P   average-pool kernel size (None = global) -> pool_size

Architecture adapted from kuangliu/pytorch-cifar (https://github.com/kuangliu/pytorch-cifar)
and He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.

WideResNet is the knowledge-distillation teacher. The teacher is trained from scratch on the same training images, as
the rules require. It is never submitted, so the parameter budget does not apply to it. With custom widths the same
pre-activation layout is also a student (PREACT_CONFIGS), and then the budget applies.
"""
import math

import torch.nn as nn
import torch.nn.functional as F

PARAM_BUDGET = 5_000_000


class SEModule(nn.Module):
    """Squeeze-and-Excitation (Hu et al., CVPR 2018): rescales each channel by a gate computed from the global
    average of all channels."""

    def __init__(self, ch, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch // reduction, 1)
        self.fc2 = nn.Conv2d(ch // reduction, ch, 1)

    def forward(self, x):
        return x * self.fc2(F.relu(self.fc1(F.adaptive_avg_pool2d(x, 1)))).sigmoid()


class BasicBlock(nn.Module):
    """Residual block computing ReLU(S(x) + F(x)), with F = conv-BN-ReLU-conv-BN (optionally followed by SE)."""

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, skip_kernel_size=1, se=False):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, stride, pad, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, 1, pad, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEModule(out_ch) if se else None
        self.relu = nn.ReLU(inplace=True)

        # Projection shortcut only when the shape changes; identity otherwise.
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, skip_kernel_size, stride, skip_kernel_size // 2, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se is not None:
            out = self.se(out)
        return self.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    """`strides[i]` is the stride of stage i's first block (default 1, 2, 2, ...: each later stage halves the
    resolution). A stride of 1 in a later stage keeps more resolution: more compute, same parameters."""

    def __init__(self, channels, blocks, kernel_size=3, skip_kernel_size=1, pool_size=None,
                 num_classes=10, dropout=0.0, strides=None, se=False):
        super().__init__()
        assert len(channels) == len(blocks), "channels and blocks must have one entry per stage"
        strides = tuple(strides) if strides else (1,) + (2,) * (len(channels) - 1)

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        stages, in_ch = [], channels[0]
        for i, (out_ch, n_blocks) in enumerate(zip(channels, blocks)):
            for j in range(n_blocks):
                stride = strides[i] if j == 0 else 1
                stages.append(BasicBlock(in_ch, out_ch, stride, kernel_size, skip_kernel_size, se))
                in_ch = out_ch
        self.layers = nn.Sequential(*stages)

        final_size = 32 // math.prod(strides)
        if pool_size is None:
            self.pool = nn.AdaptiveAvgPool2d(1)
            pooled = 1
        else:
            self.pool = nn.AvgPool2d(pool_size)
            pooled = final_size // pool_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels[-1] * pooled * pooled, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        # Zero-init the last BN in each block so every block starts as identity (Goyal et al., 2017).
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.zeros_(m.bn2.weight)

    def forward(self, x):
        out = self.layers(self.stem(x))
        out = self.pool(out).flatten(1)
        return self.fc(self.dropout(out))


class WideBasic(nn.Module):
    """Pre-activation wide residual block (Zagoruyko & Komodakis, "Wide Residual Networks", BMVC 2016)."""

    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.dropout = dropout
        self.shortcut = None if (stride == 1 and in_ch == out_ch) else nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        skip = x if self.shortcut is None else self.shortcut(out)
        out = F.relu(self.bn2(self.conv1(out)))
        out = F.dropout(out, self.dropout, self.training)
        return self.conv2(out) + skip


class WideResNet(nn.Module):
    """WRN-depth-widen for 32x32 inputs: three stages of (depth - 4) / 6 WideBasic blocks.

    `widths` and `stem` override the standard widths (16, 32, 64) * widen and the 16-channel stem.
    """

    def __init__(self, depth=28, widen=10, dropout=0.0, num_classes=10, widths=None, stem=16):
        super().__init__()
        n = (depth - 4) // 6
        widths = tuple(widths) if widths else (16 * widen, 32 * widen, 64 * widen)
        self.stem = nn.Conv2d(3, stem, 3, 1, 1, bias=False)
        blocks, in_ch = [], stem
        for i, width in enumerate(widths):
            for j in range(n):
                blocks.append(WideBasic(in_ch, width, 2 if (i > 0 and j == 0) else 1, dropout))
                in_ch = width
        self.layers = nn.Sequential(*blocks)
        self.bn = nn.BatchNorm2d(in_ch)
        self.fc = nn.Linear(in_ch, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        out = F.relu(self.bn(self.layers(self.stem(x))))
        return self.fc(F.adaptive_avg_pool2d(out, 1).flatten(1))


# Candidate architectures, all under the 5M budget.
CONFIGS = {
    # ResNet-18-like 4-stage layout, trimmed to one block in the deeper stages.
    "resnet_a": dict(channels=(64, 128, 256, 512), blocks=(2, 1, 1, 1)),
    # Wider, shallower 3-stage network.
    "resnet_b": dict(channels=(80, 160, 320), blocks=(2, 2, 2)),
    # Deeper 3-stage network at ResNet-18 widths.
    "resnet_c": dict(channels=(64, 128, 256), blocks=(3, 4, 3)),
    # resnet_c keeping 16x16 resolution in the last stage: same parameters, 1.34 vs 0.72 GMACs per image.
    "resnet_c_hr": dict(channels=(64, 128, 256), blocks=(3, 4, 3), strides=(1, 2, 1)),
    # resnet_c keeping 32x32 through stage 2 and 16x16 in stage 3: same parameters, 2.21 GMACs per image.
    "resnet_c_s112": dict(channels=(64, 128, 256), blocks=(3, 4, 3), strides=(1, 1, 2)),
    # resnet_c with Squeeze-and-Excitation in every block (+35,868 parameters).
    "resnet_c_se": dict(channels=(64, 128, 256), blocks=(3, 4, 3), se=True),
}


# Pre-activation students: the WideResNet layout with custom widths, under the budget.
PREACT_CONFIGS = {
    # Vedant's network from team runs A-C: stem 29, widths 59-118-236, 4 blocks per stage, 4,980,392 params.
    "preact_v": dict(depth=28, widths=(59, 118, 236), stem=29),
}

# Distillation teachers (exempt from the budget, never submitted).
TEACHER_CONFIGS = {
    "wrn28_10": dict(depth=28, widen=10),  # 36.5M params
    "wrn40_10": dict(depth=40, widen=10),  # 55.8M params; a deeper teacher for a more diverse ensemble
}


def build_model(name, **overrides):
    if name in TEACHER_CONFIGS:
        return WideResNet(**{**TEACHER_CONFIGS[name], **overrides})
    if name in PREACT_CONFIGS:
        model = WideResNet(**{**PREACT_CONFIGS[name], **overrides})
    else:
        model = ResNet(**{**CONFIGS[name], **overrides})
    n = count_params(model)
    assert n <= PARAM_BUDGET, f"{name} has {n:,} params, over the {PARAM_BUDGET:,} budget"
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name in [*CONFIGS, *PREACT_CONFIGS]:
        print(f"{name}: {count_params(build_model(name)):,} trainable params")
    for name in TEACHER_CONFIGS:
        print(f"{name} (teacher only): {count_params(build_model(name)):,} trainable params")
