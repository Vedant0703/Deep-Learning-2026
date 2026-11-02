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
"""
import torch.nn as nn

PARAM_BUDGET = 5_000_000


class BasicBlock(nn.Module):
    """Residual block computing ReLU(S(x) + F(x)), with F = conv-BN-ReLU-conv-BN."""

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, skip_kernel_size=1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, stride, pad, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, 1, pad, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
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
        return self.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, channels, blocks, kernel_size=3, skip_kernel_size=1, pool_size=None,
                 num_classes=10, dropout=0.0):
        super().__init__()
        assert len(channels) == len(blocks), "channels and blocks must have one entry per stage"

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # First stage keeps 32x32 resolution; every later stage halves it.
        stages, in_ch = [], channels[0]
        for i, (out_ch, n_blocks) in enumerate(zip(channels, blocks)):
            for j in range(n_blocks):
                stride = 2 if (i > 0 and j == 0) else 1
                stages.append(BasicBlock(in_ch, out_ch, stride, kernel_size, skip_kernel_size))
                in_ch = out_ch
        self.layers = nn.Sequential(*stages)

        final_size = 32 // 2 ** (len(channels) - 1)
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


# Candidate architectures, all under the 5M budget.
CONFIGS = {
    # ResNet-18-like 4-stage layout, trimmed to one block in the deeper stages.
    "resnet_a": dict(channels=(64, 128, 256, 512), blocks=(2, 1, 1, 1)),
    # Wider, shallower 3-stage network.
    "resnet_b": dict(channels=(80, 160, 320), blocks=(2, 2, 2)),
    # Deeper 3-stage network at ResNet-18 widths.
    "resnet_c": dict(channels=(64, 128, 256), blocks=(3, 4, 3)),
}


def build_model(name, **overrides):
    cfg = {**CONFIGS[name], **overrides}
    model = ResNet(**cfg)
    n = count_params(model)
    assert n <= PARAM_BUDGET, f"{name} has {n:,} params, over the {PARAM_BUDGET:,} budget"
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name in CONFIGS:
        print(f"{name}: {count_params(build_model(name)):,} trainable params")
