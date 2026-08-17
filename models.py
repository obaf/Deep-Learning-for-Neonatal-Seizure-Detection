"""Model zoo for neonatal seizure detection. Input: (B, 18, 512) = 18 bipolar
channels x 16 s @ 32 Hz, per-channel z-scored. Output: single logit per window."""
import torch
import torch.nn as nn


class ShallowCNN(nn.Module):
    """O'Shea-style fully-convolutional baseline (small, fast)."""

    def __init__(self, n_ch=18, n_t=512):
        super().__init__()
        def block(cin, cout, k, s):
            return nn.Sequential(
                nn.Conv1d(cin, cout, k, s, padding=k // 2, bias=False),
                nn.BatchNorm1d(cout), nn.ReLU(inplace=True))
        self.f = nn.Sequential(
            block(n_ch, 32, 10, 2),   # 256
            nn.MaxPool1d(2),          # 128
            block(32, 48, 7, 1),
            nn.MaxPool1d(2),          # 64
            block(48, 64, 5, 1),
            nn.MaxPool1d(2),          # 32
            block(64, 96, 3, 1),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(96, 1))

    def forward(self, x):
        return self.head(self.f(x)).squeeze(-1)


class DWBlock(nn.Module):
    """Depthwise-separable residual block (ConvNeXt-ish, 1D)."""

    def __init__(self, c, expand=4, k=7, drop=0.0):
        super().__init__()
        h = c * expand
        self.dw = nn.Conv1d(c, c, k, groups=c, padding=k // 2, bias=False)
        self.norm = nn.GroupNorm(1, c)
        self.pw1 = nn.Conv1d(c, h, 1, bias=False)
        self.act = nn.GELU()
        self.pw2 = nn.Conv1d(h, c, 1, bias=False)
        self.gamma = nn.Parameter(torch.ones(1, c, 1))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        y = self.dw(x)
        y = self.norm(y)
        y = self.pw2(self.drop(self.act(self.pw1(y))))
        return x + self.gamma * y


class ResNet1D(nn.Module):
    """Main model: stem + stages of DWBlocks with stride-2 downsampling + GAP head.

    width: base channels; depths: blocks per stage (3 stages).
    """

    def __init__(self, n_ch=18, width=48, depths=(2, 3, 4), expand=4, k=7, drop=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_ch, width, 16, 4, padding=6, bias=False),  # 512 -> 128
            nn.GroupNorm(1, width), nn.GELU(),
        )
        stages, c = [], width
        for i, d in enumerate(depths):
            blocks = [DWBlock(c, expand, k, drop) for _ in range(d)]
            stages.append(nn.Sequential(*blocks))
            if i < len(depths) - 1:
                stages.append(nn.Sequential(
                    nn.Conv1d(c, c * 2, 1, stride=2, bias=False),  # downsample
                    nn.GroupNorm(1, c * 2), nn.GELU()))
                c *= 2
        self.stages = nn.Sequential(*stages)
        self.norm = nn.GroupNorm(1, c)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(drop), nn.Linear(c, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm(x)
        return self.head(x).squeeze(-1)


class CRNN(nn.Module):
    """ResNet trunk (down to T/16) + BiGRU + temporal attention pooling."""

    def __init__(self, n_ch=18, width=32, depths=(2, 2), hidden=48, drop=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_ch, width, 16, 4, padding=6, bias=False),
            nn.GroupNorm(1, width), nn.GELU())
        mods, c = [], width
        for i, d in enumerate(depths):
            mods += [DWBlock(c, 4, 7, drop) for _ in range(d)]
            if i < len(depths) - 1:
                mods += [nn.Sequential(nn.Conv1d(c, c * 2, 1, 2, bias=False),
                                       nn.GroupNorm(1, c * 2), nn.GELU())]
                c *= 2
        self.trunk = nn.Sequential(*mods)          # (B, c, 32)
        self.rnn = nn.GRU(c, hidden, num_layers=1, bidirectional=True, batch_first=True)
        self.attn = nn.Linear(2 * hidden, 1)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * hidden, 1))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.stem(x)
        x = self.trunk(x).transpose(1, 2)          # (B, T', c)
        x, _ = self.rnn(x)
        w = torch.softmax(self.attn(self.drop(x)), dim=1)   # (B, T', 1)
        x = (w * x).sum(dim=1)
        return self.head(x).squeeze(-1)


class MultiHeadResNet(nn.Module):
    """Shared ResNet trunk with one binary head per expert annotator (A, B, C).
    forward returns (B, 3) logits; recording probability = mean of head sigmoids."""

    def __init__(self, width=32, depths=(2, 2, 2)):
        super().__init__()
        base = ResNet1D(width=width, depths=depths)
        c = base.head[-1].in_features
        base.head = nn.Identity()
        self.base = base
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(0.2)
        self.heads = nn.Linear(c, 3)

    def forward(self, x):
        f = self.base(x)
        if f.ndim == 3:
            f = self.pool(f).squeeze(-1)
        return self.heads(self.drop(f))


def build(name):
    if name == "shallow":
        return ShallowCNN()
    if name == "resnet-s":
        return ResNet1D(width=32, depths=(2, 2, 2))
    if name == "resnet":
        return ResNet1D(width=48, depths=(2, 3, 4))
    if name == "resnet-l":
        return ResNet1D(width=64, depths=(3, 4, 6))
    if name == "crnn":
        return CRNN()
    if name == "chanindep":
        return ResNet1D(n_ch=1, width=48, depths=(2, 3, 4))
    if name == "multih":
        return MultiHeadResNet(width=32, depths=(2, 2, 2))
    raise ValueError(name)
