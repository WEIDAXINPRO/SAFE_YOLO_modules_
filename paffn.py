"""PAFFN reference graph from Figs. 1, 4 and 5.

ASFF/RFAConv/EMA concepts follow ruinmessi/ASFF, Liuchen1997/RFAConv and
YOLOonMe/EMA-attention-module. PAFFN/CRDR are attributed to manuscript Ref. 13,
not claimed as newly invented here. Exact widths/depths were not reported.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .common import Conv


class EMA(nn.Module):
    def __init__(self, channels, groups=32):
        super().__init__()
        if min(channels, groups) < 1:
            raise ValueError("Channels and groups must be positive")
        self.groups = math.gcd(channels, min(channels, groups))
        self.channels = channels
        c = channels // self.groups
        self.norm = nn.GroupNorm(c, c)
        self.coordinate = nn.Conv2d(c, c, 1)
        self.local = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.channels or h * w < 2:
            raise ValueError("EMA channel mismatch or degenerate spatial input")
        z = x.reshape(b * self.groups, c // self.groups, h, w)
        zh = z.mean(3, keepdim=True)
        zw = z.mean(2, keepdim=True).transpose(2, 3)
        ah, aw = self.coordinate(torch.cat((zh, zw), 2)).split((h, w), 2)
        a = self.norm(z * ah.sigmoid() * aw.transpose(2, 3).sigmoid())
        d = self.local(z)
        qa = a.mean((2, 3)).unsqueeze(1).softmax(-1)
        qd = d.mean((2, 3)).unsqueeze(1).softmax(-1)
        response = (torch.bmm(qa, d.flatten(2)) + torch.bmm(qd, a.flatten(2)))
        return (z * response.reshape(-1, 1, h, w).sigmoid()).reshape(b, c, h, w)


class RFAConv(nn.Module):
    """Receptive-position softmax attention, not a dilated-convolution bank."""
    def __init__(self, c1, c2, kernel_size=3, stride=1):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0 or stride < 1:
            raise ValueError("Positive odd kernel and positive stride required")
        self.kernel_size = kernel_size
        n = kernel_size ** 2
        self.weights = nn.Sequential(nn.AvgPool2d(kernel_size, stride, kernel_size // 2),
                                     nn.Conv2d(c1, c1 * n, 1, groups=c1, bias=False))
        self.features = nn.Sequential(
            nn.Conv2d(c1, c1 * n, kernel_size, stride, kernel_size // 2,
                      groups=c1, bias=False), nn.BatchNorm2d(c1 * n), nn.ReLU())
        self.aggregate = nn.Sequential(nn.Conv2d(c1, c2, kernel_size, kernel_size),
                                        nn.BatchNorm2d(c2), nn.ReLU())

    def forward(self, x):
        b, c = x.shape[:2]
        k = self.kernel_size
        logits = self.weights(x)
        h, w = logits.shape[-2:]
        weights = logits.reshape(b, c, k * k, h, w).softmax(2)
        features = self.features(x).reshape(b, c, k * k, h, w)
        local = (features * weights).reshape(b, c, k, k, h, w)
        local = local.permute(0, 1, 4, 2, 5, 3).reshape(b, c, h * k, w * k)
        return self.aggregate(local)


class ConvEMA(nn.Module):
    """Figure 5: Conv2d -> BN -> SiLU -> one EMA."""
    def __init__(self, c1, c2, groups=32):
        super().__init__()
        self.conv, self.ema = Conv(c1, c2, 1), EMA(c2, groups)

    def forward(self, x):
        return self.ema(self.conv(x))


class CRDR(nn.Module):
    """Split; two Conv branches; RFA stack on one; Cat -> ConvEMA.

    No unpictured residual shortcut. depth=2 is an exposed assumption based
    on the two RFAConv boxes in Fig. 5, whose ellipsis leaves depth unspecified.
    """
    def __init__(self, channels, depth=2, kernel_size=3, ema_groups=32):
        super().__init__()
        if channels < 2 or channels % 2 or depth < 1:
            raise ValueError("Even positive channels and positive depth required")
        half = channels // 2
        self.left, self.right = Conv(half, half), Conv(half, half)
        self.rfa = nn.Sequential(*[RFAConv(half, half, kernel_size) for _ in range(depth)])
        self.fuse = ConvEMA(channels, channels, ema_groups)

    def forward(self, x):
        a, b = x.chunk(2, 1)
        return self.fuse(torch.cat((self.left(a), self.rfa(self.right(b))), 1))


class ScaleAlign(nn.Module):
    def __init__(self, c1, c2, source, target):
        super().__init__()
        distance = target - source
        if distance > 0:
            self.path = nn.Sequential(*[
                Conv(c1 if i == 0 else c2, c2, 3, 2) for i in range(distance)])
        else:
            self.path = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()

    def forward(self, x, size):
        x = self.path(x)
        return F.interpolate(x, size=size, mode="nearest") if x.shape[-2:] != size else x


class ASFF(nn.Module):
    """Pixel-wise softmax across TWO/THREE aligned feature scales."""
    def __init__(self, in_channels, target, levels=None, compress_channels=16):
        super().__init__()
        if len(in_channels) not in (2, 3) or not 0 <= target < len(in_channels):
            raise ValueError("ASFF needs 2/3 scales and a valid target")
        levels = tuple(range(len(in_channels))) if levels is None else tuple(levels)
        if len(levels) != len(in_channels) or compress_channels < 1:
            raise ValueError("Invalid scale levels or compression width")
        self.target = target
        c = in_channels[target]
        self.align = nn.ModuleList([ScaleAlign(ci, c, level, levels[target])
                                    for ci, level in zip(in_channels, levels)])
        self.describe = nn.ModuleList([Conv(c, compress_channels) for _ in in_channels])
        self.logits = nn.Conv2d(compress_channels * len(in_channels), len(in_channels), 1)
        self.expand = Conv(c, c, 3)

    def forward(self, xs, return_weights=False):
        if len(xs) != len(self.align):
            raise ValueError("ASFF scale count mismatch")
        size = xs[self.target].shape[-2:]
        aligned = [layer(x, size) for layer, x in zip(self.align, xs)]
        descriptors = torch.cat([layer(x) for layer, x in zip(self.describe, aligned)], 1)
        weights = self.logits(descriptors).softmax(1)
        y = self.expand(sum(x * weights[:, i:i+1] for i, x in enumerate(aligned)))
        return (y, weights) if return_weights else y


class PAFFN(nn.Module):
    """[P3,P4,P5] -> two ASFF-2/CRDR -> three ASFF-3/CRDR.

    Explicit latent widths avoid accidentally using unscaled 1024-channel
    base YAML widths in a width=0.5 YOLO11s. These widths are assumptions,
    not inferred historical settings. Output adapters preserve Detect widths.
    """
    def __init__(self, in_channels=(256, 256, 512), widths=(64, 128, 256),
                 out_channels=(128, 256, 512), crdr_depth=2,
                 compress_channels=16, ema_groups=32, use_crdr=True):
        super().__init__()
        if any(len(v) != 3 for v in (in_channels, widths, out_channels)):
            raise ValueError("Three input, latent and output widths required")
        if any(c < 2 or c % 2 for c in widths):
            raise ValueError("Latent widths must be positive even integers")
        self.in_channels = tuple(in_channels)
        self.lateral = nn.ModuleList([Conv(ci, co) for ci, co in zip(in_channels, widths)])
        pair = widths[1:]
        self.asff2 = nn.ModuleList([ASFF(pair, i, (1, 2), compress_channels) for i in range(2)])
        self.crdr2 = nn.ModuleList([CRDR(c, crdr_depth, ema_groups=ema_groups)
                                   if use_crdr else nn.Identity() for c in pair])
        self.asff3 = nn.ModuleList([ASFF(widths, i, compress_channels=compress_channels)
                                   for i in range(3)])
        self.crdr3 = nn.ModuleList([CRDR(c, crdr_depth, ema_groups=ema_groups)
                                   if use_crdr else nn.Identity() for c in widths])
        self.output = nn.ModuleList([Conv(ci, co) if ci != co else nn.Identity()
                                    for ci, co in zip(widths, out_channels)])

    def forward(self, xs):
        if len(xs) != 3 or any(x.ndim != 4 or x.shape[1] != c
                               for x, c in zip(xs, self.in_channels)):
            raise ValueError("Expected three NCHW inputs matching in_channels")
        if any(x.shape[0] != xs[0].shape[0] for x in xs):
            raise ValueError("All scales must have the same batch size")
        if any(xs[i].shape[-2+d] < xs[i+1].shape[-2+d] for i in range(2) for d in range(2)):
            raise ValueError("Input order must be high-to-low resolution")
        p = [layer(x) for layer, x in zip(self.lateral, xs)]
        pair = [crdr(asff(p[1:])) for asff, crdr in zip(self.asff2, self.crdr2)]
        sources = [p[0], pair[0], pair[1]]
        return [out(crdr(asff(sources)))
                for asff, crdr, out in zip(self.asff3, self.crdr3, self.output)]

