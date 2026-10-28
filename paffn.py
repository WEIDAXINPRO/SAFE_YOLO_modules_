"""PAFFN: ASFF-2/CRDR then ASFF-3/CRDR, manuscript Figs. 1, 4, 5.

ASFF, scale alignment, CRDR, RFAConv, ConvEMA and EMA are all implemented
locally; no external custom layers. RFAConv uses receptive-field-position
attention, NOT a dilated branch bank. EMA includes coordinate and 3x3 branches
and their cross-spatial interactions. Only PyTorch is required; Python >= 3.8.
Structure sources: Liuchen1997/RFAConv, YOLOonMe/EMA-attention-module,
ruinmessi/ASFF. See README for URLs, inspected versions and licence caveats.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2 if p is None else p,
                              groups=g, bias=False)
        self.bn, self.act = nn.BatchNorm2d(c2), nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class EMA(nn.Module):
    """Efficient Multi-Scale Attention with full cross-spatial learning."""
    def __init__(self, channels, factor=32):
        super().__init__()
        if channels < 1 or factor < 1:
            raise ValueError("channels and factor must be positive")
        self.groups = math.gcd(channels, min(channels, factor))
        cg = channels // self.groups
        self.channels = channels
        self.gn = nn.GroupNorm(cg, cg)
        self.conv1x1 = nn.Conv2d(cg, cg, 1, bias=True)
        self.conv3x3 = nn.Conv2d(cg, cg, 3, padding=1, bias=True)

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError("EMA channel mismatch")
        if h * w == 1:
            raise ValueError("EMA requires spatial area > 1")
        z = x.reshape(b * self.groups, c // self.groups, h, w)
        zh = z.mean(dim=3, keepdim=True)
        zw = z.mean(dim=2, keepdim=True).transpose(2, 3)
        hw = self.conv1x1(torch.cat((zh, zw), dim=2))
        ah, aw = torch.split(hw, (h, w), dim=2)
        x1 = self.gn(z * ah.sigmoid() * aw.transpose(2, 3).sigmoid())
        x2 = self.conv3x3(z)
        q1 = x1.mean(dim=(2, 3)).unsqueeze(1).softmax(dim=-1)
        q2 = x2.mean(dim=(2, 3)).unsqueeze(1).softmax(dim=-1)
        response = (torch.bmm(q1, x2.flatten(2)) +
                    torch.bmm(q2, x1.flatten(2))).reshape(b * self.groups, 1, h, w)
        return (z * response.sigmoid()).reshape(b, c, h, w)


class RFAConv(nn.Module):
    """Group-convolution RFAConv checked against the authors' implementation.

    Generate C*k*k features/weights, softmax over k*k receptive positions,
    rearrange to H*k,W*k, then contract using stride-k convolution. ReLU
    matches the upstream RFAConv; surrounding CRDR Conv blocks use SiLU.
    """
    def __init__(self, c1, c2, kernel_size=3, stride=1):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0 or stride < 1:
            raise ValueError("Requires positive odd kernel and positive stride")
        self.kernel_size = kernel_size
        n = kernel_size ** 2
        self.get_weight = nn.Sequential(
            nn.AvgPool2d(kernel_size, stride, kernel_size // 2),
            nn.Conv2d(c1, c1 * n, 1, groups=c1, bias=False))
        self.generate_feature = nn.Sequential(
            nn.Conv2d(c1, c1 * n, kernel_size, stride, kernel_size // 2,
                      groups=c1, bias=False), nn.BatchNorm2d(c1 * n), nn.ReLU())
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size, kernel_size, bias=True),
            nn.BatchNorm2d(c2), nn.ReLU())

    def forward(self, x):
        b, c = x.shape[:2]
        k = self.kernel_size
        logits = self.get_weight(x)
        h, w = logits.shape[-2:]
        weights = logits.reshape(b, c, k * k, h, w).softmax(dim=2)
        features = self.generate_feature(x).reshape(b, c, k * k, h, w)
        local = (features * weights).reshape(b, c, k, k, h, w)
        local = local.permute(0, 1, 4, 2, 5, 3).reshape(b, c, h * k, w * k)
        return self.conv(local)


class ConvEMA(nn.Module):
    """Fig. 5 inset: Conv2d -> BN2d -> SiLU -> EMA, exactly one EMA."""
    def __init__(self, c1, c2, kernel_size=1, ema_groups=32):
        super().__init__()
        self.conv = Conv(c1, c2, kernel_size)
        self.ema = EMA(c2, factor=ema_groups)

    def forward(self, x):
        return self.ema(self.conv(x))


class CRDR(nn.Module):
    """Channel split -> two 1x1 Conv -> RFA stack on one -> Cat/ConvEMA.

    No residual or extra EMA is drawn in Fig. 5. The repeated RFAConv depth is
    not specified; depth=2 is an explicit default, not an experimental fact.
    """
    def __init__(self, channels, out_channels=None, depth=2,
                 kernel_size=3, ema_groups=32):
        super().__init__()
        if channels < 2 or channels % 2:
            raise ValueError("CRDR needs even input channels >= 2")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        out_channels = channels if out_channels is None else out_channels
        self.channels = channels
        c = channels // 2
        self.left, self.right = Conv(c, c, 1), Conv(c, c, 1)
        self.rfa = nn.Sequential(*[RFAConv(c, c, kernel_size) for _ in range(depth)])
        self.fuse = ConvEMA(channels, out_channels, 1, ema_groups)

    def forward(self, x):
        if x.shape[1] != self.channels:
            raise ValueError("CRDR channel mismatch")
        a, b = x.chunk(2, dim=1)
        return self.fuse(torch.cat((self.left(a), self.rfa(self.right(b))), dim=1))


class ScaleAlign(nn.Module):
    """Channel projection and learned downsampling / nearest upsampling."""
    def __init__(self, c1, c2, source_level, target_level):
        super().__init__()
        distance = target_level - source_level
        if distance > 0:
            self.align = nn.Sequential(*[
                Conv(c1 if i == 0 else c2, c2, 3, 2) for i in range(distance)])
        else:
            self.align = Conv(c1, c2, 1)

    def forward(self, x, size):
        y = self.align(x)
        return F.interpolate(y, size=size, mode="nearest") if y.shape[-2:] != size else y


class ASFF(nn.Module):
    """ASFF-2/3: aligned inputs and pixel-wise softmax across scales.

    Local 1x1 scale descriptors are concatenated before scale-logit prediction,
    following the ASFF design; weights are spatial maps, not global scalars.
    """
    def __init__(self, in_channels, target, out_channels,
                 levels=None, compress_channels=16):
        super().__init__()
        if len(in_channels) not in (2, 3) or not 0 <= target < len(in_channels):
            raise ValueError("ASFF requires two/three inputs and a valid target")
        self.in_channels, self.target = tuple(in_channels), target
        levels = tuple(range(len(in_channels))) if levels is None else tuple(levels)
        if len(levels) != len(in_channels) or compress_channels < 1:
            raise ValueError("Invalid levels or compression width")
        self.align = nn.ModuleList([
            ScaleAlign(c, out_channels, level, levels[target])
            for c, level in zip(in_channels, levels)])
        self.weight_features = nn.ModuleList([
            Conv(out_channels, compress_channels, 1) for _ in in_channels])
        self.weight_logits = nn.Conv2d(len(in_channels) * compress_channels,
                                       len(in_channels), 1, bias=True)
        self.expand = Conv(out_channels, out_channels, 3)

    def forward(self, xs, return_weights=False):
        if len(xs) != len(self.in_channels):
            raise ValueError("Incorrect number of ASFF inputs")
        size = xs[self.target].shape[-2:]
        features = [align(x, size) for align, x in zip(self.align, xs)]
        descriptor = torch.cat([layer(x) for layer, x in
                                zip(self.weight_features, features)], dim=1)
        weights = self.weight_logits(descriptor).softmax(dim=1)
        fused = sum(x * weights[:, i:i + 1] for i, x in enumerate(features))
        output = self.expand(fused)
        return (output, weights) if return_weights else output


class PAFFN(nn.Module):
    """Progressive parallel neck matching Fig. 1's complete connectivity.

    Stage 1: two parallel ASFF-2/CRDR nodes fuse middle and deep inputs.
    Stage 2: shallow input and both stage-1 outputs feed three ASFF-3/CRDR nodes.
    Input/output order is high-to-low resolution [P3,P4,P5]. No detector/head.
    """
    def __init__(self, in_channels=(256, 512, 1024), out_channels=None,
                 crdr_depth=2, rfa_kernel=3, ema_groups=32, compress_channels=16):
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError("Expected three input levels [P3,P4,P5]")
        if out_channels is None:
            out_channels = tuple(in_channels)
        elif isinstance(out_channels, int):
            out_channels = (out_channels,) * 3
        if len(out_channels) != 3 or any(c < 2 or c % 2 for c in out_channels):
            raise ValueError("Three positive even output widths required")
        self.in_channels, self.out_channels = tuple(in_channels), tuple(out_channels)
        self.lateral = nn.ModuleList([Conv(ci, co, 1) for ci, co in
                                      zip(in_channels, out_channels)])
        pair = self.out_channels[1:]
        self.stage1_asff = nn.ModuleList([
            ASFF(pair, i, pair[i], levels=(1, 2), compress_channels=compress_channels)
            for i in range(2)])
        self.stage1_crdr = nn.ModuleList([
            CRDR(c, depth=crdr_depth, kernel_size=rfa_kernel, ema_groups=ema_groups)
            for c in pair])
        self.stage2_asff = nn.ModuleList([
            ASFF(self.out_channels, i, self.out_channels[i], levels=(0, 1, 2),
                 compress_channels=compress_channels) for i in range(3)])
        self.stage2_crdr = nn.ModuleList([
            CRDR(c, depth=crdr_depth, kernel_size=rfa_kernel, ema_groups=ema_groups)
            for c in self.out_channels])

    def forward(self, xs):
        if len(xs) != 3:
            raise ValueError("Expected [P3,P4,P5]")
        if any(x.ndim != 4 or x.shape[1] != c for x, c in zip(xs, self.in_channels)):
            raise ValueError("NCHW inputs do not match in_channels")
        if any(xs[i].shape[0] != xs[0].shape[0] for i in range(1, 3)):
            raise ValueError("Feature levels must have equal batch size")
        if any(xs[i].shape[-2 + d] < xs[i + 1].shape[-2 + d]
               for i in range(2) for d in range(2)):
            raise ValueError("Inputs must be high-to-low resolution")
        p = [conv(x) for conv, x in zip(self.lateral, xs)]
        stage1 = [crdr(asff(p[1:])) for asff, crdr in
                  zip(self.stage1_asff, self.stage1_crdr)]
        stage2 = [p[0], stage1[0], stage1[1]]
        return [crdr(asff(stage2)) for asff, crdr in
                zip(self.stage2_asff, self.stage2_crdr)]


Conv_EMA = ConvEMA

if __name__ == "__main__":
    torch.manual_seed(0)
    m = PAFFN((64, 128, 256))
    xs = [torch.randn(2, 64, 32, 32, requires_grad=True),
          torch.randn(2, 128, 16, 16, requires_grad=True),
          torch.randn(2, 256, 8, 8, requires_grad=True)]
    ys = m(xs)
    sum(y.square().mean() for y in ys).backward()
    print("PAFFN:", [tuple(y.shape) for y in ys],
          "parameters:", sum(p.numel() for p in m.parameters()))
