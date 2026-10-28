"""Figure 2 and Eqs. (1)-(13), with explicitly declared missing settings.

AKConv: offset -> bilinear resampling -> reshape -> convolution -> DSC -> SiLU.
FCA: normalized DCT-II responses -> C-vector -> MLP -> channel reweighting.
C3k_SF: Bottleneck branch || AKConv/FCA branch, then concatenate/project.
C3k2_SF: outer C2f-style multi-output concatenation shown in Fig. 2.

Reference concepts: CV-ZhangXin/LDConv and cfzd/FcaNet. This is a local
reference implementation, not evidence of the original experimental settings.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .common import Conv, Bottleneck

FREQUENCIES = {
    "F1": ((0, 0), (0, 1), (1, 0)),
    "F2": ((0, 0), (0, 1), (1, 0), (2, 2), (3, 3)),
    "F3": ((0, 0), (0, 1), (1, 0), (2, 2), (3, 3), (6, 0), (0, 6), (6, 6)),
}


class AKConv(nn.Module):
    """N-point adaptive sampling; N=5 follows the C x 5H x W in Fig. 2.

    Coordinates are N row offsets followed by N column offsets. Initial
    points form a non-centred near-square lattice, as in LDConv. Border
    replication is an implementation assumption, absent from the manuscript.
    DSC is depthwise 3x3 + pointwise 1x1, with ONE final BN/SiLU. No optional
    hidden extra layers are added for matching the manuscript's cost figures.
    """
    def __init__(self, c1, c2, num_points=5, stride=1, learn_offsets=True):
        super().__init__()
        if min(c1, c2, num_points, stride) < 1:
            raise ValueError("Channels, sampling points and stride must be positive")
        self.c1, self.c2 = c1, c2
        self.num_points, self.stride = num_points, stride
        self.learn_offsets = learn_offsets
        if learn_offsets:
            self.offset = nn.Conv2d(c1, 2 * num_points, 3, stride, 1)
            nn.init.zeros_(self.offset.weight)
            nn.init.zeros_(self.offset.bias)
        else:
            self.offset = None  # Fixed-sampling control has no unused predictor.
        width = max(1, round(math.sqrt(num_points)))
        self.register_buffer("initial_points", torch.tensor(
            [(i // width, i % width) for i in range(num_points)], dtype=torch.float32))
        self.aggregate = nn.Conv2d(c1, c2, (num_points, 1), (num_points, 1), bias=False)
        self.depthwise = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)
        self.pointwise = nn.Conv2d(c2, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def coordinates(self, x):
        b, _, h, w = x.shape
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        if self.offset is None:
            ho, wo = (h - 1) // self.stride + 1, (w - 1) // self.stride + 1
            offsets = x.new_zeros((b, 2 * self.num_points, ho, wo), dtype=dtype)
        else:
            offsets = self.offset(x).to(dtype)
            ho, wo = offsets.shape[-2:]
        yy, xx = torch.meshgrid(torch.arange(ho, device=x.device, dtype=dtype),
                                torch.arange(wo, device=x.device, dtype=dtype), indexing="ij")
        points = self.initial_points.to(dtype)
        rows = yy[None, None] * self.stride + points[:, 0][None, :, None, None]
        cols = xx[None, None] * self.stride + points[:, 1][None, :, None, None]
        return rows + offsets[:, :self.num_points], cols + offsets[:, self.num_points:]

    def resample(self, x, rows, cols):
        b, c, h, w = x.shape
        ho, wo = rows.shape[-2:]
        rows, cols = rows.clamp(0, h - 1), cols.clamp(0, w - 1)
        gy = rows * (2.0 / (h - 1)) - 1 if h > 1 else rows * 0
        gx = cols * (2.0 / (w - 1)) - 1 if w > 1 else cols * 0
        grid = torch.stack((gx, gy), -1).permute(0, 2, 3, 1, 4)
        grid = grid.reshape(b, ho, wo * self.num_points, 2)
        sampled = F.grid_sample(x.to(grid.dtype), grid, mode="bilinear",
                                padding_mode="border", align_corners=True)
        return sampled.to(x.dtype).reshape(b, c, ho, wo, self.num_points)

    def forward(self, x):
        rows, cols = self.coordinates(x)
        y = self.resample(x, rows, cols)
        b, c, h, w, n = y.shape
        y = y.permute(0, 1, 2, 4, 3).reshape(b, c, h * n, w)
        y = self.aggregate(y)
        return self.act(self.bn(self.pointwise(self.depthwise(y))))


class FCA(nn.Module):
    """One frequency per channel group, with exactly C output descriptors.

    Balanced unequal group sizes support F1/F2 without changing channel widths.
    With C % G != 0, the first C % G groups get one extra channel. This is a
    declared completion of C_i in the manuscript, not a recovered ablation log.
    DCT uses the ACTUAL H,W, with no adaptive pooling or frequency rescaling.
    """
    def __init__(self, channels, reduction=16, frequencies="F3", group_sizes=None):
        super().__init__()
        if isinstance(frequencies, str):
            frequencies = FREQUENCIES[frequencies]
        self.frequencies = tuple(tuple(pair) for pair in frequencies)
        if not self.frequencies or channels < len(self.frequencies):
            raise ValueError("At least one channel is required for every frequency")
        if reduction < 1 or channels % reduction:
            raise ValueError("C/reduction must be a positive integer, as in Eq. (13)")
        if any(len(pair) != 2 or any(not isinstance(v, int) or v < 0 for v in pair)
               for pair in self.frequencies):
            raise ValueError("DCT indices must be nonnegative integer pairs")
        g = len(self.frequencies)
        q, r = divmod(channels, g)
        self.group_sizes = tuple(group_sizes) if group_sizes is not None else tuple(
            q + int(i < r) for i in range(g))
        if (len(self.group_sizes) != g or sum(self.group_sizes) != channels or
                any(not isinstance(c, int) or c < 1 for c in self.group_sizes)):
            raise ValueError("Positive integer group sizes must sum to C")
        self.channels, self.reduction = channels, reduction
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.register_buffer("_basis", torch.empty(0), persistent=False)

    def dct_basis(self, h, w, device, dtype):
        if any(u >= h or v >= w for u, v in self.frequencies):
            raise ValueError("DCT indices exceed input size; F3 requires H,W >= 7")
        if (tuple(self._basis.shape) != (self.channels, h, w) or
                self._basis.device != device or self._basis.dtype != dtype):
            yy = torch.arange(h, device=device, dtype=dtype) + 0.5
            xx = torch.arange(w, device=device, dtype=dtype) + 0.5
            basis = []
            for (u, v), count in zip(self.frequencies, self.group_sizes):
                a = math.sqrt((1 if u == 0 else 2) / h)
                b = math.sqrt((1 if v == 0 else 2) / w)
                phi = a * torch.cos(math.pi * u * yy / h)
                psi = b * torch.cos(math.pi * v * xx / w)
                basis.append((phi[:, None] * psi[None, :])[None].expand(count, h, w))
            self._basis = torch.cat(basis, 0)
        return self._basis

    def descriptor(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("FCA expects NCHW with the configured C")
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        basis = self.dct_basis(*x.shape[-2:], x.device, dtype)
        return (x.to(dtype) * basis[None]).sum((-2, -1)).to(x.dtype)

    def forward(self, x):
        z = self.descriptor(x)
        weights = self.fc2(F.relu(self.fc1(z))).sigmoid()
        return x * weights[:, :, None, None]


class C3k_SF(nn.Module):
    """Inner Fig. 2 unit. Eqs. (1)-(3) describe this unit's two branches."""
    def __init__(self, c1, c2, n=2, shortcut=True, expansion=0.5,
                 use_akconv=True, use_fca=True, frequencies="F3", reduction=16,
                 num_points=5, learn_offsets=True):
        super().__init__()
        hidden = int(c2 * expansion)
        if hidden < 1 or n < 1:
            raise ValueError("Hidden channels and repeat count must be positive")
        self.cv1 = Conv(c1, 2 * hidden, 1)
        self.main = nn.Sequential(*[Bottleneck(hidden, shortcut) for _ in range(n)])
        self.sampling = (AKConv(hidden, hidden, num_points, learn_offsets=learn_offsets)
                         if use_akconv else Conv(hidden, hidden, 3))
        self.frequency = (FCA(hidden, reduction, frequencies) if use_fca else nn.Identity())
        self.cv2 = Conv(2 * hidden, c2, 1)

    def forward(self, x):
        fa, fb = self.cv1(x).chunk(2, 1)
        fb = self.frequency(self.sampling(fb))
        return self.cv2(torch.cat((self.main(fa), fb), 1))


class C3k2_SF(nn.Module):
    """Outer Conv/Split -> sequential C3k_SF -> multi-output Cat/Conv.

    Fourth argument is expansion, NOT the stock C3k2 c3k flag. Use keywords
    or the supplied integration helper; do not drop into an unmodified parser.
    No silent width rounding is performed.
    """
    def __init__(self, c1, c2, n=1, expansion=0.25, inner_expansion=0.5,
                 inner_depth=2, shortcut=True, use_akconv=True, use_fca=True,
                 frequencies="F3", reduction=16, num_points=5, learn_offsets=True):
        super().__init__()
        self.hidden = int(c2 * expansion)
        if self.hidden < 1 or n < 1:
            raise ValueError("Hidden channels and repeat count must be positive")
        self.cv1 = Conv(c1, 2 * self.hidden, 1)
        self.blocks = nn.ModuleList([
            C3k_SF(self.hidden, self.hidden, inner_depth, shortcut, inner_expansion,
                   use_akconv, use_fca, frequencies, reduction, num_points, learn_offsets)
            for _ in range(n)])
        self.cv2 = Conv((2 + n) * self.hidden, c2, 1)

    def forward(self, x):
        parts = list(self.cv1(x).chunk(2, 1))
        for block in self.blocks:
            parts.append(block(parts[-1]))
        return self.cv2(torch.cat(parts, 1))
