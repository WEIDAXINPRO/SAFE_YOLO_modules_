"""Local PyTorch implementations for manuscript Fig. 2 and Eqs. (1)-(13).

Contains C3k2_SF, its inner C3k_SF, AKConv and normalized DCT-II FCA.
No Ultralytics/torchvision/einops dependency. Python >= 3.8.
See README.md for interpretation choices and upstream sources.
Sources: github.com/CV-ZhangXin/LDConv/model.py and
github.com/cfzd/FcaNet/model/layer.py; source attribution/versions in README.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

PAPER_FREQUENCIES = ((0, 0), (0, 1), (1, 0), (2, 2),
                     (3, 3), (6, 0), (0, 6), (6, 6))


class Conv(nn.Module):
    """Local Conv2d -> BN2d -> SiLU."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2 if p is None else p,
                              groups=g, bias=False)
        self.bn, self.act = nn.BatchNorm2d(c2), nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=1.0):
        super().__init__()
        hidden = max(1, int(c2 * e))
        self.cv1, self.cv2 = Conv(c1, hidden, 3), Conv(hidden, c2, 3)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class AKConv(nn.Module):
    """Arbitrary-N adaptive kernel convolution with explicit bilinear sampling.

    N=5 follows the C,5H,W tensor in Fig. 2. The initial lattice follows the
    near-square non-centred layout in the authors' LDConv/AKConv repository.
    Offset order is N row offsets then N column offsets. The sampled tensor
    is rearranged to B,C,H*N,W and fused using an (N,1) convolution + BN/SiLU.
    """
    def __init__(self, c1, c2, num_param=5, stride=1, bias=False):
        super().__init__()
        if not isinstance(num_param, int) or num_param < 1:
            raise ValueError("num_param must be a positive integer")
        if not isinstance(stride, int) or stride < 1:
            raise ValueError("stride must be a positive integer")
        self.num_param, self.stride = num_param, stride
        self.p_conv = nn.Conv2d(c1, 2 * num_param, 3, stride, 1)
        nn.init.zeros_(self.p_conv.weight)
        nn.init.zeros_(self.p_conv.bias)
        width = max(1, round(math.sqrt(num_param)))
        points = [(i // width, i % width) for i in range(num_param)]
        self.register_buffer("initial_points", torch.tensor(points, dtype=torch.float32))
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, (num_param, 1), (num_param, 1), bias=bias),
            nn.BatchNorm2d(c2), nn.SiLU())

    def sampling_coordinates(self, x):
        """Pixel row/column coordinates, each B,N,Hout,Wout."""
        offset = self.p_conv(x)
        ho, wo = offset.shape[-2:]
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        offset = offset.to(dtype)
        yy, xx = torch.meshgrid(
            torch.arange(ho, device=x.device, dtype=dtype),
            torch.arange(wo, device=x.device, dtype=dtype), indexing="ij")
        points = self.initial_points.to(dtype)
        rows = yy[None, None] * self.stride + points[:, 0][None, :, None, None]
        cols = xx[None, None] * self.stride + points[:, 1][None, :, None, None]
        return rows + offset[:, :self.num_param], cols + offset[:, self.num_param:]

    def sample(self, x, rows, cols):
        """Bilinear interpolation with border clamping, Eq. (4)."""
        b, c, h, w = x.shape
        ho, wo = rows.shape[-2:]
        rows, cols = rows.clamp(0, h - 1), cols.clamp(0, w - 1)
        gy = rows * (2.0 / (h - 1)) - 1 if h > 1 else rows * 0
        gx = cols * (2.0 / (w - 1)) - 1 if w > 1 else cols * 0
        grid = torch.stack((gx, gy), dim=-1).permute(0, 2, 3, 1, 4)
        grid = grid.reshape(b, ho, wo * self.num_param, 2)
        # One grid_sample: avoids repeating the complete input N times.
        sampled = F.grid_sample(x.to(grid.dtype), grid, mode="bilinear",
                                padding_mode="border", align_corners=True)
        return sampled.to(x.dtype).reshape(b, c, ho, wo, self.num_param)

    def forward(self, x):
        rows, cols = self.sampling_coordinates(x)
        sampled = self.sample(x, rows, cols)
        b, c, ho, wo, n = sampled.shape
        sampled = sampled.permute(0, 1, 2, 4, 3).reshape(b, c, ho * n, wo)
        return self.conv(sampled)


class FCA(nn.Module):
    """Grouped normalized DCT-II attention, Eqs. (5)-(13).

    One selected frequency per equal channel group; concatenation gives B,C,
    as in Fig. 2. By default DCT is evaluated at the actual resolution.
    Optional dct_size=(7,7) enables pooled FcaNet-style evaluation instead.
    """
    def __init__(self, channels, reduction=16,
                 frequencies=PAPER_FREQUENCIES, dct_size=None):
        super().__init__()
        self.channels, self.frequencies = channels, tuple(frequencies)
        if not self.frequencies or channels % len(self.frequencies):
            raise ValueError("FCA channels must be divisible by frequency count (default 8)")
        if reduction < 1:
            raise ValueError("reduction must be positive")
        self.dct_size = tuple(dct_size) if dct_size is not None else None
        self.register_buffer("_basis", torch.empty(0), persistent=False)
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(nn.Linear(channels, hidden, bias=False),
                                nn.ReLU(), nn.Linear(hidden, channels, bias=False),
                                nn.Sigmoid())

    def dct_basis(self, h, w, device, dtype):
        if any(u < 0 or v < 0 or u >= h or v >= w for u, v in self.frequencies):
            raise ValueError("Selected DCT frequencies require sufficient size (default H,W >= 7)")
        if (self._basis.shape != (self.channels, h, w) or
                self._basis.device != device or self._basis.dtype != dtype):
            xx = torch.arange(h, device=device, dtype=dtype) + 0.5
            yy = torch.arange(w, device=device, dtype=dtype) + 0.5
            filters = []
            for u, v in self.frequencies:
                au, av = math.sqrt((1 if u == 0 else 2) / h), math.sqrt((1 if v == 0 else 2) / w)
                phi_u = au * torch.cos(math.pi * u * xx / h)
                phi_v = av * torch.cos(math.pi * v * yy / w)
                filters.append(phi_u[:, None] * phi_v[None, :])
            self._basis = torch.stack(filters).repeat_interleave(
                self.channels // len(self.frequencies), dim=0)
        return self._basis

    def descriptor(self, x):
        if x.shape[1] != self.channels:
            raise ValueError("FCA input channel mismatch")
        y = F.adaptive_avg_pool2d(x, self.dct_size) if self.dct_size else x
        dtype = torch.float64 if y.dtype == torch.float64 else torch.float32
        basis = self.dct_basis(*y.shape[-2:], y.device, dtype)
        return (y.to(dtype) * basis[None]).sum(dim=(-2, -1)).to(x.dtype)

    def forward(self, x):
        weights = self.fc(self.descriptor(x)).view(x.shape[0], self.channels, 1, 1)
        return x * weights


def _hidden(channels, expansion):
    if expansion <= 0:
        raise ValueError("expansion must be positive")
    return max(8, int(math.ceil(channels * expansion / 8.0)) * 8)


class C3k_SF(nn.Module):
    """Inner Fig. 2: Conv/Split -> Bottleneck || AKConv->FCA -> Cat/Conv."""
    def __init__(self, c1, c2, n=2, shortcut=True, e=0.5,
                 num_param=5, reduction=16, dct_size=None):
        super().__init__()
        if n < 1:
            raise ValueError("n must be >= 1")
        self.c = _hidden(c2, e)
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.bottlenecks = nn.Sequential(*[
            Bottleneck(self.c, self.c, shortcut, e=1.0) for _ in range(n)])
        self.akconv = AKConv(self.c, self.c, num_param)
        self.fca = FCA(self.c, reduction, dct_size=dct_size)
        self.cv2 = Conv(2 * self.c, c2, 1)

    def forward(self, x):
        fa, fb = self.cv1(x).chunk(2, 1)
        f_ak = self.akconv(fb)                     # Eq. (1)
        f_fca = self.fca(f_ak)                     # Eq. (2)
        return self.cv2(torch.cat((self.bottlenecks(fa), f_fca), 1))  # Eq. (3)


class C3k2_SF(nn.Module):
    """Outer C3k2 multi-output concatenation, Fig. 2.

    Both split tensors and every successive C3k_SF output enter the final
    concatenation: (2+n)*hidden channels, not just two final branches.
    """
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, inner_n=2,
                 inner_e=0.5, num_param=5, reduction=16, dct_size=None):
        super().__init__()
        if n < 1:
            raise ValueError("n must be >= 1")
        self.c = _hidden(c2, e)
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.m = nn.ModuleList([
            C3k_SF(self.c, self.c, inner_n, shortcut, inner_e,
                   num_param, reduction, dct_size) for _ in range(n)])
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

    def forward(self, x):
        parts = list(self.cv1(x).chunk(2, 1))
        for block in self.m:
            parts.append(block(parts[-1]))
        return self.cv2(torch.cat(parts, dim=1))


C3k2SF = C3k2_SF
C3K2_SF = C3k2_SF
MultiSpectralFCA = FCA

if __name__ == "__main__":
    torch.manual_seed(0)
    m = C3k2_SF(64, 128)
    x = torch.randn(2, 64, 24, 24, requires_grad=True)
    y = m(x)
    y.square().mean().backward()
    print("C3k2_SF:", tuple(y.shape), "parameters:", sum(p.numel() for p in m.parameters()))
