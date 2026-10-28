"""SICA exactly following Fig. 3 and Eqs. (14)-(18) in the latest manuscript."""
import torch
from torch import nn


class SICA(nn.Module):
    """F1 channel branch, F2 spatial branch; crossed SUM, projection, residual.

    Mc: Conv1D over the pooled channel sequence. Ms: 7x7 Conv over channel
    average/max of F2. Fused C/2 channels are projected to C with a bare 1x1
    Conv. No depthwise recoding, BN or post-residual activation is inserted.
    Conv1D kernel=3 and bias-free convolutions are declared assumptions.
    interaction=False is an explicit non-crossed control (not a training fact).
    """
    def __init__(self, channels, channel_kernel=3, spatial_kernel=7, interaction=True):
        super().__init__()
        if channels < 2 or channels % 2:
            raise ValueError("SICA requires positive even C")
        if min(channel_kernel, spatial_kernel) < 1 or channel_kernel % 2 == 0 or spatial_kernel % 2 == 0:
            raise ValueError("Attention kernels must be positive odd integers")
        self.channels, self.interaction = channels, interaction
        self.channel_conv = nn.Conv1d(1, 1, channel_kernel,
                                      padding=channel_kernel // 2, bias=False)
        self.spatial_conv = nn.Conv2d(2, 1, spatial_kernel,
                                      padding=spatial_kernel // 2, bias=False)
        self.project = nn.Conv2d(channels // 2, channels, 1, bias=False)

    def attention_maps(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("SICA expects NCHW with the configured C")
        f1, f2 = x.chunk(2, 1)
        sequence = f1.mean((2, 3)).unsqueeze(1)
        mc = self.channel_conv(sequence).sigmoid().transpose(1, 2).unsqueeze(-1)
        spatial = torch.cat((f2.mean(1, keepdim=True), f2.amax(1, keepdim=True)), 1)
        ms = self.spatial_conv(spatial).sigmoid()
        return f1, f2, mc, ms

    def forward(self, x):
        f1, f2, mc, ms = self.attention_maps(x)
        fused = f1 * ms + f2 * mc if self.interaction else f1 * mc + f2 * ms
        return x + self.project(fused)


