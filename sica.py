"""SICA, manuscript Fig. 3 and Eqs. (14)-(18); only PyTorch required.

Explicit switches expose the document's Conv1D/Conv1x1 and depthwise-recoding
ambiguities. Defaults follow Eq. (14) plus the text-described depthwise layer.
"""
import torch
from torch import nn


class SICA(nn.Module):
    """Split -> crossed modulation -> SUM -> recoding -> exact residual.

    channels: even input/output width.
    channel_mode='conv2d': single 1x1 Conv, Eq. (14).
    channel_mode='conv1d': local channel convolution, Fig. 3 label.
    use_depthwise=True: text-described depthwise spatial recoding then Eq. (17).
    use_depthwise=False: only the 1x1 projection, literal figure/equation path.
    """
    def __init__(self, channels, spatial_kernel=7, channel_mode="conv2d",
                 channel_kernel=3, use_depthwise=True):
        super().__init__()
        if channels < 2 or channels % 2:
            raise ValueError("SICA requires even channels >= 2")
        if spatial_kernel < 1 or spatial_kernel % 2 == 0:
            raise ValueError("spatial_kernel must be positive and odd")
        self.channels, self.half = channels, channels // 2
        self.channel_mode, self.use_depthwise = channel_mode, use_depthwise
        if channel_mode == "conv2d":
            self.channel_conv = nn.Conv2d(self.half, self.half, 1, bias=True)
        elif channel_mode == "conv1d":
            if channel_kernel < 1 or channel_kernel % 2 == 0:
                raise ValueError("channel_kernel must be positive and odd")
            self.channel_conv = nn.Conv1d(1, 1, channel_kernel,
                                          padding=channel_kernel // 2, bias=False)
        else:
            raise ValueError("channel_mode must be 'conv2d' or 'conv1d'")
        self.spatial_conv = nn.Conv2d(2, 1, spatial_kernel,
                                      padding=spatial_kernel // 2, bias=True)
        self.depthwise = (nn.Sequential(
            nn.Conv2d(self.half, self.half, 3, padding=1, groups=self.half, bias=False),
            nn.BatchNorm2d(self.half), nn.SiLU()) if use_depthwise else nn.Identity())
        # F_fuse has C/2 channels, projected back to C for the residual.
        self.project = nn.Conv2d(self.half, channels, 1, bias=True)

    def attention_maps(self, x):
        """F1=channel branch, F2=spatial branch; returns F1,F2,Mc,Ms."""
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("Expected NCHW input with configured channels")
        f1, f2 = x.chunk(2, dim=1)
        pooled = f1.mean(dim=(2, 3), keepdim=True)
        if self.channel_mode == "conv2d":
            mc = torch.sigmoid(self.channel_conv(pooled))
        else:
            mc = self.channel_conv(pooled.flatten(2).transpose(1, 2))
            mc = torch.sigmoid(mc.transpose(1, 2).unsqueeze(-1))
        descriptor = torch.cat((f2.mean(dim=1, keepdim=True),
                                f2.amax(dim=1, keepdim=True)), dim=1)
        ms = torch.sigmoid(self.spatial_conv(descriptor))
        return f1, f2, mc, ms

    def forward(self, x):
        f1, f2, mc, ms = self.attention_maps(x)
        fused = f1 * ms + f2 * mc                   # Eq. (16), C/2, not Cat
        f3 = self.project(self.depthwise(fused))    # text + Eq. (17)
        return x + f3                              # Eq. (18), no extra act


if __name__ == "__main__":
    torch.manual_seed(0)
    m = SICA(128)
    x = torch.randn(2, 128, 24, 24, requires_grad=True)
    y = m(x)
    y.square().mean().backward()
    print("SICA:", tuple(y.shape), "parameters:", sum(p.numel() for p in m.parameters()))
