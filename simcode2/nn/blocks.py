"""Shared network building blocks (residual conv block, patch extraction)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch, dil=1):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=dil, dilation=dil)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n = nn.GroupNorm(8, ch)

    def forward(self, x):
        return x + self.c2(F.gelu(self.n(self.c1(F.gelu(x)))))


def extract_patches(x, b, iy, ix, P):
    """x [B,C,H,W] -> patches [N,C,P,P] centred on (iy,ix), zero padded."""
    h = P // 2
    xp = F.pad(x, (h, h, h, h))
    o = torch.arange(P, device=x.device)
    yy = (iy[:, None] + o[None])[:, :, None].expand(-1, P, P)
    xx = (ix[:, None] + o[None])[:, None, :].expand(-1, P, P)
    return xp[b[:, None, None], :, yy, xx].permute(0, 3, 1, 2)
