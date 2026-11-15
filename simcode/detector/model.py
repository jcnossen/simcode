"""
The detector network of pass 1 and pass 2 (one class, `Detector`; the pass-2 detector is the same network with illumination
inputs).

Fully convolutional U-Net (3 levels, residual blocks, GroupNorm) on a T-frame window (photons [B,T,H,W]) -> per pixel of the
native 108 nm grid: detection logit, sub-pixel offset (dx, dy) in [-1.5, 1.5] px relative to the pixel centre, log-sigma
of the offset. Input: a per-frame stem (2 conv layers, weights shared over the frames) -> T x stem_ch channels + the
normalised window mean, fused by the input conv. Normalisation sign(x) log1p(|x| / 30) on photons (layers.blocks.norm_photons);
the window mean uses the same scale.
illum = True (pass 2): the illumination pattern estimated after pass 1 as input (pattern.illumination.illum_maps, [B,2T,H,W]):
  the stem's first conv gets 2 extra input channels per frame (frame_t, cos_t, sin_t), so it can demodulate frame t with its
  own illumination, and the fusion conv gets the 2T illumination maps (cin_il). Trained from a pass-1 detector with these
  extra weights zero (detector.train), i.e. starting exactly at the pass-1 detector.
Candidate selection: detector/select.py.

Checkpoints ({'model': state dict, 'args': training arguments, 'step'}) are read by load(): pass 1 or pass 2 follows from
the state dict (the illumination weights cin_il), the channel widths from the args; every detector checkpoint since
2026-11-09 has the native-grid output (args 'up' = 1); since 2026-11-15 the args carry the tag arch = ARCH.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from simcode.layers.blocks import ResBlock, norm_photons

ARCH = 'detector'       # architecture tag of the checkpoints trained since 2026-11-15 (training args 'arch'; earlier ones carry none)


class Detector(nn.Module):
    def __init__(self, T=6, ch=(48, 96, 160), stem_ch=16, illum=False, dxy_range=1.5):
        super().__init__()
        self.T, self.illum, self.dxy_range = T, bool(illum), dxy_range
        c0, c1, c2 = ch
        self.stem = nn.Sequential(nn.Conv2d(1, stem_ch, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(stem_ch, stem_ch, 3, padding=1), nn.GELU())
        self.cin = nn.Conv2d(T * stem_ch + 1, c0, 3, padding=1)
        self.e0 = nn.Sequential(ResBlock(c0), ResBlock(c0))
        self.d1 = nn.Conv2d(c0, c1, 3, stride=2, padding=1)
        self.e1 = nn.Sequential(ResBlock(c1), ResBlock(c1))
        self.d2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.e2 = nn.Sequential(ResBlock(c2, 1), ResBlock(c2, 2))
        self.u1 = nn.Conv2d(c2, c1, 1)
        self.r1 = ResBlock(c1)
        self.u0 = nn.Conv2d(c1, c0, 1)
        self.r0 = ResBlock(c0)
        self.head = nn.Sequential(nn.Conv2d(c0, c0, 3, padding=1), nn.GELU(), nn.Conv2d(c0, 4, 1))
        nn.init.constant_(self.head[-1].bias, 0.0)
        with torch.no_grad():
            self.head[-1].bias[0] = -4.0
        if illum:      # pass 2: the pass-1 layers above (same construction order) + the illumination inputs
            self.stem[0] = nn.Conv2d(3, stem_ch, 3, padding=1)
            self.cin_il = nn.Conv2d(2 * T, c0, 3, padding=1, bias=False)
            nn.init.zeros_(self.cin_il.weight)

    def load_pass1(self, sd):
        """pass-1 detector state dict -> this pass-2 detector, the illumination weights zero (= the pass-1 detector)"""
        own = self.state_dict()
        for k, v in sd.items():
            if k == 'stem.0.weight':
                w = torch.zeros_like(own[k]); w[:, :1] = v; own[k] = w
            else:
                own[k] = v
        nn.init.zeros_(self.cin_il.weight)
        own['cin_il.weight'] = torch.zeros_like(own['cin_il.weight'])
        self.load_state_dict(own)

    def forward(self, frames, illum=None):
        """frames [B,T,H,W] photons, illum [B,2T,H,W] (pass 2 only; pattern.illumination.illum_maps) -> dict logit [B,H,W],
        dxy [B,2,H,W] (px, rel. to pixel centre), log_sig [B,H,W]"""
        if self.illum and illum is None: raise ValueError('a pass-2 detector needs the illumination maps')
        B, T, H, W = frames.shape
        ph, pw = (-H) % 4, (-W) % 4
        if ph or pw:
            frames = F.pad(frames, (0, pw, 0, ph), mode='replicate')
            if self.illum: illum = F.pad(illum, (0, pw, 0, ph), mode='replicate')
        m = norm_photons(frames.mean(1, keepdim=True))
        f = norm_photons(frames)
        Hp, Wp = f.shape[-2:]
        if self.illum:
            s = self.stem(torch.stack([f, illum[:, :T], illum[:, T:]], 2).reshape(B * T, 3, Hp, Wp)).reshape(B, -1, Hp, Wp)
            x = F.gelu(self.cin(torch.cat([s, m], 1)) + self.cin_il(illum))
        else:
            s = self.stem(f.reshape(B * T, 1, Hp, Wp)).reshape(B, -1, Hp, Wp)
            x = F.gelu(self.cin(torch.cat([s, m], 1)))
        s0 = self.e0(x)
        s1 = self.e1(self.d1(F.gelu(s0)))
        x = self.e2(self.d2(F.gelu(s1)))
        x = F.interpolate(self.u1(F.gelu(x)), scale_factor=2, mode='bilinear', align_corners=False) + s1
        x = self.r1(x)
        x = F.interpolate(self.u0(F.gelu(x)), scale_factor=2, mode='bilinear', align_corners=False) + s0
        x = self.r0(x)
        o = self.head(F.gelu(x))[..., :H, :W]
        return dict(logit=o[:, 0], dxy=torch.tanh(o[:, 1:3]) * self.dxy_range, log_sig=-5.0 + 6.0 * torch.sigmoid(o[:, 3]))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def conv_flops(m, H, W, T=6):
    """multiply-accumulates x 2 of all conv layers for one [1,T,H,W] window (hooks)."""
    tot = [0]
    def hook(mod, inp, out):
        tot[0] += 2 * out.numel() * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
    hs = [mm.register_forward_hook(hook) for mm in m.modules() if isinstance(mm, nn.Conv2d)]
    dev = next(m.parameters()).device
    with torch.no_grad():
        m(torch.zeros(1, T, H, W, device=dev), torch.zeros(1, 2 * T, H, W, device=dev) if m.illum else None)
    for h_ in hs: h_.remove()
    return tot[0]


def build(args=None, illum=False):
    """a new detector with the channel widths of the training arguments (defaults if absent)"""
    args = args or {}
    return Detector(6, tuple(args.get('ch', (48, 96, 160))), args.get('stem_ch', 16), illum=illum)


def is_pass2(sd):
    """True for the state dict of a pass-2 detector (illumination inputs)"""
    return 'cin_il.weight' in sd


def load(ck, dev='cuda'):
    """detector checkpoint (path; relative paths resolved in the models directory) -> (Detector in eval mode, args).
    net.illum tells whether it needs the illumination maps (pass 2)."""
    from simcode.config import resolve
    c = torch.load(resolve(ck), map_location=dev, weights_only=False)
    a = c.get('args') or {}
    if a.get('arch', ARCH) != ARCH or a.get('up', 1) != 1 or a.get('inp', 'stem') != 'stem':
        raise ValueError(f"{ck}: detector arch {a.get('arch')!r} / output grid up={a.get('up')} / input {a.get('inp')!r} is not supported (native grid, per-frame stem)")
    net = build(a, illum=is_pass2(c['model'])).to(dev); net.load_state_dict(c['model']); net.eval()
    return net, a
