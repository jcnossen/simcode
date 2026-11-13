"""
Pattern-free dense detector for modulated T-frame windows (detector input/output ablation, critic review 22 #5).

Fully convolutional U-Net (3 levels, residual blocks, GroupNorm) -> per pixel: detection logit, sub-pixel offset
(dx, dy) in [-1.5, 1.5] px relative to the pixel centre, log-sigma of the offset. Input variants (no illumination
maps / pattern information at all):
  sum        : normalised window mean (1 ch)
  frames     : the T normalised frames as channels
  frames_sum : T frames + window mean
  stem       : a shared per-frame stem (2 conv layers, weights shared over frames) -> T x stem_ch channels,
               + the window mean, fused by the input conv
Normalisation: sign(x) log1p(|x| / 30) on photons (as the production network); the window mean (= sum / T) uses the
same scale, so a pixel of the sum image is normalised like a frame of the same mean level.

Candidate selection (select_subpixel): every pixel with p > p_min proposes the point pixel + offset; greedy NMS on
those POINTS with a sub-pixel radius (default 0.7 px; computed with shifted maps over a (2 win + 1)^2 neighbourhood,
iterated to the greedy fixed point), so two candidates can come from neighbouring pixels; then top-k (static shapes).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from simcode.nn.blocks import ResBlock

INPUTS = ('sum', 'frames', 'frames_sum', 'stem')


def norm_photons(x, scale=30.0):
    return torch.sign(x) * torch.log1p(x.abs() / scale)


class DetNet(nn.Module):
    def __init__(self, T=6, inp='frames_sum', ch=(48, 96, 160), stem_ch=16, dxy_range=1.5):
        super().__init__()
        assert inp in INPUTS
        self.T, self.inp, self.dxy_range = T, inp, dxy_range
        c0, c1, c2 = ch
        if inp == 'stem':
            self.stem = nn.Sequential(nn.Conv2d(1, stem_ch, 3, padding=1), nn.GELU(),
                                      nn.Conv2d(stem_ch, stem_ch, 3, padding=1), nn.GELU())
            cin = T * stem_ch + 1
        else:
            cin = {'sum': 1, 'frames': T, 'frames_sum': T + 1}[inp]
        self.cin = nn.Conv2d(cin, c0, 3, padding=1)
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

    def make_input(self, frames):
        """frames: photons [B,T,H,W] -> network input"""
        m = norm_photons(frames.mean(1, keepdim=True))
        if self.inp == 'sum':
            return m
        f = norm_photons(frames)
        if self.inp == 'frames':
            return f
        if self.inp == 'frames_sum':
            return torch.cat([f, m], 1)
        B, T, H, W = f.shape
        s = self.stem(f.reshape(B * T, 1, H, W)).reshape(B, -1, H, W)
        return torch.cat([s, m], 1)

    def forward(self, frames):
        """frames [B,T,H,W] photons -> dict logit [B,H,W], dxy [B,2,H,W] (px, rel. to pixel centre), log_sig [B,H,W]"""
        B, T, H, W = frames.shape
        ph, pw = (-H) % 4, (-W) % 4
        if ph or pw:
            frames = F.pad(frames, (0, pw, 0, ph), mode='replicate')
        x = F.gelu(self.cin(self.make_input(frames)))
        s0 = self.e0(x)
        s1 = self.e1(self.d1(F.gelu(s0)))
        x = self.e2(self.d2(F.gelu(s1)))
        x = F.interpolate(self.u1(F.gelu(x)), scale_factor=2, mode='bilinear', align_corners=False) + s1
        x = self.r1(x)
        x = F.interpolate(self.u0(F.gelu(x)), scale_factor=2, mode='bilinear', align_corners=False) + s0
        x = self.r0(x)
        o = self.head(F.gelu(x))[..., :H, :W]
        out = dict(logit=o[:, 0], dxy=torch.tanh(o[:, 1:3]) * self.dxy_range,
                   log_sig=-5.0 + 6.0 * torch.sigmoid(o[:, 3]))
        return psum_out(out, getattr(self, 'psum', 0.0)) if (getattr(self, 'psum', 0.0) != 0 and not self.training) else out


def psum_logit(logit, dxy, r=0.5, pmax=0.9999, mode='sum'):
    """(2026-11-10, opt-in) sub-pixel-invariant detection probability: p'_i = min(pmax, sum of p_j over the 3 x 3 pixels j
    (incl. i) whose predicted sub-pixel point lies within r px of pixel i's point). A detection split over two pixels
    (emitter near a pixel edge, nearest-pixel training target) gets its summed mass; NMS / thresholds then act on p'.
    mode 'or': p' = 1 - prod (1 - p_j) over the same pixels (fewer noise FPs than the sum; retrain_r/psum_test)."""
    lg = logit.float(); B, H, W = lg.shape
    p = torch.sigmoid(lg)
    yy, xx = torch.meshgrid(torch.arange(H, device=lg.device, dtype=torch.float32), torch.arange(W, device=lg.device, dtype=torch.float32), indexing='ij')
    X = xx + dxy[:, 0].float(); Y = yy + dxy[:, 1].float()
    pP = F.pad(p, (1, 1, 1, 1)); XP = F.pad(X, (1, 1, 1, 1), value=1e6); YP = F.pad(Y, (1, 1, 1, 1), value=1e6)
    acc = p.clone() if mode == 'sum' else 1 - p; r2 = r * r
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            if oy == 0 and ox == 0: continue
            sl = (slice(None), slice(1 + oy, 1 + oy + H), slice(1 + ox, 1 + ox + W))
            q = pP[sl] * (((XP[sl] - X) ** 2 + (YP[sl] - Y) ** 2) < r2).float()
            acc = acc + q if mode == 'sum' else acc * (1 - q)
    acc = (acc if mode == 'sum' else 1 - acc).clamp(1e-6, pmax)
    return torch.log(acc) - torch.log1p(-acc)


def psum_out(out, r):
    """r > 0: summed p within r px; r < 0: noisy-or within |r| px"""
    o = dict(out); o['logit_raw'] = out['logit']
    o['logit'] = psum_logit(out['logit'], out['dxy'], abs(r), mode='sum' if r > 0 else 'or').to(out['logit'].dtype)
    return o


@torch.no_grad()
def select_subpixel(logit, dxy, k, p_min=0.05, radius=0.7, win=3, iters=4, extra=None):
    """Greedy NMS on sub-pixel points (pixel + offset) + top-k per image.
    logit [B,H,W], dxy [B,2,H,W] -> dict b, iy, ix, x, y, logit (each [B*k]; entries below p_min have logit -inf
    and should be dropped by thresholding). extra: dict of [B,(C,)H,W] maps read out at the candidates."""
    B, H, W = logit.shape
    dev = logit.device
    yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32), indexing='ij')
    X = xx + dxy[:, 0].float(); Y = yy + dxy[:, 1].float()
    s = logit.float()
    valid = s > math.log(p_min / (1 - p_min))
    s = torch.where(valid, s, torch.full_like(s, -1e9))
    pad = lambda t, v: F.pad(t, (win, win, win, win), value=v)
    sP, XP, YP = pad(s, -1e9), pad(X, 1e6), pad(Y, 1e6)
    shifts = [(oy, ox) for oy in range(-win, win + 1) for ox in range(-win, win + 1) if (oy, ox) != (0, 0)]
    # which neighbours (shift) suppress this pixel if they are kept: higher score and point within radius
    sup_by = []
    r2 = radius * radius
    for oy, ox in shifts:
        sl = (slice(None), slice(win + oy, win + oy + H), slice(win + ox, win + ox + W))
        q_s, q_x, q_y = sP[sl], XP[sl], YP[sl]
        higher = (q_s > s) | ((q_s == s) & ((oy, ox) < (0, 0)))
        sup_by.append(higher & ((q_x - X) ** 2 + (q_y - Y) ** 2 < r2) & (q_s > -1e8))
    keep = valid.clone()
    for _ in range(iters):
        kP = F.pad(keep, (win, win, win, win), value=False)
        sup = torch.zeros_like(keep)
        for (oy, ox), m in zip(shifts, sup_by):
            sup |= m & kP[:, win + oy: win + oy + H, win + ox: win + ox + W]
        new = valid & ~sup
        if torch.equal(new, keep):
            break
        keep = new
    sc = torch.where(keep, s, torch.full_like(s, -float('inf')))
    top = sc.reshape(B, -1).topk(min(k, H * W), dim=1)
    kk = top.indices.shape[1]
    b = torch.arange(B, device=dev)[:, None].expand(B, kk).reshape(-1)
    idx = top.indices.reshape(-1)
    iy, ix = idx // W, idx % W
    out = dict(b=b, iy=iy, ix=ix, x=X[b, iy, ix], y=Y[b, iy, ix], logit=top.values.reshape(-1))
    for nm, mp in (extra or {}).items():
        out[nm] = mp[b, ..., iy, ix] if mp.dim() == 3 else mp[b, :, iy, ix]
    return out


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
        m(torch.zeros(1, T, H, W, device=dev))
    for h_ in hs: h_.remove()
    return tot[0]


def build(args):
    return DetNet(args.get('T', 6), args['inp'], tuple(args.get('ch', (48, 96, 160))), args.get('stem_ch', 16))


def load(ck, dev='cuda'):
    from simcode.config import resolve
    c = torch.load(resolve(ck), map_location=dev, weights_only=False)
    net = build(c['args']).to(dev); net.load_state_dict(c['model']); net.eval()
    return net, c['args']
