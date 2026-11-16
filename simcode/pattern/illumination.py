"""
Illumination inputs for the pass-2 networks.

Pattern format (as the acquisition simulator / processing.common.pattern_frames): per frame (kx, ky, phase, depth, relint), illumination
relint (1 + depth cos(kx x + ky y - phase)), x = column, y = row (pixel centres at integer coordinates).

illum_maps: per frame two channels, the zero-mean modulation  relint (1 + m cos(.)) - 1  and the quadrature  relint m sin(.)
(the pitch is 2.07 px, just above the pixel Nyquist limit; the quadrature makes the local phase unambiguous at each pixel).
-> [nb, 2T, H, W] (channels 0..T-1 cos part, T..2T-1 sin part).

perturb: a realistic pattern-estimation error applied to the TRUE pattern (per window, per angle), pivoting at the window
centre so that a k error does not move the phase at the centre:
  pitch    relative k scale  N(0, PITCH_SD)        (smoke test M7: pitch within 0.0024 px of 2.07 -> 1.2e-3)
  angle    k rotation        N(0, ANG_SD) rad      (M7: angle within 0.012 deg = 2.1e-4 rad)
  phase    common per angle  N(0, PH_SD) rad       (0.10 rad = 3.6 nm; per-window drift residual + global, task 2-4 nm)
           per frame         N(0, PHF_SD) rad      (within-window drift residual; sim pipeline frame sd 1.8 nm ~ 0.05 rad)
  depth    common offset U(DEP_LO, DEP_HI) + per angle N(0, 0.01)
           (pass-1 network-intensity estimator is biased low: 0.875 / 0.889 vs 0.931 / 0.955 true on the sim movie)
  relint   per frame x (1 + N(0, 0.02))
"""
import math
import torch

PITCH_SD = 1.5e-3
ANG_SD = 3.5e-4
PH_SD = 0.10
PHF_SD = 0.04
DEP_LO, DEP_HI = -0.08, 0.03
REL_SD = 0.02
# 'real' error model (2026-11-09/10, training of the earlier default pass-2 network archive_superseded/previous_default/emitter_pass2.pt; critic 33 #5 + M7 of the full real runs): phase total ~0.17 rad rms
# (drift eps(t) scale: per angle 0.16 + per frame 0.06), angle 0.04 deg (M7 2.5 nM: ours - SIMCODE 0.025 deg, - calib up to
# 0.07 deg), pitch 1.5e-3 (M7 <= 0.0024 px = 1.2e-3), depth U(-0.10, +0.04) + 0.02 per angle (M7 ours - calib -0.065..-0.025,
# - SIMCODE -0.04..+0.02), relint 2 % per frame + 3 % per angle (M7 per-angle ours vs SIMCODE 0.5-2 %)
MODELS = dict(
    v1=dict(pitch=PITCH_SD, ang=ANG_SD, ph=PH_SD, phf=PHF_SD, dlo=DEP_LO, dhi=DEP_HI, dang=0.01, rel=REL_SD, rela=0.0),
    real=dict(pitch=1.5e-3, ang=7.0e-4, ph=0.16, phf=0.06, dlo=-0.10, dhi=0.04, dang=0.02, rel=0.02, rela=0.03))


def angle_of(patt):
    """[..., T, 5] -> angle index per frame (0: |kx| > |ky|)"""
    return (patt[..., 0].abs() < patt[..., 1].abs()).long()


def perturb(patt, H, W, scale=1.0, model='v1'):
    """patt [nb,T,5] (true) -> perturbed estimate [nb,T,5]; scale multiplies all error sds / ranges; model 'v1' (default,
    the first pass-2 training) or 'real' (see MODELS)"""
    M = MODELS[model]
    PITCH_SD, ANG_SD, PH_SD, PHF_SD, DEP_LO, DEP_HI, REL_SD = (M[k] for k in ('pitch', 'ang', 'ph', 'phf', 'dlo', 'dhi', 'rel'))
    nb, T, _ = patt.shape
    dev = patt.device
    ang = angle_of(patt)
    g = lambda *s: torch.randn(*s, device=dev)
    s = 1 + g(nb, 2) * PITCH_SD * scale
    th = g(nb, 2) * ANG_SD * scale
    sa, ta = s.gather(1, ang), th.gather(1, ang)
    kx, ky = patt[..., 0], patt[..., 1]
    kx2 = (kx * torch.cos(ta) - ky * torch.sin(ta)) * sa
    ky2 = (kx * torch.sin(ta) + ky * torch.cos(ta)) * sa
    cx, cy = (W - 1) / 2, (H - 1) / 2
    ph = patt[..., 2] + (kx2 - kx) * cx + (ky2 - ky) * cy
    ph = ph + (g(nb, 2) * PH_SD * scale).gather(1, ang) + g(nb, T) * PHF_SD * scale
    dc = DEP_LO * scale + (DEP_HI - DEP_LO) * scale * torch.rand(nb, 1, device=dev)
    m = (patt[..., 3] + dc + (g(nb, 2) * M['dang'] * scale).gather(1, ang)).clamp(0.05, 1.0)
    rel = patt[..., 4] * (1 + g(nb, T) * REL_SD * scale)
    if M['rela'] > 0:
        rel = rel * (1 + (g(nb, 2) * M['rela'] * scale).gather(1, ang))
    return torch.stack([kx2, ky2, ph, m, rel], -1)


def illum_maps(patt, H, W, x0=0.0, y0=0.0):
    """patt [nb,T,5] -> [nb,2T,H,W] (cos part rel (1 + m cos) - 1, sin part rel m sin); pixel (i, j) at (x0 + j, y0 + i)"""
    dev = patt.device
    yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32) + y0,
                            torch.arange(W, device=dev, dtype=torch.float32) + x0, indexing='ij')
    q = patt.float()[..., None, None]
    a = q[..., 0, :, :] * xx + q[..., 1, :, :] * yy - q[..., 2, :, :]
    rel, m = q[..., 4, :, :], q[..., 3, :, :]
    return torch.cat([rel * (1 + m * torch.cos(a)) - 1, rel * m * torch.sin(a)], 1)
