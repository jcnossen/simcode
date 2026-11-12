"""
Shared pieces of the per-candidate head and the explicit pattern solve wrapper (sc_solve).
  detector + top-k  : pattern-free detector (simcode2/det/up2.py) with sub-pixel NMS -> candidates (b, pixel, sub-pixel point)
  per-candidate head: the neighbour-aware SCHead4 (simcode2/hd/sc4.py) on a 21 x 21 x T patch at the candidate pixel. NO
                      pixel fit: network intensities I_1..I_T with a FULL T x T covariance (Cholesky factor), the
                      pattern-free position prior (dx, dy) +- sigma, per-frame bg. Here: patch extraction (patches), the PSF
                      template (PSFCol), apply_head / nbr_for (head call on detector candidates) and load().
  pattern solve     : sc_solve = the explicit intensity-domain pattern solve on the network intensities with their
                      covariance (simcode2/hd/msolve.py: fringe-start posterior, kinetic model selection, nuisance
                      covariance Sigma(mu) with c / c_a, chi2-aware calibrated sigma). The fast drop-in is
                      simcode2/hd/msolve_fast.sc_solve_fast.
"""
import argparse, json, math, os, time
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from simcode2.nn.blocks import ResBlock

dev = 'cuda'
PP = 21


def load_det_front(path=None):
    """candidate detector: path, else env SC_DET, else the production det_n1 (models/det_n1.pt; DetUp, loaded with
    simcode2/det/up2.load; native grid, logit / dxy outputs)"""
    from simcode2 import config
    path = config.resolve(path or os.environ.get('SC_DET') or config.DET)
    a = torch.load(path, map_location='cpu', weights_only=False)['args']
    if 'up' in a:
        from simcode2.det.up2 import load
    else:
        from simcode2.det.model import load
    return load(path)


def norm_photons(x, scale=30.0):
    return torch.sign(x) * torch.log1p(x.abs() / scale)


def patches(x, b, iy, ix, P=PP):
    """x [B,C,H,W] -> [N,C,P,P] centred on (iy, ix), zero padded"""
    h = P // 2
    xp = F.pad(x, (h, h, h, h))
    o = torch.arange(P, device=x.device)
    yy = (iy[:, None] + o[None])[:, :, None].expand(-1, P, P)
    xx = (ix[:, None] + o[None])[:, None, :].expand(-1, P, P)
    return xp[b[:, None, None], :, yy, xx].permute(0, 3, 1, 2)


class PSFCol:
    """unit-sum-free PSF columns on the PP x PP patch from the calibrated 15 x 15 LUT (z fixed 50 nm)"""
    def __init__(self, sim, z=0.05):
        from simcode2.psf.lut import make_lut
        self.lut = make_lut(sim, PP); self.z = z

    def __call__(self, off):
        return self.lut.evaluate(torch.cat([off, torch.full_like(off[:, :1], self.z)], 1))


def apply_head(head, col, frames, rn2, b, iy, ix, x, y, nbr=None, p=None):
    off = torch.stack([x - ix.float(), y - iy.float()], 1)
    pt = patches(frames, b, iy, ix)
    if getattr(head, 'nbr', False):     # sc4n (simcode2/hd/sc4.py): neighbour splat + detector-probability channels
        from simcode2.hd.sc4 import nbr_patches
        o = head(pt, None, off, col(off) if col is not None else None, nbr_patches(nbr, b, iy, ix, x, y, p))
    else:
        o = head(pt, None, off, col(off) if col is not None else None)
    o['x'] = x + o['dxy'][:, 0]; o['y'] = y + o['dxy'][:, 1]
    o['x_det'] = x; o['y_det'] = y          # detector point (extra solve start, cal['ms_xdet'])
    return o


@torch.no_grad()
def nbr_for(head, dd_logit, b, iy, ix, x, y, lg):
    """neighbour input maps for an sc4n head (None for the other heads); candidates = the kept p >= THR0 list"""
    if not getattr(head, 'nbr', False): return None
    from simcode2.hd.sc4 import nbr_map
    return nbr_map(dd_logit.float(), b, iy, ix, x, y, lg)


@torch.no_grad()
def sc_solve(o, pt_win, b, cal=None, nf=1, thr=9.21):
    """explicit pattern solve on the head's intensities with their full covariance (simcode2/hd/msolve.py).
    Returns r [n,2], s [n,2] (Fisher), s_cal (chi2-aware), q / dof, p. The calibration must set 'mix' (fringe-start
    posterior; all shipped calibrations do) or 'box' / 'ms_xdet' (multi-start NLL selection)."""
    from simcode2.core import chi2_sf
    cal = cal or {}
    I = o['I'].float(); L = o['L'].float()
    ph = I.sum(1)
    # 2026-11-09 (critic 32 #5): per-photon-bin prior / intensity scales (cal 'kap_bins': {'ph': bin centres (network window
    # photons), 'kap_p': [...], 'kap_I2': [...]}, log-photon interpolation, constant outside) replace the global kap_p / kap_I2
    kI2 = _binval(cal['kap_bins'], 'kap_I2', ph) if 'kap_bins' in cal and 'kap_I2' in cal['kap_bins'] else cal.get('kap_I2', 1.0)
    kp = _binval(cal['kap_bins'], 'kap_p', ph) if 'kap_bins' in cal and 'kap_p' in cal['kap_bins'] else cal.get('kap_p', 1.0)
    kI2 = kI2[:, None, None] if torch.is_tensor(kI2) else kI2
    kp = kp[:, None] if torch.is_tensor(kp) else kp
    Sig = L @ L.transpose(1, 2) * kI2 * float(cal.get('sig_scale', 1.0)) + 1e-3 * torch.eye(I.shape[1], device=I.device)
    # sig_scale (2026-11-09, critic 33 #2): per-dataset global scale of the network intensity covariance (automatic (scale, c))
    cm = float(cal.get('cmult', 0.0))      # nuisance-aware Sigma(mu) inside the solve (msolve.pattern_solve_all cmult)
    if cal.get('kap_mult', 0.0) > 0:
        # iteration 32 (critic 31 #2): intensity-domain multiplicative noise (per-frame / per-step emitter brightness
        # variation) var = (kap_mult I)^2 on the diagonal; I = the observed network I_t ('I') or its window mean ('mean')
        Ib = I.clamp(min=0) if cal.get('kap_mult_src', 'I') == 'I' else I.mean(1, keepdim=True).clamp(min=0).expand_as(I)
        Sig = Sig + torch.diag_embed((cal['kap_mult'] * Ib) ** 2)
    r0 = torch.stack([o['x'], o['y']], 1).float()
    if 'prior_off' in cal:     # 2026-11-09 (critic 32 #7): constant prior-position bias (px), measured on gate iso, removed
        r0 = r0 - torch.tensor(cal['prior_off'], device=r0.device, dtype=r0.dtype)[None]
    sp = o['lsxy'].float().exp() * kp
    if cal.get('mix'):
        # iteration 31 (critic 30 #1): fringe-start posterior (msolve.fringe_posterior, temperature cal['mix_T']);
        # 'pm' = posterior-mean position with the mixture sigma, 'inf' = MAP position with the inflated sigma
        from simcode2.hd import msolve as MS
        bases = [r0] + ([torch.stack([o['x_det'], o['y_det']], 1).float()] if cal.get('ms_xdet') else [])
        s = MS.fringe_posterior(I, Sig, r0, sp, pt_win[b], bases, nf=nf, box=cal.get('box', 0.75), thr=thr, temp=cal.get('mix_T', 1.0),
                                cmult=cm, logdet=bool(cal.get('cmult_logdet', True)), cang=float(cal.get('cmult_ang', 0.0)))
        qd = s['q'] / s['dof'].clamp(min=1)
        f = (cal.get('sig_a', 1.0) * qd.clamp(min=1.0) ** cal.get('sig_g', 0.0))[:, None]
        if cal['mix'] == 'pm':
            r, s_cal = s['r_pm'], (f * s['s_pm'] ** 2 + s['spread_pm'] ** 2).sqrt()
        else:
            r, s_cal = s['r'], (f * s['s'] ** 2 + s['spread'] ** 2).sqrt()
        if 'sig_ph' in cal: s_cal = s_cal * _binval(cal['sig_ph'], 'f', ph)[:, None]
        return dict(r=r, s=s['s'], s_cal=s_cal, q=qd, p=chi2_sf(s['q'], s['dof']), kin=s['kin'], r_map=s['r'],
                    s_map_cal=s['s'] * f.sqrt(), dnll=s['dnll'], r_prior=r0, s_prior=sp,
                    mu=s['mu'], Sig=Sig, dof=s['dof'], chi2=s['q'])      # + mu / Sigma_net / dof / raw chi2 (automatic c, 2026-11-09)
    if not (cal.get('box') is not None or cal.get('ms_xdet')):
        raise ValueError("sc_solve: the calibration sets neither 'mix' nor 'box' (the legacy single-start solve was removed)")
    # solver options (msolve.solve_ms): position box half-width, extra start set at the detector point (each with its fringe
    # shifts), selected by the posterior NLL
    from simcode2.hd import msolve as MS
    bases = [r0] + ([torch.stack([o['x_det'], o['y_det']], 1).float()] if cal.get('ms_xdet') else [])
    s = MS.solve_ms(I, Sig, r0, sp, pt_win[b], bases, nf=nf, box=cal.get('box', 0.75), thr=thr, cmult=cm, cang=float(cal.get('cmult_ang', 0.0)))['nll']
    qd = s['q'] / s['dof'].clamp(min=1)
    f = cal.get('sig_a', 1.0) * qd.clamp(min=1.0) ** cal.get('sig_g', 0.0)
    s_cal = s['s'] * f.sqrt()[:, None]
    if 'sig_ph' in cal: s_cal = s_cal * _binval(cal['sig_ph'], 'f', ph)[:, None]
    return dict(r=s['r'], s=s['s'], s_cal=s_cal, q=qd, p=chi2_sf(s['q'], s['dof']), kin=s['kin'])


def _binval(tab, key, ph):
    """per-photon-bin calibration value: tab['ph'] bin centres (network window photons), tab[key] values; piecewise
    linear in log photons, constant beyond the end bins -> tensor [n]"""
    xp = torch.log(torch.tensor(tab['ph'], device=ph.device, dtype=torch.float32))
    fp = torch.tensor(tab[key], device=ph.device, dtype=torch.float32)
    x = torch.log(ph.float().clamp(min=1.0)).clamp(xp[0], xp[-1])
    i = torch.searchsorted(xp, x).clamp(1, len(xp) - 1)
    w = (x - xp[i - 1]) / (xp[i] - xp[i - 1]).clamp(min=1e-9)
    return fp[i - 1] + w * (fp[i] - fp[i - 1])


def mod_slope(I, Ig):
    """regression slope of the mean-removed (per emitter) network intensities on the mean-removed GT intensities"""
    a = I - I.mean(1, keepdim=True); g = Ig - Ig.mean(1, keepdim=True)
    return float((a * g).sum() / (g * g).sum().clamp(min=1e-9))


def load(ck):
    from simcode2.config import resolve
    c = torch.load(resolve(ck), map_location=dev, weights_only=False)
    if c['args'].get('arch') != 'sc4n':
        raise ValueError(f"{ck}: head arch {c['args'].get('arch')!r} is not supported (only the neighbour-aware 'sc4n' head, simcode2/hd/sc4.py)")
    from simcode2.hd.sc4 import SCHead4
    h = SCHead4(6, c['args'].get('ch', 96), P=PP, bf16=bool(c['args'].get('bf16', 1))).to(dev)
    h.load_state_dict(c['model']); h.eval()
    return h

