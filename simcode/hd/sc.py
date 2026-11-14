"""
Shared pieces of the per-candidate head and the explicit pattern solve wrapper (sc_solve).
  detector + top-k  : pattern-free detector (simcode/det/up2.py) with sub-pixel NMS -> candidates (b, pixel, sub-pixel point)
  per-candidate head: the neighbour-aware SCHead4 (simcode/hd/sc4.py) on a 21 x 21 x T patch at the candidate pixel. NO
                      pixel fit: network intensities I_1..I_T with a FULL T x T covariance (Cholesky factor), the
                      pattern-free position prior (dx, dy) +- sigma, per-frame bg. Here: patch extraction (patches), the PSF
                      template (PSFCol), apply_head / nbr_for (head call on detector candidates) and load().
  pattern solve     : sc_solve = the explicit intensity-domain pattern solve on the network intensities with their
                      covariance (simcode/hd/msolve.py: fringe-start posterior, kinetic model selection, nuisance
                      covariance Sigma(mu) with c / c_a, chi2-aware calibrated sigma). The fast drop-in is
                      simcode/hd/msolve_fast.sc_solve_fast.
"""
import torch
import torch.nn.functional as F

dev = 'cuda'
PP = 21


def load_det_front(path=None):
    """candidate detector: path, else the default model's detector (config.DET; DetUp, loaded with simcode/det/up2.load; native
    grid, logit / dxy outputs). (The env SC_DET override was replaced by the --det arguments, 2026-11-14.)"""
    from simcode import config
    path = config.resolve(path or config.DET)
    a = torch.load(path, map_location='cpu', weights_only=False)['args']
    if 'up' in a:
        from simcode.det.up2 import load
    else:
        from simcode.det.model import load
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
        from simcode.psf import spline
        self.lut = spline(sim.psf, sim.zmax, roisize=PP); self.z = z

    def __call__(self, off):
        return self.lut.render(torch.cat([off, torch.full_like(off[:, :1], self.z)], 1))


def apply_head(head, col, frames, rn2, b, iy, ix, x, y, nbr=None, p=None):
    off = torch.stack([x - ix.float(), y - iy.float()], 1)
    pt = patches(frames, b, iy, ix)
    if getattr(head, 'nbr', False):     # sc4n (simcode/hd/sc4.py): neighbour splat + detector-probability channels
        from simcode.hd.sc4 import nbr_patches
        o = head(pt, None, off, col(off) if col is not None else None, nbr_patches(nbr, b, iy, ix, x, y, p))
    else:
        o = head(pt, None, off, col(off) if col is not None else None)
    o['x'] = x + o['dxy'][:, 0]; o['y'] = y + o['dxy'][:, 1]
    o['x_det'] = x; o['y_det'] = y          # detector point
    return o


@torch.no_grad()
def nbr_for(head, dd_logit, b, iy, ix, x, y, lg):
    """neighbour input maps for an sc4n head (None for the other heads); candidates = the kept p >= THR0 list"""
    if not getattr(head, 'nbr', False): return None
    from simcode.hd.sc4 import nbr_map
    return nbr_map(dd_logit.float(), b, iy, ix, x, y, lg)


@torch.no_grad()
def sc_solve(o, pt_win, b, cal, nf=1, thr=9.21):
    """explicit pattern solve on the head's intensities with their full covariance (simcode/hd/msolve.py): fringe-start
    posterior (posterior-mean position with the mixture sigma), kinetic model selection, nuisance covariance Sigma(mu).
    Returns r [n,2], s [n,2] (Fisher), s_cal (chi2-aware, calibrated), q / dof, p. Reference implementation (test oracle of
    simcode/hd/msolve_fast.sc_solve_fast, which the pipeline uses).
    Calibration keys (all required): kap_bins {'ph', 'kap_p', 'kap_I2'} (per-photon-bin prior / intensity covariance scales,
    log-photon interpolation, constant outside), prior_off (constant prior-position bias, px, removed), cmult / cmult_ang
    (nuisance covariance c, c_a), box (start box half-width, px), mix_T (posterior temperature), sig_a / sig_g (chi2-aware
    sigma), sig_ph {'ph', 'f'} (per-photon-bin sigma factor)."""
    from simcode.core import chi2_sf
    from simcode.hd import msolve as MS
    I, Sig, r0, sp, ph = solve_inputs(o, cal)
    s = MS.fringe_posterior(I, Sig, r0, sp, pt_win[b], [r0], nf=nf, box=cal['box'], thr=thr, temp=cal['mix_T'],
                            cmult=float(cal['cmult']), cang=float(cal['cmult_ang']))
    qd = s['q'] / s['dof'].clamp(min=1)
    f = (cal['sig_a'] * qd.clamp(min=1.0) ** cal['sig_g'])[:, None]
    s_cal = (f * s['s_pm'] ** 2 + s['spread_pm'] ** 2).sqrt() * _binval(cal['sig_ph'], 'f', ph)[:, None]
    return dict(r=s['r_pm'], s=s['s'], s_cal=s_cal, q=qd, p=chi2_sf(s['q'], s['dof']), kin=s['kin'], r_map=s['r'],
                s_map_cal=s['s'] * f.sqrt(), dnll=s['dnll'], r_prior=r0, s_prior=sp,
                mu=s['mu'], Sig=Sig, dof=s['dof'], chi2=s['q'])


def solve_inputs(o, cal, mm=torch.matmul):
    """head outputs + calibration -> (I [n,T], Sig [n,T,T] = L L^T kap_I2 + 1e-3, prior position r0 [n,2] (prior_off removed),
    prior sigma sp [n,2] = exp(lsxy) kap_p, window photons ph [n]); shared by sc_solve and msolve_fast.sc_solve_fast"""
    I = o['I'].float(); L = o['L'].float()
    ph = I.sum(1)
    kI2 = _binval(cal['kap_bins'], 'kap_I2', ph)[:, None, None]
    kp = _binval(cal['kap_bins'], 'kap_p', ph)[:, None]
    Sig = mm(L, L.transpose(1, 2)) * kI2 + 1e-3 * torch.eye(I.shape[1], device=I.device)
    r0 = torch.stack([o['x'], o['y']], 1).float() - torch.tensor(cal['prior_off'], device=I.device, dtype=torch.float32)[None]
    sp = o['lsxy'].float().exp() * kp
    return I, Sig, r0, sp, ph


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
    from simcode.config import resolve
    c = torch.load(resolve(ck), map_location=dev, weights_only=False)
    if c['args'].get('arch') != 'sc4n':
        raise ValueError(f"{ck}: head arch {c['args'].get('arch')!r} is not supported (only the neighbour-aware 'sc4n' head, simcode/hd/sc4.py)")
    from simcode.hd.sc4 import SCHead4
    h = SCHead4(6, c['args'].get('ch', 96), P=PP, bf16=bool(c['args'].get('bf16', 1))).to(dev)
    h.load_state_dict(c['model']); h.eval()
    return h

