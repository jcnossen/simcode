"""
Per-candidate head base (SCHead) and the explicit pattern solve wrapper (sc_solve).
  detector + top-k  : pattern-free detector (det_n1, simcode2/det/up2.py) with sub-pixel NMS -> candidates (b, pixel,
                      sub-pixel point)
  per-candidate head: a CNN on a 21 x 21 x T patch at the candidate pixel. NO pixel fit: a learned linear intensity layer
                      I_lin,t = sum_q w_q (y_tq - bg_t) with a network-predicted weight map w and network bg_t, plus a learned
                      correction; outputs I_1..I_T with a FULL T x T covariance (Cholesky factor), the pattern-free position
                      prior (dx, dy) +- sigma, per-frame bg. The production head is the neighbour-aware SCHead4
                      (simcode2/hd/sc4.py, models/hd_sc5nb.pt); SCHead here is its base class.
  pattern solve     : sc_solve = the explicit intensity-domain pattern solve on the network intensities with their
                      covariance (simcode2/hd/msolve.py: fringe-start posterior, kinetic model selection, nuisance
                      covariance Sigma(mu) with c / c_a, chi2-aware calibrated sigma). The fast drop-in is
                      simcode2/hd/msolve_fast.sc_solve_fast.
train(): the original SCHead trainer (kept for reference; the production head is trained with simcode2/hd/sc4.py).
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


def wls(y, msk, rn2, psf):
    """per-frame WLS of [psf, 1]: y [n,T,Q], msk [n,1,Q], rn2 [n,1,Q], psf [n,Q] -> I [n,T], bg [n,T] (2 passes)"""
    n, T, Q = y.shape
    lvl = torch.quantile(y, 0.5, dim=-1, keepdim=True).clamp(min=1.0)
    w = msk / (lvl + rn2)
    p = psf[:, None]
    for _ in range(2):
        a = (w * p * p).sum(-1); bq = (w * p).sum(-1); cc = w.sum(-1)
        r1 = (w * p * y).sum(-1); r2 = (w * y).sum(-1)
        det = (a * cc - bq * bq).clamp(min=1e-9)
        I = (cc * r1 - bq * r2) / det; bg = (a * r2 - bq * r1) / det
        mu = (I[..., None] * p + bg[..., None]).clamp(min=1e-2)
        w = msk / (mu + rn2)
    return I, bg


class SCHead(nn.Module):
    """per-candidate head (no pixel fit anywhere): a CNN on the P x P x T patch predicts
      - a learned spatial weight map w (P x P, shared by the frames) and per-frame bg_t: the LEARNED linear intensity
        layer I_lin,t = sum_q w_q (y_tq - bg_t) (adapts to neighbours / wings / bg structure; linear in the data, so it
        does not shrink the frame-to-frame modulation unless the weights do; initialised at a Gaussian matched filter
        template at the candidate point)
      - a correction I_t = I_lin,t + c dI_t, the Cholesky factor L of the full T x T covariance (scale c = photon noise)
      - the pattern-free position refinement (dx, dy) +- sigma"""
    def __init__(self, T=6, ch=64, nblk=4, P=21):
        super().__init__()
        self.T, self.P = T, P
        cin = 2 * T + 4
        self.cin = nn.Conv2d(cin, ch, 3, padding=1)
        self.blk = nn.Sequential(*[ResBlock(ch, 1 + (i % 2)) for i in range(nblk)])
        self.wmap = nn.Sequential(nn.Conv2d(ch, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.wmap[-1].weight); nn.init.zeros_(self.wmap[-1].bias)
        self.down = nn.Sequential(nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.GELU(), nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.GELU(),
                                  nn.AdaptiveAvgPool2d(4))
        nsc = 2 * T + 2
        self.mlp = nn.Sequential(nn.Linear(ch * 16 + nsc, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU())
        nt = T * (T + 1) // 2
        self.out = nn.Linear(256, T + nt + 4 + T)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.tril = torch.tril_indices(T, T, -1)

    def forward(self, pt, rn2p, off, psf=None):
        n, T, P, _ = pt.shape
        flat = pt.reshape(n, T, -1)
        # bg start: per-frame median of the patch border (a robust statistic, not a fit)
        border = torch.cat([pt[:, :, 0], pt[:, :, -1], pt[:, :, 1:-1, 0], pt[:, :, 1:-1, -1]], -1)
        b0 = border.median(-1).values                                          # [n,T]
        lvl = b0.mean(1)[:, None, None, None]
        o = torch.arange(P, device=pt.device, dtype=torch.float32) - P // 2
        dx = o[None, None, :] - off[:, 0, None, None]; dy = o[None, :, None] - off[:, 1, None, None]
        cx = dx.expand(n, P, P) / (P // 2); cy = dy.expand(n, P, P) / (P // 2)
        lin = (pt - lvl) / 100.0
        x = torch.cat([norm_photons(pt), norm_photons(pt.mean(1, keepdim=True)), lin, lin.mean(1, keepdim=True),
                       cx[:, None], cy[:, None]], 1)
        hmap = self.blk(F.gelu(self.cin(x)))
        # matched-filter template g at the point, weights normalised so that sum(w g) = 1 (unbiased for the template PSF);
        # the network learns the correction of the map (a template, not a fit: nothing is solved from the data)
        if psf is not None:      # calibrated PSF template at the point (z 50 nm; simulator normalisation: phot x psf)
            g = psf
        else:
            g = torch.exp(-0.5 * (dx ** 2 + dy ** 2) / 1.3 ** 2).expand(n, P, P)
            g = 0.7 * g / g.sum((1, 2), keepdim=True)
        w0 = g / (g * g).sum((1, 2), keepdim=True)
        w = w0 * torch.exp(self.wmap(hmap)[:, 0].clamp(-5, 5))
        w = w / (w * g).sum((1, 2), keepdim=True)                             # keep the response to the template at 1
        h = self.down(hmap).flatten(1)
        sc = torch.cat([norm_photons(b0), off, torch.zeros(n, T, device=pt.device)], 1)
        q = self.out(self.mlp(torch.cat([h, sc], 1)))
        nt = T * (T + 1) // 2
        bg = b0 + q[:, T + nt + 4:] * (b0.abs() + 1).sqrt()
        Ilin = (w.reshape(n, 1, -1) * (flat - bg[..., None])).sum(-1)
        c = (Ilin.clamp(min=0).sum(1, keepdim=True) / T + bg.clamp(min=0).mean(1, keepdim=True) * 9).sqrt().clamp(min=3.0)
        dI = q[:, :T]
        ld = q[:, T:2 * T]; od = q[:, 2 * T:T + nt]
        L = torch.zeros(n, T, T, device=pt.device)
        L[:, torch.arange(T), torch.arange(T)] = c * torch.exp(ld.clamp(-4, 4)) * 1.5
        L[:, self.tril[0], self.tril[1]] = c * od
        I = Ilin + c * dI
        dxy = 0.5 * torch.tanh(q[:, T + nt:T + nt + 2])
        lsxy = -4.5 + 5.0 * torch.sigmoid(q[:, T + nt + 2:T + nt + 4])
        return dict(I=I, L=L, dxy=dxy, lsxy=lsxy, bg=bg, I_lin=Ilin, w=w)


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
    if getattr(head, 'bgmap', False):   # sc7bg (simcode2/hd/sc7bg.py, 2026-11-09): + the dense BgNet bg patch
        from simcode2.hd.sc4 import nbr_patches
        with torch.no_grad():
            bgp = patches(head.bgnet(frames), b, iy, ix)
        o = head(pt, None, off, col(off), nbr_patches(nbr, b, iy, ix, x, y, p), bgp)
    elif getattr(head, 'nbr', False):     # sc4n (simcode2/hd/sc4.py): neighbour splat + detector-probability channels
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


def train():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--ch', type=int, default=64)
    ap.add_argument('--det', default='det_n1.pt')
    ap.add_argument('--pair_frac', type=float, default=0.35)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--w_mr', type=float, default=0.5)
    ap.add_argument('--w_slope', type=float, default=0.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); json.dump(vars(a), open(a.out + '/args.json', 'w'), indent=1)
    torch.manual_seed(a.seed)
    from simcode2.hd import data as D
    from simcode2.det.model import load as load_det, select_subpixel
    from scipy.optimize import linear_sum_assignment
    gen = D.Gen(64, 64, 6, seed=a.seed)
    det, _ = load_det_front(a.det)
    head = SCHead(6, a.ch, P=PP).to(dev)
    col = PSFCol(gen.rs['0.5nM'].sim)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps + 1, pct_start=0.05)
    log = open(a.out + '/log.jsonl', 'a'); t0 = time.time(); acc = {}; IS, GS = [], []
    for step in range(a.steps + 1):
        with torch.no_grad():
            o = gen.batch(a.batch, pair_frac=a.pair_frac, pair_curr=True)
            fr = o['frames']; nb = fr.shape[0]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                dd = det(fr)
            c = select_subpixel(dd['logit'].float(), dd['dxy'].float(), 400, p_min=0.5, radius=0.7)
            keep = torch.isfinite(c['logit'])
            cb, cy_, cx_, xx, yy = c['b'][keep], c['iy'][keep], c['ix'][keep], c['x'][keep], c['y'][keep]
            G = o['gt']; act = (G['onf'].sum(1) > 0) & (G['phot'].sum(1) >= 50)
            gi = torch.nonzero(act)[:, 0]
            # one-to-one matching per FOV (< 1 px)
            mi, mg = [], []
            gb, gx, gy = G['fov'][gi].cpu().numpy(), G['x'][gi].cpu().numpy(), G['y'][gi].cpu().numpy()
            pb, px, py = cb.cpu().numpy(), xx.cpu().numpy(), yy.cpu().numpy()
            for f in range(nb):
                ip = np.nonzero(pb == f)[0]; ig = np.nonzero(gb == f)[0]
                if len(ip) == 0 or len(ig) == 0: continue
                Dm = (px[ip, None] - gx[None, ig]) ** 2 + (py[ip, None] - gy[None, ig]) ** 2
                Cc = np.where(Dm < 1, Dm, 1e6); r_, c_ = linear_sum_assignment(Cc); ok = Cc[r_, c_] < 1
                mi.append(ip[r_[ok]]); mg.append(ig[c_[ok]])
            if not mi: continue
            mi = torch.tensor(np.concatenate(mi), device=dev); mg = gi[torch.tensor(np.concatenate(mg), device=dev)]
            ok_in = (cy_[mi] >= 0) & (cx_[mi] >= 0)
            rn2 = (o['rn'] ** 2).expand(nb, 1, 64, 64)
            Ig = G['phot'][mg]; tx, ty = G['x'][mg], G['y'][mg]
            Dg = torch.cdist(torch.stack([tx, ty], 1), torch.stack([G['x'][gi], G['y'][gi]], 1))
            Dg = torch.where(G['fov'][mg][:, None] == G['fov'][gi][None], Dg, torch.full_like(Dg, 1e9))
            iso_w = ((Dg < 2.0).sum(1) <= 1).float()
            bgt = o['bg'][G['fov'][mg], :, torch.round(ty).long().clamp(0, 63), torch.round(tx).long().clamp(0, 63)]
        out = apply_head(head, col, fr, rn2, cb[mi], cy_[mi], cx_[mi], xx[mi], yy[mi])
        L = out['L']; e = (Ig - out['I'])[..., None]
        z = torch.linalg.solve_triangular(L, e, upper=False)[..., 0]
        nll_I = (0.5 * (z ** 2).sum(1) + torch.log(L.diagonal(dim1=1, dim2=2).abs().clamp(min=1e-6)).sum(1)
                 - torch.log((Ig.clamp(min=0).sum(1) / 6 + 30).sqrt()) * 6).mean()
        ex = (tx - out['x']); ey = (ty - out['y'])
        ls = out['lsxy']
        nll_p = (0.5 * (ex / ls[:, 0].exp()) ** 2 + ls[:, 0] + 0.5 * (ey / ls[:, 1].exp()) ** 2 + ls[:, 1]).mean()
        l_bg = (((out['bg'] - bgt) ** 2) / (bgt + 1.0)).mean()
        # anti-shrinkage: squared error of the mean-removed per-frame intensities in units of the photon-noise scale
        cn = (Ig.clamp(min=0).sum(1, keepdim=True) / 6 + 30).sqrt()
        mr = ((out['I'] - out['I'].mean(1, keepdim=True)) - (Ig - Ig.mean(1, keepdim=True))) / cn
        l_mr = F.huber_loss(mr, torch.zeros_like(mr), delta=3.0)   # robust (merged / contaminated candidates are heavy-tailed)
        # batch modulation-slope penalty (gate: slope of the mean-removed I_net on the mean-removed I_gt >= 0.97), on the
        # candidates whose matched GT has no other GT within 2 px (merged pairs legitimately differ)
        am = out['I'] - out['I'].mean(1, keepdim=True); gm = Ig - Ig.mean(1, keepdim=True)
        slope_b = (am * gm * iso_w[:, None]).sum() / (gm * gm * iso_w[:, None]).sum().clamp(min=1e-6)
        l_sl = (1 - slope_b) ** 2
        loss = nll_I / 6 + nll_p + 0.1 * l_bg + a.w_mr * l_mr + a.w_slope * l_sl
        opt.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
        sched.step()
        with torch.no_grad():
            IS.append(out['I'].detach()); GS.append(Ig)
            for k, v in dict(slope_iso=slope_b.item(), mr=l_mr.item(), nll_I=nll_I.item(), nll_p=nll_p.item(), bg=l_bg.item(), n=len(mi),
                             zI=float((z ** 2).mean().sqrt()), pos_rmse=float(torch.sqrt((ex ** 2 + ey ** 2).mean() / 2))).items():
                acc[k] = acc.get(k, 0) + v
        if step % 200 == 0:
            nn_ = 1 if step == 0 else 200
            Ia, Ga = torch.cat(IS), torch.cat(GS); IS, GS = [], []
            Iw = None
            r = dict(step=step, t=round(time.time() - t0), lr=sched.get_last_lr()[0], slope=mod_slope(Ia, Ga),
                     **{k: v / nn_ for k, v in acc.items()})
            acc = {}
            log.write(json.dumps(r) + '\n'); log.flush(); print(json.dumps(r), flush=True)
        if step % 5000 == 0 and step > 0:
            torch.save(dict(model=head.state_dict(), args=vars(a), step=step), a.out + f'/ckpt_{step}.pt')
    torch.save(dict(model=head.state_dict(), args=vars(a), step=a.steps), a.out + '/final.pt')


def load(ck):
    from simcode2.config import resolve
    c = torch.load(resolve(ck), map_location=dev, weights_only=False)
    if c['args'].get('arch') == 'sc7bg':    # simcode2/hd/sc7bg.py: sc4n head + dense BgNet bg (2026-11-09)
        from simcode2.hd.sc7bg import SCHead7
        h = SCHead7(6, 96, P=PP, bf16=True).to(dev)
    elif c['args'].get('arch') == 'sc4n':     # simcode2/hd/sc4.py neighbour-aware head
        from simcode2.hd.sc4 import SCHead4
        h = SCHead4(6, c['args'].get('ch', 96), P=PP, bf16=bool(c['args'].get('bf16', 1))).to(dev)
    else:
        h = SCHead(6, c['args'].get('ch', 64), P=PP).to(dev)
    h.load_state_dict(c['model']); h.eval()
    return h


if __name__ == '__main__':
    train()
