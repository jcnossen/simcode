"""Step 4: refit of the pass-1 detections with the fitted summed PSF (spline LUT of the in-situ vectorial fit) and
per-frame, per-channel amplitude fits with the PSF shape fixed.

PSF MLE: theta = (x, y, z, N, bg) on the 15 x 15 registered 6-frame 4-channel sum, 4 z starts (-0.45, -0.15, 0.15,
0.45 um) + the Gaussian/astigmatism start; best log-likelihood kept. CRLB from the Fisher matrix.
Amplitudes: per (frame t, channel c): data = the unwarped 11 x 11 channel image, model A_tc g_c + b_tc where g_c is the
summed PSF at the fitted (x, y, z) mapped into channel c's pixel grid (PSF normalised to the full plane); weighted
linear LSQ (weights 1 / (mu + rn^2), 3 reweighting passes, A unconstrained), errors from the 2 x 2 Fisher matrix.
usage: python refit.py <cell> <file_index> <psf_knots.npz>
"""
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/jc/simcode2/analysis/baselines'); sys.path.insert(0, '/home/jc/simcode2')
from zpsf import ZStackLUT
from mle import PSFModel, lm_fit, loglik
from register import PolyMap

P1 = '/home/jc/simcode2/out/4pi/smlm/p1'
RN2 = 1.3


class LUTPSF:
    def __init__(self, fn, P_list=(15, 11)):
        d = np.load(fn)
        self.z0, self.hz = float(d['z0']), float(d['hz'])
        self.zmax = self.z0 + self.hz * (d['zstack'].shape[0] - 1)
        ks = torch.as_tensor(d['zstack'], dtype=torch.float32)
        self.L = {P: ZStackLUT(ks, int(d['S']), float(d['u0']), self.z0, self.hz, P) for P in P_list}

    def eval3(self, x, y, z, P):
        pos = torch.stack([x, y, z], 1).float()
        g, gx, gy, gz = self.L[P].evaluate(pos, grad=True)
        return g.double(), gx.double(), gy.double(), gz.double()

    def img(self, x, y, z, P):
        pos = torch.stack([x, y, z], 1).float()
        return self.L[P].evaluate(pos).double()


def psf_refit(psf, roi, g0, chunk=4000, zgrid=np.linspace(-0.8, 0.8, 17), iters=30):
    """roi [n,15,15] e; g0 Gaussian fit [n,6] (x,y,N,bg,sx,sy). z start = best of a coarse LUT grid (x, y, N, bg from
    the Gaussian fit), then one LM run. Returns th [n,5], crlb, ll, chi2, dev"""
    P = roi.shape[-1]
    zl = (psf.z0 + 2 * psf.hz, psf.zmax - 2 * psf.hz)
    m = PSFModel(psf, P, zl)
    outs = []
    for s in range(0, len(roi), chunk):
        d = torch.as_tensor(roi[s:s + chunk], dtype=torch.float64)
        g = torch.as_tensor(g0[s:s + chunk], dtype=torch.float64)
        x, y = g[:, 0].clamp(-P / 2 + 2, P / 2 - 2), g[:, 1].clamp(-P / 2 + 2, P / 2 - 2)
        best = torch.full((len(d),), -float('inf'), dtype=torch.float64); bz = torch.zeros(len(d), dtype=torch.float64)
        for z0 in zgrid:
            zz = torch.full_like(x, float(z0))
            gi = psf.img(x, y, zz, P)
            # linear N, bg (unweighted LSQ) for this z
            gf = gi.flatten(1); df = d.flatten(1)
            a11 = (gf * gf).sum(1); a12 = gf.sum(1); a22 = float(P * P); b1 = (gf * df).sum(1); b2 = df.sum(1)
            det = a11 * a22 - a12 ** 2
            N = ((a22 * b1 - a12 * b2) / det).clamp(min=1); bg = ((a11 * b2 - a12 * b1) / det).clamp(min=1e-3)
            mu = N[:, None, None] * gi + bg[:, None, None]
            ll = loglik(d, mu, 24 * RN2)[0]
            upd = ll > best; best = torch.where(upd, ll, best); bz = torch.where(upd, zz, bz)
        th0 = torch.stack([x, y, bz, g[:, 2], g[:, 3]], 1)
        r = lm_fit(m, d, th0, 24 * RN2, iters=iters)
        outs.append([r[0].numpy(), r[2].numpy(), r[1].numpy(), r[3].numpy(), r[4].numpy()])
    return [np.concatenate([o[i] for o in outs]) for i in range(5)]


def amp_fit(psf, pm, roi_ch, corner, x, y, z, chunk=4000, iters=3):
    """roi_ch [n,6,4,11,11]; corner [n,4,2] (ix, iy); x, y (ch0 coords), z. Returns A [n,6,4], sA, bg [n,6,4], chi2 [n,6,4]"""
    n = len(x); P = roi_ch.shape[-1]; h = (P - 1) / 2
    A = np.zeros((n, 6, 4)); sA = np.zeros((n, 6, 4)); B = np.zeros((n, 6, 4)); C2 = np.zeros((n, 6, 4))
    for c in range(4):
        xc, yc = pm.inv(c, x, y)
        rx = xc - (corner[:, c, 0] + h); ry = yc - (corner[:, c, 1] + h)
        for s in range(0, n, chunk):
            sl = slice(s, s + chunk)
            g = psf.img(torch.as_tensor(rx[sl]), torch.as_tensor(ry[sl]), torch.as_tensor(z[sl]), P)      # [b,P,P]
            d = torch.as_tensor(roi_ch[sl, :, c].astype(np.float32), dtype=torch.float64)                    # [b,6,P,P]
            gf = g.flatten(1)[:, None].expand(-1, 6, -1); df = d.flatten(2)
            w = torch.ones_like(df)
            for _ in range(iters):
                a11 = (w * gf * gf).sum(-1); a12 = (w * gf).sum(-1); a22 = w.sum(-1)
                b1 = (w * gf * df).sum(-1); b2 = (w * df).sum(-1)
                det = a11 * a22 - a12 ** 2
                Aa = (a22 * b1 - a12 * b2) / det; bb = (a11 * b2 - a12 * b1) / det
                mu = Aa[..., None] * gf + bb[..., None]
                w = 1.0 / (mu.clamp(min=1e-3) + RN2)
            var = a22 / det
            A[sl, :, c] = Aa.numpy(); sA[sl, :, c] = var.sqrt().numpy(); B[sl, :, c] = bb.numpy()
            C2[sl, :, c] = (((df - mu) ** 2) * w).sum(-1).numpy()
    return A, sA, B, C2


def link_on(w, x, y, r=1.0):
    """True where the same emitter (within r px) is also detected in windows w - 1 and w + 1 (on through the window)"""
    from scipy.spatial import cKDTree
    on = np.zeros(len(w), bool)
    order = np.argsort(w); ws = w[order]
    starts = np.searchsorted(ws, np.arange(w.max() + 2)); 
    trees = {}
    def tree(k):
        if k not in trees:
            idx = order[starts[k]:starts[k + 1]] if 0 <= k <= w.max() else np.array([], int)
            trees[k] = (cKDTree(np.c_[x[idx], y[idx]]) if len(idx) else None)
        return trees[k]
    for k in range(w.max() + 1):
        idx = order[starts[k]:starts[k + 1]]
        if len(idx) == 0: continue
        tp, tn = tree(k - 1), tree(k + 1)
        if tp is None or tn is None: continue
        q = np.c_[x[idx], y[idx]]
        dp, _ = tp.query(q); dn, _ = tn.query(q)
        on[idx] = (dp < r) & (dn < r)
    return on


def run(cell, fi, knots, out=None):
    """PSF refit of detections with Gaussian N > 600 and the Gaussian centre within 2 px of the ROI centre; amplitude fits
    for the diagnostic subset: refit N > 800, linked on in windows w-1 and w+1, nearest other detection >= 5 px."""
    t0 = time.time()
    D = np.load(f'{P1}/p1_{cell}_{fi:03d}.npz')
    g = D['gfit']
    s1 = np.nonzero((g[:, 2] > 600) & (np.abs(g[:, 0]) < 2) & (np.abs(g[:, 1]) < 2))[0]
    psf = LUTPSF(knots)
    pm = PolyMap.load(f'/home/jc/simcode2/out/4pi/smlm/reg/reg_insitu_{cell}.json')
    th, cr, ll, chi2, dev = psf_refit(psf, D['roi_sum'][s1], g[s1])
    print(f'{cell} {fi}: PSF refit {len(th)} in {time.time() - t0:.0f}s', flush=True)
    x = D['cx'][s1] + th[:, 0]; y = D['cy'][s1] + th[:, 1]
    w = D['w'][s1]
    on = link_on(w, x, y)
    s2 = np.nonzero(on & (D['nn'][s1] >= 5) & (th[:, 3] > 800))[0]
    A, sA, B, C2 = amp_fit(psf, pm, D['roi_ch'][s1[s2]], D['corner'][s1[s2]], x[s2], y[s2], th[s2, 2])
    out = out or f'{P1}/p2_{cell}_{fi:03d}.npz'
    np.savez(out, idx=s1, w=w, x=x.astype(np.float32), y=y.astype(np.float32), pfit=th.astype(np.float32), pcrlb=cr.astype(np.float32),
             pll=ll, pchi2=chi2, pdev=dev, on=on, gfit=g[s1], gcrlb=D['gcrlb'][s1], gll=D['gll'][s1], gchi2=D['gchi2'][s1],
             gdev=D['gdev'][s1], nn=D['nn'][s1], gx=D['x'][s1], gy=D['y'][s1], knots=knots,
             aidx=s2, A=A.astype(np.float32), sA=sA.astype(np.float32), B=B.astype(np.float32), achi2=C2.astype(np.float32))
    print(f'{cell} {fi}: done; {len(s2)} amplitude fits; {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(int(os.environ.get('NT', '2')))
    run(sys.argv[1], int(sys.argv[2]), sys.argv[3])
