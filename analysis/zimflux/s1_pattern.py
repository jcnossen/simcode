"""S1 step 3: illumination pattern from the per-frame intensities (paper 3.5.2, eq. 31-32) with the astigmatic positions
(raw, not drift corrected: the pattern is fixed to the optics).
r_vk = N_vk / sum_k N_vk = eta_k (1 + m_k cos(q . r_xy + k_z z - psi_k(T))), 10 time bins: for given (q, k_z) the per-bin,
per-slot model is linear in (eta, eta m cos psi, eta m sin psi) (eq. 32). (q, k_z) by variable projection: grid over k_z
(both signs), then L-BFGS on (q, k_z) through the inner least squares; 2 rounds with outlier trimming (spots not ON in all
3 frames). Note: k_z from this regression is attenuated by the astigmatic z error (errors in variables); s1_zf.py refines
it by the profile likelihood of the joint fit.
usage: python s1_pattern.py <gpu> -> out/zimflux/s1/pattern.json"""
import sys, json, numpy as np, torch
sys.path.insert(0, '/home/jc/simcode2/analysis/zimflux')
from s1_common import *
NB = 10


def good_mask(A):
    th = A['th']; Nk = A['Nk']
    return ((np.abs(th[:, :2]) < 6).all(1) & (th[:, 2] > ZLO + 0.005) & (th[:, 2] < ZHI - 0.005) & (A['dev'] < 3)
            & (Nk.sum(1) > 0))


def load(dev, A, sel):
    th = A['th'][sel]; c = A['corner'][sel]
    t = lambda a: torch.tensor(a, device=dev, dtype=torch.float32)
    xa = t(c[:, 1] + 7.5 + th[:, 0]); ya = t(c[:, 0] + 7.5 + th[:, 1]); z = t(th[:, 2])
    Nk = t(A['Nk'][sel]); Nv = Nk.sum(1)
    return xa, ya, z, Nk / Nv[:, None], Nv, t(A['cycle'][sel].astype(np.float32))


def inner(q, kz, xa, ya, z, r, Nv, tbin, w=None):
    """per (bin, slot) weighted LS of r on [1, cos th, sin th]; returns coef [NB,K,3], residual sum"""
    th = q[0] * xa + q[1] * ya + kz * z
    X = torch.stack([torch.ones_like(th), torch.cos(th), torch.sin(th)], 1)
    W = Nv if w is None else Nv * w
    coef, res = [], 0
    for b in range(NB):
        m_ = tbin == b; Xb = X[m_] * W[m_, None].sqrt(); yb = r[m_] * W[m_, None].sqrt()
        cb = torch.linalg.solve(Xb.T @ Xb, Xb.T @ yb)                  # [3,K]
        res = res + ((Xb @ cb - yb) ** 2).sum()
        coef.append(cb.T)
    return torch.stack(coef), res


def to_pattern(q, kz, coef, ncyc):
    eta = coef[..., 0]; mc, ms = coef[..., 1], coef[..., 2]
    m = torch.sqrt(mc ** 2 + ms ** 2) / eta; psi = torch.atan2(ms, mc)          # [NB,K]
    tb = (torch.arange(NB, dtype=torch.float32) + 0.5) * ncyc / NB
    return {'q': q.tolist(), 'kz': float(kz), 'eta': (eta.mean(0) / eta.mean(0).sum()).tolist(), 'm': m.mean(0).tolist(),
            'psi': psi.T.tolist(), 'tb': tb.tolist(), 'eta_bins': eta.tolist(), 'm_bins': m.tolist()}


def fit_pattern(dev, A, sel, ncyc, q0, kz_fixed=None, kz_grid=None, verbose=True):
    xa, ya, z, r, Nv, t = load(dev, A, sel)
    tbin = (t / ncyc * NB).long().clamp(0, NB - 1)
    w = torch.ones_like(Nv)
    q = torch.tensor(q0, device=dev, dtype=torch.float32)
    if kz_fixed is None:
        grid = kz_grid if kz_grid is not None else np.arange(-24, 24.01, 0.5)
        rs = [float(inner(q, torch.tensor(k, device=dev), xa, ya, z, r, Nv, tbin)[1]) for k in grid]
        kz0 = float(grid[int(np.argmin(rs))])
        if verbose: print('kz grid min', kz0, 'res', min(rs), 'at 0:', rs[len(grid) // 2], flush=True)
    else:
        kz0 = kz_fixed
    for rnd in range(2):
        pq = q.clone().requires_grad_(True); pk = torch.tensor(kz0, device=dev, requires_grad=kz_fixed is None)
        opt = torch.optim.LBFGS([pq] + ([pk] if kz_fixed is None else []), lr=0.5, max_iter=50, line_search_fn='strong_wolfe')
        def clo():
            opt.zero_grad(); _, res = inner(pq, pk, xa, ya, z, r, Nv, tbin, w); res.backward(); return res
        opt.step(clo)
        q = pq.detach(); kz0 = float(pk.detach())
        coef, res = inner(q, torch.tensor(kz0, device=dev), xa, ya, z, r, Nv, tbin, w)
        # trim: per-spot residual (weighted), drop the worst 10 %
        th = q[0] * xa + q[1] * ya + kz0 * z
        X = torch.stack([torch.ones_like(th), torch.cos(th), torch.sin(th)], 1)
        pred = torch.einsum('nj,nkj->nk', X, coef[tbin])
        e = ((r - pred) ** 2).sum(1) * Nv
        w = (e < torch.quantile(e, 0.9)).float()
        if verbose: print('round', rnd, 'q', q.tolist(), 'kz', kz0, 'res', float(res), flush=True)
    p = to_pattern(q.cpu(), kz0, coef.cpu(), ncyc)
    p['n_spots'] = int(len(Nv)); p['res'] = float(res)
    return p


if __name__ == '__main__':
    dev = torch.device(f'cuda:{sys.argv[1]}')
    A = dict(np.load(WORK + 'astig.npz'))
    ncyc = int(np.load(WORK + 'rois.npz')['ncycles'])
    g = good_mask(A) & (A['Nk'].sum(1) > 1000)
    q0 = json.load(open('/home/jc/simcode2/out/zimflux/inventory/pattern_jan13.json'))['orientations'][0]['k_rad_px']
    p = fit_pattern(dev, A, np.nonzero(g)[0], ncyc, q0)
    print({k: p[k] for k in ('q', 'kz', 'eta', 'm', 'n_spots')}, 'pitch nm', 2 * np.pi / np.hypot(*p['q']) * PX,
          'p_ax nm', 2 * np.pi / abs(p['kz']) * 1000, flush=True)
    json.dump(p, open(WORK + 'pattern.json', 'w'), indent=1)
