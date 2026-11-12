"""4Pi-SIMFLUX high-density showcase: ground-truth evaluation of all methods on one showcase movie.

Region: x, y in [8, W - 8) px (candidates need a full 15 px patch). GT unit = one binding event within one 6-frame
window (simeval4pi.gt_units). Matching (simeval4pi.match): per window, greedy nearest within 0.5 px (64 nm) laterally.
One matching of the localizations against ALL GT units (any photon count, region + 1 px margin); recall over GT units in the
region with >= 300 e per window; precision = matched / localizations in the region (a localization of a dim < 300 e
unit counts as correct; simeval4pi.py counts it as false, so its precision is lower at high density).
Errors after removing the median offset; robust = 1.4826 MAD; RMS without fringe errors (|dz| <= 100 nm); fringe error
rate = |dz| > 100 nm; CRLB = isolated-emitter pixel CRLB with the true per-frame photons (simeval4pi.crlb_units).
Methods (same det4pi candidates for all):
  net        head v10 + explicit solve, pattern and 4Pi phase field estimated in situ, simeval4pi filters (sigma_z < 25 nm,
             sigma_xy < 0.12 px, N > 300) + chi2 selection 'good' (chi2/dof < 1.6)
  net_bound  the same with the TRUE pattern and phase field
  net_nochi2 net without the chi2 selection;  net_all  every candidate
  cl_pub     classical published-style (SIMFLUX two-step x, y + 4Pi moment phase z) with the published rejections
  cl_pix     classical 24-image pixel MLE (+ its own convergence / bound / chi2 rejection)
  cl_pub_nofilt  cl_pub without rejections
  astig      summed-SMLM MLE, z from the astigmatic (incoherent) PSF only, simeval4pi's fit rejection (chi2/px <= 4, no
             bound, converged, sigma_xy < 0.12 px, N > 300)
Also: crowding (other emitters in the 15 x 15 px patch), localizations per um of tube, FRC (x-y and x-z), tube
cross-section profiles (localizations relative to the nearest GT tube axis away from crossings).
Output: out/4pi/showcase/<tag>/eval.json, locs_<method>.npz (filtered, region), profiles.npz, frc.npz
Usage: CUDA_VISIBLE_DEVICES=1 python analysis/fourpi/showcase4pi_eval.py --tag sc_d1
"""
import argparse, json, os, sys, time
import numpy as np, torch
from scipy.spatial import cKDTree
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.fourpi.setup import P4
from simmovie4pi import PX
from simeval4pi import gt_units, match, crlb_units, EDGES
from simcode.fourpi.lut4pi import LUT4Pi
from simcode.fourpi.lm import lm_poisson_j, fisher_j, crlb

NM = PX * 1000


def astig_fit(F, lut, dev, chunk=20000):
    """the summed-SMLM part of simeval4pi.classical (identical model, starts, iterations and quality flags), without
    its aperture-photometry 4Pi z (not used here; that loop costs ~45 ms per candidate)"""
    v = 24 * 1.3
    def fj(th):
        x, y, z, N, b = th.unbind(1)
        out = [L.evaluate(torch.stack([x, y, z], 1), True) for L in [lut.luts[0][c] for c in range(4)]]
        g = sum(o[0] for o in out); gx = sum(o[1] for o in out); gy = sum(o[2] for o in out); gz = sum(o[3] for o in out)
        mu = N[:, None, None] * g + b[:, None, None]
        J = torch.stack([N[:, None, None] * gx, N[:, None, None] * gy, N[:, None, None] * gz, g, torch.ones_like(g)], -1)
        return mu.reshape(len(th), -1), J.reshape(len(th), -1, 5)
    lo = torch.tensor([-3, -3, -0.6, 1, 0], device=dev, dtype=torch.float32); hi = torch.tensor([3, 3, 0.6, 1e6, 1e3], device=dev, dtype=torch.float32)
    R = {k: [] for k in ('chi2px', 'conv', 'at_bound', 'x', 'y', 'z', 'sx', 'sy', 'sz', 'N')}
    roi_all = F['roi_sum']
    for i in range(0, len(roi_all), chunk):
        roi = torch.tensor(roi_all[i:i + chunk], device=dev); n = len(roi)
        dd = roi.reshape(n, -1)
        bg0 = roi[:, [0, -1]].median(-1).values.median(-1).values
        N0 = (dd - bg0[:, None]).sum(1).clamp(min=50)
        best = None
        for zs in (-0.3, 0.0, 0.3):
            th0 = torch.stack([torch.zeros(n, device=dev), torch.zeros(n, device=dev), torch.full((n,), zs, device=dev), N0, bg0], 1)
            th, ll, _ = lm_poisson_j(fj, dd, th0, v, lo=lo, hi=hi, iters=30)
            if best is None: best = (th, ll)
            else:
                b = ll > best[1]; best = (torch.where(b[:, None], th, best[0]), torch.where(b, ll, best[1]))
        th_s = best[0]
        cr = crlb(fisher_j(fj, th_s, v))
        mu_f, _ = fj(th_s)
        R['chi2px'].append((((dd - mu_f) ** 2) / (mu_f.clamp(min=1e-6) + v)).mean(1))
        th_c, _, _ = lm_poisson_j(fj, dd, th_s, v, lo=lo, hi=hi, iters=10)
        R['conv'].append(((th_c[:, :2] - th_s[:, :2]).abs().max(1).values < 0.01) & ((th_c[:, 2] - th_s[:, 2]).abs() < 0.002))
        R['at_bound'].append((th_s[:, :2].abs() >= 2.95).any(1) | (th_s[:, 2].abs() >= 0.595) | (th_s[:, 3] <= 1.5) | (th_s[:, 4] <= 1e-3) | (th_s[:, 4] >= 999))
        for k, t_ in (('x', th_s[:, 0]), ('y', th_s[:, 1]), ('z', th_s[:, 2]), ('sx', cr[:, 0]), ('sy', cr[:, 1]), ('sz', cr[:, 2]), ('N', th_s[:, 3])):
            R[k].append(t_)
    R = {k: torch.cat(v_).cpu().numpy() for k, v_ in R.items()}
    R['x'] = F['cx'] + R['x']; R['y'] = F['cy'] + R['y']; R['w'] = F['w']
    return R


def frc(xa, ya, xb, yb, ext_a, ext_b, pix=4.0):
    """FRC of two localization sets (nm coordinates), image extent ext_a, ext_b (nm) -> (freq 1/nm, frc, resolution nm)"""
    na, nb = int(ext_a / pix), int(ext_b / pix)
    def img(x, y):
        ok = (x >= 0) & (x < ext_a) & (y >= 0) & (y < ext_b)
        I = np.zeros((nb, na)); np.add.at(I, ((y[ok] / pix).astype(int), (x[ok] / pix).astype(int)), 1); return I
    win = np.outer(np.hanning(nb), np.hanning(na)) ** 0.5
    A = np.fft.fft2(img(xa, ya) * win); B = np.fft.fft2(img(xb, yb) * win)
    fy = np.fft.fftfreq(nb, pix); fx = np.fft.fftfreq(na, pix)
    fr = np.hypot(fy[:, None], fx[None])
    df = 1.0 / (max(na, nb) * pix)
    k = (fr / df).astype(int).ravel(); nmax = int(0.5 / pix / df)
    num = np.bincount(k, (A * B.conj()).real.ravel(), nmax + 1)[:nmax]
    da = np.bincount(k, (np.abs(A) ** 2).ravel(), nmax + 1)[:nmax]; db = np.bincount(k, (np.abs(B) ** 2).ravel(), nmax + 1)[:nmax]
    c = num / np.sqrt(np.maximum(da * db, 1e-30))
    f = (np.arange(nmax) + 0.5) * df
    cs = np.convolve(c, np.ones(5) / 5, 'same')
    lo = np.nonzero((cs < 1 / 7) & (f > 1 / 300))[0]
    res = float(1 / f[lo[0]]) if len(lo) else float('nan')
    return f, c, res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='sc_d1'); p.add_argument('--model', default='v3L_det')
    p.add_argument('--sz_max', type=float, default=0.025); p.add_argument('--sxy_max', type=float, default=0.12)
    p.add_argument('--chi2_max', type=float, default=4.0)
    a = p.parse_args()
    from simcode.util.proctitle import auto; auto('4pi-sceval')
    d = f'{P4}/simmovie/{a.tag}'; dev = 'cuda'
    out = f'{P4}/showcase/{a.tag}'; os.makedirs(out, exist_ok=True)
    meta = json.load(open(f'{d}/meta.json')); H, W, T = meta['H'], meta['W'], meta['T']
    LO, HI = 8.0, W - 8.0
    st = json.load(open(f'{P4}/psf4pi/{meta["psf"]}.json'))['psf']
    lut = LUT4Pi(st, P=15, dev=dev)
    t0 = time.time()
    U = gt_units(d, T)
    inr = lambda x, y, m=0.0: (x >= LO - m) & (x < HI + m) & (y >= LO - m) & (y < HI + m)
    gall = inr(U['x'], U['y'], 1.0)
    GA = {k: v[gall] for k, v in U.items()}                       # every GT unit (any photons), region + 1 px
    gsel = inr(GA['x'], GA['y']) & (GA['N'] >= 300)                # the recall set
    # ---- crowding: other emitters in the 15 x 15 px patch (same window: any photons; same frame: ON in that frame)

    def crowd(w, x, y, wa, xa, ya):
        cnt = np.zeros(len(w))
        order = np.argsort(wa); bnd = np.searchsorted(wa[order], np.arange(wa.max() + 2))
        ordw = np.argsort(w); bw = np.searchsorted(w[ordw], np.arange(wa.max() + 2))
        for v in range(wa.max() + 1):
            A_ = order[bnd[v]:bnd[v + 1]]; B_ = ordw[bw[v]:bw[v + 1]]
            if len(A_) == 0 or len(B_) == 0: continue
            dx = np.abs(x[B_][:, None] - xa[A_][None]); dy = np.abs(y[B_][:, None] - ya[A_][None])
            cnt[B_] = ((dx < 7.5) & (dy < 7.5)).sum(1) - 1
        return cnt
    Ua = {k: v[inr(U['x'], U['y'], 8.0)] for k, v in U.items()}
    cw = crowd(GA['w'][gsel], GA['x'][gsel], GA['y'][gsel], Ua['w'], Ua['x'], Ua['y'])
    g = np.load(f'{d}/gt_frames.npz'); gf_in = inr(g['x'], g['y'])
    cf = crowd(g['frame'][gf_in], g['x'][gf_in], g['y'][gf_in], g['frame'], g['x'], g['y'])
    ax_ = np.load(f'{d}/gt_axis.npz'); AX = ax_['axis']
    axin = inr(AX[:, 0] / PX, AX[:, 1] / PX)
    tube_um = float(axin.sum() * np.median(np.linalg.norm(np.diff(AX[:, :3], axis=0), axis=1)))
    res = dict(tag=a.tag, dens=meta['dens_actual'], T=T, region_um=(HI - LO) * PX, tube_um_region=tube_um,
               gt_units=int(gsel.sum()), gt_units_per_um=float(gsel.sum() / tube_um),
               crowd_window_mean=float(cw.mean()), crowd_frame_mean=float(cf.mean()),
               crowd_window_frac_gt3=float(np.mean(cw > 3)), on_per_frame=float(len(g['frame']) / T))
    print(f"{a.tag}: dens {res['dens']:.2f}, GT units {res['gt_units']} ({res['gt_units_per_um']:.0f}/um tube), "
          f"others in 15px patch: window {res['crowd_window_mean']:.2f}, frame {res['crowd_frame_mean']:.2f}", flush=True)
    # ---- methods
    F = np.load(f'{d}/front_SIM_000_{a.model}.npz')
    fc = f'{d}/classical_{a.model}_q.npz'
    fa = f'{out}/astig.npz'
    if os.path.exists(fc): C = dict(np.load(fc))          # simeval4pi.classical cache (same fields)
    elif os.path.exists(fa): C = dict(np.load(fa))
    else:
        C = astig_fit(F, lut, dev); np.savez(fa, **C)
    print(f'astig baseline done {time.time() - t0:.0f}s', flush=True)
    S = {}
    for nm, pt in (('net', 'iterv10'), ('net_bound', 'oraclev10_phiO')):
        L = dict(np.load(f'{d}/locs_SIM_000_{a.model}_{pt}.npz'))
        base = (L['sz'] < a.sz_max) & (L['sx'] < a.sxy_max) & (L['sy'] < a.sxy_max) & (L['N'] > 300)
        S[nm] = (L['x'], L['y'], L['z'], L['w'], base & L['good'])
        if nm == 'net':
            S['net_nochi2'] = (L['x'], L['y'], L['z'], L['w'], base)
            S['net_all'] = (L['x'], L['y'], L['z'], L['w'], np.ones(len(L['x']), bool))
    fk = f'{P4}/showcase/cl/{a.tag}/cl_{a.tag}.npz'
    if os.path.exists(fk):
        K = dict(np.load(fk)); res['classical_failed'] = False
    else:   # sim_cl.py stopped (no emitter passes its selection for the pattern / phase-field fit): no classical output
        z0 = np.zeros(0); K = dict(x=z0, y=z0, z=z0, w=z0.astype(int), keep=z0.astype(bool), xp=z0, yp=z0, zp=z0, keep_pix=z0.astype(bool))
        res['classical_failed'] = True
    S['cl_pub'] = (K['x'], K['y'], K['z'], K['w'], K['keep'])
    S['cl_pix'] = (K['xp'], K['yp'], K['zp'], K['w'], K['keep_pix'])
    S['cl_pub_nofilt'] = (K['x'], K['y'], K['z'], K['w'], np.isfinite(K['z']))
    qok = (C['chi2px'] <= a.chi2_max) & ~C['at_bound'] & C['conv'] & (C['sx'] < a.sxy_max) & (C['sy'] < a.sxy_max) & (C['N'] > 300)
    S['astig'] = (C['x'], C['y'], C['z'], C['w'], qok)
    # ---- CRLB (cached)
    fcr = f'{out}/crlb.npy'
    if os.path.exists(fcr): cr = np.load(fcr)
    else:
        G = {k: v[gsel] for k, v in GA.items()}
        cr = crlb_units(G, np.arange(len(G['x'])), meta, lut, dev) * np.array([NM, NM, 1000]); np.save(fcr, cr)
    print(f'crlb done {time.time() - t0:.0f}s', flush=True)
    Gx, Gy, Gz, GN = GA['x'][gsel], GA['y'][gsel], GA['z'][gsel], GA['N'][gsel]
    # ---- per-method metrics
    res['methods'] = {}
    blo = np.r_[EDGES[:-1], 1000]; bhi = np.r_[EDGES[1:], 30000]       # bins + pooled >= 1000 e
    for nm, (x, y, z, w, ok) in S.items():
        ok = ok & inr(x, y, 1.0) & np.isfinite(z) & np.isfinite(x) & np.isfinite(y)
        x, y, z, w = x[ok], y[ok], z[ok], w[ok].astype(np.int64)
        if len(x):
            mg, ml = match(GA['w'], GA['x'], GA['y'], w, x, y)
        else:
            mg, ml = np.full(len(GA['w']), -1), np.zeros(0, int)
        lin = inr(x, y)
        prec = float((ml[lin] >= 0).mean()) if lin.any() else float('nan')
        m = mg[gsel]; mm = m >= 0
        ex = (x[m[mm]] - Gx[mm]) * NM; ey = (y[m[mm]] - Gy[mm]) * NM; ez = (z[m[mm]] - Gz[mm]) * 1000
        bias = [float(np.median(e)) for e in (ex, ey, ez)] if mm.any() else [np.nan] * 3
        rows = []
        for lo, hi in zip(blo, bhi):
            sg = (GN >= lo) & (GN < hi); sm = sg[mm]
            r = dict(lo=int(lo), hi=int(hi), n_gt=int(sg.sum()), n_matched=int(sm.sum()), recall=float(mm[sg].mean()) if sg.any() else np.nan)
            if sm.sum() >= 30:
                rb = lambda e: float(1.4826 * np.median(np.abs(e[sm] - np.median(e[sm]))))
                gz = sm & (np.abs(ez) <= 100)
                r.update(x_rob=rb(ex), y_rob=rb(ey), z_rob=rb(ez),
                         x_rms=float(np.sqrt(np.mean((ex[gz] - bias[0]) ** 2))), y_rms=float(np.sqrt(np.mean((ey[gz] - bias[1]) ** 2))),
                         z_rms=float(np.sqrt(np.mean((ez[gz] - bias[2]) ** 2))), fringe=float(np.mean(np.abs(ez[sm]) > 100)),
                         crlb=[float(np.sqrt(np.mean(cr[mm][sm][:, k] ** 2))) for k in range(3)])
            rows.append(r)
        nloc = int(lin.sum())
        res['methods'][nm] = dict(n_locs=nloc, precision=prec, recall=float(mm.mean()), bias_nm=bias, rows=rows,
                                  locs_per_um=float(nloc / tube_um), correct_locs_per_um=float(nloc * prec / tube_um))
        np.savez(f'{out}/locs_{nm}.npz', x=x[lin], y=y[lin], z=z[lin], w=w[lin], match=ml[lin] >= 0)
        r24 = [r for r in rows if r['lo'] == 2000][0]
        print(f"{nm:14s} locs {nloc:7d} prec {prec:.3f} recall {mm.mean():.3f} | 2-4k: recall {r24['recall']:.2f} "
              + (f"rob {r24['x_rob']:.1f}/{r24['y_rob']:.1f}/{r24['z_rob']:.1f} RMS {r24['x_rms']:.1f}/{r24['y_rms']:.1f}/{r24['z_rms']:.1f} "
                 f"CRLB {r24['crlb'][0]:.1f}/{r24['crlb'][1]:.1f}/{r24['crlb'][2]:.1f} fringe {r24['fringe']:.3f}" if 'x_rob' in r24 else '(< 30 matched)'), flush=True)
    np.savez(f'{out}/locs_gt.npz', x=Gx, y=Gy, z=Gz, w=GA['w'][gsel], N=GN)
    # ---- tube cross-section profiles: locs relative to the nearest GT axis point (3D), away from crossings
    t = AX[:, 3:6]; zh = np.array([0.0, 0.0, 1.0])
    n1 = np.cross(t, zh); n1 /= np.linalg.norm(n1, axis=1, keepdims=True); n2 = np.cross(t, n1)
    n2 *= np.sign(n2[:, 2:3])
    fil = AX[:, 6].astype(int)
    cross = np.zeros(len(AX), bool)
    for f_ in np.unique(fil):
        tr = cKDTree(AX[fil != f_, :2]); dd, _ = tr.query(AX[fil == f_, :2]); cross[fil == f_] = dd < 0.3
    usable = axin & ~cross
    tr3 = cKDTree(AX[:, :3])
    prof = {}
    for nm in ['gt'] + list(S):
        L = np.load(f'{out}/locs_{nm}.npz')
        P3 = np.c_[L['x'] * PX, L['y'] * PX, L['z']]
        dd, ii = tr3.query(P3)
        okp = (dd < 0.12) & usable[ii]
        dv = P3[okp] - AX[ii[okp], :3]
        u = (dv * n1[ii[okp]]).sum(1) * 1000; v = (dv * n2[ii[okp]]).sum(1) * 1000
        prof[nm] = (u, v)
        np.savez(f'{out}/prof_{nm}.npz', u=u, v=v, fil=fil[ii[okp]])
    rr = float(ax_['r']) * 1000
    def dip(e):
        h, b = np.histogram(e, np.arange(-80, 80.1, 4.0)); c = 0.5 * (b[1:] + b[:-1])
        h = np.convolve(h, [0.25, 0.5, 0.25], 'same')
        pk = max(h[(c < -rr * 0.5) & (c > -rr * 1.6)].max(), 1e-9), max(h[(c > rr * 0.5) & (c < rr * 1.6)].max(), 1e-9)
        return float(h[np.abs(c) < 4].mean() / np.mean(pk))
    for nm, (u, v) in prof.items():
        rho = np.hypot(u, v)
        res['methods'].setdefault(nm, {})['tube'] = dict(n=int(len(u)), dip_lat=dip(u) if len(u) > 100 else np.nan,
                                                         dip_ax=dip(v) if len(u) > 100 else np.nan,
                                                         radius_med=float(np.median(rho)) if len(u) else np.nan,
                                                         radius_rsd=float(1.4826 * np.median(np.abs(rho - np.median(rho)))) if len(u) else np.nan)
    # ---- FRC (x-y and x-z projection), halves = alternating blocks of 20 windows
    frcs = {}
    for nm in ['gt'] + list(S):
        L = np.load(f'{out}/locs_{nm}.npz')
        if len(L['x']) < 200: continue
        h = (L['w'] // 20) % 2 == 0
        X = (L['x'] - LO) * NM; Y = (L['y'] - LO) * NM; Z = (L['z'] + 0.35) * 1000
        ext = (HI - LO) * NM
        f, c, rxy = frc(X[h], Y[h], X[~h], Y[~h], ext, ext)
        f2, c2, rxz = frc(X[h], Z[h], X[~h], Z[~h], ext, 700.0)
        frcs[nm] = dict(f=f, c=c, f_xz=f2, c_xz=c2)
        res['methods'].setdefault(nm, {})['frc_nm'] = dict(xy=rxy, xz=rxz)
    np.savez(f'{out}/frc.npz', **{f'{k}__{kk}': vv for k, v in frcs.items() for kk, vv in v.items()})
    json.dump(res, open(f'{out}/eval.json', 'w'), indent=1, default=float)
    for nm, r in res['methods'].items():
        print(f"{nm:14s} tube: n {r['tube']['n']} dip lat {r['tube']['dip_lat']:.2f} ax {r['tube']['dip_ax']:.2f} radius {r['tube']['radius_med']:.1f} "
              f"(rsd {r['tube']['radius_rsd']:.1f}) | FRC xy/xz {r.get('frc_nm', {}).get('xy', np.nan):.1f}/{r.get('frc_nm', {}).get('xz', np.nan):.1f} nm")
    print(f'total {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
