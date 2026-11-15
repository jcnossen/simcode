"""Covariance ablation, part 1: what does the head's 6 x 6 intensity covariance look like, and is its correlation structure real?
Gate scenes (GT): dumps out/psfmatch/fe_m1809a (+ out/ablation_cov/fe_m1809a for the dense scene), fronts s1 (pass 1) / s2 (pass 2),
seeds 301-303, detector-matched targets (plib.targets). Real: captured pass-2 head outputs of 1809_2 (capture.py), good rows (p >= 0.63).
Frames are re-ordered per row to (angle 1 steps 1-3, angle 2 steps 1-3) with the pattern's angle index (windows starting at frame % 6 == 3
have the angles swapped). Per group (scene / front / isolated vs neighbour within 3 px / photon bin):
  Rpred   mean of the per-emitter predicted correlation matrices  C_ij = S_ij / sqrt(S_ii S_jj), S = L L^T
  Remp    correlation of the actual errors normalised by the predicted sd, z_t = (I_t - I_t,true) / sqrt(S_tt): E[z_i z_j] / sqrt(E z_i^2 E z_j^2)
          (rows with max |z| > 6 dropped; gate only), with a 20-block jackknife se of the mean within- / cross-angle off-diagonal
  off-diagonal coefficients: histogram of all 15 per emitter, per-emitter mean within-angle (6 pairs) and cross-angle (9 pairs)
  mode    top eigenvector of the predicted correlation matrix: variance fraction, |cos| with the uniform vector (additive common mode:
          background) and with the intensity vector I / |I| in sd units (multiplicative common mode: position / PSF-shape / brightness)
  whitening check (gate): sd of L^-1 e (full) and its residual correlation, vs the diagonal-only normalisation
-> out/ablation_cov/cov_desc.json / cov_desc.npz"""
import glob, json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'analysis', 'calib'))
import plib as P
from simcode.solve import pattern as C
from simcode.util import proctitle
proctitle.set_title('covabl')
OUT = os.path.join(REPO, 'out', 'ablation_cov')
PHB = [(200, 700), (700, 2000), (2000, 1e9)]
WI = np.zeros((6, 6), bool); WI[:3, :3] = True; WI[3:, 3:] = True; np.fill_diagonal(WI, False)
CR = np.zeros((6, 6), bool); CR[:3, 3:] = True; CR[3:, :3] = True
IU = np.triu_indices(6, 1)
HB = np.linspace(-1, 1, 81)


def reorder(ang):
    """stable per-row frame order: angle 0 frames first"""
    return np.argsort(ang, axis=1, kind='stable')


def take(A, o):
    if A.ndim == 2: return np.take_along_axis(A, o, 1)
    A = np.take_along_axis(A, o[:, :, None], 1)
    return np.take_along_axis(A, o[:, None, :], 2)


def group_stats(Sg, I, e=None):
    """Sg [n,6,6] covariance (frames re-ordered), I [n,6] network intensities, e [n,6] errors or None"""
    n = len(Sg); d = np.sqrt(np.einsum('nii->ni', Sg)); Cm = Sg / (d[:, :, None] * d[:, None, :])
    r = dict(n=int(n), Rpred=Cm.mean(0), within=float(Cm[:, WI].mean()), cross=float(Cm[:, CR].mean()),
             within_q=np.percentile(Cm[:, WI].mean(1), [5, 25, 50, 75, 95]), cross_q=np.percentile(Cm[:, CR].mean(1), [5, 25, 50, 75, 95]),
             hist_within=np.histogram(Cm[:, WI].ravel(), HB)[0], hist_cross=np.histogram(Cm[:, CR].ravel(), HB)[0],
             frac_abs_gt02=float((np.abs(Cm[:, IU[0], IU[1]]) > 0.2).mean()), sd_rel=float(np.median(d.sum(1) / np.clip(I.sum(1), 1, None))))
    w, V = np.linalg.eigh(Cm); v = V[:, :, -1]
    u = np.ones(6) / np.sqrt(6); zi = np.clip(I, 0, None) / d; zi = zi / np.linalg.norm(zi, axis=1, keepdims=True).clip(1e-9)
    r.update(top_frac=float((w[:, -1] / 6).mean()), cos_uniform=float(np.abs(v @ u).mean()), cos_intensity=float(np.abs((v * zi).sum(1)).mean()))
    # additive vs multiplicative common mode: pooled least squares of the off-diagonal covariances S_ij on [1, I_i I_j]
    # -> a (ph^2: variance of an error common to all frames, e.g. background) and b^2 (relative variance of a common factor)
    Ip = np.clip(I, 0, None); X = (Ip[:, IU[0]] * Ip[:, IU[1]]).ravel(); Y = Sg[:, IU[0], IU[1]].ravel()
    A = np.c_[np.ones(len(X)), X]; (a_, b2_), *_ = np.linalg.lstsq(A, Y, rcond=None)
    r.update(add_var=float(a_), mult_var=float(b2_), add_share=float(a_ / max(Y.mean(), 1e-9)))
    if e is not None:
        Ye = (e[:, IU[0]] * e[:, IU[1]]); oke = np.abs(e / d).max(1) < 6
        (ae, be), *_ = np.linalg.lstsq(A[np.repeat(oke, 15)], Ye[oke].ravel(), rcond=None)
        r.update(emp_add_var=float(ae), emp_mult_var=float(be))
    if e is not None:
        z = e / d; ok = np.abs(z).max(1) < 6
        r['frac_z6'] = float(1 - ok.mean()); z = z[ok]
        def emp(z):
            M = z.T @ z / len(z); s = np.sqrt(np.diag(M)); return M / np.outer(s, s), s
        Re, s = emp(z); r['Remp'] = Re; r['z_sd'] = s; r['emp_within'] = float(Re[WI].mean()); r['emp_cross'] = float(Re[CR].mean())
        blk = np.arange(len(z)) % 20; jw, jc = [], []
        for b in range(20):
            Rb, _ = emp(z[blk != b]); jw.append(Rb[WI].mean()); jc.append(Rb[CR].mean())
        r['emp_within_se'] = float(np.sqrt(19 / 20 * np.sum((np.array(jw) - np.mean(jw)) ** 2))); r['emp_cross_se'] = float(np.sqrt(19 / 20 * np.sum((np.array(jc) - np.mean(jc)) ** 2)))
        # predicted on the same rows
        Co = Cm[ok]; r['pred_within_ok'] = float(Co[:, WI].mean()); r['pred_cross_ok'] = float(Co[:, CR].mean()); r['Rpred_ok'] = Co.mean(0)
        # full whitening
        Lc = np.linalg.cholesky(Sg[ok] + 1e-6 * np.eye(6)); wz = np.linalg.solve(Lc, e[ok][..., None])[..., 0]
        k2 = np.abs(wz).max(1) < 6; wz = wz[k2]
        Mw = wz.T @ wz / len(wz); sw = np.sqrt(np.diag(Mw)); Rw = Mw / np.outer(sw, sw)
        r['white_sd'] = float(sw.mean()); r['white_rob_sd'] = float(1.4826 * np.median(np.abs(wz))); r['white_offdiag_rms'] = float(np.sqrt((Rw[IU] ** 2).mean()))
        r['diag_rob_sd'] = float(1.4826 * np.median(np.abs(z))); r['diag_offdiag_rms'] = float(np.sqrt((Re[IU] ** 2).mean()))
        # quadratic form (chi2_6) calibration: median of e^T S^-1 e vs the chi2_6 median 5.348 for full and diagonal-only S
        r['q_full_med'] = float(np.median((wz ** 2).sum(1))); r['q_diag_med'] = float(np.median((z ** 2).sum(1)))
        r['q_full_p90'] = float(np.percentile((wz ** 2).sum(1), 90)); r['q_diag_p90'] = float(np.percentile((z ** 2).sum(1), 90))   # chi2_6 p90 = 10.64
    return r


def splits(nn, ph):
    out = {}
    for sn, ms in (('all', np.ones(len(nn), bool)), ('iso', ~(nn < 3.0)), ('nbr', nn < 3.0)):
        for lo, hi in PHB + [(200, 1e9)]:
            out[f'{sn}|{lo:g}-{hi:g}'] = ms & (ph >= lo) & (ph < hi)
    return out


def gate(sc, front):
    d = f'{OUT}/fe_m1809a' if sc.startswith('rs25') else os.path.join(REPO, 'out', 'psfmatch', 'fe_m1809a')
    seeds = tuple(s for s in (301, 302, 303) if os.path.exists(f'{d}/{sc}__s{s}_{front}.npz'))
    if not seeds: return None
    R = P.load(sc, front, seeds=seeds, d=d); ti, oi = P.targets(R)
    ang = C.angle_k(torch.tensor(R['patt_est'], dtype=torch.float32))[1].numpy()[R['sc_b'][oi].astype(int)]
    o = reorder(ang)
    L = R['sc_L'][oi].astype(np.float64); Sg = take(L @ L.transpose(0, 2, 1), o)
    I = take(R['sc_I'][oi].astype(np.float64), o); e = I - take(R['gt_I'][ti].astype(np.float64), o)
    nn = np.nan_to_num(R['gt_nn'][ti], nan=1e9); ph = R['gt_phot'][ti]
    return {k: group_stats(Sg[m], I[m], e[m]) for k, m in splits(nn, ph).items() if m.sum() >= 150}


def real(chunks, pmin=0.63):
    from scipy.spatial import cKDTree
    Sg, I, nn, ph = [], [], [], []
    for fn in chunks:
        H = np.load(fn)
        ws = H['wstart']; x, y = H['x_det'], H['y_det']
        Pts = np.c_[x, y, ws.astype(np.float64) * 1e3]; tr = cKDTree(Pts)
        dd, _ = tr.query(Pts, k=2); dn = dd[:, 1]
        k = np.nonzero(H['p'] >= pmin)[0]
        if len(k) > 150000: k = np.sort(np.random.default_rng(0).choice(k, 150000, replace=False))
        ang = C.angle_k(torch.tensor(H['PW'], dtype=torch.float32))[1].numpy()[H['bw'][k]]
        o = reorder(ang); L = H['o_L'][k].astype(np.float64)
        Sg.append(take(L @ L.transpose(0, 2, 1), o)); I.append(take(H['o_I'][k].astype(np.float64), o)); nn.append(dn[k]); ph.append(H['o_I'][k].sum(1))
    Sg, I, nn, ph = (np.concatenate(v) for v in (Sg, I, nn, ph))
    return {k: group_stats(Sg[m], I[m]) for k, m in splits(nn, ph).items() if m.sum() >= 150}


def main():
    res = {}
    for front in ('s2', 's1'):
        for sc in ('iso', 'pair2', 'pair1', 'rs05qm', 'rs05pm', 'rs25qm'):
            g = gate(sc, front)
            if g is None: continue
            res[f'gate:{sc}:{front}'] = g
            for k in ('all|200-1e+09', 'iso|200-1e+09', 'nbr|200-1e+09', 'all|200-700', 'all|700-2000', 'all|2000-1e+09'):
                if k in g:
                    r = g[k]
                    print(f'{sc:7s} {front} {k:16s} n {r["n"]:6d} pred within {r["within"]:+.3f} cross {r["cross"]:+.3f} | emp within {r["emp_within"]:+.3f}+-{r["emp_within_se"]:.3f} '
                          f'cross {r["emp_cross"]:+.3f}+-{r["emp_cross_se"]:.3f} | top {r["top_frac"]:.2f} cosU {r["cos_uniform"]:.2f} cosI {r["cos_intensity"]:.2f} | '
                          f'add sd {np.sign(r["add_var"])*abs(r["add_var"])**.5:.1f} ph mult {np.sign(r["mult_var"])*abs(r["mult_var"])**.5:.3f} emp {np.sign(r["emp_add_var"])*abs(r["emp_add_var"])**.5:.1f} / {np.sign(r["emp_mult_var"])*abs(r["emp_mult_var"])**.5:.3f} | white sd {r["white_rob_sd"]:.2f} q50 full {r["q_full_med"]:.2f} diag {r["q_diag_med"]:.2f} (5.35) p90 {r["q_full_p90"]:.1f}/{r["q_diag_p90"]:.1f} (10.6)', flush=True)
    ch = sorted(glob.glob(f'{OUT}/head_real/head_*.npz'), key=lambda f: int(f.rsplit('_', 1)[1][:-4]))
    if ch:
        sel = [ch[i] for i in sorted(set(np.linspace(0, len(ch) - 1, 4).round().astype(int)))]
        g = real(sel); res['real:1809_2:s2'] = g; res['real_chunks'] = [os.path.basename(f) for f in sel]
        for k, r in g.items():
            print(f'real s2 {k:16s} n {r["n"]:7d} pred within {r["within"]:+.3f} cross {r["cross"]:+.3f} | top {r["top_frac"]:.2f} cosU {r["cos_uniform"]:.2f} cosI {r["cos_intensity"]:.2f} |r|>0.2 {r["frac_abs_gt02"]:.2f} add sd {np.sign(r["add_var"])*abs(r["add_var"])**.5:.1f} ph mult {np.sign(r["mult_var"])*abs(r["mult_var"])**.5:.3f}', flush=True)
    tolist = lambda v: v.tolist() if isinstance(v, np.ndarray) else v
    J = {k: ({kk: {k3: tolist(v3) for k3, v3 in vv.items()} for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in res.items()}
    json.dump(J, open(f'{OUT}/cov_desc.json', 'w'))
    print('written', f'{OUT}/cov_desc.json')


if __name__ == '__main__':
    main()
