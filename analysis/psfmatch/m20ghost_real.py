"""m20ghost (2026-11-13): do the m25 detector ghosts (seen on 20 nM-PSF simulations) survive the paper selection on REAL 20nM_2 data?
Tables (pass 2, window starts frame < 5994 = windows inside the first 6000 frames):
  m25  out/psfmatch/m20check/real6k_m25/locs_p2.npz        (m25 set, production flags, --nframes 6000; strict_u from these 6000 frames)
  mix  out/psfmatch/m20check/ref6k_mix_locs_p2.npz         (subset of out/paper2/20nM_2/v2/locs_p2.npz, mixture set; stored strict_u
                                                             = tile ranks over the full 120k-frame movie) and mix6k = same rows with
                                                             strict_u recomputed on these 6000 frames (simcode.strict.strict_product on
                                                             the un-rescaled sigma), the like-for-like comparison with m25.
Test: parents = rows of a class with window photons in a bin; neighbours = other rows of the same class in the same window at 2-8 px.
Counts in small disks at the ghost offsets vs their three mirrors (point -v, x-mirror, y-mirror); excess = n - mean(mirrors),
Poisson error sqrt(n + sum(mirrors) / 9). Ghost offsets: 'A' = (+1.5, +2.4) px r 1.5 (mean offset of the dense-sim ghosts, as
specified), 'L1' = (+3.9, +4.3) px r 1.0 (the main ghost lobe of the isolated-emitter sims, fe_m20a iso s2: 69 % of the ghosts, std
0.2-0.4 px), 'L2' = (-2.1, -0.5) r 1.0, 'L3' = (-2.3, +6.1) r 1.0 (minor lobes). Also: neighbours with photons < 0.15 x parent
(ghosts carry ~6 % of the parent's photons), and the strongest point-asymmetric 1-px bins of the offset map.
usage: python analysis/psfmatch/m20ghost_real.py  ->  out/psfmatch/m20check/ghost_real.txt / .json"""
import json, numpy as np
from scipy.spatial import cKDTree
from simcode.util import proctitle
from simcode import strict as SP
proctitle.set_title('m20ghost')
O = 'out/psfmatch/m20check'
FMAX = 6000 - 6
BINS = [(1000, 2000), (2000, 4000), (4000, 8000), (8000, 1e9), (4000, 1e9)]
REG = dict(A=((1.5, 2.4), 1.5), L1=((3.9, 4.3), 1.0), L2=((-2.1, -0.5), 1.0), L3=((-2.3, 6.1), 1.0))
L = []; RES = {}
def pr(s=''): print(s, flush=True); L.append(s)


def load(path, recompute_u=None):
    Z = np.load(path); f = Z['frame']; k = f < FMAX
    T = {n: Z[n][k] for n in ('frame', 'x', 'y', 'sx', 'sy', 'photons', 'p', 'good', 'sel', 'strict_u', 'strict')}
    if recompute_u:
        g = json.load(open(recompute_u)); u0 = T['strict_u'].astype(float); fu = np.isfinite(u0)
        sig = 0.5 * (T['sx'] + T['sy']).astype(np.float64)
        sig[fu] /= np.interp(u0[fu], g['u_mid'], g['g'])            # undo the sig_u rescale -> the sigma the ranks were built on
        u, st, info = SP.strict_product(T['x'], T['y'], sig, T['good'].astype(bool), float(Z['px_nm']), frame=T['frame'], tag='est2')
        T['strict_u'] = u.astype(np.float32); T['strict'] = st; T['_info'] = info
    return T


def ghost_test(T, cls, k):
    fr, x, y, ph = T['frame'][k].astype(np.int64), T['x'][k], T['y'][k], T['photons'][k]
    tr = cKDTree(np.c_[x, y, fr * 1e4])
    out = {}
    for lo, hi in BINS:
        par = np.nonzero((ph >= lo) & (ph < hi))[0]
        key = f'{lo}-{int(hi) if hi < 1e9 else "inf"}'
        if not len(par): out[key] = dict(n_par=0); continue
        nb = tr.query_ball_point(np.c_[x[par], y[par], fr[par] * 1e4], 8.0)
        ip = np.repeat(par, [len(v) for v in nb]); jn = np.concatenate([np.asarray(v, int) for v in nb]) if len(par) else np.zeros(0, int)
        dx, dy = x[jn] - x[ip], y[jn] - y[ip]; r = np.hypot(dx, dy)
        m = (r >= 2) & (r <= 8); dx, dy, rat = dx[m], dy[m], ph[jn[m]] / ph[ip[m]]
        R = dict(n_par=int(len(par)), n_annulus=int(m.sum()), annulus_per_parent=float(m.sum() / len(par)))
        for faint in (False, True):
            sel = rat < 0.15 if faint else np.ones(len(dx), bool)
            ddx, ddy = dx[sel], dy[sel]
            for nm, ((cx, cy), rr) in REG.items():
                c = lambda a, b: int((np.hypot(ddx - a, ddy - b) <= rr).sum())
                n, mir = c(cx, cy), [c(-cx, -cy), c(cx, -cy), c(-cx, cy)]
                exc = n - np.mean(mir); err = np.sqrt(n + sum(mir) / 9)
                R[f'{nm}{"_faint" if faint else ""}'] = dict(n=n, point=mir[0], xmir=mir[1], ymir=mir[2], excess=float(exc), err=float(err),
                                                            excess_per_parent=float(exc / len(par)), err_per_parent=float(err / len(par)),
                                                            excess_vs_point=float(n - mir[0]), err_vs_point=float(np.sqrt(n + mir[0])))
        # strongest point-asymmetric 1-px bins (orientation-free search for any ghost lobe)
        H, _, _ = np.histogram2d(dy, dx, bins=16, range=[[-8, 8], [-8, 8]]); D = H - H[::-1, ::-1]; S = D / np.sqrt(H + H[::-1, ::-1] + 1)
        top = np.argsort(S.ravel())[::-1][:3]
        R['top_asym_bins'] = [dict(dx=float(-8 + j % 16 + 0.5), dy=float(-8 + j // 16 + 0.5), n=int(H.ravel()[j]), n_point=int(H[::-1, ::-1].ravel()[j]),
                                   z=float(S.ravel()[j])) for j in top]
        out[key] = R
    return out


def phot_summary(T, k):
    ph = T['photons'][k]
    return dict(n=int(k.sum()), pct=dict(zip(('p10', 'p50', 'p90', 'p99', 'p99.9'), np.percentile(ph, [10, 50, 90, 99, 99.9]).round(0).tolist())) if k.any() else {},
                n_ge4000=int((ph >= 4000).sum()), n_ge8000=int((ph >= 8000).sum()))


tabs = dict(m25=load(f'{O}/real6k_m25/locs_p2.npz'), mix=load(f'{O}/ref6k_mix_locs_p2.npz'),
            mix6k=load(f'{O}/ref6k_mix_locs_p2.npz', recompute_u='models/calib/sigu/sigu_p2_new.json'))
pr(f'# m20ghost real 20nM_2, pass 2, window starts < {FMAX} (first 6000 frames); {len(np.unique(tabs["m25"]["frame"]))} / {len(np.unique(tabs["mix"]["frame"]))} window starts (m25 / mix)')
if '_info' in tabs['mix6k']: pr(f'# mix6k strict recomputed on 6k frames: q {tabs["mix6k"]["_info"]["q"]:.3f}, density {tabs["mix6k"]["_info"].get("density_per_um2_per_start", 0):.3f}/um2/start')
pr('\n## rows and photons (window photons)')
for nm, T in tabs.items():
    u = T['strict_u']; cl = dict(all=np.ones(len(u), bool), sel=T['sel'].astype(bool), good=T['good'].astype(bool), u045=np.isfinite(u) & (u <= 0.45), strict=T['strict'].astype(bool))
    RES.setdefault(nm, {})['rows'] = {c: phot_summary(T, k) for c, k in cl.items()}
    for c, s in RES[nm]['rows'].items():
        pr(f'{nm:6s} {c:6s} rows {s["n"]:8d}  photons p10/50/90/99/99.9 {list(s["pct"].values())}  >=4000 {s["n_ge4000"]:6d}  >=8000 {s["n_ge8000"]:5d}')
pr('\n## ghost test (counts of neighbours in the ghost disk / point mirror / x-mirror / y-mirror; excess = n - mean(mirrors) +- Poisson; per parent)')
for nm, T in tabs.items():
    u = T['strict_u']
    for cls, k in (('good', T['good'].astype(bool)), ('u045', np.isfinite(u) & (u <= 0.45))):
        if nm == 'mix6k' and cls == 'good': continue
        R = ghost_test(T, cls, k); RES[nm][f'ghost_{cls}'] = R
        for b, r in R.items():
            if not r.get('n_par'): continue
            s = f'{nm:6s} {cls:5s} parents {b:10s} n {r["n_par"]:6d} annulus/parent {r["annulus_per_parent"]:.3f}'
            for reg in ('A', 'L1', 'L2', 'L3', 'A_faint', 'L1_faint'):
                q = r[reg]; s += f' | {reg} {q["n"]}/{q["point"]}/{q["xmir"]}/{q["ymir"]} exc {q["excess_per_parent"]:+.4f}+-{q["err_per_parent"]:.4f}'
            pr(s)
            pr('        top point-asymmetric bins: ' + '; '.join(f'({t["dx"]:+.1f},{t["dy"]:+.1f}) {t["n"]} vs {t["n_point"]} z {t["z"]:.1f}' for t in r['top_asym_bins']))
json.dump(RES, open(f'{O}/ghost_real.json', 'w'), indent=1, default=float)
open(f'{O}/ghost_real.txt', 'w').write('\n'.join(L) + '\n')
