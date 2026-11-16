"""Re-evaluation of the 4Pi density sweep for the SI figure si_4pi_sim (CPU, from the saved per-movie data; no new runs):

  time to fidelity  GT-FRC x-y and x-z sections on the localizations with frame < N (2^(1/4) grid of N, 500 .. T), the
                    same renders / 5-shell smoothed FRC as sweep4pi_eval.py, but the resolution is the INTERPOLATED crossing
                    of the smoothed curve (linear in frequency between the two shells that bracket 1/7) by the sustained rule
                    of analysis/papersim/metrics.frc_sustained (first frequency after which the curve stays below 1/7 for
                    0.01 / nm: the hollow-tube spectral null at 1 / 64 nm, where the curve dips and recovers, is not taken
                    as the resolution), and the resolution-vs-N curve is made non-increasing (running minimum: adding frames
                    cannot lose resolution). A step from ~65 to ~35 nm remains: the hollow-tube spectrum has a null at
                    1 / 64 nm (J0 zero, tube radius 24.6 nm), so the curve crosses 1/7 either before the null or, once the
                    second lobe rises above 1/7, after it; this is the structure, not noise.
  false positives   a kept localization is a false positive when no emitter that is ON in its own window lies within the
                    fixed 3D gate 30 / 30 / 75 nm (x / y / z; = gtfrc.ghost_fraction with sigma 10 / 10 / 25 nm, k = 3), OR
                    when it is matched laterally (simeval4pi.match, 64 nm, per window) to a GT emitter but placed one 4Pi
                    fringe off in z (|dz| > 100 nm). FP fraction = false positives / all kept localizations in the region.
Methods: the figure's network selection (net_sig1 = sigma_xy < 5 nm, sigma_z < 12 nm, every density) and the classical published-style table (cl_pub).
Input:  <work>/sweep4pi/<tag>/locs_<method>.npz (eval's selected region rows), <work>/simmovie/<tag>/gt_*.npz
Output: <work>/sweep4pi/<tag>/ttf_fp.json
usage: python analysis/fourpi_sim/sweep4pi_ttf_fp.py --work <4Pi work folder> [--tags mtc_d01 ...]
"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep4pi_eval as E                       # constants, frc_sections, gtfrc module (E.G), gt_units, match
G = E.G; NM = E.NM
BEST = {t: 'net_sig1' for t in ('mtc_d01', 'mtc_d03', 'mtc_d05', 'mtc_d075', 'mtc_d15')}   # one selection (si_4pi_sim)


def crossing_interp(freq, curve, thr=G.THRESHOLD, start=3, width=0.01):
    f = np.asarray(freq); c = np.asarray(curve)
    if len(c) <= start: return float('nan')
    w = max(int(width / max(f[1] - f[0], 1e-12)), 1)
    for i in range(start, len(c)):
        seg = c[i:min(i + w, len(c))]
        if (seg < thr).all():
            if i == start or c[i - 1] < thr: return float(1 / f[i])
            u = (c[i - 1] - thr) / (c[i - 1] - c[i])                     # linear between shells i-1 (>= thr) and i (< thr)
            return float(1 / (f[i - 1] + u * (f[i] - f[i - 1])))
    return 2 * E.PIX                                                       # never below: Nyquist


def runmin(r):
    r = np.array(r, float); out = r.copy(); m = np.inf
    for i, v in enumerate(r):
        if np.isfinite(v): m = min(m, v)
        out[i] = m if np.isfinite(m) else np.nan
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tags', nargs='*', default=list(BEST))
    p.add_argument('--work', required=True, help='4Pi work folder'); a = p.parse_args()
    P4 = a.work
    for tag in a.tags:
        t0 = time.time()
        d = f'{P4}/simmovie/{tag}'; out = f'{P4}/sweep4pi/{tag}'
        meta = json.load(open(f'{d}/meta.json')); T = meta['T']
        ev = json.load(open(f'{out}/eval.json'))
        grid = [int(round(500 * 2 ** (k / 4))) for k in range(60) if 500 * 2 ** (k / 4) < T] + [T]      # 2^(1/4) grid (eval: 2^(1/2))
        LO, HI = E.LO, E.HI
        inr = lambda x, y, m=0.0: (x >= LO - m) & (x < HI + m) & (y >= LO - m) & (y < HI + m)
        lo, hi = (LO * NM, LO * NM), (HI * NM, HI * NM)
        S = np.load(f'{d}/gt_sites.npz')['sites']; sreg = inr(S[:, 0], S[:, 1])
        GT = np.c_[S[sreg, 0] * NM, S[sreg, 1] * NM, S[sreg, 2] * 1000]
        gimg_xy = G.render(GT[:, :2], lo, hi, E.PIX)
        ys = np.arange(lo[1], hi[1], E.SLAB)
        gimg_xz = [G.render(GT[(GT[:, 1] >= y0) & (GT[:, 1] < y0 + E.SLAB)][:, [0, 2]], (lo[0], E.ZF[0]), (hi[0], E.ZF[1]), E.PIX) for y0 in ys]
        U = E.gt_units(d, T)
        GA = {k: v[inr(U['x'], U['y'], 1.0)] for k, v in U.items()}
        EW = {}
        o = np.argsort(U['w'], kind='stable'); ws = U['w'][o]; cut = np.r_[0, np.nonzero(np.diff(ws))[0] + 1, len(ws)]
        for i0, i1 in zip(cut[:-1], cut[1:]):
            ii = o[i0:i1]; EW[int(ws[i0])] = np.c_[U['x'][ii] * NM, U['y'][ii] * NM, U['z'][ii] * 1000]
        res = dict(tag=tag, dens_fov=ev['dens_fov'], grid=grid, methods={})
        for nm in (BEST[tag], 'cl_pub'):
            L = np.load(f'{out}/locs_{nm}.npz'); P3 = np.c_[L['x'], L['y'], L['z']]; w = L['w'].astype(np.int64)
            n = len(P3)
            # false positives
            gh = np.ones(n, bool)
            if n:
                sg = np.tile([10.0, 10.0, 25.0], (n, 1))
                oo = np.argsort(w, kind='stable'); fs = w[oo]; cc = np.r_[0, np.nonzero(np.diff(fs))[0] + 1, len(fs)]
                for i0, i1 in zip(cc[:-1], cc[1:]):
                    Ew = EW.get(int(fs[i0]))
                    if Ew is None or not len(Ew): continue
                    ii = oo[i0:i1]
                    d2 = (((P3[ii, None, :] - Ew[None]) / sg[ii, None, :]) ** 2).sum(-1)
                    gh[ii] = d2.min(1) > 9.0
                mg, ml = E.match(GA['w'], GA['x'], GA['y'], w, P3[:, 0] / NM, P3[:, 1] / NM)
                wf = np.zeros(n, bool); mm = ml >= 0
                wf[mm] = np.abs(P3[mm, 2] - GA['z'][ml[mm]] * 1000) > 100
            else:
                wf = np.zeros(0, bool); mm = np.zeros(0, bool)
            fp = gh | wf
            r = dict(n_locs=int(n), fp_fraction=float(fp.mean()) if n else float('nan'), fp_gate_only=float(gh.mean()) if n else float('nan'),
                     fp_wrong_fringe_only=float((wf & ~gh).mean()) if n else float('nan'), wrong_fringe_of_kept=float(wf.mean()) if n else float('nan'),
                     check_ghost_fixed_eval=ev['methods'][nm]['ghost_fixed']['ghost_fraction'])
            # time to fidelity
            frm = 6 * w
            for key, fn in (('xy', lambda q: G.gt_frc(q[:, :2], None, lo, hi, E.PIX, 1.0, gt_img=gimg_xy)), ('xz', lambda q: E.frc_sections(q, gimg_xz, lo, hi, E.SLAB))):
                raw = []
                for N in grid:
                    c = fn(P3[frm < N])
                    raw.append(crossing_interp(c['freq'], c['curve']) if len(c['freq']) and c['n'] >= 50 else float('nan'))
                r[f'ttf_{key}_interp'] = raw; r[f'ttf_{key}'] = runmin(raw).tolist()
                r[f'n_by_grid'] = [int((frm < N).sum()) for N in grid]
                r[f'frames_to_{key}'] = E.ttf(r[f'ttf_{key}'], grid, E.TARGETS_XY if key == 'xy' else E.TARGETS_XZ)
            res['methods'][nm] = r
            print(f"{tag} {nm}: n {n}, FP {r['fp_fraction']:.3f} (gate {r['fp_gate_only']:.3f} + fringe-only {r['fp_wrong_fringe_only']:.3f}; eval ghost {r['check_ghost_fixed_eval']:.3f}) "
                  f"xy {np.round(r['ttf_xy'], 0).tolist()} xz {np.round(r['ttf_xz'], 0).tolist()} [{time.time() - t0:.0f}s]", flush=True)
        json.dump(res, open(f'{out}/ttf_fp.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
