"""m20check part 2 (2026-11-12): the m25 network set on the standard gate scenes rendered with the averaged 20 nM in-situ PSF
(out/psfmatch/gate_m20a; front-end dumps in out/psfmatch/m20check/fe_m20a = links to out/psfmatch/fe_m20a + any new dumps).
Same procedure as analysis/psfmatch/accept3.py / analysis/psf3d_nets/accept.py (plib: detector-matched targets, explicit pattern solve,
RMS / CRLB_SF_multi, FOV bootstrap), run on the CPU (plib.solve dev='cpu'; no GPU use).
Fronts (those whose dumps exist are used):
  s1 / s2   m25 set pass 1 / 2 with its production sim calibration        s1r / s2r  the same dumps, calibration refit on these scenes
  x1 / x2   retired mixture set (its training mixture contained the 20 nM PSF: the nearest thing to a 20 nM-matched set)   x1r / x2r refit
  r1 / r2   m1809 set (for scale)                                          r1r / r2r  refit
  c1 / c2   the old default set (0.5 nM PSF)
Refit = analysis/psfmatch/fit_cal_set.py procedure (plib.fit_cal on tuning seed 202, scene rs25qm, prior_off from iso), written to
out/psfmatch/m20check/calibs/.  Evaluation seeds 301-303.
Reference for "what a matched set would reach": the m25 set on its OWN PSF scenes (out/psfmatch/fe_m25a, s1 / s2), unpaired.
usage: python analysis/psfmatch/m20check_gates.py -> out/psfmatch/m20check/gates_m20a.txt / .json"""
import functools, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'calib')); sys.path.insert(0, os.path.join(HERE, '..', 'psf3d_nets'))
import plib as P
import accept as A
from simcode2 import config
from simcode2.util import proctitle

proctitle.set_title('m20check')
import torch; torch.set_num_threads(4)
P.solve = functools.partial(P.solve, dev=os.environ.get('M20_DEV', 'cpu'))          # fit_cal / run_front look the name up in the module
O = os.path.join(config.root('out'), 'psfmatch'); OUT = f'{O}/m20check'; FE = f'{OUT}/fe_m20a'; CD = f'{OUT}/calibs'
SCN = os.environ.get('M20_SCN', 'iso,rs25qm,pair1,pair2').split(',')   # one scene per process (CPU solve of all candidates is slow)
SUF = '_' + '_'.join(SCN) if 'M20_SCN' in os.environ else ''
HEAVY = ('s1', 'x1', 'c1', 's2', 'x2', 'r1', 'r2')   # fronts with the detection sweep (all candidates solved); others precision only
CAL = dict(c1='calib/calib_sc_hd_sc5nbb15_pm_bright.json', c2='calib/calib_p2L.json', x1='calib/calib_sc_hd_psf3dmix.json',
           x2='calib/calib_p2L_psf3dmix.json', s1='calib/calib_sc_hd_m25.json', s2='calib/calib_p2L_m25.json',
           r1='calib/calib_sc_hd_m1809.json', r2='calib/calib_p2L_m1809.json')
BASE = {'1': 'calib/calib_sc_hd_sc5nbb15_pm_bright.json', '2': 'calib/calib_p2L.json'}
BINS = ['200-2000', '200-400', '400-700', '700-1200', '1200-2000', '2000-4000']
COMP = [('s1', 'x1'), ('s1r', 'x1r'), ('s1', 'c1'), ('r1', 's1'), ('r1r', 's1r'), ('s2', 'x2'), ('s2r', 'x2r'), ('s2', 'c2'), ('r2', 's2'), ('r2r', 's2r'),
        ('s2', 's1')]


def have(d, f):
    return os.path.exists(f'{d}/iso__s301_{f}.npz')


def refit(front):
    fn = f'{CD}/calib_{front}r_m20a.json'
    if not os.path.exists(fn):
        B = json.load(open(config.resolve(BASE[front[1]])))
        Rs = [P.load('rs25qm', front, seeds=(202,), d=FE)]; iso = P.load('iso', front, seeds=(202,), d=FE)
        cal = P.fit_cal(Rs, B, iso=iso)
        cal['note'] = f'm20check 2026-11-12: plib.fit_cal of front {front} on seed 202 of the averaged 20 nM PSF gate scenes (rs25qm; prior_off from iso); base {BASE[front[1]]}'
        os.makedirs(CD, exist_ok=True); json.dump(cal, open(fn, 'w'), indent=1)
    return json.load(open(fn))


def own(R, E, nboot=200):
    """per-front stats on its own matched targets (resolvable: finite CRLB < 25 nm), FOV-bootstrap CI of RMS / CRLB"""
    ti = E['ti']; c = R['gt_sf_m'][ti]; ok = np.isfinite(c).all(1) & (np.hypot(*c.T) / np.sqrt(2) * P.PX < 25) & np.isfinite(E['e']).all(1)
    ph = R['gt_phot'][ti]; fov = R['gt_fov'][ti]; out = {}
    for b in BINS:
        lo, hi = [float(v) for v in b.split('-')]
        m = ok & (ph >= lo) & (ph < hi)
        if m.sum() < 20: continue
        s = P.stats(E['e'][m], c[m], E['s'][m])
        bs = [P.rr(E['e'][m], c[m], w) for w in P.boot_w(fov[m], nboot)]
        s['ci'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]; s['se'] = float(np.std(bs))
        s['crlb_nm'] = float(np.sqrt(np.mean(c[m] ** 2)) * P.PX)
        out[b] = s
    return out


def run_light(sc, f, d, cal):
    R = P.load(sc, f, d=d); ti, oi = P.targets(R)
    return R, P.errs(R, ti, oi, P.solve(R, oi, cal)), None


def run_set(d, cals, scn, heavy=()):
    F = {}
    for sc in scn:
        F[sc] = {}
        for f, cal in cals.items():
            F[sc][f] = A.run_front(sc, f, d, cal) if f in heavy else run_light(sc, f, d, cal)
            print('solved', os.path.basename(d), sc, f, flush=True)
    return F


def main():
    # links <scene>__s<seed>_<f>r.npz -> the same dump (only the calibration differs)
    cals = {}
    for f, cp in CAL.items():
        if not have(FE, f): continue
        cals[f] = json.load(open(config.resolve(cp)))
        if f[0] in 'sxr':
            cals[f + 'r'] = refit(f)
            for fn in os.listdir(FE):
                if fn.endswith(f'_{f}.npz'):
                    ln = os.path.join(FE, fn[:-len(f'_{f}.npz')] + f'_{f}r.npz')
                    if not os.path.lexists(ln): os.symlink(fn, ln)
    F = run_set(FE, cals, SCN, HEAVY)
    res = dict(fronts=sorted(cals), m20a={}, paired={}, ref_m25a={}, ref_m1809a={})
    L = [f'# m20check gates: averaged 20 nM PSF scenes (gate_m20a), eval seeds 301-303; fronts {sorted(cals)}']
    L.append('\n## per front on its own matched targets (RMS / CRLB_SF_multi with FOV-bootstrap 95 % CI; detection at score >= 0.5; recall@FPx = recall at the FP count of x of the same pass)')
    for sc in SCN:
        res['m20a'][sc] = {}
        for f in sorted(cals):
            R, E, S = F[sc][f]
            st = own(R, E)
            ref = 'x' + f[1] + ('r' if f.endswith('r') else '')
            fpx = A.det_sweep(F[sc][ref][0], F[sc][ref][2])['at0.5']['FP'] if ref in F[sc] and F[sc][ref][2] is not None else None
            NOD = dict(recall=np.nan, FP=-1, fp_per_fov=np.nan)
            dt = A.det_sweep(R, S, fp_ref=fpx) if S is not None else {'at0.5': NOD}
            res['m20a'][sc][f] = dict(stats=st, det=dt)
            a = st.get('200-2000', {}); h = st.get('2000-4000', {}); r5 = dt['at0.5']; rf = dt.get('at_fp_ref')
            L.append(f'{sc:7s} {f:4s} n {a.get("n", 0):5d}  RMS/CRLB 200-2000 {a.get("rms", np.nan):.3f} [{a.get("ci", [np.nan] * 2)[0]:.3f}, {a.get("ci", [np.nan] * 2)[1]:.3f}]  '
                     f'nm {a.get("nm", np.nan):6.2f}  zcal {a.get("zcal", np.nan):.2f}  c5 {a.get("c5", np.nan):5.2f} %  | 2000-4000 {h.get("rms", np.nan):.3f}  '
                     f'| recall {r5["recall"]:.4f} FP {r5["FP"]:5d} ({r5["fp_per_fov"]:.2f}/FOV)' + (f'  recall@FPx {rf["recall"]:.4f}' if rf else ''))
    L.append('\n## paired (common targets): A - B of RMS / CRLB, 200-2000 ph (and 2000-4000), 95 % CI')
    for sc in SCN:
        res['paired'][sc] = {}
        for a, b in COMP:
            if a not in F[sc] or b not in F[sc]: continue
            pr = P.paired(F[sc][b][0], F[sc][a][1], F[sc][b][1]); res['paired'][sc][f'{a}_vs_{b}'] = pr
            v = pr.get('200-2000'); h = pr.get('2000-4000')
            if v: L.append(f'{sc:7s} {a:4s} - {b:4s} n {v["A"]["n"]:5d}  A {v["A"]["rms"]:.3f} B {v["B"]["rms"]:.3f}  d {v["d"]:+.3f} [{v["ci"][0]:+.3f}, {v["ci"][1]:+.3f}]'
                           + (f'  | 2000-4000 d {h["d"]:+.3f} [{h["ci"][0]:+.3f}, {h["ci"][1]:+.3f}]' if h else ''))
    # reference: the sets on their own PSF scenes (unpaired)
    for tag, key, fr, scn0 in (('m25a', 'ref_m25a', dict(s1=CAL['s1'], s2=CAL['s2']), SCN), ('m1809a', 'ref_m1809a', dict(s1='calib/calib_sc_hd_m1809.json', s2='calib/calib_p2L_m1809.json'), ['iso', 'pair1', 'pair2'])):
        d = f'{O}/fe_{tag}'; scn = [q for q in scn0 if q in SCN]
        G = run_set(d, {f: json.load(open(config.resolve(c))) for f, c in fr.items() if have(d, f)}, scn)
        L.append(f'\n## reference: the matched set on its own PSF scenes ({tag}), same statistics')
        for sc in scn:
            res[key][sc] = {}
            for f in G[sc]:
                R, E, S = G[sc][f]; st = own(R, E); dt = {'at0.5': dict(recall=np.nan, FP=-1, fp_per_fov=np.nan)}; res[key][sc][f] = dict(stats=st, det=dt)
                a = st.get('200-2000', {}); h = st.get('2000-4000', {}); r5 = dt['at0.5']
                L.append(f'{sc:7s} {f:4s} n {a.get("n", 0):5d}  RMS/CRLB 200-2000 {a.get("rms", np.nan):.3f} [{a.get("ci", [np.nan] * 2)[0]:.3f}, {a.get("ci", [np.nan] * 2)[1]:.3f}]  '
                         f'nm {a.get("nm", np.nan):6.2f}  CRLB nm {a.get("crlb_nm", np.nan):5.2f}  zcal {a.get("zcal", np.nan):.2f}  c5 {a.get("c5", np.nan):5.2f} %  | 2000-4000 {h.get("rms", np.nan):.3f}  '
                         f'| recall {r5["recall"]:.4f} FP {r5["FP"]:5d} ({r5["fp_per_fov"]:.2f}/FOV)')
    L.append('\n## m25 set: 20 nM PSF scenes minus its own (2.5 nM) PSF scenes, RMS / CRLB (unpaired; +- 2 se from the two FOV bootstraps) = the most a 20 nM-matched set could gain')
    res['gap'] = {}
    for sc in SCN:
        for f in ('s1', 's1r', 's2', 's2r'):
            f0 = f[:2]
            if f not in res['m20a'][sc] or f0 not in res['ref_m25a'].get(sc, {}): continue
            for b in BINS:
                a = res['m20a'][sc][f]['stats'].get(b); r = res['ref_m25a'][sc][f0]['stats'].get(b)
                if not a or not r: continue
                g = dict(d=a['rms'] - r['rms'], se=float(np.hypot(a['se'], r['se'])), on20=a['rms'], own=r['rms'], crlb20=a['crlb_nm'], crlb25=r['crlb_nm'],
                         nm20=a['nm'], nm25=r['nm'])
                res['gap'][f'{sc}|{f}|{b}'] = g
                if b in ('200-2000', '2000-4000'):
                    L.append(f'{sc:7s} {f:4s} {b:9s}  on 20 nM {g["on20"]:.3f} (CRLB {g["crlb20"]:.2f} nm, RMS {g["nm20"]:.2f} nm)  own {g["own"]:.3f} (CRLB {g["crlb25"]:.2f} nm, RMS {g["nm25"]:.2f} nm)  '
                             f'd {g["d"]:+.3f} +- {2 * g["se"]:.3f}')
            a = res['m20a'][sc][f]['det']['at0.5']; r = res['ref_m25a'][sc][f0]['det']['at0.5']
            L.append(f'{sc:7s} {f:4s} detection  on 20 nM recall {a["recall"]:.4f} FP/FOV {a["fp_per_fov"]:.2f}   own recall {r["recall"]:.4f} FP/FOV {r["fp_per_fov"]:.2f}')
    json.dump(res, open(f'{OUT}/gates_m20a{SUF}.json', 'w'), indent=1, default=float)
    open(f'{OUT}/gates_m20a{SUF}.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
