"""nr80 evaluation: pass 2 (front s2) minus pass 1 (front s1) on IDENTICAL gate-scene targets, from p2.dump outputs.
Per scene (seeds 301-303 pooled), each front solved with its own nr80 sim calibration (out/nanoruler/models/calib):
  paired RMS / CRLB per GT-photon band on the targets found by both fronts (plib.paired, FOV bootstrap 95 % CI of s2 - s1),
  the position-prior RMS (nm per axis, same targets), recall of the targets at detector score >= 0.5 and >= 0.63 ('good'),
  and the same split by the window's first pattern step (0: starts with orientation 0 = cycle-aligned as the movies; 3).
usage: python ev_p12.py <fe dir> <scenes, comma> [A=s2] [B=s1]  (px 65 nm)"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'calib'))
import plib as P
from simcode.util.proctitle import set_title
set_title('nr-eval'); P.PX = 65.0
MC = '/home/jc/simcode2/out/nanoruler/models/calib'
CAL = dict(s1=f'{MC}/calib_sc_hd_nr80.json', s2=f'{MC}/calib_p2L_nr80.json')
BANDS = [(200, 2000), (800, 2500), (1200, 3500), (2000, 4000)]
d, scen = sys.argv[1], sys.argv[2].split(','); fa = sys.argv[3] if len(sys.argv) > 3 else 's2'; fb = sys.argv[4] if len(sys.argv) > 4 else 's1'
for k in sys.argv[5:]:
    a, b = k.split('='); CAL[a] = b
rms = lambda e: float(np.sqrt(np.mean(e ** 2)) * P.PX)
print(f'# {d}: {fa} (A) minus {fb} (B) on identical targets; calibs {CAL[fa]} / {CAL[fb]}')
for sc in scen:
    F = {}
    for f in (fa, fb):
        R = P.load(sc, f, d=d); ti, oi = P.targets(R); sol = P.solve(R, oi, json.load(open(CAL[f]))); F[f] = (R, P.errs(R, ti, oi, sol))
    Ra, Ea = F[fa]; Rb, Eb = F[fb]
    assert np.allclose(Ra['gt_x'], Rb['gt_x'])
    kx = Rb['patt_true'][:, 0, 0]; ky = Rb['patt_true'][:, 0, 1]; ang0 = (np.abs(kx) > np.abs(ky)).astype(int)     # orientation of frame 0 per FOV
    k0 = np.asarray(json.load(open(os.environ.get('NR_PRESET', '/home/jc/simcode2/out/nanoruler/nr80_preset.json')))['common']['k'])
    a_first = int(abs(k0[0][0]) > abs(k0[0][1])); step0 = np.where(ang0 == a_first, 0, 3)
    sp = np.remainder(Rb['patt_true'][:, 1, 2] - Rb['patt_true'][:, 0, 2], 2 * np.pi)
    print(f'\n== {sc}: targets {int((Rb["gt_target"] > 0).sum())}, FOVs {Rb["n_fov"]}; true phase step frame 1 - frame 0 (mod 2 pi) mean {sp.mean():.3f} rad [{sp.min():.2f}, {sp.max():.2f}]')
    nt = int((Rb['gt_target'] > 0).sum())
    for f, (R, E) in F.items():
        g5 = len(P.targets(R, 0.5)[0]); g63 = len(P.targets(R, 0.63)[0])
        print(f'  {f}: recall of targets at score >= 0.5: {g5 / nt:.4f}, >= 0.63 (good): {g63 / nt:.4f}; candidates {len(R["sc_score"])}')
    fov_step = dict(zip(Rb['pfov'].astype(int), step0))
    for name, selfun in (('all', None), ('step-0 starts', 0), ('step-3 starts', 3)):
        if selfun is None: A_, B_ = Ea, Eb
        else:
            ma = np.array([fov_step[int(v)] == selfun for v in Ra['gt_fov'][Ea['ti']]]); mb = np.array([fov_step[int(v)] == selfun for v in Rb['gt_fov'][Eb['ti']]])
            A_ = {k: v[ma] for k, v in Ea.items()}; B_ = {k: v[mb] for k, v in Eb.items()}
        pr = P.paired(Rb, A_, B_, bins=BANDS)
        com, ia, ib = np.intersect1d(A_['ti'], B_['ti'], return_indices=True); ph = Rb['gt_phot'][com]
        c = Rb['gt_sf_m'][com]; ok = np.isfinite(c).all(1) & (np.hypot(*c.T) / np.sqrt(2) * P.PX < 25)
        for (lo, hi) in BANDS:
            key = f'{lo}-{hi:g}'
            if key not in pr: continue
            v = pr[key]; m = ok & (ph >= lo) & (ph < hi)
            print(f'  {name:14s} {key:>10s} n {v["A"]["n"]:5d}  RMS/CRLB A {v["A"]["rms"]:.3f} B {v["B"]["rms"]:.3f} d {v["d"]:+.3f} [{v["ci"][0]:+.3f}, {v["ci"][1]:+.3f}]'
                  f'  nm A {v["A"]["nm"]:.2f} B {v["B"]["nm"]:.2f}  zcal A {v["A"]["zcal"]:.2f} B {v["B"]["zcal"]:.2f} | prior RMS nm A {rms(A_["ep"][ia][m]):.2f} B {rms(B_["ep"][ib][m]):.2f}')
