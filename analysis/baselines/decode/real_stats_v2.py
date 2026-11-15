"""Real-data ranges for the DECODE v3 training sims in camera-v2 units (PSF v2 runs of SIMCODE v2; real_stats_paper2.py logic).
Source: pass1_<f0>.npz chunks of the SIMCODE v2 PSF-v2 runs (network per-frame intensity I = intensity parameter of the PSF v2
model, bg photons / px / frame, camera v2), rows p >= 0.63 inside the crop, disjoint windows; chunks at ~10 / 50 / 90 % of the
movie (only the needed members are read). Per 6-frame sum: photons (sum of the 6 I_t), bg (sum of the 6 bg), emitters per
40 x 40 px per sum, height z (um).
usage: python analysis/baselines/decode/real_stats_v2.py [out.json]"""
import os, sys, glob, json, zipfile
import numpy as np
RUNS = {'1809_2': '/mnt/data/dmdsf/bn_dnapaint/1809/cover1/2/runs/simcode_v2_psfv2_20261115',
        '2.5nM_2': '/mnt/data/dmdsf/bn_dnapaint/dnapaint/2.5nM/2/runs/simcode_v2_psfv2_final',
        '20nM_2': '/mnt/data/dmdsf/bn_dnapaint/dnapaint/20nM/2/runs/simcode_v2_psfv2_final'}
Q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
KEYS = ('p', 'crop', 'x', 'y', 'X0', 'Y0', 'wstart', 'I', 'bg', 'z')


def load(f):
    with zipfile.ZipFile(f) as zf:
        return {k: np.load(zf.open(k + '.npy')) for k in KEYS}


out = {}
for nm, d in RUNS.items():
    fs = sorted(glob.glob(f'{d}/pass1_*.npz'), key=lambda f: int(f.split('_')[-1][:-4]))
    fs = [fs[int(round(q * (len(fs) - 1)))] for q in (0.1, 0.5, 0.9)]
    ph, bg, dens, zz = [], [], [], []
    for f in fs:
        z = load(f)
        x0, y0, sz = [int(v) for v in z['crop']]
        X, Y = z['x'] + z['X0'], z['y'] + z['Y0']
        inc = (X >= x0) & (X < x0 + sz) & (Y >= y0) & (Y < y0 + sz)
        m = (z['p'] >= 0.63) & inc & (z['wstart'] % 6 == 0)
        ph.append(z['I'][m].sum(1)); bg.append(z['bg'][m].sum(1)); zz.append(z['z'][m])
        nwin = len(np.unique(z['wstart'][z['wstart'] % 6 == 0]))
        dens.append(m.sum() / max(nwin, 1) / sz ** 2 * 1600)
    ph, bg, zz = np.concatenate(ph), np.concatenate(bg), np.concatenate(zz)
    qs = lambda v: {str(q): round(float(np.quantile(v, q)), 3) for q in Q}
    out[nm] = dict(chunks=[os.path.basename(f) for f in fs], n=int(len(ph)), photons_sum6=qs(ph), log_photons_sd=float(np.std(np.log(ph[ph > 50]))),
                   bg_sum6_per_px=qs(bg), z_um=qs(zz), emitters_per_40x40_per_sum=dict(mean=float(np.mean(dens)), min=float(np.min(dens)), max=float(np.max(dens))))
    print(nm, json.dumps(out[nm]), flush=True)
if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], 'w'), indent=1)
