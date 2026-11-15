"""Event-level gap300 FRC on identical rows (ours vs reference) with block-phase replicates: the 300-frame block boundaries
shifted by 0 / 75 / 150 / 225 frames (b = (f + offset) // 300); Delta = ours - reference per phase, and the +3 nm iid
sensitivity per phase (ours identical rows + 3 nm per row per axis). Pairing: exact window start, mutual nearest < 0.5 px.
usage: python analysis/evalproto/blockphase.py <dataset.json> [out_dir]   -> blockphase_<dataset>.txt / .json"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets as DS
from simcode.evaluation import metrics as E
spec = sys.argv[1]
OFFS = (0, 75, 150, 225)
LINES = []; OUT = {}
def log(s=''):
    print(s, flush=True); LINES.append(s)


def std(D, grid):
    k = D['f'] % grid == 0
    return dict(x=D['x'][k], y=D['y'][k], f=D['f'][k].astype(np.int64))


def gap300(x, y, f, off):
    b = (f + off) // 300; r = []
    for o, o2 in ((0, 2), (1, 3)):
        a = b % 4 == o; cc = b % 4 == o2
        r.append(E.frc(x[a], y[a], x[cc], y[cc]))
    return float(np.nanmean(r))


S0, MS0, META = DS.load(spec)
grid = META['grid']; c = META['name']
MS = {k: std(v, grid) for k, v in MS0.items()}; S = std(S0, grid)

rng = np.random.default_rng(37)
log(f'# {c}, G{grid}: gap300 FRC (nm) on identical rows, block phase offsets {OFFS} frames')
log('| method | offset | ours | SC | Delta ours - SC | ours + 3 nm iid (shift) |'); log('|---|---|---|---|---|---|')
for k, A in MS.items():
    ia, ib = E.pair(A['x'], A['y'], A['f'], S['x'], S['y'], S['f'])
    xa, ya, fa = A['x'][ia], A['y'][ia], A['f'][ia]; xb, yb, fb = S['x'][ib], S['y'][ib], S['f'][ib]
    xi = xa + rng.normal(0, 3 / E.PX, len(xa)); yi = ya + rng.normal(0, 3 / E.PX, len(ya))
    R = []
    for off in OFFS:
        a, b, s3 = gap300(xa, ya, fa, off), gap300(xb, yb, fb, off), gap300(xi, yi, fa, off)
        R.append(dict(off=off, A=a, SC=b, d=a - b, inj3=s3 - a))
        log(f'| {k} | {off} | {a:.2f} | {b:.2f} | {a - b:+.2f} | {s3 - a:+.2f} |')
    d = np.array([r['d'] for r in R]); s3 = np.array([r['inj3'] for r in R])
    log(f'| {k} | **mean ± sd (4 phases)** | {np.mean([r["A"] for r in R]):.2f} | {np.mean([r["SC"] for r in R]):.2f} | **{d.mean():+.2f} ± {d.std(ddof=1):.2f}** (range {d.min():+.2f} … {d.max():+.2f}) | {s3.mean():+.2f} ± {s3.std(ddof=1):.2f} |')
    OUT[k] = dict(n=int(len(ia)), phases=R, d_mean=float(d.mean()), d_sd=float(d.std(ddof=1)), inj3_mean=float(s3.mean()))
fn = DS.out_path(spec, 'blockphase', sys.argv[2] if len(sys.argv) > 2 else None)
json.dump(OUT, open(fn + '.json', 'w'), indent=1, default=float)
open(fn + '.txt', 'w').write('\n'.join(LINES) + '\n')
