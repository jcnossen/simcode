"""True brightness variation of the held-out simulated movies (2026-11-17), for the comparison with the brightness noise c, c_a that
the pipeline estimates with the pattern (kin_c_compare.py). CPU only, ground truth only (no run needed).
GT per-frame photons of every 6-frame window an event is ON through (truth_frames.npz: incl. the brightness factors and the pattern
field) against mu_t = B il_t(x) with the TRUE illumination at the event (the frame's pattern + the event's field dph / dmf, so the
pattern-field part is not counted as brightness noise), B fitted per window (least squares). Within-angle part of the relative
residual -> c, angle means -> c_a (mu-weighted: c^2 = sum (I - mu - angle part)^2 / sum mu^2 x 3/2, c_a^2 = 2 var_w(angle mean) - c^2 / 3).
Also the simulator's knobs: c = sqrt(frame_sd^2 + step_sd^2) (per-frame and per-step factors, both independent between the 6
frames of a window), c_a = ang_sd / sqrt(2) (per-angle factor normalised over the 2 angles; ang_sd of the preset).
usage: python gt_brightness.py <out json> <sim> [...]"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from evalsim import load_gt, SJ

ANG_SD = {'0.5nM': 0.14, '2.5nM': 0.2}


def one(sim, nmax=400000, seed=0):
    d = f'{SJ}/truth/{sim}'
    G = np.load(f'{d}/sim_gt.npz'); TF = np.load(f'{d}/truth_frames.npz')
    x, y, t0, t1, nf = load_gt(f'{d}/sim_gt.npz')
    # windows (start % 3 == 0) an event is ON through
    ev, w = [], []
    for e in range(len(x)):
        a = int(np.ceil(t0[e] / 3.0)) * 3
        b = min(int(np.floor(t1[e])), nf) - 6
        if b >= a:
            s = np.arange(a, b + 1, 3); ev.append(np.full(len(s), e)); w.append(s)
    ev = np.concatenate(ev); w = np.concatenate(w)
    rng = np.random.default_rng(seed)
    if len(ev) > nmax:
        k = rng.choice(len(ev), nmax, replace=False); ev, w = ev[k], w[k]
    key = TF['ev'] * nf + TF['frame']; o = np.argsort(key); key, ph = key[o], TF['ph'][o]
    q = ev[:, None] * nf + w[:, None] + np.arange(6)[None]
    j = np.clip(np.searchsorted(key, q), 0, len(key) - 1); I = np.where(key[j] == q, ph[j], np.nan)
    fr = w[:, None] + np.arange(6)[None]; ang = (fr % 6) // 3
    pt = G['patt'][fr].astype(np.float64).copy()
    if 'dph' in TF.files:
        pt[..., 2] += TF['dph'][ev[:, None], ang]; pt[..., 3] = np.minimum(pt[..., 3] * TF['dmf'][ev[:, None], ang], 1.0)
    il = pt[..., 4] * (1 + pt[..., 3] * np.cos(pt[..., 0] * x[ev, None] + pt[..., 1] * y[ev, None] - pt[..., 2]))
    ok = np.isfinite(I).all(1) & (np.nansum(I, 1) > 100)
    I, il, ang = I[ok], il[ok], ang[ok]
    B = (I * il).sum(1) / (il ** 2).sum(1); mu = B[:, None] * il
    r = I - mu
    ma = np.stack([np.where(ang == a, r, 0).sum(1) / np.where(ang == a, mu, 0).sum(1) for a in (0, 1)], 1)        # relative angle offsets
    within = r - np.take_along_axis(ma, ang, 1) * mu
    c2 = 1.5 * (within ** 2).sum() / (mu ** 2).sum()
    wa = np.stack([np.where(ang == a, mu, 0).sum(1) for a in (0, 1)], 1)
    ca2 = 2 * (wa * ma ** 2).sum() / wa.sum() - c2 / 3
    pre = str(G['preset']) if 'preset' in G.files else ''
    res = dict(sim=sim, n_windows=int(ok.sum()), c_gt=float(np.sqrt(max(c2, 0))), c_a_gt=float(np.sqrt(max(ca2, 0))),
               frame_sd=float(G['frame_sd']), step_sd=float(G['step_sd']), preset=pre)
    res['c_knobs'] = float(np.hypot(res['frame_sd'], res['step_sd']))
    if pre in ANG_SD: res['c_a_knobs'] = ANG_SD[pre] / np.sqrt(2)
    return res


if __name__ == '__main__':
    out = [one(s) for s in sys.argv[2:]]
    for r in out: print(json.dumps(r))
    json.dump(out, open(sys.argv[1], 'w'), indent=1)
