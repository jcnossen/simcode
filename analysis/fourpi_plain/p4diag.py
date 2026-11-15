"""Where is the PSF mismatch? Excess deviance per spot (central 9 x 9 px of the 4 channels, minus the floor 1 / px) per
1000 photons, binned by lateral position, time (file / frame block), z, the total 4Pi phase, background.

  python analysis/fourpi_plain/p4diag.py [--state <Model4Pi state json> | --knots <refined .pt>] [--n 16000]
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from insitu4pi import load_spots, theta0, localize, evaluate, load_psf0, Model4Pi, dev
from p4insitu_fit import select, split

p = argparse.ArgumentParser(); p.add_argument('--state', default=''); p.add_argument('--knots', default=''); p.add_argument('--n', type=int, default=16000)
p.add_argument('--json', default='')
a = p.parse_args()
D = load_spots(sel=select)
sp = split(D); j = np.concatenate([sp['test'], sp['file3']])
j = np.sort(np.random.default_rng(7).choice(j, min(a.n, len(j)), replace=False))
D = {k: (v[j] if isinstance(v, (np.ndarray, torch.Tensor)) and len(v) == len(D['x']) else v) for k, v in D.items()}
if a.knots:
    from p4spline_refine import load_refined; S = load_refined(a.knots); beta = S.beta
else:
    m = Model4Pi.from_state(json.load(open(a.state)), roisize=17) if a.state else load_psf0(17); S = m.spline(); beta = float(m.beta)
th, _ = localize(S, D, theta0(D), iters=25)
ev = evaluate(S, D, th)
N = th[:, 4].cpu().numpy(); e = (ev['ctr'].cpu().numpy() - 1.0) * 324 / (N / 1000)
z = th[:, 2].cpu().numpy(); ph = ((th[:, 3].cpu().numpy() + (2 * np.pi / 0.239 + beta) * z) % (2 * np.pi))
t = D['fidx'] * 3000 + D['frame']
out = {}
def rep(name, v, edges):
    b = np.digitize(v, edges) - 1
    r = [(f'{edges[i]:.3g}', int((b == i).sum()), round(float(np.median(e[b == i])), 2)) for i in range(len(edges) - 1) if (b == i).sum() > 50]
    out[name] = r; print(name, r)
rep('x px', D['x'], np.linspace(0, 192, 7)); rep('y px', D['y'], np.linspace(0, 168, 7))
rep('frame', t, np.arange(0, 12001, 1500)); rep('z um', z, np.array([-0.9, -0.7, -0.55, -0.4, -0.25, -0.1, 0.05]))
rep('phase', ph, np.linspace(0, 2 * np.pi, 7)); rep('N', N, np.array([2500, 4000, 6000, 9000, 14000, 50000]))
rep('bg sum', th[:, 5:].sum(1).cpu().numpy(), np.array([0, 50, 65, 80, 100, 1000]))
print('median excess deviance per 1000 photons', float(np.median(e)))
if a.json: json.dump(out, open(a.json, 'w'), indent=1)
