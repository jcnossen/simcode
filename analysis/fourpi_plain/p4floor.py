"""Deviance floor of the in-situ PSF evaluation: spots simulated from a model at the fitted spot parameters (Poisson +
Gaussian read noise, quantized to the camera's ADU step) and scored exactly as the real spots.

  python analysis/fourpi_plain/p4floor.py [--state <Model4Pi state json>]
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from insitu4pi import load_spots, theta0, localize, evaluate, expected, load_psf0, Model4Pi, dev
from p4insitu_fit import select, split, summarize

p = argparse.ArgumentParser(); p.add_argument('--state', default=''); p.add_argument('--n', type=int, default=8000)
a = p.parse_args()
D = load_spots(sel=select)
j = split(D)['test']; j = np.sort(np.random.default_rng(5).choice(j, a.n, replace=False))
D = {k: (v[j] if isinstance(v, (np.ndarray, torch.Tensor)) and len(v) == len(D['x']) else v) for k, v in D.items()}
m = Model4Pi.from_state(json.load(open(a.state)), roisize=17) if a.state else load_psf0(17)
S = m.spline()
th, _ = localize(S, D, theta0(D), iters=25)
r_real = summarize(evaluate(S, D, th), D['N'], np.arange(len(th)))
with torch.no_grad():
    mu = torch.cat([expected(S, D, th[s:s + 2000], torch.arange(s, min(s + 2000, len(th)), device=dev)) for s in range(0, len(th), 2000)])
g = 0.455
sim = torch.poisson(mu.clamp(min=0)) + torch.randn_like(mu) * np.sqrt(D['var'])
sim = torch.round(sim / g) * g
D2 = dict(D, d=sim)
th2, _ = localize(S, D2, th, iters=25)
r_sim = summarize(evaluate(S, D2, th2), D['N'], np.arange(len(th)))
print('real :', json.dumps({k: r_real[k] for k in ('ctr_median', 'ctr_bins', 'all_median', 'outer_median', 'outer_bins')}))
print('sim  :', json.dumps({k: r_sim[k] for k in ('ctr_median', 'ctr_bins', 'all_median', 'outer_median', 'outer_bins')}))
