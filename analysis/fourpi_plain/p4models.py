"""PSF models on the held-out Cell11 spots: deviance per pixel (centre 9 x 9 of the 4 channels, the background ring > 6 px
from the centre, whole ROI) per photon bin, and the floor (spots simulated from the model at the fitted parameters:
Poisson + read noise + ADU quantization, scored the same way).

  python analysis/fourpi_plain/p4models.py --models bead:- v1:<state json> r3:<knots .pt> [--set test|file3] [--n 8000]
      [--floor 1] [--json out.json]
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from insitu4pi import load_spots, theta0, localize, evaluate, expected, dev
from p4insitu_fit import select, split, summarize
from p4insitu_resid import get_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', required=True); p.add_argument('--set', nargs='+', default=['test', 'file3'])
    p.add_argument('--n', type=int, default=8000); p.add_argument('--floor', type=int, default=1); p.add_argument('--json', default='')
    a = p.parse_args()
    D0 = load_spots(sel=select)
    sp = split(D0)
    out = {}
    for sname in a.set:
        j = np.sort(np.random.default_rng(5).choice(sp[sname], min(a.n, len(sp[sname])), replace=False))
        D = {k: (v[j] if isinstance(v, (np.ndarray, torch.Tensor)) and len(v) == len(D0['x']) else v) for k, v in D0.items()}
        for spec in a.models:
            name, S = get_model(spec)
            th, _ = localize(S, D, theta0(D), iters=25)
            r = summarize(evaluate(S, D, th), D['N'], np.arange(len(th)))
            o = {k: r[k] for k in ('ctr_median', 'ctr_bins', 'outer_median', 'outer_bins', 'all_median', 'ch_median')}
            if a.floor:
                with torch.no_grad():
                    mu = torch.cat([expected(S, D, th[s:s + 2000], torch.arange(s, min(s + 2000, len(th)), device=dev)) for s in range(0, len(th), 2000)])
                g = 0.455
                sim = torch.round((torch.poisson(mu.clamp(min=0)) + torch.randn_like(mu) * np.sqrt(D['var'])) / g) * g
                D2 = dict(D, d=sim)
                th2, _ = localize(S, D2, th, iters=25)
                r2 = summarize(evaluate(S, D2, th2), D['N'], np.arange(len(th)))
                o['floor'] = {k: r2[k] for k in ('ctr_median', 'ctr_bins', 'outer_median', 'all_median')}
            out[f'{sname}/{name}'] = o
            print(sname, name, json.dumps({k: (np.round(v, 4).tolist() if not isinstance(v, dict) else {kk: np.round(vv, 4).tolist() for kk, vv in v.items()}) for k, v in o.items()}), flush=True)
    if a.json: json.dump(out, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
