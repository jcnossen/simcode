"""Residual images of 4Pi PSF models on the Cell11 spots: per z bin and channel the mean data / N and the mean residual
(data - model) / N after the per-spot LM on each model, and the excess deviance per spot vs photons.

  python analysis/fourpi_plain/p4insitu_resid.py --models bead:- insitu:<state.json> [spline:<knots.pt>] --set test
      --out <png> [--json <json>]
Models: 'bead' (the bead-stack PSF, '-'), a Model4Pi state json, or a refined spline (p4spline_refine.py output .pt).
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from insitu4pi import load_spots, theta0, load_psf0, localize, expected, deviance, Model4Pi, OUT, dev
from p4insitu_fit import select, split

ZB = np.array([-0.9, -0.7, -0.55, -0.4, -0.25, -0.1, 0.05])


def spline_model(path):
    from p4spline_refine import load_refined
    return load_refined(path)


def get_model(spec):
    name, src = spec.split(':', 1)
    if name == 'bead' or src == '-':
        return name, load_psf0(17).spline()
    if src.endswith('.pt'):
        return name, spline_model(src)
    return name, Model4Pi.from_state(json.load(open(src)), roisize=17).spline()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--set', default='test')
    p.add_argument('--n', type=int, default=12000)
    p.add_argument('--phi', default='field')
    p.add_argument('--out', required=True)
    p.add_argument('--json', default='')
    a = p.parse_args()
    D = load_spots(sel=select)
    j = split(D)[a.set]
    j = np.sort(np.random.default_rng(5).choice(j, min(a.n, len(j)), replace=False))
    D = {k: (v[j] if isinstance(v, (np.ndarray, torch.Tensor)) and len(v) == len(D['x']) else v) for k, v in D.items()}
    res = {}
    for spec in a.models:
        name, S = get_model(spec)
        th, L = localize(S, D, theta0(D), fix_phi=a.phi == 'field', iters=25)
        with torch.no_grad():
            mus = torch.cat([expected(S, D, th[s:s + 2000], torch.arange(s, min(s + 2000, len(th)), device=dev)) for s in range(0, len(th), 2000)])
        dv = deviance(mus, D['d'], D['var'])
        N = th[:, 4]; z = th[:, 2].cpu().numpy()
        r = ((D['d'] - mus) / N[:, None, None, None]).cpu().numpy()
        dn = ((D['d'] - th[:, 5:, None, None]) / N[:, None, None, None]).cpu().numpy()
        zb = np.digitize(z, ZB) - 1
        R = np.stack([r[zb == b].mean(0) if (zb == b).sum() > 20 else np.zeros_like(r[0]) for b in range(len(ZB) - 1)])
        Dm = np.stack([dn[zb == b].mean(0) if (zb == b).sum() > 20 else np.zeros_like(r[0]) for b in range(len(ZB) - 1)])
        cnt = [int((zb == b).sum()) for b in range(len(ZB) - 1)]
        exc = (dv.sum((1, 2, 3)) - dv[0].numel()).cpu().numpy()            # excess deviance per spot
        Nn = N.cpu().numpy()
        nb = [(2500, 5000), (5000, 10000), (10000, 1e9)]
        res[name] = dict(R=R, Dm=Dm, cnt=cnt, th=th.cpu().numpy(),
                         excess_bins=[float(np.median(exc[(Nn >= lo) & (Nn < hi)])) for lo, hi in nb],
                         dev_ctr_median=float(dv[:, :, 4:13, 4:13].mean((1, 2, 3)).median()),
                         dev_all_mean=float(dv.mean()),
                         resid_rms_zbin=[float(np.sqrt((R[b] ** 2).mean())) for b in range(len(cnt))])
        print(name, 'excess deviance per spot (2.5-5k, 5-10k, >10k):', np.round(res[name]['excess_bins'], 1),
              'dev/px ctr', round(res[name]['dev_ctr_median'], 4), 'resid rms per z bin', np.round(res[name]['resid_rms_zbin'], 5), flush=True)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    names = list(res); nz = len(ZB) - 1; C = 4
    fig, ax = plt.subplots(nz, C * (1 + len(names)), figsize=(1.25 * C * (1 + len(names)), 1.35 * nz))
    vmax = max(np.abs(res[n]['R'][:, :, 3:14, 3:14]).max() for n in names)
    for b in range(nz):
        for c in range(C):
            A = ax[b, c]; A.imshow(res[names[0]]['Dm'][b, c, 3:14, 3:14], cmap='magma'); A.set_xticks([]); A.set_yticks([])
            if b == 0: A.set_title(f'data ch{c}', fontsize=7)
            if c == 0: A.set_ylabel(f'z {ZB[b]:.2f}..{ZB[b + 1]:.2f}\nn={res[names[0]]["cnt"][b]}', fontsize=6)
            for k, n in enumerate(names):
                A = ax[b, C * (k + 1) + c]
                A.imshow(res[n]['R'][b, c, 3:14, 3:14], cmap='RdBu_r', vmin=-vmax, vmax=vmax); A.set_xticks([]); A.set_yticks([])
                if b == 0: A.set_title(f'{n} resid ch{c}', fontsize=7)
    fig.suptitle(f'Cell11 spots ({a.set}, {len(j)}): mean (data - model) / N per z bin and channel, colour range +-{vmax:.4f} per px', fontsize=8)
    fig.tight_layout(); fig.savefig(a.out, dpi=130)
    if a.json:
        json.dump({n: {k: v for k, v in r.items() if k not in ('R', 'Dm', 'th')} for n, r in res.items()}, open(a.json, 'w'), indent=1)
    np.savez(a.out.replace('.png', '.npz'), **{f'{n}_R': r['R'] for n, r in res.items()}, **{f'{n}_th': r['th'] for n, r in res.items()}, idx=D['idx'])


if __name__ == '__main__':
    main()
