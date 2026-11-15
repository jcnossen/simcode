"""In-situ refinement of the FourPiSplinePSF knots (the psf.zip 'phase_spline' channels: inc / cr / ci per channel, the
spot phase Phi analytic) on the Cell11 spots, started from the spline table of the in-situ FourPiPSF fit.

Knots K = K0 + dK on a reduced table (oversample S, dz, offset limit xy_limit: only the knots the ROIs can reach), with
per outer iteration: per-spot LM on the current knots (x, y, z, N, bg_c; Phi = the SIMCODE field value) -> L-BFGS on dK
at fixed spot parameters (the model is linear in the knots: a convex subproblem). Loss per train spot:
    NLL + lam_s |second differences of dK|^2 / s + lam_r |dK|^2 / s + lam_g (gauge terms) / s
s = (max |K0|)^2; gauge terms (unobservable or degenerate in blinking data, kept at the start model's values): per z plane
the sum of the inc knots over the channels (the photon normalization, trades with N) and its first lateral moments (a
z-dependent lateral shift of the whole PSF cannot be seen in single frames).
Held-out spots (other tiles, file 3) are localized and scored after every outer iteration.

  python analysis/fourpi_plain/p4spline_refine.py --tag r1 --init <insitu state.json> [--lam_s 1e3] [--lam_r 10]
         [--S 2] [--dz 0.02] [--outer 6]
  -> <OUT>/spline/<tag>/{knots.pt (K, geometry, reg, beta), eval.json, log.txt}
  load_refined(knots.pt) -> SplineModel
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from insitu4pi import (load_spots, theta0, localize, evaluate, nll, Model4Pi, SplineModel, OUT, dev, ZR, FourPiSplinePSF,
                       baked_spline, simcode_state)
from p4insitu_fit import select, split, summarize


def refine_geometry(model, S=2, dz=0.01, xy_limit=2.5):
    """the reduced knot table with the model's registration offsets and beta baked in (insitu4pi.baked_spline)"""
    return baked_spline(model, P=17, z_range=ZR, oversample=S, dz_um=dz, xy_limit=xy_limit)


def save_refined(path, spl, K, vector_state, extra=None):
    """K = baked table + dK; vector_state = the SIMCODE state of the vectorial model the table was built from (LUT4Pi
    'spline' / 'spline_base', simcode/fourpi/lut4pi.py)"""
    torch.save(dict(K=K.detach().cpu(), roisize=spl.roisize, z_range=spl.z_range, S=spl.S, dz=spl.hz, zpad=spl.zpad,
                    xy_limit=spl.xy_limit, tau=spl.tau.cpu(), pixel_um=spl.pixel_um, vector_state=vector_state,
                    source_state=spl.source_state, **(extra or {})), path)


def load_refined(path, device=dev):
    s = torch.load(path, weights_only=False)
    spl = FourPiSplinePSF(s['roisize'], s['z_range'], s['K'], oversample=s['S'], dz_um=s['dz'], zpad=s['zpad'],
                          xy_limit=s['xy_limit'], tau=s['tau'], device=device, pixel_um=s['pixel_um'],
                          source_state=s['source_state'])
    return SplineModel(spl, torch.zeros(spl.n_channels, 2, device=device), 0.0)


def gauge_terms(dK, spl):
    """per z plane: sum over channels of the inc knots and its first lateral moments -> [3, nz]"""
    inc = dK[0].sum(0)                                   # [nz, n, n]
    n = inc.shape[-1]
    u = (torch.arange(n, device=dK.device, dtype=dK.dtype) - (n - 1) / 2) / spl.S
    m0 = inc.sum((1, 2)); mx = (inc * u[None, None, :]).sum((1, 2)); my = (inc * u[None, :, None]).sum((1, 2))
    return torch.stack([m0, mx, my])


def upsample_z(dKc, nz):
    """correction knots on every second z plane [.., nzc, n, n] -> all nz planes (Catmull-Rom midpoints; edges clamped)"""
    nzc = dKc.shape[-3]
    pad = torch.cat([dKc[..., :1, :, :], dKc, dKc[..., -1:, :, :], dKc[..., -1:, :, :]], -3)      # index i -> i + 1
    mid = (-pad[..., :-3, :, :] + 9 * pad[..., 1:-2, :, :] + 9 * pad[..., 2:-1, :, :] - pad[..., 3:, :, :]) / 16   # between i, i + 1
    out = torch.stack([dKc, mid[..., :nzc, :, :]], -3).flatten(-4, -3)
    return out[..., :nz, :, :]


def smooth_terms(dK):
    dz = dK[:, :, 2:] - 2 * dK[:, :, 1:-1] + dK[:, :, :-2]
    dy = dK[:, :, :, 2:] - 2 * dK[:, :, :, 1:-1] + dK[:, :, :, :-2]
    dx = dK[..., 2:] - 2 * dK[..., 1:-1] + dK[..., :-2]
    return (dz ** 2).sum() + (dy ** 2).sum() + (dx ** 2).sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--init', required=True, help='Model4Pi state json (in-situ fit)')
    p.add_argument('--S', type=int, default=2)
    p.add_argument('--dz', type=float, default=0.01, help='z spacing of the table (the correction: every second plane)')
    p.add_argument('--lam_s', type=float, default=1.0)
    p.add_argument('--lam_r', type=float, default=0.3)
    p.add_argument('--lam_g', type=float, default=1.0)
    p.add_argument('--outer', type=int, default=6)
    p.add_argument('--inner', type=int, default=40)
    p.add_argument('--ntrain', type=int, default=20000)
    p.add_argument('--neval', type=int, default=8000)
    p.add_argument('--batch', type=int, default=4000)
    a = p.parse_args()
    od = f'{OUT}/spline/{a.tag}'; os.makedirs(od, exist_ok=True)
    logf = open(f'{od}/log.txt', 'a')

    def log(s):
        print(s, flush=True); logf.write(s + '\n'); logf.flush()
    log(f'== {time.ctime()} {vars(a)}')
    D = load_spots(sel=select)
    sp = split(D)
    rng = np.random.default_rng(1)                       # the same train / eval spots as p4insitu_fit.py
    tr = np.sort(rng.choice(sp['train'], min(a.ntrain, len(sp['train'])), replace=False))
    ev_sets = {k: np.sort(rng.choice(sp[k], min(a.neval, len(sp[k])), replace=False)) for k in ('test', 'file3')}
    ev_sets['train'] = tr[:a.neval]
    use = np.unique(np.concatenate([tr] + list(ev_sets.values())))
    D = {k: (v[use] if isinstance(v, (np.ndarray, torch.Tensor)) and len(v) == len(D['x']) else v) for k, v in D.items()}
    pos_of = {k: torch.tensor(np.searchsorted(use, v), device=dev) for k, v in ev_sets.items()}
    tr_t = torch.tensor(np.searchsorted(use, tr), device=dev)
    m = Model4Pi.from_state(json.load(open(a.init)), roisize=17)
    spl = refine_geometry(m, a.S, a.dz)
    reg, beta = torch.zeros_like(m.offsets()).detach(), 0.0          # baked into the table
    vstate = simcode_state(m)
    K0 = spl.knots.to(dev)
    log(f'knot table {tuple(K0.shape)} ({K0.numel() / 1e6:.2f} M knots), spots {len(use)}')
    # accuracy of the reduced table against the fine table of the same model
    fine = m.spline()
    th = theta0(D)
    th, _ = localize(fine, D, th, iters=25)
    SM = SplineModel(spl, reg, beta)
    with torch.no_grad():
        j = torch.nonzero(th[:, :2].abs().max(1).values < 2.0)[:2000, 0]
        g1 = fine.images(D['pos0'][j] + th[j, None, :2], th[j, 2], th[j, 3]); g2 = SM.images(D['pos0'][j] + th[j, None, :2], th[j, 2], th[j, 3])
    log(f'reduced table vs fine table: relative L2 {float((g1 - g2).norm() / g1.norm()):.2e}')
    sc = float(K0.abs().max()) ** 2
    nz = K0.shape[2]
    ks = float(K0.abs().max())          # the optimizer works on u = dK / ks (L-BFGS stops on absolute step sizes ~1e-9)
    u = torch.zeros(*K0.shape[:2], (nz + 1) // 2, *K0.shape[3:], device=dev, requires_grad=True)
    dKc = lambda: u * ks
    log(f'correction knots {tuple(u.shape)} ({u.numel() / 1e6:.2f} M)')
    full = lambda: K0 + upsample_z(dKc(), nz)
    rec = dict(args=vars(a), iters=[])
    t0 = time.time()

    def score(it, th):
        SMc = SplineModel(spl, reg, beta, knots=full().detach())
        th, L = localize(SMc, D, th, iters=25 if it == 0 else 10)
        r = dict(it=it, nll_train=float(L[tr_t].mean()))
        for k, j in pos_of.items():
            r[k] = summarize(evaluate(SMc, D, th, j), D['N'], j.cpu().numpy())
        dKf = full().detach() - K0
        r['dK_rel'] = float(dKf.norm() / K0.norm())
        r['dK_comp_ch'] = (dKf.pow(2).sum((2, 3, 4)).sqrt() / K0.pow(2).sum((2, 3, 4)).sqrt()).cpu().tolist()
        rec['iters'].append(r)
        log(f'it {it} ({time.time() - t0:.0f}s): dev/px ctr median train {r["train"]["ctr_median"]:.4f} test {r["test"]["ctr_median"]:.4f} file3 {r["file3"]["ctr_median"]:.4f}; '
            f'bins test {np.round(r["test"]["ctr_bins"], 3).tolist()} file3 {np.round(r["file3"]["ctr_bins"], 3).tolist()}; |dK|/|K0| {r["dK_rel"]:.4f}')
        return th
    th = score(0, th)
    save_refined(f'{od}/knots_it0.pt', spl, K0, vstate)
    best = (rec['iters'][0]['test']['ctr_median'], 0)
    for it in range(1, a.outer + 1):
        if it == 1:            # one L-BFGS (its history) over all outer iterations; a fresh one stalls on the tiny first step
            opt = torch.optim.LBFGS([u], max_iter=a.inner, history_size=20, line_search_fn='strong_wolfe')
        ntr = len(tr_t)

        def closure():
            opt.zero_grad()
            tot = 0.0
            K = full()
            SMk = SplineModel(spl, reg, beta, knots=K)
            for s in range(0, ntr, a.batch):
                j = tr_t[s:s + a.batch]; t = th[j]
                g = SMk.images(D['pos0'][j] + t[:, None, :2], t[:, 2], t[:, 3])
                mu = t[:, 4, None, None, None] * g + t[:, 5:, None, None]
                l = nll(mu, D['d'][j], D['var']).sum() / ntr
                l.backward(retain_graph=True)
                tot += float(l.detach())
            dK = dKc()
            reg_t = (a.lam_s * smooth_terms(dK) + a.lam_r * (dK ** 2).sum() + a.lam_g * (gauge_terms(dK, spl) ** 2).sum()) / sc
            reg_t.backward()
            return torch.tensor(tot + float(reg_t.detach()), device=dev)
        l = opt.step(closure)
        log(f'  knots L-BFGS: loss {float(l):.3f}')
        th = score(it, th)
        save_refined(f'{od}/knots_it{it}.pt', spl, full().detach(), vstate)
        if rec['iters'][-1]['test']['ctr_median'] < best[0]: best = (rec['iters'][-1]['test']['ctr_median'], it)
        json.dump(rec, open(f'{od}/eval.json', 'w'), indent=1)
    rec['best'] = best
    json.dump(rec, open(f'{od}/eval.json', 'w'), indent=1)
    if best[1] > 0:
        import shutil; shutil.copy(f'{od}/knots_it{best[1]}.pt', f'{od}/knots.pt')
    log(f'best held-out (test tiles) at it {best[1]}: dev/px ctr {best[0]:.4f}; done {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
