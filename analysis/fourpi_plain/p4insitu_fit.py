"""In-situ FourPiPSF fit on the Cell11 spots (analysis/fourpi_plain/insitu4pi.py), started from the bead-stack PSF.

Spots: SIMCODE localizations with N > nmin photons, nearest other candidate > 10 px, sigma_xy < 25 nm, z within 120 nm of
the astigmatic prior (no period slip). Split: files 0-2 checkerboard tiles (tile px) -> train (even) / test (odd);
file 3 -> held-out file. Phi = the SIMCODE phase field value Phi0 (fixed, as in the solve), or free per spot (--phi free).

Per outer iteration: spline table of the current model -> per-spot LM (x, y, z, [Phi], N, bg_c) -> Adam on the PSF
parameters (minibatches of the train spots, vectorial model).

  python analysis/fourpi_plain/p4insitu_fit.py --tag v1 [--outer 6] [--steps 150] [--ntrain 20000] [--phi field|free]
      [--groups zern,phi,tau,a,gamma,dbot,blur,reg,beta] [--zidx 21] [--init <state json>]
  -> <OUT>/insitu/<tag>/{state.json, eval.json, log.txt, theta.npz}
"""
import argparse, json, os, time, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from insitu4pi import (load_spots, theta0, load_psf0, localize, psf_step, evaluate, Model4Pi, OUT, dev, ZR,
                       FourPiPSF)

PH_BINS = (2500, 5000, 10000, 1e9)


def select(Z, nmin=2500):
    return ((Z['N'] > nmin) & (Z['nn'] > 10) & (np.maximum(Z['sx'], Z['sy']) * 128 < 25)
            & (np.abs(Z['z'] - Z['zprior']) < 0.12) & np.isfinite(Z['zprior']))


def split(D, tile=16):
    t = ((np.floor(D['x'] / tile) + np.floor(D['y'] / tile)) % 2).astype(int)
    f3 = D['fidx'] == 3
    return dict(train=np.nonzero(~f3 & (t == 0))[0], test=np.nonzero(~f3 & (t == 1))[0], file3=np.nonzero(f3)[0])


def summarize(ev, N, idx):
    o = {}
    for k in ('all', 'ctr', 'outer'):
        v = ev[k].cpu().numpy()
        o[k + '_median'] = float(np.median(v)); o[k + '_mean'] = float(np.mean(v))
        o[k + '_bins'] = [float(np.median(v[(N[idx] >= a) & (N[idx] < b)])) for a, b in zip(PH_BINS[:-1], PH_BINS[1:])]
    o['ch_median'] = ev['ch'].median(0).values.cpu().tolist()
    return o


def widen_zernikes(m, nmax):
    """a copy of the model with Noll 5..nmax for both arms (new coefficients 0)"""
    st = m.state(); s = st['psf']
    old = list(s['meta']['zidx']); new = list(range(5, nmax + 1))
    if new == old: return m
    s['meta']['zidx'] = new
    s['zt'] = [s['zt'][old.index(j)] if j in old else 0.0 for j in new]
    s['zb'] = [s['zb'][old.index(j)] if j in old else 0.0 for j in new]
    s.pop('raw', None)
    raw = m.psf.state()['raw']; s['raw'] = raw
    return Model4Pi.from_state(st, roisize=m.psf.roisize)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--outer', type=int, default=6)
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--batch', type=int, default=1500)
    p.add_argument('--ntrain', type=int, default=20000)
    p.add_argument('--neval', type=int, default=8000)
    p.add_argument('--nmin', type=float, default=2500)
    p.add_argument('--phi', default='field', choices=('field', 'free'))
    p.add_argument('--groups', default='zern,phi,tau,a,gamma,dbot,blur,reg,beta')
    p.add_argument('--zidx', type=int, default=21)
    p.add_argument('--init', default='')
    p.add_argument('--lr', type=float, default=1.0)
    p.add_argument('--xband', type=float, nargs=2, default=None, help='only spots with x in [x0, x1) px (field-dependence test)')
    p.add_argument('--pupil_smooth', type=float, default=1e3, help='Laplacian penalty of the free pupil maps (group pupil)')
    a = p.parse_args()
    od = f'{OUT}/insitu/{a.tag}'; os.makedirs(od, exist_ok=True)
    logf = open(f'{od}/log.txt', 'a')

    def log(s):
        print(s, flush=True); logf.write(s + '\n'); logf.flush()
    log(f'== {time.ctime()} {vars(a)}')
    xb = a.xband
    D = load_spots(sel=lambda Z: select(Z, a.nmin) & ((Z['x'] >= xb[0]) & (Z['x'] < xb[1]) if xb else True))
    sp = split(D)
    rng = np.random.default_rng(1)
    tr = np.sort(rng.choice(sp['train'], min(a.ntrain, len(sp['train'])), replace=False))
    ev_sets = {k: np.sort(rng.choice(sp[k], min(a.neval, len(sp[k])), replace=False)) for k in ('test', 'file3')}
    ev_sets['train'] = tr[:a.neval]
    use = np.unique(np.concatenate([tr] + list(ev_sets.values())))
    D = {k: (v[use] if isinstance(v, (np.ndarray, torch.Tensor)) and len(v) == len(D['x']) else v) for k, v in D.items()}
    pos_of = {k: torch.tensor(np.searchsorted(use, v), device=dev) for k, v in ev_sets.items()}
    tr_t = torch.tensor(np.searchsorted(use, tr), device=dev)
    log(f'spots: {len(use)} (train {len(tr)}, eval {[(k, len(v)) for k, v in ev_sets.items()]}), N median {np.median(D["N"]):.0f}')
    if a.init:
        m = Model4Pi.from_state(json.load(open(a.init)), roisize=17)
    else:
        m = load_psf0(17)
    m = widen_zernikes(m, a.zidx)
    if 'pupil' in a.groups.split(','): m.free_pupil()
    for q in m.parameters(): q.requires_grad_(False)
    fix_phi = a.phi == 'field'
    th = theta0(D)
    groups = a.groups.split(',')
    record = dict(args=vars(a), iters=[])

    def do_eval(m, th, it):
        S = m.spline()
        th, L = localize(S, D, th, fix_phi=fix_phi, iters=25 if it == 0 else 12)
        r = dict(it=it)
        for k, j in pos_of.items():
            r[k] = summarize(evaluate(S, D, th, j), D['N'], j.cpu().numpy())
        r['nll_train'] = float(L[tr_t].mean())
        return th, r, S
    t0 = time.time()
    th, r, S = do_eval(m, th, 0)
    record['iters'].append(r); record['bead'] = r
    log(f'it 0 (bead PSF): dev/px ctr median train {r["train"]["ctr_median"]:.4f} test {r["test"]["ctr_median"]:.4f} file3 {r["file3"]["ctr_median"]:.4f}; bins test {np.round(r["test"]["ctr_bins"], 3).tolist()}; ch {np.round(r["test"]["ch_median"], 3).tolist()}')
    for it in range(1, a.outer + 1):
        psf_step(m, D, th, tr_t, groups, steps=a.steps, batch=a.batch, lr_scale=a.lr * (0.7 ** (it - 1)), seed=it, log=log, pupil_smooth=a.pupil_smooth)
        th, r, S = do_eval(m, th, it)
        st = m.state(); r['state_brief'] = dict(phi=st['psf']['phi'], tau=st['psf']['tau'], a=st['psf']['a'], gamma=st['psf']['gamma'],
                                                d_bot=st['psf']['d_bot'], blur=st['psf']['blur_px2'], reg=st['reg'], beta=st['beta'])
        record['iters'].append(r)
        log(f'it {it} ({time.time() - t0:.0f}s): dev/px ctr median train {r["train"]["ctr_median"]:.4f} test {r["test"]["ctr_median"]:.4f} file3 {r["file3"]["ctr_median"]:.4f}; bins test {np.round(r["test"]["ctr_bins"], 3).tolist()}; ch {np.round(r["test"]["ch_median"], 3).tolist()}')
        log(f'   NA {[round(st["arms"][k]["NA"], 4) for k in ("top", "bot")]} apod {[np.round(st["arms"][k]["apod"], 3).tolist() for k in ("top", "bot")]}')
        log(f'   phi {np.round(st["psf"]["phi"], 3).tolist()} tau {np.round(st["psf"]["tau"], 3).tolist()} a {st["psf"]["a"]:.3f} gamma {st["psf"]["gamma"]:.3f} d_bot {np.round(st["psf"]["d_bot"], 3).tolist()} blur {st["psf"]["blur_px2"]:.3f} reg {np.round(st["reg"], 3).tolist()} beta {st["beta"]:.3f}')
        json.dump(st, open(f'{od}/state.json', 'w'))
        json.dump(record, open(f'{od}/eval.json', 'w'), indent=1)
    np.savez(f'{od}/theta.npz', th=th.cpu().numpy(), idx=D['idx'], use=use, train=tr, **{f'ev_{k}': v for k, v in ev_sets.items()})
    log(f'done {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
