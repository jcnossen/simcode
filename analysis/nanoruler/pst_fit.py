"""
Part 1: shared / per-step / hybrid pattern fits of the 80 nm nanoruler in-situ intensities, on all emitters and on disjoint
random halves of the binding sites (two independent splits), for several knot spacings. Held-out evaluation: pattern fitted
on one half, amplitudes refitted and the robust residual evaluated on the other half.
Output: out/nanoruler/perstep/fits.pt  {configs: {name: {full, halves[split][half]: state, heldout: ...}}}
usage: python analysis/nanoruler/pst_fit.py [nproc]
"""
import sys, time
import numpy as np, torch
sys.path.insert(0, '/home/jc/simcode2/analysis/nanoruler')
import pst_lib as L

CONFIGS = [('shared', kf, None) for kf in (3000, 1500, 750, 300, 150, 60)] + \
          [('perstep', kf, None) for kf in (3000, 1500, 750, 300, 150, 60)] + \
          [('hybrid', 300, 3000), ('hybrid', 300, 1500), ('hybrid', 150, 1500), ('hybrid', 150, 750), ('hybrid', 60, 1500)]
NSPLIT = 2
name = lambda c: f'{c[0]}_{c[1]}' + (f'_{c[2]}' if c[2] else '')


def job(arg):
    cfg, split, half = arg
    from simcode2.util.proctitle import set_title; set_title('nr-pstep')
    torch.set_num_threads(2)
    D = L.load_data(); n = len(D['I'])
    if split < 0: sel = np.ones(n, bool)
    else:
        h0 = L.site_halves(D, seed=split); sel = h0 if half == 0 else ~h0
    t = time.time()
    P, loss = L.fit(D, sel, cfg[0], cfg[1], cfg[2])
    out = dict(state=P.state(), train_loss=loss)
    if split >= 0:
        hl, z, dof = L.heldout(D, ~sel, P)
        out.update(held_huber=hl.astype(np.float32), held_z=z.astype(np.float32), held_dof=dof)
    print(f'{name(cfg)} split {split} half {half}: train {loss:.5f}' + (f' held {out["held_huber"].sum() / dof:.5f}' if split >= 0 else '')
          + f' ({time.time() - t:.0f} s)', flush=True)
    return arg, out


if __name__ == '__main__':
    import multiprocessing as mp
    from simcode2.util.proctitle import set_title; set_title('nr-pstep')
    jobs = [(c, -1, 0) for c in CONFIGS] + [(c, sp, h) for c in CONFIGS for sp in range(NSPLIT) for h in (0, 1)]
    with mp.get_context('spawn').Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as pool:
        res = pool.map(job, jobs, chunksize=1)
    R = {name(c): dict(cfg=c, halves={}) for c in CONFIGS}
    for (c, sp, h), o in res:
        if sp < 0: R[name(c)]['full'] = o
        else: R[name(c)]['halves'][(sp, h)] = o
    torch.save(R, L.OUT + 'fits.pt')
    print('saved')
