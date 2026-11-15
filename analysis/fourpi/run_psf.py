"""Summed-channel PSF: bead z-stack fit -> in-situ fit on bright isolated 6-frame 4-channel sums -> held-out deviance.
usage: python run_psf.py <cell> <tag> [--px 0.128 --NA 1.35 --lam 0.60 --nmed 1.406 --two_arm 0 --ntrain 3000 --nheld 2000]
Outputs out/4pi/smlm/psf/<cell>_<tag>.json (+ _knots.npz with --knots 1).
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.config import root; P4 = root('fourpi')
import os, sys, json, argparse, time, glob
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from psffit import SumPSF, fit, z_grid_init, summary, render_knots, nll, varpro_Nbg

ap = argparse.ArgumentParser()
ap.add_argument('cell'); ap.add_argument('tag')
ap.add_argument('--px', type=float, default=0.128); ap.add_argument('--NA', type=float, default=1.35)
ap.add_argument('--lam', type=float, default=0.60); ap.add_argument('--nmed', type=float, default=1.406)
ap.add_argument('--nimm', type=float, default=1.406); ap.add_argument('--depth', type=float, default=0.0)
ap.add_argument('--two_arm', type=int, default=0)
ap.add_argument('--ntrain', type=int, default=3000); ap.add_argument('--nheld', type=int, default=2000)
ap.add_argument('--iters', type=int, default=300); ap.add_argument('--knots', type=int, default=0)
ap.add_argument('--init', default=''); ap.add_argument('--threads', type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(0)
OUT = f'{P4}/smlm/psf'; os.makedirs(OUT, exist_ok=True)
t0 = time.time()
res = dict(args=vars(a))
kw = dict(px=a.px, NA=a.NA, n_imm=a.nimm, n_med=a.nmed, lam=a.lam, depth=a.depth)

# ---------------- bead z-stack ----------------
Rb = 21
B = np.load(f'{P4}/smlm/zcal/zstack_rois_{a.cell}.npy').sum(1)       # [61,21,21] e (4-ch sum)
zc = json.load(open(f'{P4}/smlm/zcal/zcal_{a.cell}.json'))
zst = np.array(zc['zstage']); good = np.array(zc['good']) & (np.abs(zst) <= 0.7)
db = torch.as_tensor(B[good], dtype=torch.float32)
xyb = torch.as_tensor(np.c_[zc['x'], zc['y']][good], dtype=torch.float32)
best = None
for s in (1.0, -1.0):
    mb = SumPSF(Rb, **kw)
    with torch.no_grad():
        mb.p1.zcoef[0] = 0.1 * s   # Noll 5 start: some astigmatism (sign set by the bead's sx/sy order)
    r = fit(mb, db, xyb, 0.0, 4 * 1.3, iters=500, lr_pos=0.02, lr_glob=0.01, bead=dict(zstage=zst[good], s=s))
    L = r['nll'].sum().item()
    print(f'bead fit s={s:+.0f}: deviance/2 {L:.1f} over {len(db)} slices, z0 {r["zoff"]:+.3f}', flush=True)
    if best is None or L < best[0]: best = (L, s, mb, r)
L, s, mb, rb = best
res['bead'] = dict(s=s, deviance_half=L, n=int(len(db)), z0=rb['zoff'], psf=summary(mb),
                   dev_per_slice=rb['nll'].tolist(), zstage=zst[good].tolist(), npix=Rb * Rb)
print('bead fit:', json.dumps(summary(mb)), f'{time.time() - t0:.0f}s', flush=True)

# ---------------- in-situ samples ----------------
fs = sorted(glob.glob(f'{P4}/smlm/p1/p1_{a.cell}_*.npz'))
rois, xy0 = [], []
for f in fs:
    D = np.load(f)
    g = D['gfit']; dev = D['gdev']
    sel = (D['nn'] >= 8) & (g[:, 2] > 1500) & (g[:, 4] < 3.5) & (g[:, 5] < 3.5) & (np.abs(g[:, 0]) < 1) & (np.abs(g[:, 1]) < 1)
    sel &= (-dev / 225 < 2.0)
    idx = np.nonzero(sel)[0]
    rois.append(D['roi_sum'][idx]); xy0.append(g[idx, :2])
rois = np.concatenate(rois); xy0 = np.concatenate(xy0)
rng = np.random.default_rng(1); perm = rng.permutation(len(rois))
tr, he = perm[:a.ntrain], perm[a.ntrain:a.ntrain + a.nheld]
res['n_pool'] = int(len(rois))
R = rois.shape[-1]
m = SumPSF(R, two_arm=bool(a.two_arm), **kw)
sd = {k: v for k, v in mb.state_dict().items() if k.startswith('p1.zcoef') or k == 'logblur'}
m.load_state_dict(sd, strict=False)
if a.init:
    m.load_state_dict(torch.load(a.init), strict=False)
if a.two_arm:
    with torch.no_grad():
        m.p2.zcoef.copy_(m.p1.zcoef); m.p2.zcoef.add_(0.02 * torch.randn_like(m.p2.zcoef))
v = 24 * 1.3
d = torch.as_tensor(rois[tr], dtype=torch.float32); xy = torch.as_tensor(xy0[tr], dtype=torch.float32)
z0 = z_grid_init(m, d, xy, v)
r = fit(m, d, xy, z0, v, iters=a.iters, lr_pos=0.02, lr_glob=0.005, log=True)
# second round: re-init z by grid with the updated PSF (escape wrong astigmatic branch), short refit
z1 = z_grid_init(m, d, r['xy'], v)
r = fit(m, d, r['xy'], z1, v, iters=a.iters // 2, lr_pos=0.02, lr_glob=0.003, log=True)
res['insitu'] = dict(psf=summary(m), train_dev_half_mean=float(r['nll'].mean()), z_train=r['z'].tolist()[:2000],
                     N_train_median=float(r['N'].median()), npix=R * R)
torch.save(m.state_dict(), f'{OUT}/{a.cell}_{a.tag}_state.pt')
print('in-situ fit:', json.dumps(summary(m)), f'{time.time() - t0:.0f}s', flush=True)
# ---------------- held-out (globals fixed) ----------------
dh = torch.as_tensor(rois[he], dtype=torch.float32); xyh = torch.as_tensor(xy0[he], dtype=torch.float32)
zh = z_grid_init(m, dh, xyh, v)
rh = fit(m, dh, xyh, zh, v, iters=120, lr_pos=0.02, fit_glob=False)
res['heldout'] = dict(dev_half_mean=float(rh['nll'].mean()), dev_half_median=float(rh['nll'].median()), n=int(len(dh)),
                      z=rh['z'].tolist(), N=rh['N'].tolist(), bg=rh['bg'].tolist())
# residual images by z bin (held-out): mean (data - model) / mean model peak
resid = (dh - rh['mu']).numpy(); zz = rh['z'].numpy()
edges = np.quantile(zz, np.linspace(0, 1, 7))
res['heldout']['resid_zbins'] = []
for i in range(6):
    s_ = (zz >= edges[i]) & (zz <= edges[i + 1])
    mr = resid[s_].mean(0); mm = rh['mu'][s_].numpy().mean(0)
    res['heldout']['resid_zbins'].append(dict(z=[float(edges[i]), float(edges[i + 1])], n=int(s_.sum()),
                                             rel_rms=float(np.sqrt((mr ** 2).mean()) / (mm.max() - np.median(mm))),
                                             mean_resid=mr.round(3).tolist(), mean_model=mm.round(3).tolist()))
print(f'held-out deviance/2 per sample: mean {res["heldout"]["dev_half_mean"]:.2f} median {res["heldout"]["dev_half_median"]:.2f} '
      f'(npix {R * R}); rel resid rms by z bin', [round(b['rel_rms'], 4) for b in res['heldout']['resid_zbins']], f'{time.time() - t0:.0f}s', flush=True)
json.dump(res, open(f'{OUT}/{a.cell}_{a.tag}.json', 'w'))
if a.knots:
    G, u0, zmin, hz = render_knots(m, -0.9, 0.9, hz=0.01, S=4, P=15)
    np.savez_compressed(f'{OUT}/{a.cell}_{a.tag}_knots.npz', zstack=G.numpy().astype(np.float32), u0=u0, S=4, z0=zmin, hz=hz)
    print('knots saved', tuple(G.shape), f'{time.time() - t0:.0f}s', flush=True)
