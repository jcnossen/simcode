"""Fit the vectorial PSF model to the bead z-stacks in PSF_calib/raw (data root).
The shipped fit models/insitu/beads/psf_fit_sign-1_na149_ball_drift.* (the Zernike init of the in-situ fit) was made with
  python -m simcode.psf.fit_beads --sign -1 --nball 32 --tag _na149_ball_drift
(per-stack xy drift is always fitted; stack 7 held out)."""
import argparse, json, os
import numpy as np, tifffile, torch
from scipy import ndimage as ndi
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from simcode.psf.vectorial import VectorialPSF
from simcode import config

ap = argparse.ArgumentParser()
ap.add_argument('--roi', type=int, default=26)
ap.add_argument('--dz', type=float, default=0.010)
ap.add_argument('--sign', type=float, default=1.0)   # stage direction
ap.add_argument('--gain', type=float, default=0.56)
ap.add_argument('--offset', type=float, default=106.34)
ap.add_argument('--iters', type=int, default=300)
ap.add_argument('--zmax', type=int, default=21)       # highest Noll index
ap.add_argument('--holdout', type=int, default=7)      # zstack index kept out of the fit (0 = none)
ap.add_argument('--na', type=float, default=1.49)
ap.add_argument('--fit-na', action='store_true')
ap.add_argument('--wl', type=float, default=0.70)       # emission wavelength [um]
ap.add_argument('--nmed', type=float, default=1.33)
ap.add_argument('--fit-nmed', action='store_true')
ap.add_argument('--nball', type=int, default=1)          # >1: model bead as uniformly stained sphere, K sample points
ap.add_argument('--chunk', type=int, default=4)
ap.add_argument('--tag', default='')
ap.add_argument('--out', default=os.path.join(config.out_dir('insitu'), 'psf_fit'))
ap.add_argument('--stacks', nargs='+', default=[f'PSF_calib/raw/zstack{i}-1.tif' for i in range(2, 8)],
                help='bead z-stack tifs (absolute or relative to the data root); stack ids are 2,3,... in this order (so --holdout 7 = 6th stack, as before)')
ap.add_argument('--pixelsize', type=float, default=0.10833)
ap.add_argument('--min_rel', type=float, default=0.3)   # bead detection: max-projection peak > bg + min_rel*(max-bg)
a = ap.parse_args()
a.stacks = [config.resolve(s, 'data') for s in a.stacks]
dev = 'cuda'
R = a.roi

# --- collect bead ROIs
rois, stack_id = [], []
for i, fn in enumerate(a.stacks, start=2):
    s = tifffile.imread(fn).astype(np.float32)
    mp = s.max(0); bgv = np.median(s)
    loc = (mp == ndi.maximum_filter(mp, 15)) & (mp > bgv + a.min_rel * (mp.max() - bgv))
    for y, x in zip(*np.nonzero(loc)):
        if R // 2 <= y < s.shape[1] - R // 2 and R // 2 <= x < s.shape[2] - R // 2:
            rois.append(s[:, y - R // 2:y - R // 2 + R, x - R // 2:x - R // 2 + R]); stack_id.append(i)
rois = np.stack(rois)                      # [nb, nz, R, R]
stack_id = np.array(stack_id)
nb, nz = rois.shape[:2]
data = torch.tensor((rois - a.offset) * a.gain, device=dev).clamp(min=0)   # photons (approx)
train = torch.tensor(stack_id != a.holdout, device=dev)
stack_t = torch.tensor(stack_id, device=dev)
print(f'{nb} beads, {nz} slices, holdout stack {a.holdout}: {int((~train).sum())} beads')

# --- init per-bead params
peak = data.amax((2, 3)).argmax(1).float()
bg0 = data[:, :, :2, :].median(-1).values.median(-1).values.median(-1).values
I0 = (data[torch.arange(nb), peak.long()] - bg0[:, None, None]).sum((1, 2)) * 1.3
model = VectorialPSF(R, pixelsize=a.pixelsize, oversample=2, zernike_indices=range(5, a.zmax + 1), fit_NA=a.fit_na, NA=a.na, wavelength=a.wl,
                     n_med=a.nmed, fit_nmed=a.fit_nmed, device=dev)
# quasi-uniform points in the unit ball (fixed), used to average the PSF over the bead volume
g = torch.Generator().manual_seed(0)
ball = torch.randn(4096, 3, generator=g); ball = ball / ball.norm(dim=1, keepdim=True) * torch.rand(4096, 1, generator=g) ** (1 / 3)
ball = ball[:a.nball].to(dev) if a.nball > 1 else torch.zeros(1, 3, device=dev)
p = dict(
    xy=torch.zeros(nb, 2, device=dev),
    zoff=((peak - (nz - 1) / 2) * a.dz * a.sign),
    logI=torch.log(I0.clamp(min=100)),
    bleach=torch.zeros(nb, device=dev),
    bg=bg0.clone(),
    ze=torch.tensor(0.05, device=dev),   # bead radius (bead center height above coverslip) [um]
    logdz=torch.tensor(0.0, device=dev),
    slice_logI=torch.zeros(int(stack_id.max()) + 1, nz, device=dev),
    drift=torch.zeros(int(stack_id.max()) + 1, 2, 4, device=dev),   # per-stack xy stage drift, cubic in slice [px]
)
for v in p.values(): v.requires_grad_(True)
slices = (torch.arange(nz, device=dev) - (nz - 1) / 2) * a.dz * a.sign

def render(idx):
    b = idx.numel()
    r = p['ze'].clamp(0, 0.5)
    tt = torch.linspace(-1, 1, nz, device=dev)
    basis = torch.stack([tt, tt ** 2, tt ** 3, tt ** 4], 0)          # no constant term (that is the bead xy)
    dxy = torch.einsum('bcd,dz->bzc', p['drift'][stack_t[idx]], basis)  # [b, nz, 2]
    xy = (p['xy'][idx, None, :] + dxy).reshape(-1, 2)
    pos = torch.cat([xy, r.expand(b * nz, 1)], 1)
    focus = (slices[None] * torch.exp(p['logdz']) - p['zoff'][idx, None]).reshape(-1) + r
    K = ball.shape[0]
    off = ball * r * torch.tensor([1 / model.pixelsize, 1 / model.pixelsize, 1.0], device=dev)  # px, px, um
    pos = (pos[:, None] + off[None]).reshape(-1, 3)
    focus = focus.repeat_interleave(K)
    sl = p['slice_logI'] - p['slice_logI'].mean(1, keepdim=True)
    I = torch.exp(p['logI'][idx, None] + sl[stack_t[idx]] - p['bleach'][idx, None] * torch.arange(nz, device=dev)[None] / nz).reshape(-1)
    mu = model(pos, focus).reshape(b, nz, K, R, R).mean(2)
    mu = mu * I.reshape(b, nz, 1, 1) + p['bg'][idx].reshape(b, 1, 1, 1)
    return mu

def nll(idx):
    mu = render(idx).clamp(min=1e-3)
    d = data[idx]
    return (mu - d * torch.log(mu)).sum() / idx.numel()

tr_idx = torch.nonzero(train)[:, 0]
params = list(p.values()) + [model.zcoef, model.NA]
opt = torch.optim.Adam([{'params': list(p.values()), 'lr': 0.02},
                        {'params': [model.zcoef], 'lr': 0.005},
                        {'params': [model.NA, model.n_med], 'lr': 0.002}])
for it in range(a.iters):
    opt.zero_grad()
    loss = 0
    for chunk in tr_idx.split(a.chunk):
        l = nll(chunk) * chunk.numel() / tr_idx.numel()
        l.backward(); loss += l.item()
    opt.step()
    with torch.no_grad():
        model.NA.clamp_(1.0, 1.515); model.n_med.clamp_(1.0, 1.52)
    if it % 25 == 0 or it == a.iters - 1:
        print(f'{it:4d} nll/bead {loss:.1f}  NA {model.NA.item():.4f}  ze {p["ze"].item():.3f}  '
              f'|zcoef| {model.zcoef.norm().item():.3f}  n_med {model.n_med.item():.3f}  dz {a.dz*torch.exp(p["logdz"]).item()*1e3:.2f}nm')

# holdout: refit only per-bead params with global PSF frozen
ho_idx = torch.nonzero(~train)[:, 0]
if len(ho_idx):
    model.zcoef.requires_grad_(False); model.NA.requires_grad_(False)
    model.n_med.requires_grad_(False)
    opt2 = torch.optim.Adam([p['xy'], p['zoff'], p['logI'], p['bleach'], p['bg'], p['slice_logI'], p['drift']], lr=0.02)
    for it in range(200):
        opt2.zero_grad()
        for c in ho_idx.split(a.chunk):
            l = nll(c); l.backward()
        opt2.step()

# --- report: chi2-like residual stats (Poisson deviance per pixel) for train / holdout
with torch.no_grad():
    mu = torch.cat([render(c) for c in torch.arange(nb, device=dev).split(a.chunk)]).clamp(min=1e-3)
    d = data
    dev_px = 2 * (mu - d + torch.where(d > 0, d * torch.log(d / mu), torch.zeros_like(d)))
    for name, sel in [('train', train), ('holdout', ~train)]:
        if sel.any():
            print(f'{name}: mean Poisson deviance/pixel {dev_px[sel].mean().item():.3f} (≈1 if model+noise exact; '
                  f'gain/EM excess noise inflates this)')
res = dict(drift_px_rms=float(p['drift'].detach().norm(dim=(1,2)).max()), n_med=model.n_med.item(), dz_nm=a.dz*np.exp(p['logdz'].item())*1e3, NA=model.NA.item(), ze=p['ze'].item(), zernike={int(j): float(c) for j, c in zip(model.zidx, model.zcoef.detach().cpu())},
           sign=a.sign, final_nll=loss)
print(json.dumps(res, indent=1))
json.dump(res, open(a.out + f'_sign{int(a.sign)}{a.tag}.json', 'w'), indent=1)
torch.save(dict(n_med=model.n_med.detach().cpu(), zcoef=model.zcoef.detach().cpu(), NA=model.NA.detach().cpu(), ze=p['ze'].detach().cpu()), a.out + f'_sign{int(a.sign)}{a.tag}.pt')

# --- plot xz sections data vs model for 3 beads (incl. holdout)
sel = list(tr_idx[:2].cpu().numpy()) + (list(ho_idx[:1].cpu().numpy()) if len(ho_idx) else [])
fig, ax = plt.subplots(len(sel), 3, figsize=(12, 3.2 * len(sel)))
for r, b in enumerate(sel):
    dd = d[b, :, R // 2].cpu().numpy(); mm = mu[b, :, R // 2].cpu().numpy()
    vm = dd.max()
    ax[r, 0].imshow(dd.T, aspect='auto', vmax=vm); ax[r, 0].set_title(f'bead {b} (stack {stack_id[b]}) data xz')
    ax[r, 1].imshow(mm.T, aspect='auto', vmax=vm); ax[r, 1].set_title('model xz')
    ax[r, 2].imshow((dd - mm).T, aspect='auto', cmap='RdBu', vmin=-vm / 5, vmax=vm / 5); ax[r, 2].set_title('residual')
plt.tight_layout(); plt.savefig(a.out + f'_sign{int(a.sign)}{a.tag}_xz.png', dpi=70)
