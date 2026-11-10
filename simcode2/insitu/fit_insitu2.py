"""
In-situ stage A with variable projection: per-frame intensity + per-(sample, pattern angle) background are solved
exactly (weighted LSQ) at every iteration; per-sample x,y,z and global PSF/camera parameters by Adam.
Afterwards computes per-frame intensities for all samples with the final PSF (input for fit_pattern.py).
"""
import argparse, json, math
import numpy as np, torch
from simcode2.psf.vectorial import VectorialPSF
from simcode2.insitu.varpro import solve_linear
from simcode2 import config
from simcode2.insitu import out

ap = argparse.ArgumentParser()
ap.add_argument('--tracks', default=out('tracks_0.5nM.npz'))
ap.add_argument('--psf', default='')                  # optional init (zcoef) from a previous fit
ap.add_argument('--nsamples', type=int, default=8000)
ap.add_argument('--iters', type=int, default=600)
ap.add_argument('--gain', type=float, default=0.56)
ap.add_argument('--zmax', type=float, default=0.2)
ap.add_argument('--fit-zernike', type=int, default=1)
ap.add_argument('--fit-apod', type=int, default=0)
ap.add_argument('--fit-na', type=int, default=0)
ap.add_argument('--out', default=out('insitu2_0.5nM'))
a = ap.parse_args()
if a.psf: a.psf = config.resolve(a.psf)
dev = 'cuda'
torch.manual_seed(0); np.random.seed(0)

D = np.load(a.tracks)
Ball = len(D['rois'])
R = D['rois'].shape[-1]; c = (R - 1) / 2
model = VectorialPSF(R, oversample=2, NA=1.49, wavelength=0.70, n_med=1.33, device=dev)
if a.psf:
    with torch.no_grad(): model.zcoef.copy_(torch.load(a.psf)['zcoef'].to(dev))
model.zcoef.requires_grad_(bool(a.fit_zernike)); model.apod.requires_grad_(bool(a.fit_apod)); model.NA.requires_grad_(bool(a.fit_na))
G = dict(focus=torch.tensor(0.0, device=dev), logrn=torch.tensor(math.log(1.9), device=dev), offset=torch.tensor(99.3, device=dev))
G['focus'].requires_grad_(True)  # camera (offset, read noise) fixed from fit_insitu (degenerate with bg here)
g = a.gain


def load(idx):
    adu = torch.tensor(D['rois'][idx].astype(np.float32), device=dev)
    return dict(adu=adu, mask=torch.tensor(D['mask'][idx], device=dev), frames=torch.tensor(D['frames'][idx], device=dev),
                corner=torch.tensor(D['corner'][idx], device=dev, dtype=torch.float32),
                xy=(torch.tensor(D['xy_init'][idx], device=dev, dtype=torch.float32) - c).requires_grad_(True),
                z=torch.zeros(len(idx), device=dev, requires_grad=True))


def loss_and_linear(S, sl=slice(None)):
    adu, mask, frames = S['adu'][sl], S['mask'][sl], S['frames'][sl]
    b = adu.shape[0]
    psf = model(torch.cat([S['xy'][sl], a.zmax * torch.sigmoid(S['z'][sl])[:, None]], 1), G['focus'].expand(b))
    data = (adu - G['offset']) * g
    rn2 = torch.exp(2 * G['logrn'])
    group = torch.where(mask, (frames % 6) // 3, torch.zeros_like(frames))
    I, bg = solve_linear(psf.reshape(b, -1).detach(), data.reshape(b, adu.shape[1], -1).detach(), mask, group, rn2.detach())
    mu = (I[..., None, None] * psf[:, None] + torch.gather(bg, 1, group)[..., None, None]).clamp(min=1e-3)
    var = mu + rn2
    nll = 0.5 * (data - mu) ** 2 / var + 0.5 * torch.log(2 * math.pi * var)
    nll = torch.where(mask[..., None, None], nll, torch.zeros_like(nll)).sum()
    return nll, I, bg, mu, var, data


# ---- stage A on a subset
sel = np.sort(np.random.permutation(Ball)[:a.nsamples])
S = load(sel)
npx = S['mask'].sum().item() * R * R
opt = torch.optim.Adam([{'params': [S['xy'], S['z']], 'lr': 0.03},
                        {'params': [G['focus']], 'lr': 0.005},
                        {'params': [model.zcoef], 'lr': 0.002}, {'params': [model.apod, model.NA], 'lr': 0.003}])
chunks = torch.arange(len(sel), device=dev).split(2000)
for it in range(a.iters):
    opt.zero_grad(); tot = 0
    for ch in chunks:
        l = loss_and_linear(S, ch)[0] / npx; l.backward(); tot += l.item()
    if it < 100:
        for v in list(G.values()) + [model.zcoef, model.apod, model.NA]: v.grad = None
    opt.step()
    with torch.no_grad(): model.NA.clamp_(1.0, 1.515)
    if it % 100 == 0 or it == a.iters - 1:
        print(f'{it:5d} nll/px {tot:.4f}  rn {torch.exp(G["logrn"]).item():.2f}e  offset {G["offset"].item():.2f}  '
              f'focus {G["focus"].item():.3f}  z med {(a.zmax*torch.sigmoid(S["z"])).median().item():.3f}  '
              f'NA {model.NA.item():.3f}  apod {[round(v, 2) for v in model.apod.tolist()]}  |Z| {model.zcoef.norm().item():.3f}', flush=True)

with torch.no_grad():
    rs = []
    for ch in chunks:
        _, I, bg, mu, var, data = loss_and_linear(S, ch)
        rs.append(((data - mu) / var.sqrt())[S['mask'][ch]])
    rs = torch.cat(rs)
print(f'normalized residual var {rs.var().item():.3f}')
res = dict(gain=g, readnoise_e=torch.exp(G['logrn']).item(), offset=G['offset'].item(), focus=G['focus'].item(),
           NA=model.NA.item(), apod=model.apod.tolist(), zernike=model.zcoef.tolist(), resid_var=rs.var().item(), zmax=a.zmax)
json.dump(res, open(a.out + '_psf.json', 'w'), indent=1)
torch.save(dict(zcoef=model.zcoef.detach().cpu(), NA=model.NA.detach().cpu(), apod=model.apod.detach().cpu(),
                **{k: v.detach().cpu() for k, v in G.items()}, gain=g, zmax=a.zmax), a.out + '_psf.pt')
print(json.dumps({k: v for k, v in res.items() if k != 'zernike'}))

# ---- intensities for all samples (PSF frozen; per-sample xy,z refit)
for prm in [model.zcoef, model.apod, model.NA] + list(G.values()): prm.requires_grad_(False)
Iall = torch.zeros(Ball, D['rois'].shape[1]); rall = torch.zeros(Ball, 2); zall = torch.zeros(Ball)
for start in range(0, Ball, 8000):
    idx = np.arange(start, min(Ball, start + 8000))
    S2 = load(idx)
    o2 = torch.optim.Adam([S2['xy'], S2['z']], lr=0.03)
    for it in range(150):
        o2.zero_grad()
        for ch in torch.arange(len(idx), device=dev).split(4000):
            loss_and_linear(S2, ch)[0].backward()
        o2.step()
    with torch.no_grad():
        for ch in torch.arange(len(idx), device=dev).split(4000):
            _, I, bg, *_ = loss_and_linear(S2, ch)
            Iall[idx[ch.cpu().numpy()]] = I.cpu()
        rall[idx] = (S2['xy'] + c + S2['corner']).detach().cpu()
        zall[idx] = (a.zmax * torch.sigmoid(S2['z'])).detach().cpu()
mask = torch.tensor(D['mask']); frames = torch.tensor(D['frames'])
med = (Iall * mask).sum(1) / mask.sum(1)
bright = (Iall > 0.25 * med[:, None]) & mask
full = bright.clone(); full[:, 1:] &= bright[:, :-1]; full[:, :-1] &= bright[:, 1:]; full[:, 0] = False; full[:, -1] = False
torch.save(dict(I=Iall, r=rall, z=zall, frames=frames, mask=mask, full=full), a.out + '_intensities.pt')
print('saved intensities', a.out + '_intensities.pt', 'median I', Iall[mask].median().item())
