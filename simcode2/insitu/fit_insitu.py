"""
In-situ calibration on isolated emitter tracks.

Stage A (PSF + camera): per sample x,y,z (constant over the track), per-frame intensity and background;
  global Zernike coefficients, focal plane position, camera gain, read noise, offset.
  Likelihood: Gaussian approximation of Poisson + read noise, in photons, with Jacobian for the gain:
     d = (ADU - offset) * g ;  d ~ N(mu, mu + rn^2)  ;  logp(ADU) = logN(d) + log g
Stage B (pattern): from per-frame intensities, fit
     I[b,t] = A[b,angle] * relint[s] * (1 + m[s] cos(k[angle].r_b - phi[s] - dphi[angle](t)))
  on frames where the emitter is on for the full frame (neighbouring frames also bright).
"""
import argparse, json, math
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from simcode2 import config
from simcode2.psf.vectorial import VectorialPSF
from simcode2.insitu import BEAD_PSF, out

ap = argparse.ArgumentParser()
ap.add_argument('--tracks', default=out('tracks_0.5nM.npz'))
ap.add_argument('--psf', default=BEAD_PSF, help='initial Zernikes (bead fit, psf/fit_beads.py)')
ap.add_argument('--nsamples', type=int, default=8000)
ap.add_argument('--iters', type=int, default=1500)
ap.add_argument('--fit-zernike', type=int, default=1)
ap.add_argument('--gain', type=float, default=0.0)   # >0: fix camera gain [e/ADU]
ap.add_argument('--zmax', type=float, default=0.0)    # >0: emitter z constrained to [0, zmax] um (TIRF)
ap.add_argument('--fit-apod', type=int, default=0)
ap.add_argument('--fit-na', type=int, default=0)
ap.add_argument('--out', default=out('insitu_0.5nM'))
a = ap.parse_args()
a.psf = config.resolve(a.psf)
dev = 'cuda'
torch.manual_seed(0); np.random.seed(0)

D = np.load(a.tracks)
sel = np.random.permutation(len(D['rois']))[:a.nsamples]
adu = torch.tensor(D['rois'][sel].astype(np.float32), device=dev)
mask = torch.tensor(D['mask'][sel], device=dev)
frames = torch.tensor(D['frames'][sel], device=dev)
corner = torch.tensor(D['corner'][sel], device=dev, dtype=torch.float32)
B, T, R, _ = adu.shape
print(f'{B} samples, T={T}, roi {R}')

psf0 = torch.load(a.psf)
model = VectorialPSF(R, oversample=2, NA=1.49, wavelength=0.70, n_med=1.33, device=dev)
with torch.no_grad():
    model.zcoef.copy_(psf0['zcoef'].to(dev))
zc_bead = model.zcoef.detach().clone()
model.zcoef.requires_grad_(bool(a.fit_zernike))
model.apod.requires_grad_(bool(a.fit_apod)); model.NA.requires_grad_(bool(a.fit_na))
zfun = (lambda v: a.zmax * torch.sigmoid(v)) if a.zmax > 0 else torch.nn.functional.softplus

# ---- parameters
g0, off0 = (a.gain if a.gain > 0 else 0.5), 100.0
d0 = (adu - off0) * g0
bg_init = torch.quantile(d0.reshape(B, T, -1), 0.3, dim=2).clamp(min=0.5)
I_init = (d0 - bg_init[..., None, None]).clamp(min=0).sum((2, 3)) * 1.4
c = (R - 1) / 2
P = dict(
    xy=torch.tensor(D['xy_init'][sel], device=dev, dtype=torch.float32) - c,
    z=torch.full((B,), -2.0, device=dev),                 # softplus -> ~0.13 um
    Iraw=I_init.clamp(min=1.0) + torch.log(-torch.expm1(-I_init.clamp(min=1.0))),  # stable softplus^-1
    bg=bg_init.clone(),
)
G = dict(
    focus=torch.tensor(0.0, device=dev),
    loggain=torch.tensor(math.log(g0), device=dev),
    logrn=torch.tensor(math.log(1.5), device=dev),
    offset=torch.tensor(off0, device=dev),
)
for v in list(P.values()) + list(G.values()): v.requires_grad_(True)
sp = torch.nn.functional.softplus


def expected(idx):
    xy, z = P['xy'][idx], zfun(P['z'][idx])
    psf = model(torch.cat([xy, z[:, None]], 1), G['focus'].expand(len(idx)))   # [b,R,R], same PSF all frames
    return sp(P['Iraw'][idx])[..., None, None] * psf[:, None] + P['bg'][idx][..., None, None]


def negloglik(idx):
    mu = expected(idx).clamp(min=1e-3)
    g = torch.exp(G['loggain']); rn2 = torch.exp(2 * G['logrn'])
    d = (adu[idx] - G['offset']) * g
    var = mu + rn2
    nll = 0.5 * (d - mu) ** 2 / var + 0.5 * torch.log(2 * math.pi * var) - G['loggain']
    return torch.where(mask[idx][..., None, None], nll, torch.zeros_like(nll)).sum()


npx = mask.sum().item() * R * R
opt = torch.optim.Adam([
    {'params': [P['xy'], P['z'], P['Iraw'], P['bg']], 'lr': 0.03},
    {'params': [G['focus'], G['logrn']] + ([] if a.gain > 0 else [G['loggain']]), 'lr': 0.005},
    {'params': [G['offset']], 'lr': 0.05},
    {'params': [model.zcoef], 'lr': 0.002},
    {'params': [model.apod, model.NA], 'lr': 0.003},
])
chunks = torch.arange(B, device=dev).split(2000)
for it in range(a.iters):
    opt.zero_grad(); tot = 0
    for ch in chunks:
        l = negloglik(ch) / npx; l.backward(); tot += l.item()
    if it < 200:  # warm-up: emitter params only
        for v in G.values(): v.grad = None
        model.zcoef.grad = None; model.apod.grad = None; model.NA.grad = None
    opt.step()
    with torch.no_grad(): model.NA.clamp_(1.0, 1.515)
    if it % 100 == 0 or it == a.iters - 1:
        print(f'{it:5d} nll/px {tot:.4f}  gain {torch.exp(G["loggain"]).item():.3f} e/ADU  rn {torch.exp(G["logrn"]).item():.2f} e  '
              f'offset {G["offset"].item():.2f}  focus {G["focus"].item():.3f}um  z med {zfun(P["z"]).median().item():.3f}um  NA {model.NA.item():.3f} apod {[round(v,2) for v in model.apod.tolist()]}  '
              f'|dZ| {(model.zcoef - zc_bead).norm().item():.3f}', flush=True)

# ---- goodness of fit: normalized residuals should be ~N(0,1)
with torch.no_grad():
    res = []
    for ch in chunks:
        mu = expected(ch); g = torch.exp(G['loggain']); rn2 = torch.exp(2 * G['logrn'])
        d = (adu[ch] - G['offset']) * g
        r = ((d - mu) / torch.sqrt(mu.clamp(min=1e-3) + rn2))[mask[ch]]
        res.append(r)
    res = torch.cat(res)
    print(f'normalized residual: mean {res.mean().item():.3f}  var {res.var().item():.3f} (1 = model matches noise)')
    Ifit = sp(P['Iraw']).detach(); xyg = (P['xy'].detach() + c + corner)
    zfit = zfun(P['z']).detach()

out = dict(gain=torch.exp(G['loggain']).item(), readnoise_e=torch.exp(G['logrn']).item(), offset=G['offset'].item(),
           focus=G['focus'].item(), zernike=model.zcoef.detach().cpu().tolist(), zernike_bead=zc_bead.cpu().tolist(),
           NA=model.NA.item(), apod=model.apod.tolist(), z_median=zfit.median().item(), resid_var=res.var().item())
json.dump(out, open(a.out + '_stageA.json', 'w'), indent=1)
torch.save(dict(I=Ifit.cpu(), bg=P['bg'].detach().cpu(), xy=xyg.cpu(), z=zfit.cpu(), frames=frames.cpu(), mask=mask.cpu(),
                zcoef=model.zcoef.detach().cpu(), NA=model.NA.detach().cpu(), apod=model.apod.detach().cpu(), **{k: v.detach().cpu() for k, v in G.items()}), a.out + '_stageA.pt')
plt.figure(figsize=(10, 4))
plt.subplot(121); plt.hist(zfit.cpu().numpy(), 60); plt.xlabel('fitted emitter z [um]')
plt.subplot(122); m0 = mask.cpu().numpy(); plt.hist(Ifit.cpu().numpy()[m0], 100, range=(0, 3000)); plt.xlabel('photons/frame')
plt.tight_layout(); plt.savefig(a.out + '_stageA.png', dpi=70)
print(json.dumps(out))
