"""(2026-11-11 in-situ 3D PSF) Per-pattern-angle PSF quantification of a fitted in-situ PSF (fit_insitu2 _psf.pt with a
per-angle dipole tensor / blur in 'extras').
 1. shape per angle at z = 0 / 0.05 / 0.1 um: dM = Mxx - Myy (r < 3 px aperture, px^2 and nm^2), FWHM_x / FWHM_y (central
    profiles of the PSF sampled at 1/8 px, no pixel integration), vs the angle-averaged PSF;
 2. difference image PSF_0 - PSF_1 (pixel-integrated, 11 x 11) -> png;
 3. CRLB of x, y for one emitter in a 6-frame window (per-frame I free, bg per angle, z free; pattern-free, 1000 ph / frame,
    bg 10 ph / px) with the per-angle PSFs vs with the angle average as the truth;
 4. misspecification bias: noise-free expected 6-frame data rendered with the per-angle PSFs, fitted (Poisson MLE by LBFGS on
    x, y, z, I_1..6, bg_a) with the angle-averaged PSF, over sub-pixel positions and z: bias of x / y (nm).
usage: python analysis/psf3d/perangle.py <fit_psf.pt> <out prefix>"""
import sys, json, math
import numpy as np, torch
from simcode2.psf.vectorial import VectorialPSF

dev = 'cuda'
fit, outp = sys.argv[1], sys.argv[2]
S = torch.load(fit, weights_only=False); ex = S.get('extras', {})
focus = float(S['focus'])


def mk(R, os_=3, px=0.10833):
    m = VectorialPSF(R, pixelsize=px, oversample=os_, NA=float(S['NA']), wavelength=0.70, n_med=1.33, device=dev)
    with torch.no_grad():
        m.zcoef.copy_(S['zcoef'].to(dev)); m.apod.copy_(S['apod'].to(dev))
        if 'saf' in ex: m.saf.copy_(torch.tensor(ex['saf'], device=dev))
    for p_ in m.parameters(): p_.requires_grad_(False)
    return m


def kw_angle(k):
    kw = {}
    if 'dipole_M' in ex: kw['dipM'] = torch.tensor(ex['dipole_M'][k], device=dev)
    if 'dipole_w' in ex: kw['dipw'] = torch.tensor(ex['dipole_w'][min(k, len(ex['dipole_w']) - 1)], device=dev)
    if 'blur_px2' in ex: kw['blur'] = torch.tensor(ex['blur_px2'][min(k, len(ex['blur_px2']) - 1)], device=dev)
    return kw


KW = [kw_angle(0), kw_angle(1)]
def kw_mean():
    kw = {}
    if 'dipole_M' in ex: kw['dipM'] = torch.tensor(ex['dipole_M'], device=dev).mean(0)
    if 'dipole_w' in ex: kw['dipw'] = torch.tensor(ex['dipole_w'], device=dev).mean(0)
    if 'blur_px2' in ex: kw['blur'] = torch.tensor(ex['blur_px2'], device=dev).mean(0)
    return kw
KWM = kw_mean()
res = dict(fit=fit, extras=ex, shape={})

# ---- 1. shape
P11 = mk(11)
o = torch.arange(11, device=dev, dtype=torch.float32) - 5
r = torch.sqrt(o[None] ** 2 + o[:, None] ** 2); q = (r < 3).float()
fine = mk(41, os_=1, px=0.10833 / 8)                       # 1/8-px samples over +-2.5 px
of = (torch.arange(41, device=dev, dtype=torch.float32) - 20) / 8


def fwhm(prof):
    p = prof.cpu().numpy(); x = of.cpu().numpy(); h = p.max() / 2; i0 = int(p.argmax())
    l = i0
    while l > 0 and p[l] > h: l -= 1
    rr = i0
    while rr < len(p) - 1 and p[rr] > h: rr += 1
    xl = x[l] + (h - p[l]) / (p[l + 1] - p[l]) * (x[l + 1] - x[l]); xr = x[rr - 1] + (h - p[rr - 1]) / (p[rr] - p[rr - 1]) * (x[rr] - x[rr - 1])
    return float(xr - xl)


with torch.no_grad():
    for z in (0.0, 0.05, 0.1):
        pos = torch.tensor([[0.0, 0.0, z]], device=dev)
        row = {}
        for nm, kw in (('angle0', KW[0]), ('angle1', KW[1]), ('mean', KWM), ('iso_noextra', {})):
            im = P11(pos, focus, **kw)[0]; Sq = (im * q).sum()
            dM = float(((im * q) * (o[None] ** 2 - o[:, None] ** 2)).sum() / Sq)
            fi = fine(pos, focus, **{k: (v * 64 if k == 'blur' else v) for k, v in kw.items()})[0]   # blur in px^2 of the fine grid
            row[nm] = dict(dM_px2=dM, dM_nm2=dM * 108.33 ** 2, fwhm_x_px=fwhm(fi[20]), fwhm_y_px=fwhm(fi[:, 20]))
            row[nm]['fwhm_ratio_xy'] = row[nm]['fwhm_x_px'] / row[nm]['fwhm_y_px']
        row['dM_angle1_minus_angle0_px2'] = row['angle1']['dM_px2'] - row['angle0']['dM_px2']
        res['shape'][f'z{z:g}'] = row
        print(f'z {z:.2f}: ' + ' | '.join(f"{k} dM {v['dM_px2']:+.4f} px2 FWHM x/y {v['fwhm_x_px']:.3f}/{v['fwhm_y_px']:.3f}" for k, v in row.items() if isinstance(v, dict)))
    # ---- 2. difference image
    pos = torch.tensor([[0.0, 0.0, 0.05]], device=dev)
    A0, A1 = P11(pos, focus, **KW[0])[0], P11(pos, focus, **KW[1])[0]
    D = (A0 - A1).cpu().numpy(); res['diff_max_rel_peak'] = float(np.abs(D).max() / A0.max().item())
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    for a_, im, t in ((ax[0], A0.cpu().numpy(), 'angle 0 (k ~ x, E ~ y)'), (ax[1], A1.cpu().numpy(), 'angle 1 (k ~ y, E ~ x)')):
        h = a_.imshow(im, cmap='magma'); a_.set_title(t); fig.colorbar(h, ax=a_, shrink=0.8)
    v = np.abs(D).max(); h = ax[2].imshow(D / A0.max().item(), cmap='RdBu_r', vmin=-v / A0.max().item(), vmax=v / A0.max().item())
    ax[2].set_title('(PSF0 - PSF1) / peak'); fig.colorbar(h, ax=ax[2], shrink=0.8)
    for a_ in ax: a_.set_xlabel('x (px)'); a_.set_ylabel('y (px)')
    fig.suptitle(f'per-angle PSF, z 0.05 um ({fit.split("/")[-1]})'); fig.tight_layout(); fig.savefig(outp + '_diff.png', dpi=110)

# ---- 3. CRLB (pattern-free; per-frame I, bg per angle, x, y, z)
I0, B0 = 1000.0, 10.0
def crlb(xy, z, kws):
    pos = torch.tensor([[xy[0], xy[1], z]], device=dev, requires_grad=True)
    ims, J = [], []
    for k in range(6):
        kw = kws[k // 3]
        f = lambda p: P11(p, focus, **kw)[0].reshape(-1)
        im = f(pos); jac = torch.autograd.functional.jacobian(f, pos)[:, 0, :]            # [121, 3]
        ims.append(im.detach()); J.append(jac.detach())
    n = 3 + 6 + 2
    F = torch.zeros(n, n, device=dev, dtype=torch.float64)
    for k in range(6):
        mu = I0 * ims[k] + B0
        g = torch.zeros(121, n, device=dev, dtype=torch.float64)
        g[:, :3] = (I0 * J[k]).double(); g[:, 3 + k] = ims[k].double(); g[:, 9 + k // 3] = 1.0
        F += g.T @ (g / mu.double()[:, None])
    C = torch.linalg.inv(F)
    return C.diagonal()[:2].sqrt().cpu().numpy() * 108.33
cr = {}
for z in (0.02, 0.06, 0.12):
    a_ = crlb((0.2, -0.3), z, KW); b_ = crlb((0.2, -0.3), z, [KWM, KWM])
    cr[f'z{z:g}'] = dict(perangle_xy_nm=a_.tolist(), anglemean_xy_nm=b_.tolist(), ratio=(a_ / b_).tolist())
    print(f'CRLB z {z}: per-angle {a_.round(3)} nm, angle-mean {b_.round(3)} nm, ratio {(a_ / b_).round(4)}')
res['crlb_1000ph_bg10'] = cr

# ---- 4. misspecification bias (fit the angle-average PSF to per-angle expected data)
IPAT = {'equal': [1.0] * 6, 'angle0_only': [1.0, 1.0, 1.0, 0.05, 0.05, 0.05]}   # per-frame I / I0 (a blinking emitter on in angle 0 only)
def bias_grid(ip):
    sub = [(-0.4, -0.4), (0.0, 0.0), (0.25, -0.1), (0.5, 0.5), (0.45, 0.0), (0.0, 0.45)]
    zs = [0.02, 0.06, 0.12]
    out = []
    for z in zs:
        for sx, sy in sub:
            pos = torch.tensor([[sx, sy, z]], device=dev)
            with torch.no_grad():
                mu = torch.stack([I0 * IPAT[ip][k] * P11(pos, focus, **KW[k // 3])[0] + B0 for k in range(6)])   # [6,R,R]
            th = torch.tensor([sx, sy], device=dev, requires_grad=True); zl = torch.tensor([math.log(z / (0.2 - z))], device=dev, requires_grad=True)
            lI = torch.tensor([math.log(I0 * v) for v in IPAT[ip]], device=dev, requires_grad=True); lb = torch.full((2,), math.log(B0), device=dev, requires_grad=True)
            opt = torch.optim.Adam([th, zl, lI, lb], lr=3e-3)
            sat = (mu - mu * torch.log(mu)).sum()                                # deviance offset (saturated model)
            for it in range(2500):
                opt.zero_grad()
                p = torch.cat([th, 0.2 * torch.sigmoid(zl)])[None]
                m = torch.exp(lI)[:, None, None] * P11(p, focus, **KWM)[0][None] + torch.exp(lb)[[0, 0, 0, 1, 1, 1]][:, None, None]
                L = ((m - mu * torch.log(m)).sum() - sat); L.backward(); opt.step()
                if it == 1500:
                    for g_ in opt.param_groups: g_['lr'] = 5e-4
            d = (th.detach().cpu().numpy() - np.array([sx, sy])) * 108.33
            out.append(dict(ipat=ip, x=sx, y=sy, z=z, bias_x_nm=float(d[0]), bias_y_nm=float(d[1]), z_fit=float(0.2 * torch.sigmoid(zl).item())))
    return out
B = bias_grid('equal') + bias_grid('angle0_only'); res['misspec_bias'] = B
bx = np.array([b['bias_x_nm'] for b in B]); by = np.array([b['bias_y_nm'] for b in B])
for ip in IPAT:
    k_ = np.array([b['ipat'] == ip for b in B])
    print(f'misspecification bias ({ip}; angle-mean PSF fitted to per-angle data): max |x| {np.abs(bx[k_]).max():.3f} nm, max |y| {np.abs(by[k_]).max():.3f} nm, '
          f'max |dz| {max(abs(b["z_fit"] - b["z"]) for b, kk in zip(B, k_) if kk) * 1e3:.2f} nm')
res['misspec_bias_max_nm'] = [float(np.abs(bx).max()), float(np.abs(by).max())]
json.dump(res, open(outp + '.json', 'w'), indent=1)
