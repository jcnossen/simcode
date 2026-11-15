"""
In-situ stage A with variable projection: per-frame intensity + per-(sample, pattern angle) background are solved
exactly (weighted LSQ) at every iteration; per-sample x,y,z and global PSF/camera parameters by Adam.
Afterwards computes per-frame intensities for all samples with the final PSF (input for fit_pattern.py).
(2026-11-11 in-situ 3D PSF) extra PSF terms, used in the stages of the paper's PSF calibration (out/psf3d/run_final*.sh: base,
then --blur line, then --init-fit <bl> --freeze-base 1 with --blur angle_delta or --photosel 3):
  --blur line: image-plane line blur v v^T (VectorialPSF.blur, covariance sxx, syy, sxy px^2; a strictly anisotropic, PSD blur);
  --blur angle_delta (with --init-fit): the init fit's blur +- a traceless per-angle part (d, -d, e) (angle 0: -, angle 1: +);
  --photosel 3: excitation photoselection per pattern angle a, M_a = (1 - gamma) diag((1 - zeta) / 2, (1 - zeta) / 2, zeta) +
    gamma e_a e_a^T, e_a the in-plane unit vector perpendicular to k_a (s-polarised excitation; k from --pol-calib's pattern
    block), one shared gamma and z-fraction zeta;
  --init-fit <_psf.pt> [--freeze-base 1]: start from (and optionally freeze) a previous fit, to test one added term cleanly.
--split cb0 | cb1: train only on the samples of one checkerboard parity of --tile px tiles (emitter position), the other parity
is the held-out set of psf_heldout.py. --intensities 1 adds the all-sample intensity stage (input of fit_pattern.py).
(2026-11-14) Removed options no run used: --psf, --fit-zernike, --fit-apod, --fit-na, --dipole, --fit-saf, --gamma-init,
--photosel 1 / 2, --blur global / angle / angle_line; --intensities defaults to 0 (every run passed it).
"""
import argparse, json, math
import numpy as np, torch
from simcode.psf import VectorialPSF
from simcode.psf import solve_linear
from simcode import config
from simcode.calibration import out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracks', default=out('tracks_0.5nM.npz'))
    ap.add_argument('--nsamples', type=int, default=8000)
    ap.add_argument('--iters', type=int, default=600)
    ap.add_argument('--gain', type=float, default=0.56)
    ap.add_argument('--zmax', type=float, default=0.2)
    ap.add_argument('--out', default=out('insitu2_0.5nM'))
    ap.add_argument('--blur', default='none', choices=['none', 'line', 'angle_delta'])
    ap.add_argument('--init-fit', default='')     # start from a previous fit_psf_tracks _psf.pt (Zernikes, focus, its global blur as fixed base)
    ap.add_argument('--freeze-base', type=int, default=0)   # with --init-fit: train only the new extra terms
    ap.add_argument('--photosel', type=int, default=0, choices=[0, 3])
    ap.add_argument('--pol-calib', default='insitu/calib_0.5nM.json')
    ap.add_argument('--split', default='', choices=['', 'cb0', 'cb1'])
    ap.add_argument('--tile', type=int, default=16)
    ap.add_argument('--intensities', type=int, default=0)
    a = ap.parse_args()
    dev = 'cuda'
    torch.manual_seed(0); np.random.seed(0)

    D = np.load(a.tracks)
    Ball = len(D['rois'])
    R = D['rois'].shape[-1]; c = (R - 1) / 2
    model = VectorialPSF(R, oversample=2, NA=1.49, wavelength_um=0.70, n_med=1.33, device=dev)
    model.zcoef.requires_grad_(True); model.apod.requires_grad_(False); model.NA.requires_grad_(False)
    G = dict(focus=torch.tensor(0.0, device=dev), logrn=torch.tensor(math.log(1.9), device=dev), offset=torch.tensor(99.3, device=dev))
    G['focus'].requires_grad_(True)  # camera (offset, read noise) fixed from fit_insitu (degenerate with bg here)
    g = a.gain
    BL0 = None
    if a.init_fit:
        S0_ = torch.load(config.resolve(a.init_fit), weights_only=False)
        with torch.no_grad():
            model.zcoef.copy_(S0_['zcoef'].to(dev)); G['focus'].fill_(float(S0_['focus']))
        if 'blur_px2' in S0_.get('extras', {}): BL0 = torch.tensor(S0_['extras']['blur_px2'][0], device=dev)
        if a.freeze_base:
            model.zcoef.requires_grad_(False); G['focus'].requires_grad_(False)
    BL = BLD = None   # blur parameters: line vector [1, 2]; per-angle traceless delta [2]
    if a.blur == 'angle_delta':
        assert BL0 is not None, '--blur angle_delta needs --init-fit with a global blur'
        BLD = torch.zeros(2, device=dev, requires_grad=True)
    elif a.blur == 'line':
        BL = torch.zeros(1, 2, device=dev)
        BL[:, 0] = 0.05      # line blur v = (vx, vy): start off zero (the gradient of v v^T vanishes at 0)
        BL.requires_grad_(True)
    GAM = None
    if a.photosel:
        kc = torch.tensor(json.load(open(config.resolve(a.pol_calib)))['pattern']['k_rad_per_px'], device=dev)
        KDIR = kc / kc.norm(dim=1, keepdim=True)                                    # [2 angles, (x, y)]
        EDIR = torch.stack([-KDIR[:, 1], KDIR[:, 0]], 1)                            # s-pol: in-plane, perpendicular to k
        GAM = torch.tensor(0.05, device=dev, requires_grad=True)
    ZETA = torch.tensor(1 / 3, device=dev, requires_grad=True) if a.photosel == 3 else None
    EXTRA = [v for v in (BL, GAM, ZETA, BLD) if v is not None]


    def dip_tensor(k):
        e = torch.zeros(3, device=dev); e[:2] = EDIR[k]
        base = torch.diag(torch.stack([(1 - ZETA) / 2, (1 - ZETA) / 2, ZETA]))
        return (1 - GAM) * base + GAM * e[:, None] * e[None, :]


    def blur_cov(k):
        if BLD is not None:
            sg = -1.0 if k == 0 else 1.0
            return BL0 + sg * torch.stack([BLD[0], -BLD[0], BLD[1]])
        if BL is None:
            return BL0
        v = BL[k]
        return v if v.shape[0] == 3 else torch.stack([v[0] ** 2, v[1] ** 2, v[0] * v[1]])


    def psf_of(xyz, focus):
        """[b,R,R] (no per-angle term) or [b,2,R,R] (per-angle blur / photoselection)"""
        if BL is None and GAM is None and BL0 is None:
            return model(xyz, focus)
        na = max(BL.shape[0] if BL is not None else 1, 2 if (GAM is not None or BLD is not None) else 1)
        out = []
        for k in range(na):
            kw = {}
            if BL is not None: kw['blur'] = blur_cov(min(k, BL.shape[0] - 1))
            elif BL0 is not None: kw['blur'] = blur_cov(k)
            if GAM is not None: kw['dipM'] = dip_tensor(k)
            out.append(model(xyz, focus, **kw))
        return out[0] if na == 1 else torch.stack(out, 1)


    def extras_json():
        r = {}
        if BL is not None: r['blur_px2'] = [blur_cov(k).detach().cpu().tolist() for k in range(BL.shape[0])]
        elif BLD is not None: r['blur_px2'] = [blur_cov(k).detach().cpu().tolist() for k in range(2)]
        elif BL0 is not None: r['blur_px2'] = [BL0.cpu().tolist()]
        if GAM is not None:
            r['gamma'] = GAM.item(); r['dipole_M'] = [dip_tensor(k).detach().cpu().tolist() for k in range(2)]
            r['e_dir'] = EDIR.cpu().tolist()
            if ZETA is not None: r['zeta'] = ZETA.item()
        return r


    def load(idx):
        adu = torch.tensor(D['rois'][idx].astype(np.float32), device=dev)
        return dict(adu=adu, mask=torch.tensor(D['mask'][idx], device=dev), frames=torch.tensor(D['frames'][idx], device=dev),
                    corner=torch.tensor(D['corner'][idx], device=dev, dtype=torch.float32),
                    xy=(torch.tensor(D['xy_init'][idx], device=dev, dtype=torch.float32) - c).requires_grad_(True),
                    z=torch.zeros(len(idx), device=dev, requires_grad=True))


    def loss_and_linear(S, sl=slice(None)):
        adu, mask, frames = S['adu'][sl], S['mask'][sl], S['frames'][sl]
        b = adu.shape[0]
        psf = psf_of(torch.cat([S['xy'][sl], a.zmax * torch.sigmoid(S['z'][sl])[:, None]], 1), G['focus'].expand(b))
        data = (adu - G['offset']) * g
        rn2 = torch.exp(2 * G['logrn'])
        group = torch.where(mask, (frames % 6) // 3, torch.zeros_like(frames))
        if psf.dim() == 4:   # per-angle PSF -> per-frame PSF [b,T,R,R]
            psf = torch.gather(psf, 1, group[..., None, None].expand(-1, -1, psf.shape[-2], psf.shape[-1]))
            I, bg = solve_linear(psf.reshape(b, adu.shape[1], -1).detach(), data.reshape(b, adu.shape[1], -1).detach(), mask, group, rn2.detach())
            mu = (I[..., None, None] * psf + torch.gather(bg, 1, group)[..., None, None]).clamp(min=1e-3)
        else:
            I, bg = solve_linear(psf.reshape(b, -1).detach(), data.reshape(b, adu.shape[1], -1).detach(), mask, group, rn2.detach())
            mu = (I[..., None, None] * psf[:, None] + torch.gather(bg, 1, group)[..., None, None]).clamp(min=1e-3)
        var = mu + rn2
        nll = 0.5 * (data - mu) ** 2 / var + 0.5 * torch.log(2 * math.pi * var)
        nll = torch.where(mask[..., None, None], nll, torch.zeros_like(nll)).sum()
        return nll, I, bg, mu, var, data


    # ---- stage A on a subset
    if a.split:   # checkerboard parity of the emitter position (tile px); permutation drawn over that parity only
        pos_ = D['corner'] + D['xy_init']
        par = ((pos_[:, 0] // a.tile).astype(int) + (pos_[:, 1] // a.tile).astype(int)) % 2
        cand = np.flatnonzero(par == int(a.split[2]))
        sel = np.sort(cand[np.random.permutation(len(cand))[:a.nsamples]])
    else:
        sel = np.sort(np.random.permutation(Ball)[:a.nsamples])
    S = load(sel)
    npx = S['mask'].sum().item() * R * R
    opt = torch.optim.Adam([{'params': [S['xy'], S['z']], 'lr': 0.03},
                            {'params': [G['focus']], 'lr': 0.005},
                            {'params': [model.zcoef], 'lr': 0.002}, {'params': [model.apod, model.NA], 'lr': 0.003}]
                           + ([{'params': [BL], 'lr': 0.003}] if BL is not None else [])
                           + ([{'params': [GAM], 'lr': 0.003}] if GAM is not None else [])
                           + ([{'params': [ZETA], 'lr': 0.003}] if ZETA is not None else [])
                           + ([{'params': [BLD], 'lr': 0.002}] if BLD is not None else []))
    chunks = torch.arange(len(sel), device=dev).split(2000)
    for it in range(a.iters):
        opt.zero_grad(); tot = 0
        for ch in chunks:
            l = loss_and_linear(S, ch)[0] / npx; l.backward(); tot += l.item()
        if it < 100:
            for v in list(G.values()) + [model.zcoef, model.apod, model.NA] + EXTRA: v.grad = None
        opt.step()
        with torch.no_grad(): model.NA.clamp_(1.0, 1.515)
        if it % 100 == 0 or it == a.iters - 1:
            print(f'{it:5d} nll/px {tot:.4f}  rn {torch.exp(G["logrn"]).item():.2f}e  offset {G["offset"].item():.2f}  '
                  f'focus {G["focus"].item():.3f}  z med {(a.zmax*torch.sigmoid(S["z"])).median().item():.3f}  '
                  f'NA {model.NA.item():.3f}  apod {[round(v, 2) for v in model.apod.tolist()]}  |Z| {model.zcoef.norm().item():.3f}'
                  + (f'  extras {json.dumps(extras_json())}' if EXTRA else ''), flush=True)

    with torch.no_grad():
        rs = []
        for ch in chunks:
            _, I, bg, mu, var, data = loss_and_linear(S, ch)
            rs.append(((data - mu) / var.sqrt())[S['mask'][ch]])
        rs = torch.cat(rs)
    print(f'normalized residual var {rs.var().item():.3f}')
    res = dict(gain=g, readnoise_e=torch.exp(G['logrn']).item(), offset=G['offset'].item(), focus=G['focus'].item(),
               NA=model.NA.item(), apod=model.apod.tolist(), zernike=model.zcoef.tolist(), resid_var=rs.var().item(), zmax=a.zmax)
    if EXTRA or a.split or BL0 is not None:   # (opt-in fields; absent in the default run's json)
        res.update(blur_mode=a.blur, split=a.split, tile=a.tile, nll_per_px=tot,
                   n_train=int(len(sel)), **extras_json())
    json.dump(res, open(a.out + '_psf.json', 'w'), indent=1)
    torch.save(dict(zcoef=model.zcoef.detach().cpu(), NA=model.NA.detach().cpu(), apod=model.apod.detach().cpu(),
                    **{k: v.detach().cpu() for k, v in G.items()}, gain=g, zmax=a.zmax,
                    **({'extras': extras_json()} if (EXTRA or BL0 is not None) else {})), a.out + '_psf.pt')
    if not a.intensities: raise SystemExit
    print(json.dumps({k: v for k, v in res.items() if k != 'zernike'}))

    # ---- intensities for all samples (PSF frozen; per-sample xy,z refit)
    for prm in [model.zcoef, model.apod, model.NA] + list(G.values()) + EXTRA: prm.requires_grad_(False)
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
