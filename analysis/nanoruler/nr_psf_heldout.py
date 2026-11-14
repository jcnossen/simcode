"""NANORULER COPY of simcode/insitu/psf_heldout.py (package file not edited). Only change: the optics / camera constants that are
hard-coded for the DMD-SIMFLUX setup come from the environment: NR_PX (pixel size um, 0.065), NR_WL (emission wavelength um, 0.68),
NR_OFFSET (ADU, 100.0), NR_RN (read noise e, start / fixed value of the fit, 1.6). Original docstring follows.
(2026-11-11 in-situ 3D PSF) Held-out evaluation of an in-situ PSF on isolated emitter track samples.

The PSF (and camera: gain / offset / read noise of the stage-A fit) is frozen. Per held-out sample, x, y, z are refit by
Adam (same parametrisation as fit_insitu2: z = zmax sigmoid), the per-frame I and per-(sample, angle) bg are solved exactly
(varpro). Per sample it stores the Gaussian-approximation Poisson NLL (as fit_insitu2) and second moments of the data and of
the noise-free model about the fitted centre, after removing the fitted bg, in apertures r < 2.5 / 3 / 4 px, per pattern
angle: S = sum v, Q = sum v (dx^2 - dy^2), W = sum v (dx^2 + dy^2), X = sum v 2 dx dy (dM = Q / S = Mxx - Myy).

PSF sources: --fit <prefix>_psf.pt (fit_insitu2 output, with its opt-in 'extras') or --calib <calib json> (zernike_waves,
focus_um, apod, NA and the opt-in psf_* keys). Test set: --test cb0 | cb1 = samples of that checkerboard parity (--tile px),
'' = all; then --nsamples drawn with --seed (the same set for every PSF given the same tracks / test / seed).
usage: python -m simcode.insitu.psf_heldout --tracks T.npz --fit F_psf.pt --test cb1 --out H.npz
"""
import argparse, json, math
import numpy as np, torch
from simcode import psf as spsf
from simcode.psf import VectorialPSF
from simcode.psf import solve_linear
from simcode import config

APERTURES = (2.5, 3.0, 4.0)


def build_psf(R, fit=None, calib=None, dev='cuda', os_=2):
    """-> (model, focus, cam dict, per-angle kwargs list [angle0, angle1])"""
    if fit:
        S = torch.load(config.resolve(fit), weights_only=False)
        import os as _os
        model = VectorialPSF(R, pixel_um=float(_os.environ.get('NR_PX', 0.065)), oversample=os_, NA=float(S['NA']), wavelength_um=float(_os.environ.get('NR_WL', 0.68)), n_med=1.33, device=dev)
        with torch.no_grad():
            model.zcoef.copy_(S['zcoef'].to(dev)); model.apod.copy_(S['apod'].to(dev))
        ex = S.get('extras', {})
        focus = float(S['focus']); cam = dict(gain=float(S['gain']), offset=float(S['offset']), rn=float(torch.exp(S['logrn'])),
                                               zmax=float(S['zmax']))
        if 'saf' in ex:
            with torch.no_grad(): model.saf.copy_(torch.tensor(ex['saf'], device=dev))
        bl, dw, dM = ex.get('blur_px2'), ex.get('dipole_w'), ex.get('dipole_M')
    else:
        c = json.load(open(config.resolve(calib)))
        model = spsf.PSF.load(config.resolve(calib), roisize=R, oversample=os_, device=dev)   # with the opt-in psf_* terms
        focus = float(c['focus_um']); import os as _os
        cam = dict(gain=c['camera']['gain_e_per_adu'], offset=float(_os.environ.get('NR_OFFSET', 100.0)), rn=float(_os.environ.get('NR_RN', 1.6)), zmax=0.2)
        bl = c.get('psf_blur_angle_px2'); dw = c.get('psf_dipole_w_angle'); dM = c.get('psf_dipole_M_angle')
    kws = []
    for k in range(2):
        kw = {}
        if bl is not None: kw['blur'] = torch.tensor(bl[min(k, len(bl) - 1)], device=dev)
        if dw is not None: kw['dipw'] = torch.tensor(dw[min(k, len(dw) - 1)], device=dev)
        if dM is not None: kw['dipM'] = torch.tensor(dM[min(k, len(dM) - 1)], device=dev)
        kws.append(kw)
    return model, focus, cam, kws


def select(D, test, tile, nsamples, seed):
    B = len(D['rois'])
    if test:
        pos = D['corner'] + D['xy_init']
        par = ((pos[:, 0] // tile).astype(int) + (pos[:, 1] // tile).astype(int)) % 2
        cand = np.flatnonzero(par == int(test[2]))
    else:
        cand = np.arange(B)
    rng = np.random.default_rng(seed)
    return np.sort(cand[rng.permutation(len(cand))[:nsamples]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracks', required=True)
    ap.add_argument('--fit', default='')
    ap.add_argument('--calib', default='')
    ap.add_argument('--test', default='', choices=['', 'cb0', 'cb1'])
    ap.add_argument('--tile', type=int, default=16)
    ap.add_argument('--nsamples', type=int, default=6000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--iters', type=int, default=300)
    ap.add_argument('--zstarts', default='0.02,0.08,0.15')   # multi-start in z (um): per sample the best NLL is kept
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = 'cuda'; torch.manual_seed(0)
    D = np.load(config.resolve(a.tracks))
    R = D['rois'].shape[-1]; c = (R - 1) / 2
    model, focus, cam, kws = build_psf(R, a.fit, a.calib, dev)
    for p_ in model.parameters(): p_.requires_grad_(False)
    per_angle = any(kws[0].get(k) is not None and not torch.equal(kws[0][k], kws[1][k]) for k in kws[0])
    g, off, rn2, zmax = cam['gain'], cam['offset'], cam['rn'] ** 2, cam['zmax']
    sel = select(D, a.test, a.tile, a.nsamples, a.seed)
    oy, ox = torch.meshgrid(torch.arange(R, device=dev, dtype=torch.float32), torch.arange(R, device=dev, dtype=torch.float32), indexing='ij')

    def evaluate(xy, z, adu, mask, frames):
        b, T = adu.shape[:2]
        xyz = torch.cat([xy, zmax * torch.sigmoid(z)[:, None]], 1)
        group = torch.where(mask, (frames % 6) // 3, torch.zeros_like(frames))
        if per_angle:
            ps = torch.stack([model(xyz, focus, **kws[k]) for k in range(2)], 1)
            psf = torch.gather(ps, 1, group[..., None, None].expand(-1, -1, R, R))                  # [b,T,R,R]
        else:
            psf = model(xyz, focus, **kws[0])[:, None].expand(-1, T, -1, -1)
        data = (adu - off) * g
        I, bg = solve_linear(psf.reshape(b, T, -1).detach(), data.reshape(b, T, -1), mask, group, rn2)
        bgf = torch.gather(bg, 1, group)[..., None, None]
        sig = I[..., None, None] * psf
        mu = (sig + bgf).clamp(min=1e-3); var = mu + rn2
        nll = 0.5 * (data - mu) ** 2 / var + 0.5 * torch.log(2 * math.pi * var)
        nll = torch.where(mask[..., None, None], nll, torch.zeros_like(nll))
        evaluate.last = (I, bg)
        return nll, data, sig, bgf, group, var, mu

    n = len(sel)
    out = {k: np.zeros(n) for k in ('nll', 'npix', 'chi2', 'z', 'x', 'y', 'Isum')}
    Tn = D['rois'].shape[1]
    out['I'] = np.zeros((n, Tn)); out['bg'] = np.zeros((n, 2))
    M = np.zeros((n, 2, 2, len(APERTURES), 4))     # [sample, data|model, angle, aperture, S Q W X]
    for s0 in range(0, n, 4000):
        idx = sel[s0:s0 + 4000]
        adu = torch.tensor(D['rois'][idx].astype(np.float32), device=dev); mask = torch.tensor(D['mask'][idx], device=dev)
        frames = torch.tensor(D['frames'][idx], device=dev)
        best = None
        for zs in [float(v) for v in a.zstarts.split(',')]:
            xy_ = (torch.tensor(D['xy_init'][idx], device=dev, dtype=torch.float32) - c).requires_grad_(True)
            z_ = torch.full((len(idx),), math.log(zs / (zmax - zs)), device=dev, requires_grad=True)
            opt = torch.optim.Adam([xy_, z_], lr=0.03)
            for it in range(a.iters):
                opt.zero_grad()
                for ch in torch.arange(len(idx), device=dev).split(2000):
                    evaluate(xy_[ch], z_[ch], adu[ch], mask[ch], frames[ch])[0].sum().backward()
                opt.step()
            with torch.no_grad():
                L = torch.cat([evaluate(xy_[ch], z_[ch], adu[ch], mask[ch], frames[ch])[0].sum((1, 2, 3))
                               for ch in torch.arange(len(idx), device=dev).split(2000)])
                if best is None:
                    best, xy, z = L, xy_.detach().clone(), z_.detach().clone()
                else:
                    b_ = L < best; best = torch.where(b_, L, best)
                    xy[b_] = xy_[b_].detach(); z[b_] = z_[b_].detach()
        with torch.no_grad():
            for ch in torch.arange(len(idx), device=dev).split(2000):
                nll, data, sig, bgf, group, var, mu = evaluate(xy[ch], z[ch], adu[ch], mask[ch], frames[ch])
                j = ch.cpu().numpy() + s0
                m = mask[ch]
                out['I'][j] = evaluate.last[0].cpu().numpy(); out['bg'][j] = evaluate.last[1].cpu().numpy()
                out['nll'][j] = nll.sum((1, 2, 3)).cpu().numpy()
                out['npix'][j] = (m.sum(1) * R * R).cpu().numpy()
                out['chi2'][j] = torch.where(m[..., None, None], (data - mu) ** 2 / var, torch.zeros_like(var)).sum((1, 2, 3)).cpu().numpy()
                out['z'][j] = (zmax * torch.sigmoid(z[ch])).cpu().numpy()
                out['x'][j] = (xy[ch, 0] + c).cpu().numpy() + D['corner'][idx[ch.cpu().numpy()], 0]
                out['y'][j] = (xy[ch, 1] + c).cpu().numpy() + D['corner'][idx[ch.cpu().numpy()], 1]
                dx = ox[None] - (xy[ch, 0] + c)[:, None, None]; dy = oy[None] - (xy[ch, 1] + c)[:, None, None]
                r = torch.sqrt(dx ** 2 + dy ** 2)
                Isum = (sig.sum((2, 3)) * m).sum(1); out['Isum'][j] = Isum.cpu().numpy()
                for di, img in enumerate((data - bgf, sig)):
                    for an in range(2):
                        fm = (m & (group == an)).float()[..., None, None]
                        v = (img * fm).sum(1)                                           # [b,R,R] summed over the angle's frames
                        for ai, rad in enumerate(APERTURES):
                            q = (r < rad).float(); vq = v * q
                            M[j, di, an, ai] = torch.stack([vq.sum((1, 2)), (vq * (dx ** 2 - dy ** 2)).sum((1, 2)),
                                                            (vq * (dx ** 2 + dy ** 2)).sum((1, 2)), (vq * 2 * dx * dy).sum((1, 2))], 1).cpu().numpy()
        print(f'{s0 + len(idx)}/{n} done', flush=True)
    np.savez(a.out, sel=sel, M=M, apertures=np.array(APERTURES), **out)
    tot = out['nll'].sum() / out['npix'].sum()
    dm = lambda di, ai: M[:, di, :, ai, 1].sum() / M[:, di, :, ai, 0].sum()
    print(json.dumps(dict(n=n, nll_per_px=tot, chi2_per_px=out['chi2'].sum() / out['npix'].sum(), z_median=float(np.median(out['z'])),
                          dM_data_r3=dm(0, 1), dM_model_r3=dm(1, 1), ddM_r3=dm(0, 1) - dm(1, 1))))


if __name__ == '__main__':
    main()
