"""Bead z-stacks -> vectorial PSF with torch_psf.estimate_psf_beads (Poisson likelihood; Zernike coefficients; per bead x, y
and per plane photons and background; a linear xy drift per stack; one stack held out), for the setup's optics and camera.

  python -m simcode.psf.beads --stacks PSF_calib/raw/zstack{2..7}-1.tif --dz 0.010 --sign -1 --holdout 7 --out <prefix>

Writes <prefix>_sign<sign><tag>.json (zernike {noll: waves}, NA, n_med, nll_per_px, heldout_nll_per_px, n_beads, ...) and
<prefix>_sign<sign><tag>.pt (zcoef, NA, n_med): the bead stage of the automatic pipeline (simcode/auto/pipeline.py) and the
Zernike start of the in-situ fit. Stacks are numbered 2, 3, ... in the order given (--holdout 7 = the 6th stack; 0 = none).
The shipped models/insitu/beads/psf_fit_sign-1_na149_ball_drift.* was made by the earlier SIMCODE bead fit (per-plane
intensity factors, bleaching, a fitted z step and bead height, a finite bead volume); torch_psf fits a point emitter on the
coverslip at the nominal z step."""
import argparse, json, os
import numpy as np, torch
from simcode import psf as spsf
from simcode.psf import fit_beads
from simcode.psf import poisson_nll
from simcode import config


def heldout_nll(est, sel, iters=200):
    """NLL per pixel of the beads `sel` with the PSF frozen and their own x, y, photons, bg fitted (torch_psf leaves the
    per-bead parameters of held-out beads at their start values)"""
    d, focus = est.data['rois'][sel], est.data['focus_um']
    r = fit_beads(d, focus, est.psf, iters=iters, lr=0.0, stack_id=est.data['stack_id'][sel], drift=True)
    nb, nk, P = d.shape[0], d.shape[1], d.shape[-1]
    with torch.no_grad():
        pos = torch.cat([r.data['xy'][:, None].expand(-1, nk, -1), torch.zeros(nb, nk, 1, device=d.device)], -1).reshape(-1, 3)
        mu = r.data['photons'].reshape(-1, 1, 1) * r.psf(pos, focus[None].expand(nb, -1).reshape(-1)) + r.data['bg'].reshape(-1, 1, 1)
        return float(poisson_nll(d.reshape(-1, P, P), mu).sum()) / d.numel()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--stacks', nargs='+', default=[f'PSF_calib/raw/zstack{i}-1.tif' for i in range(2, 8)],
                    help='bead z-stack tifs (absolute or relative to the data root)')
    ap.add_argument('--roi', type=int, default=26)
    ap.add_argument('--dz', type=float, default=0.010)
    ap.add_argument('--sign', type=float, default=1.0)       # stage direction
    ap.add_argument('--gain', type=float, default=0.56)
    ap.add_argument('--offset', type=float, default=106.34)
    ap.add_argument('--iters', type=int, default=300)
    ap.add_argument('--zmax', type=int, default=21)          # highest Noll index
    ap.add_argument('--holdout', type=int, default=7)        # stack id kept out of the fit (0 = none)
    ap.add_argument('--na', type=float, default=1.49)
    ap.add_argument('--wl', type=float, default=0.70)        # emission wavelength [um]
    ap.add_argument('--nmed', type=float, default=1.33)
    ap.add_argument('--pixelsize', type=float, default=0.10833)
    ap.add_argument('--nbeads', type=int, default=32)        # beads per stack (brightest isolated)
    ap.add_argument('--oversample', type=int, default=2)
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', default=os.path.join(config.out_dir('insitu'), 'psf_fit'))
    a = ap.parse_args(argv)
    stacks = [config.resolve(s, 'data') for s in a.stacks]
    optics = spsf.Optics(pixel_um=a.pixelsize, NA=a.na, wavelength_um=a.wl, n_med=a.nmed)
    camera = spsf.Camera(gain_e_per_adu=a.gain, offset_adu=a.offset)
    ho = a.holdout - 2 if a.holdout else None
    log = lambda it, nll, t, zc: (it % 25 == 0 or it == a.iters - 1) and print(f'{it:4d} nll {nll:.1f}  |zcoef| {zc.norm().item():.3f}  {t:.0f} s')
    est = spsf.estimate_psf_beads(stacks, a.dz, optics=optics, camera=camera, roisize=a.roi, sign=a.sign, nbeads=a.nbeads,
                                iters=a.iters, holdout=ho, drift=True, zernike_indices=tuple(range(5, a.zmax + 1)),
                                oversample=a.oversample, log=log)
    m = est.psf
    rep = dict(est.report)
    if ho is not None:
        rep['heldout_nll_per_px'] = heldout_nll(est, est.data['stack_id'] == ho)
        print(f"holdout: NLL/pixel {rep['heldout_nll_per_px']:.4f} (PSF frozen, per-bead parameters fitted)")
    res = dict(NA=float(m.NA), n_med=float(m.n_med), zernike={int(j): float(c) for j, c in zip(m.zidx, m.zcoef.detach().cpu())},
               sign=a.sign, dz_um=a.dz, nll_per_px=rep['nll_per_px'], heldout_nll_per_px=rep.get('heldout_nll_per_px'),
               n_beads=int(rep['n_beads']), n_planes=int(rep['n_planes']), holdout_stack=a.holdout, method='torch_psf.estimate_psf_beads')
    print(json.dumps(res, indent=1))
    pre = a.out + f'_sign{int(a.sign)}{a.tag}'
    json.dump(res, open(pre + '.json', 'w'), indent=1)
    torch.save(dict(n_med=m.n_med.detach().cpu(), zcoef=m.zcoef.detach().cpu(), NA=m.NA.detach().cpu()), pre + '.pt')
    return est, res


if __name__ == '__main__':
    main()
