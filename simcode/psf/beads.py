"""Bead z-stacks -> vectorial PSF with torch_psf.ZStackEstimator (Poisson likelihood; Zernike coefficients; per bead x, y and
photons, a per-plane intensity factor shared by the beads (bleaching / illumination drift) and background; a linear xy drift
per stack; a fitted z-step scale and bead height; one stack held out and scored with its beads' own position / photons / bg
refitted), for the setup's optics and camera.

  python -m simcode.psf.beads --stacks PSF_calib/raw/zstack{2..7}-1.tif --dz 0.010 --sign -1 --holdout 7 --out <prefix>

Writes <prefix>_sign<sign><tag>.json (zernike {noll: waves}, NA, n_med, dz_nm (fitted z step), ze (fitted bead height, um),
nll_per_px, heldout_nll_per_px, n_beads, ...) and <prefix>_sign<sign><tag>.pt (zcoef, NA, n_med): the bead stage of the
automatic pipeline (simcode/pipeline/localization_pipeline.py) and the Zernike start of the in-situ fit. Stacks are numbered 2, 3, ... in the
order given (--holdout 7 = the 6th stack; 0 = none). --bead_diameter: finite bead volume (um; default a point emitter);
--fit_dz_scale 0 / --fit_height 0 / --intensity free switch the fitted z step / bead height / per-plane factors off.
The shipped models/insitu/beads/psf_fit_sign-1_na149_ball_drift.* was made by the earlier SIMCODE bead fit (the same terms,
with a per-stack cubic drift and the bead radius tied to its height)."""
import argparse, json, os
import numpy as np, torch
from simcode import psf as spsf
from simcode import config


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
    ap.add_argument('--fit_dz_scale', type=int, default=1)  # fit a scale of the z step
    ap.add_argument('--fit_height', type=int, default=1)    # fit the bead height above the coverslip
    ap.add_argument('--intensity', default='bead_x_plane', choices=['bead_x_plane', 'free'])   # per-plane factors or free photons
    ap.add_argument('--bead_diameter', type=float, default=0.0)   # um; 0 = point emitter
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', default=os.path.join(config.out_dir('insitu'), 'psf_fit'))
    a = ap.parse_args(argv)
    stacks = [config.resolve(s, 'data') for s in a.stacks]
    optics = spsf.Optics(pixel_um=a.pixelsize, NA=a.na, wavelength_um=a.wl, n_med=a.nmed)
    camera = spsf.Camera(gain_e_per_adu=a.gain, offset_adu=a.offset)
    ho = a.holdout - 2 if a.holdout else None
    log = lambda stage, it, nll, t: (it % 25 == 0 or it == a.iters - 1) and print(f'{it:4d} nll {nll:.1f}  {t:.0f} s')
    est = spsf.ZStackEstimator('vectorial', optics=optics, camera=camera, roisize=a.roi, sign=a.sign, nbeads=a.nbeads,
                               iters=a.iters, holdout=ho, drift=True, fit_dz_scale=bool(a.fit_dz_scale),
                               fit_height=bool(a.fit_height), intensity=a.intensity,
                               bead_diameter_um=a.bead_diameter or None, zernike_indices=tuple(range(5, a.zmax + 1)),
                               oversample_pupil=a.oversample, progress=log)
    m = est.estimate(stacks, a.dz)
    rep = dict(est.report)
    if ho is not None:
        print(f"holdout: NLL/pixel {rep['heldout_nll_per_px']:.4f} (PSF frozen, per-bead parameters fitted)")
    res = dict(NA=float(m.NA), n_med=float(m.n_med), zernike={int(j): float(c) for j, c in zip(m.zidx, m.zcoef.detach().cpu())},
               sign=a.sign, dz_um=a.dz, dz_nm=1e3 * a.dz * rep.get('dz_scale', 1.0), ze=rep.get('height_um'),
               nll_per_px=rep['nll_per_px'], heldout_nll_per_px=rep.get('heldout_nll_per_px'),
               n_beads=int(rep['n_beads']), n_planes=int(rep['n_planes']), holdout_stack=a.holdout,
               intensity=a.intensity, bead_diameter_um=a.bead_diameter or None, method='torch_psf.ZStackEstimator')
    print(json.dumps(res, indent=1))
    pre = a.out + f'_sign{int(a.sign)}{a.tag}'
    json.dump(res, open(pre + '.json', 'w'), indent=1)
    torch.save(dict(n_med=m.n_med.detach().cpu(), zcoef=m.zcoef.detach().cpu(), NA=m.NA.detach().cpu()), pre + '.pt')
    return est, res


if __name__ == '__main__':
    main()
