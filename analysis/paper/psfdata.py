"""The data side of the PSF panel of the experimental figures (fig_exp_new.py --psf): the photons, background, read noise
and emitter height at which analysis/papersim/psfpanel.py draws the PSF and computes the SMLM / SIMFLUX CRLBs.

  data_conditions(run_dir, rows, calib) -> dict
      photons_median   median photons per 6-frame window of `rows` (the plotted SIMCODE v2 set; the network's photons =
                       the PSF model's intensity parameter in the units of the run's camera calibration, see panel_info)
      bg_px_frame      median network background (photons / px / frame, mean over the 6 frames) of the pass-1 rows with
                       p >= 0.63, disjoint windows (start % 6 == 0), inside the run's crop (pass1_*.npz 'bg'; the pass-2
                       table has no background column)
      rn2_e            mean read-noise variance (e^2 / px / frame) of the calibration's camera map over the crop
      z_median_um      median emitter height of `rows` (pass-1 z, px relative to the calibration's focal plane ->
                       focus_um + z * pixelsize_um); z_quantiles_um q05 / q25 / q75 / q95; z_nan_frac (rows without a pass-1 z)
      pattern          pitch / depth / angles of the calibration's pattern (for the SIMFLUX CRLB)
  panel_info(calib, cond)  -> psfpanel.psf_info at those conditions (intensity parameter = the network photons, read noise,
                              z range 0.40 um) + collected fraction / detected-photon equivalent"""
import os, sys, json, glob, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, 'analysis', 'papersim'))


def _calib(calib):
    from simcode.config import resolve
    p = resolve(calib)
    return p, json.load(open(p))


def data_conditions(run_dir, rows, calib, crop=(5.0, 5.0, 110.0), p_min=0.63):
    cp, c = _calib(calib)
    x0, y0, s = crop
    bg = []
    for f in sorted(glob.glob(os.path.join(run_dir, 'pass1_*.npz'))):
        d = np.load(f)
        x, y, w, p = d['x'], d['y'], d['wstart'], d['p']
        m = (p >= p_min) & (w % 6 == 0) & (x >= x0) & (x < x0 + s) & (y >= y0) & (y < y0 + s)
        bg.append(d['bg'][m].mean(1))
    bg = np.concatenate(bg)
    rn = np.load(os.path.join(os.path.dirname(cp), c['camera']['maps']))['readnoise_e'].astype(np.float64)
    i0, i1 = int(y0), int(y0 + s); j0, j1 = int(x0), int(x0 + s)
    rn2 = float((rn[i0:i1, j0:j1] ** 2).mean())
    F = rows.numpy()
    ph = F['photons'].astype(np.float64)
    z = F['pos'][:, 2].astype(np.float64) if F['pos'].shape[1] > 2 else np.full(len(ph), np.nan)
    zu = c['focus_um'] + z * c['pixelsize_um']
    q = np.nanpercentile(zu, [5, 25, 50, 75, 95]) if np.isfinite(zu).any() else [np.nan] * 5
    pat = c['pattern']
    k = np.asarray(pat['k_rad_per_px'], float)
    pattern = dict(pitch_nm=float(np.mean(pat['pitch_nm'])), depth=float(np.mean(pat['depth'])),
                   angles_deg=tuple(float(math.degrees(math.atan2(kk[1], kk[0]))) for kk in k))
    return dict(calib=cp, photons_median=float(np.median(ph)), n_rows=int(len(ph)), bg_px_frame=float(np.median(bg)),
                bg_n=int(len(bg)), bg_q25_q75=np.percentile(bg, [25, 75]).tolist(), rn2_e=rn2,
                z_median_um=float(q[2]), z_quantiles_um=dict(q05=float(q[0]), q25=float(q[1]), q75=float(q[3]), q95=float(q[4])),
                z_nan_frac=float(np.mean(~np.isfinite(zu))), pattern=pattern, gain_e_per_adu=c['camera']['gain_e_per_adu'],
                focus_um=c['focus_um'])


def panel_info(calib, cond, device=None):
    """CRLB with the intensity parameter = the network's photons: SIMCODE v2's photons are in the label units of its training
    simulations, i.e. the intensity parameter of the calibration's PSF model (rendered as I * psf; the PSF's sum over a 41 x 41 px
    image at height z is the collected fraction, 0.73 at 112 nm for the 1809_2 PSF v2), not detected photons. info['collected'] /
    info['detected_photons'] give the detected equivalent (photons x collected)."""
    import torch, psfpanel, zpsf
    info = psfpanel.psf_info(cond['calib'], cond['photons_median'], cond['bg_px_frame'], z_um=cond['z_median_um'],
                             pattern=cond['pattern'], detected=False, rn2_e=cond['rn2_e'], zmax=0.4, device=device)
    dev = psfpanel._dev(device)
    psf = zpsf.load(cond['calib'], device=dev, zmax=0.4); zero = torch.zeros(1, device=dev)
    coll = float(psf.eval(zero, zero, 41, torch.full((1,), info['z_um'], device=dev))[0].sum())
    info['collected'] = coll; info['detected_photons'] = coll * cond['photons_median']
    return info
