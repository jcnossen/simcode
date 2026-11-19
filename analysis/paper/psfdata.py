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
      pattern          the run's OWN estimated pattern (<run_dir>/pattern_est.pt, run_pattern): pitch / depth / angles
                       (means over the 2 orientations) + per orientation pitch and depth + the 6 relative step intensities,
                       for the SIMFLUX CRLB. (Before 2026-11-15 late: the calibration file's pattern block, a constant
                       shared by all calibration files (224.19 nm, depth 0.94): wrong for every run.)
  run_pattern(run_dir, px_nm)  -> the pattern dict of a run's pattern_est.pt
  refresh(values_json, ...)    recompute the psf_panel block of an existing figure values json (and its <out>_psf_table) from
                               its stored conditions with the run's own pattern, without reading any localization table
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
    from simcode.psf import read_calib
    return p, read_calib(p)


def run_pattern(run_dir, px_nm):
    """the run's estimated pattern (pattern_est.pt of simcode.processing.pattern_estimation: k [2, 2] rad / px (x = col,
    y = row), m [6] depth per frame (one per orientation), logrel [6] log relative step intensity) -> dict with the means over
    the 2 orientations (pitch_nm, depth), angles_deg, and per orientation pitch_nm_per_angle / depth_per_angle, relint [6]"""
    import torch
    f = os.path.join(run_dir, 'pattern_est.pt')
    P = torch.load(f, map_location='cpu', weights_only=False)
    k = P['k'].double().numpy(); m = P['m'].double().numpy(); rel = np.exp(P['logrel'].double().numpy())
    na = k.shape[0]; T = len(m) // na
    pitch = [float(2 * np.pi / np.hypot(*kk) * px_nm) for kk in k]
    depth = [float(m[a * T:(a + 1) * T].mean()) for a in range(na)]
    return dict(pitch_nm=float(np.mean(pitch)), depth=float(np.mean(depth)),
                angles_deg=tuple(float(math.degrees(math.atan2(kk[1], kk[0]))) for kk in k),
                pitch_nm_per_angle=pitch, depth_per_angle=depth, relint=(rel / rel.mean()).tolist(), source=f)


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
    from simcode.psf import camera_maps
    rn = camera_maps(cp, c)['readnoise_e'].astype(np.float64)
    i0, i1 = int(y0), int(y0 + s); j0, j1 = int(x0), int(x0 + s)
    rn2 = float((rn[i0:i1, j0:j1] ** 2).mean())
    F = rows.numpy()
    ph = F['photons'].astype(np.float64)
    z = F['pos'][:, 2].astype(np.float64) if F['pos'].shape[1] > 2 else np.full(len(ph), np.nan)
    zu = c['focus_um'] + z * c['pixelsize_um']
    q = np.nanpercentile(zu, [5, 25, 50, 75, 95]) if np.isfinite(zu).any() else [np.nan] * 5
    pattern = run_pattern(run_dir, c['pixelsize_um'] * 1e3)
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
                             pattern=cond['pattern'], detected=False, rn2_e=cond['rn2_e'], device=device)
    dev = psfpanel._dev(device)
    psf = zpsf.load(cond['calib'], device=dev); zero = torch.zeros(1, device=dev)
    coll = float(psf.eval(zero, zero, 41, torch.full((1,), info['z_um'], device=dev))[0].sum())
    info['collected'] = coll; info['detected_photons'] = coll * cond['photons_median']
    return info


def refresh(values_json, run_dir=None, device=None):
    """recompute vals['psf_panel'] of a fig_exp_new values json with the run's own pattern (run_dir: default vals['paths']['v2'])
    from the stored conditions (calib, photons, background, read noise, height: unchanged, no table read). Rewrites the json and,
    when they exist, <out>_psf_table.{md,csv}. Returns (old crlb_nm, old pattern, new crlb_nm, new pattern)."""
    import fig_exp_new as FE
    V = json.load(open(values_json)); P = V['psf_panel']; cond = dict(P['conditions'])
    old = (P['crlb_nm'], cond['pattern'])
    cp, c = _calib(cond['calib'])
    cond['pattern'] = run_pattern(run_dir or V['paths']['v2'], c['pixelsize_um'] * 1e3)
    info = panel_info(cond['calib'], cond, device=device)
    V['psf_panel'] = dict(P, conditions=cond, rms_mwaves=info['rms_mwaves'], z_um=info['z_um'], collected=info['collected'],
                          intensity_param=info['intensity_param'], detected_photons=info['detected_photons'],
                          crlb_nm=FE.jsonable(info['crlb']))
    json.dump(V, open(values_json, 'w'), indent=1)
    out = values_json[:-len('_values.json')]
    if os.path.exists(out + '_psf_table.md'):
        sc = V['labels'][-1]
        md, csv_s = FE.psf_table(info, cond, int(V['n_frc'][-1]), sc=sc)
        txt = FE.paper_text if 'v2' not in sc else (lambda t: t)
        open(out + '_psf_table.md', 'w').write(txt(md)); open(out + '_psf_table.csv', 'w').write(txt(csv_s))
    return old + (V['psf_panel']['crlb_nm'], cond['pattern'])


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='recompute the PSF-panel CRLBs of figure values json files with each run\'s own pattern')
    ap.add_argument('values', nargs='+'); a = ap.parse_args()
    sys.path.insert(0, HERE)
    for f in a.values:
        oc, op, nc, npat = refresh(f)
        print(f); print('  pattern old', {k: op[k] for k in ('pitch_nm', 'depth')}, 'new', {k: npat[k] for k in ('pitch_nm', 'depth', 'depth_per_angle', 'pitch_nm_per_angle')})
        for k in ('smlm', 'sf'):
            print(f'  CRLB {k} x/y/z old', [round(oc[k][q], 2) for q in 'xyz'], 'new', [round(nc[k][q], 2) for q in 'xyz'],
                  '| xy z known old', [round(v, 2) for v in oc[k]['xy_zfixed']], 'new', [round(v, 2) for v in nc[k]['xy_zfixed']])
