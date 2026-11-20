"""The paper PSFs as psf.zip (torch-psf docs/PSF_FORMAT.md, format 0.2.0), next to their json in the models directory (the
json files stay; SIMCODE reads both through simcode.psf.read_calib / PSF.load / fourpi_state):

  insitu/calib_<movie>_psf_v2.json (1809_2, 2p5nM_m2, 20nM_m2) -> insitu/calib_<movie>_psf_v2.psf.zip
      torch_psf convert_calibration (lossless: PSF, camera with its offset / read-noise maps, pattern -> patterns,
      tirf_depth_um / emitter_z_* -> acquisition, source / psf_v2 -> provenance, the json in provenance.converted_from;
      calibrated camera scalars, free-form camera keys and extra npz entries kept in the camera block).
      Report: the InSituEstimator report and settings of the in-situ run (<results>/result_<movie>_v2p.json, --results:
      spot counts, deviance rule, held-out NLL; no estimator data: the stored PSF is the polished branch of
      out/psf_v2/psf_refit.py, which is not re-run here) -> psf.json report {estimator, results, settings} and the
      report pages; provenance: the final branch values and a note.
  insitu/calib_gauss1.3.json -> insitu/calib_gauss1.3.psf.zip (GaussianPSF; model pages; NA / wavelength / n in the
      top-level optics, the json's unused vectorial keys in provenance.unused_model_keys)
  insitu/beads/psf_fit_sign-1_na149_ball_drift.json -> .psf.zip: VectorialPSF of the bead z-stack fit (the earlier
      SIMCODE bead fit, not reproducible here: model pages; z = -focus of the bead stack; bead height, z step, stage
      sign in acquisition, the fit json in provenance.converted_from)
  <work>/psf4pi/NPC_f_dxy15.json (--work = the 4Pi work folder; the 4Pi PSF of the paper: also the state inside head4pi_v10.pt / v3L.pt) ->
      fourpi/NPC_f_dxy15.psf.zip: FourPiPSF (coupled model, 4 source_output channels with their camera crop / mirror;
      the fitted state verbatim plus torch_psf's 'raw' parameters, fourpi_psf below);
      fourpi/NPC_f_dxy15_spline.psf.zip (--spline): its FourPiSplinePSF as the 4Pi heads build it
      (simcode.psf.fourpi_spline(state, 15): 15 px, 4 x oversampled, z -0.6 .. 0.6 um, 10 nm, zpad 3) with the FourPiPSF as
      source; --spline-knots rebuild (default: no knots stored, the reader tabulates the source on the grid; on the GPU
      the knots equal fourpi_spline on the GPU bit for bit, tests/test_psfzip.py) or float32 (~120 MB, not in git)
  model folders: m1809/calib_1809_2_psf_v2.psf.zip, m25/calib_2p5nM_m2_psf_v2.psf.zip, g13/calib_gauss1.3.psf.zip
      (byte copies of the insitu/ files)

  CUDA_VISIBLE_DEVICES= python analysis/psfzip/convert_paper_psfs.py --results <psf_v2 results folder> --work <4Pi work folder>
      [--only calib,gauss,beads,fourpi,copies] [--spline] [--spline-knots rebuild|float32|float16]
      (--results is needed for calib, --work for fourpi)
"""
import argparse, json, os, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from simcode import config
from simcode import psf as spsf

V2 = ('1809_2', '2p5nM_m2', '20nM_m2')
COPIES = {'m1809': 'calib_1809_2_psf_v2.psf.zip', 'm25': 'calib_2p5nM_m2_psf_v2.psf.zip', 'g13': 'calib_gauss1.3.psf.zip'}
# the 4Pi channels on the camera chip (simcode/pipeline/configs/fourpi.py): raw columns, 360 x 200 px, channels 2, 3 mirrored
FOURPI_X0, FOURPI_MIRROR = (217, 652, 1420, 1855), ('none', 'none', 'lr', 'lr')


def calib_v2(name, dev, results):
    js = config.model(f'insitu/calib_{name}_psf_v2.json')
    R = json.load(open(os.path.join(results, f'result_{name}_v2p.json')))
    settings = dict(estimator='InSituEstimator', script='out/psf_v2/psf_refit.py', movie=R['movie'], frames=[0, 30000],
                    roisize=15, z_range_um=[0.0, 0.4], nspots=8000, nheldout=4000, iters=80, refine=False,
                    focus_um=R['focus_um'], camera=R['camera'])
    prov = dict(estimator='InSituEstimator', final=dict(branch=R['branch'], **R['final']),
                report_note=('report = the InSituEstimator run (branch A, neutral start) of out/psf_v2/psf_refit.py; the PSF '
                             f"stored here is branch {R['branch']} after the restart polish (final: held-out NLL / px "
                             f"{R['final']['heldout_nll_per_px']:.4f}); no estimator data (spots) are stored"))
    rep = dict(report=R['report'], settings=settings, estimator='InSituEstimator')
    out = spsf.write_calib_zip(js, provenance=prov, report=rep, device=dev)
    print(out, flush=True)


def gauss(dev):
    print(spsf.write_calib_zip(config.model('insitu/calib_gauss1.3.json'), device=dev), flush=True)


def beads(dev):
    js = config.model('insitu/beads/psf_fit_sign-1_na149_ball_drift.json')
    B = json.load(open(js))
    noll = sorted(int(k) for k in B['zernike'])
    optics = spsf.Optics(pixel_um=0.10833, NA=B['NA'], wavelength_um=0.70, n_med=B['n_med'])   # simcode.psf.beads defaults
    vec = spsf.VectorialPSF.from_optics(optics, 26, zernike_indices=tuple(noll), focus_um=0.0, z_range=(-1.0, 1.0),
                                        device=dev)
    import torch
    with torch.no_grad():
        vec.zcoef.copy_(torch.tensor([B['zernike'][str(j)] for j in noll], device=vec.dev))
    prov = dict(estimator='earlier SIMCODE bead z-stack fit (vectorial, per-stack cubic drift, bead radius tied to its height)',
                converted_from=dict(file=os.path.basename(js), json=B),
                note='bead z-stacks: z = -focus (stage displacement); NA and n_med fitted, wavelength 0.70 um and pixel '
                     '0.10833 um as simcode.psf.beads; the Zernike start of the in-situ PSF fit',
                report=dict(final_nll=B.get('final_nll'), drift_px_rms=B.get('drift_px_rms')))
    acq = dict(bead_height_um=B.get('ze'), dz_nm=B.get('dz_nm'), stage_sign=B.get('sign'))
    out = os.path.splitext(js)[0] + '.psf.zip'
    spsf.save_zip(out, vec, provenance=prov, acquisition=acq)
    print(out, flush=True)


def fourpi_psf(state, device=None, z_range=(-0.6, 0.6), **kw):
    """torch_psf FourPiPSF of a 4Pi PSF state (the model of simcode.psf.fourpi_psf) whose state() is the
    given state verbatim plus torch_psf's 'raw' block (the internal parameters, which a reader takes for a bit-exact
    model): torch_psf's own state() (still at 4d806d0, formerly 1052362 / 0fee3c3) recomputes tau = softmax(log tau) and float32-rounds NA / n, so it does not
    reproduce the fitted values; this way save_zip stores the fitted state as it is and the exact parameters."""
    p = spsf.FourPiPSF.from_state(state, device=device, z_range=z_range, **kw)
    s0 = json.loads(json.dumps(state)); s0.pop('raw', None)
    raw = p.state().get('raw')
    if raw is not None: s0['raw'] = raw
    p.state = lambda: json.loads(json.dumps(s0))
    return p


def fourpi(dev, spline, work, knots='rebuild'):
    """work: the 4Pi work folder (<work>/psf4pi/NPC_f_dxy15.json)"""
    P4 = work
    js = os.path.join(P4, 'psf4pi', 'NPC_f_dxy15.json')
    J = json.load(open(js)); st = J['psf']
    p = fourpi_psf(st, device=dev)
    chans = [dict(name=f'c{i}', crop=dict(x0=x, y0=0, w=360, h=200), flip=m) for i, (x, m) in enumerate(zip(FOURPI_X0, FOURPI_MIRROR))]
    prov = dict(estimator='4Pi bead z-stack fit (analysis/fourpi/fit_psf4pi.py; NPC cell, dxy 15)',
                converted_from=dict(file=os.path.basename(js), json=J),
                note='the 4Pi PSF of the paper runs (MT and NPC; also the state stored in head4pi_v10.pt / head4pi_v3L.pt); '
                     'channel crops / mirrors of the 4Pi-SIMFLUX camera (simcode/pipeline/configs/fourpi.py), no registration')
    d = config.model('fourpi'); os.makedirs(d, exist_ok=True)
    out = os.path.join(d, 'NPC_f_dxy15.psf.zip')
    spsf.save_zip(out, p, provenance=prov, channels=chans)
    print(out, flush=True)
    if spline:
        S = p.to_spline(15, (-0.6, 0.6), value='pixel', **spsf.FOURPI_KIND, oversample=4, dz_um=0.01, zpad=3)   # = fourpi_spline's table
        out = os.path.join(d, 'NPC_f_dxy15_spline.psf.zip')
        spsf.save_zip(out, S, source=p, channels=chans, spline_knots=knots,
                      provenance=dict(prov, spline='simcode.psf.fourpi_spline(state, 15) grid (formerly simcode/fourpi/spline_table.LUT4Pi(state, P=15)): 4 x oversampled, z -0.6 .. 0.6 um, '
                                                   '10 nm, zpad 3, offset limit 8.5 px' + ('' if knots == 'rebuild' else f'; tabulated on {dev}')))
        print(out, flush=True)


def copies():
    for m, f in COPIES.items():
        src = config.model(f'insitu/{f}'); dst = config.model(f'{m}/{f}')
        shutil.copyfile(src, dst); print(dst, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='calib,gauss,beads,fourpi,copies')
    ap.add_argument('--spline', action='store_true', help='also the 4Pi spline zip')
    ap.add_argument('--spline-knots', default='rebuild', choices=('rebuild', 'float32', 'float16'),
                    help="knots of the spline zip: 'rebuild' (none stored; tabulated on load) or stored (float32: ~120 MB)")
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--results', default=None, help='folder of the in-situ PSF v2 results (result_<movie>_v2p.json; on qnano1 ~/simcode2/out/psf_v2); required for calib')
    ap.add_argument('--work', default=None, help='4Pi work folder (psf4pi/NPC_f_dxy15.json); required for fourpi')
    a = ap.parse_args()
    todo = a.only.split(',')
    if 'calib' in todo and not a.results: ap.error('calib needs --results <folder of the in-situ PSF v2 results>')
    if 'fourpi' in todo and not a.work: ap.error('fourpi needs --work <4Pi work folder>')
    if 'calib' in todo:
        for n in V2: calib_v2(n, a.device, a.results)
    if 'gauss' in todo: gauss(a.device)
    if 'beads' in todo: beads(a.device)
    if 'fourpi' in todo: fourpi(a.device, a.spline, a.work, a.spline_knots)
    if 'copies' in todo: copies()
