"""SMLM and SIMFLUX baselines on a papersim movie: analysis/baselines/paper_pipeline.py (the paper's classical pipeline with
the torch LM-MLE on a zpsf PSF; every setting and filter as documented there) driven on a simulated movie instead of one of
the 3 real datasets. Nothing of paper_pipeline.py is changed; this wrapper imports its fitter, SIMFLUX fit and pattern
estimate and repeats its main() with the movie's own camera (calib_sim.json: gain 1, flat offset map).

Differences to paper_pipeline.main (all sim-specific):
  * movie, camera and PSF from the movie dir (PSF default = common.PSF_CALIB, the averaged 1809/2 PSF that rendered the movie;
    2D fits at its emitter plane 0.055 um = the simulated z, as the paper's fixed-sigma Gaussian fits).
  * smlm_spots_per_bin 10000 (the preprint's simulation call of run_pipeline.process, fig_densities2.py; the real-data runs use
    20000).
  * two SIMFLUX variants: 'sf' = the pattern estimated from the SMLM fits (mp_smlm.const_phase_offsets(), as on real data) and
    'sf_gtp' = the ground-truth pattern (the preprint's simulations: process(use_gt_mp=True) set mp_sf = mp_gt). Both use the
    modulation-error filter with their own pattern, the same SIMFLUX LM and the |SIMFLUX - SMLM| < 0.5 px filter.
usage: CUDA_VISIBLE_DEVICES=0 python analysis/papersim/run_baselines.py <movie dir> [--maxframes N] [--steps smlm,sf,sf_gtp]
          [--psf spec] [--out <movie dir>/base]
outputs: <out>/smlm/smlm.npy (+ hdf5), mp_smlm.pickle, info.json; <out>/sf/sf.hdf5, info.json; <out>/sf_gtp/sf.hdf5, info.json;
         <out>/run.json (timing). Resumable: the smlmtorch ROI / fit caches under <out>/work are reused."""
import os, sys, json, time, argparse, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import common as C
_s = importlib.util.spec_from_file_location('paper_pipeline', os.path.join(REPO, 'analysis', 'baselines', 'paper_pipeline.py'))
PP = importlib.util.module_from_spec(_s); _s.loader.exec_module(PP)
import numpy as np, torch
from smlmtorch import config_dict
from smlmtorch.simflux.pattern_estimator import ModulationPattern

SPOTS_PER_BIN_SIM = 10000


def sf_step(sfloc, psf, mp, smlm_ds, name, out, T, init='fixed'):
    t2 = time.time()
    me = mp.mod_error(smlm_ds); me_sel = me < PP.PAPER['mod_error_threshold']
    sf_ds, sdiag = PP.fit_simflux_torch(sfloc, psf, mp, smlm_ds[me_sel], PP.PAPER['sf_iterations'], PP.PAPER['distFromSMLM'], init=init)
    torch.cuda.synchronize(); T[f'{name}_fit_s'] = time.time() - t2
    os.makedirs(f'{out}/{name}', exist_ok=True)
    sf_ds['pixelsize'] = PP.PAPER['pixelsize']
    sf_ds.save(f'{out}/{name}/sf.hdf5'); mp.save(f'{out}/{name}/mp_sf.pickle')
    sinfo = dict(n_modsel=int(me_sel.sum()), n_in=int(len(smlm_ds)), n_out=int(len(sf_ds)), lm=sdiag, pattern=dict(
        pitch_nm=(2 * np.pi / np.linalg.norm(mp.mod['k'][:, :2], axis=1) * PP.PAPER['pixelsize']).round(2).tolist(),
        depth=np.asarray(mp.mod['depth']).tolist()), **PP.crlb_summary(sf_ds))
    print(name, json.dumps(sinfo), flush=True)
    json.dump(sinfo, open(f'{out}/{name}/info.json', 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('movie_dir')
    ap.add_argument('--psf', default=C.PSF_CALIB)
    ap.add_argument('--psf_z', type=float, default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--maxframes', type=int, default=None)
    ap.add_argument('--steps', default='smlm,sf,sf_gtp')
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--spots_per_bin', type=int, default=SPOTS_PER_BIN_SIM)
    a = ap.parse_args()
    from simcode.util.proctitle import auto; auto('psim-base')
    md = os.path.abspath(a.movie_dir)
    out = os.path.abspath(a.out or os.path.join(md, 'base'))
    os.makedirs(out, exist_ok=True)
    tif = os.path.join(md, 'movie.tif')
    cam = np.load(os.path.join(md, 'camera_sim.npz'))
    gain, offset = float(cam['gain']), cam['offset_adu'].astype(np.float32)
    run = 'torch_sim' + (f'_{a.maxframes}' if a.maxframes else '')
    psf = PP.zpsf.load(a.psf, z_fit=a.psf_z)
    rel = lambda p: os.path.relpath(p, os.path.dirname(tif))
    T = {}; t0 = time.time()
    kw = dict(pattern_frames=PP.PAPER['pattern_frames'], pixelsize=PP.PAPER['pixelsize'], detection_threshold=PP.PAPER['detection_threshold'],
              roisize=PP.PAPER['roisize'], device='cuda:0', zrange=PP.PAPER['zrange'], gain=gain, offset=offset, maxframes=a.maxframes,
              psf_sigma_binsize=None, cache_name=rel(f'{out}/work/{run}/cache'), result_dir=rel(f'{out}/work/{run}/results'))
    sfloc = PP.PSFLocalizer(tif, psf, iters=a.iters, det_template='paper', psf_calib=[1.3, 1.3], **kw)
    cfg = config_dict(dict(psf_calib=psf.desc, roisize=PP.PAPER['roisize'], detection_threshold=PP.PAPER['detection_threshold'],
                           pattern_frames=PP.PAPER['pattern_frames'], gain=gain, offset='camera_sim.npz', pixelsize=PP.PAPER['pixelsize'],
                           zrange=PP.PAPER['zrange'], result_dir=sfloc.result_dir, device='cuda:0'))
    sfloc.detect_spots(ignore_cache=False, moving_window=True)
    torch.cuda.synchronize(); T['detect_s'] = time.time() - t0
    smlm_ds = sfloc.fit_smlm(max_crlb_xy=None, ignore_cache=False, max_chisq=PP.PAPER['smlm_max_chisq'])
    torch.cuda.synchronize(); T['smlm_fit_s'] = time.time() - t0 - T['detect_s']
    lm_stats = getattr(sfloc.smlm_fitter, 'stats', {})
    os.makedirs(f'{out}/smlm', exist_ok=True)
    smlm_ds['pixelsize'] = PP.PAPER['pixelsize']
    smlm_ds.save(f'{out}/smlm/smlm.npy'); smlm_ds.save(f'{out}/smlm/smlm.hdf5')
    info = dict(movie=md, psf=psf.desc, psf_z_fit_um=getattr(psf, 'z_fit', None), maxframes=a.maxframes, numrois=int(sfloc.numrois),
                n_after_chisq=int(len(smlm_ds)), lm=lm_stats, chisq_per_px_median=float(np.median(smlm_ds.chisq) / PP.PAPER['roisize'] ** 2),
                **PP.crlb_summary(smlm_ds))
    print('SMLM:', json.dumps(info), flush=True)
    json.dump(info, open(f'{out}/smlm/info.json', 'w'), indent=1)
    steps = a.steps.split(',')
    if 'sf' in steps:
        t1 = time.time()
        mp_smlm = PP.estimate_mp_smlm(smlm_ds, cfg, a.spots_per_bin, PP.PAPER['mod_error_threshold'], PP.PAPER['fix_depth'])
        mp_smlm.save(f'{out}/smlm/mp_smlm.pickle')
        mp_sf = mp_smlm.const_phase_offsets(); mp_sf.save(f'{out}/smlm/mp_sf.pickle')
        T['pattern_s'] = time.time() - t1
        sf_step(sfloc, psf, mp_sf, smlm_ds, 'sf', out, T)
    if 'sf_gtp' in steps:
        mp_gt = ModulationPattern.load(os.path.join(md, 'mp_gt.pickle'))
        sf_step(sfloc, psf, mp_gt, smlm_ds, 'sf_gtp', out, T)
    T['total_s'] = time.time() - t0
    json.dump(dict(args=vars(a), timing=T, run=run, gain=gain, offset='camera_sim.npz'), open(f'{out}/run.json', 'w'), indent=1)
    print('timing', T, flush=True)


if __name__ == '__main__':
    main()
