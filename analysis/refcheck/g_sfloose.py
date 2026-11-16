"""Referee check (g): SIMFLUX with a looser modulation-error filter on 2.5nM_2 (Fig 4 keeps 6 % of the raw detections).

sf.hdf5 holds only the rows AFTER the filter (modulation error < 0.1, then the SIMFLUX fit and its limits / |SF - SMLM| < 0.5 px),
so the looser sets need the SIMFLUX fit of the extra ROIs. Refit-free part (CPU): the modulation error of every accepted SMLM
row (smlm.npy, the figure's SMLM table) under the figure's SIMFLUX pattern (sf/mp_sf.pickle) -> counts at < 0.1 (asserted = the
pipeline's n_modsel), 0.15, 0.2, 0.3, also on the fair-FRC windows / crop.
--fit (GPU, minutes): only the SIMFLUX LM step of analysis/baselines/paper_pipeline.py (fit_simflux_torch, the same PSF, camera,
fit plane, pattern, iterations, limits and |SF - SMLM| < 0.5 px) on the rows with modulation error < 0.3 in the fair-FRC windows (start % 6 == 0) and crop (the only rows the FRC / RSP use), reading the ROIs from
the figure run's smlmtorch cache READ-ONLY (cache writes / rebuilds are patched to raise); every output goes to
out/refcheck/sf_loose/<ds>/: sf_lt<thr>.hdf5 per threshold (rows of the < 0.3 fit with their own modulation error < thr).
usage: CUDA_VISIBLE_DEVICES=0 python analysis/refcheck/g_sfloose.py 2.5nM_2 [--fit]"""
import os, sys, json, time, argparse, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(REPO, 'analysis', 'baselines'), REPO]
THR = (0.1, 0.15, 0.2, 0.3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('ds'); ap.add_argument('--fit', action='store_true')
    ap.add_argument('--base', default=None, help='baseline dir (default out/paper2/<ds>/baselines_psfv2)')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    base = a.base or os.path.join(REPO, 'out', 'paper2', a.ds, 'baselines_psfv2')
    od = os.path.join(REPO, 'out', 'refcheck', 'sf_loose', a.ds + a.tag); os.makedirs(od, exist_ok=True)
    import paper_pipeline as PPL
    from smlmtorch.simflux.dataset import SFDataset
    import smlmtorch.simflux.pattern_estimator as pe
    t0 = time.time()
    # (2026-11-15 22:07 another session replaced <base>/smlm by a link to the fit-plane-0.318 refit and moved the Fig 4 table
    # to <base>/smlm_z086; the SIMFLUX fit belongs to the latter)
    sd = os.path.join(base, 'smlm_z086') if os.path.isdir(os.path.join(base, 'smlm_z086')) else os.path.join(base, 'smlm')
    smlm = SFDataset.load(os.path.join(sd, 'smlm.npy')); res_smlm = sd
    mp = pe.ModulationPattern.load(os.path.join(base, 'sf', 'mp_sf.pickle'))
    me = mp.mod_error(smlm)
    sinfo = json.load(open(os.path.join(base, 'sf', 'info.json')))
    f = np.asarray(smlm.frame); x, y = smlm.pos[:, 0], smlm.pos[:, 1]
    fair = (f % 6 == 0) & (x >= 5) & (x < 115) & (y >= 5) & (y < 115)
    res = dict(base=base, smlm_table=res_smlm, n_smlm=int(len(smlm)), n_modsel_pipeline=sinfo['n_modsel'], counts={}, counts_fair_windows={},
               me_quantiles=np.quantile(me, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist())
    for t in THR:
        res['counts'][str(t)] = int((me < t).sum()); res['counts_fair_windows'][str(t)] = int(((me < t) & fair).sum())
    assert res['counts']['0.1'] == sinfo['n_modsel'], (res['counts']['0.1'], sinfo['n_modsel'])
    print(json.dumps(res), f'({time.time() - t0:.0f} s)', flush=True)
    json.dump(res, open(os.path.join(od, 'counts.json'), 'w'), indent=1)
    if not a.fit: return
    # ---- SIMFLUX LM on the < 0.3 rows, ROIs from the figure run's cache (read only)
    import smlmtorch.util.caching as cache
    _read = cache.read

    def read_only(cache_fn, cfg, func, rebuild=False, verbose=False):
        def refuse(): raise SystemExit(f'cache {cache_fn} would be rebuilt: refusing (read-only use)')
        return _read(cache_fn, cfg, refuse, rebuild=False, verbose=True)
    cache.read = read_only
    cache.save_cache_cfg = lambda *q, **k: (_ for _ in ()).throw(SystemExit('cache cfg write refused (read-only use)'))
    run = json.load(open(os.path.join(base, 'run.json'))); A = run['args']
    D = PPL.DATASETS[a.ds]
    cc = json.load(open(A['psf']))['camera']; cam = np.load(os.path.join(os.path.dirname(os.path.abspath(A['psf'])), cc['maps']))
    gain, offset = float(cc['gain_e_per_adu']), cam['offset_adu'].astype(np.float32)
    psf = PPL.zpsf.load(A['psf'], z_fit=run['psf_z_fit_um'], zmax=A['psf_zmax'])
    tif = os.path.join(REPO, 'out', 'paper2', a.ds, 'movie', os.path.basename(D['tif']))
    rel = lambda p: os.path.relpath(p, os.path.dirname(tif))
    P = PPL.PAPER
    kw = dict(pattern_frames=P['pattern_frames'], pixelsize=P['pixelsize'], detection_threshold=P['detection_threshold'],
              roisize=P['roisize'], device='cuda:0', zrange=P['zrange'], gain=gain, offset=offset, maxframes=None,
              psf_sigma_binsize=None, cache_name=rel(f"{base}/work/{run['run']}/cache"), result_dir=rel(f'{od}/results'))
    sfloc = PPL.PSFLocalizer(tif, psf, iters=A['iters_smlm'], det_template=A['det_template'], psf_calib=[1.3, 1.3], **kw)
    sfloc.detect_spots(ignore_cache=False, moving_window=True)
    smlm_ds = sfloc.fit_smlm(max_crlb_xy=None, ignore_cache=False, max_chisq=P['smlm_max_chisq'])
    assert len(smlm_ds) == len(smlm), (len(smlm_ds), len(smlm))
    me2 = mp.mod_error(smlm_ds); assert np.allclose(me2, me)
    fd = np.asarray(smlm_ds.frame); px_, py_ = smlm_ds.pos[:, 0], smlm_ds.pos[:, 1]
    sel = (me2 < max(THR)) & (fd % 6 == 0) & (px_ >= 5) & (px_ < 115) & (py_ >= 5) & (py_ < 115)   # only the fair-FRC windows / crop
    t1 = time.time()
    sf_ds, diag = PPL.fit_simflux_torch(sfloc, psf, mp, smlm_ds[sel], P['sf_iterations'], P['distFromSMLM'], init=A['sf_init'])
    sf_ds['pixelsize'] = P['pixelsize']
    me_sf = me2[sel][np.asarray(sf_ds.roi_id)]        # fit_simflux_torch stores the index into its input rows as roi_id
    out = dict(fit_s=time.time() - t1, diag=diag, n_fit_in=int(sel.sum()), n_out=int(len(sf_ds)), tables={})
    for t in THR:
        m = me_sf < t
        fn = os.path.join(od, f'sf_lt{t:g}.hdf5'); sf_ds[m].save(fn)
        out['tables'][str(t)] = dict(path=fn, n=int(m.sum()))
    print(json.dumps(out), flush=True)
    res['fit'] = out
    json.dump(res, open(os.path.join(od, 'counts.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
