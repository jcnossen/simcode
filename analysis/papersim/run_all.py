"""One command per papersim movie: all methods + metrics, resumable (a finished step is skipped; --redo <step> forces it).

  CUDA_VISIBLE_DEVICES=1 python analysis/papersim/run_all.py <movie dir> [--nframes N] [--run <run dir>]
        [--steps base,decode,v2,metrics] [--model m1809] [--cal_kind sim] [--decode_name dec_1809_2] [--redo v2]
Run dir default: <movie dir>/full (all frames) or <movie dir>/cut<N> (first N frames). Steps:
  base     run_baselines.py: SMLM, SIMFLUX (estimated pattern), SIMFLUX (GT pattern)   -> <run>/base/{smlm,sf,sf_gtp}
  decode   run_decode.py                                                               -> <run>/decode/locs.npz
  v1       run_simcode_v1.py: the preprint's SIMCODE (paper defaults)                  -> <run>/v1/locs.npz
  v1gt     the same with the ground-truth pattern (diagnostic)                     -> <run>/v1gt/locs.npz
  v2       run_simcode.py (network set --model; + the GT-pattern diagnostic)           -> <run>/v2_<model>/locs_p2.hdf5
  hollow   hollow_analysis.py (hollow-tube movies only): cross-section profiles, dip ratio, two-line vs one-line fit
  metrics  metrics.py (density: FRC, Jaccard vs frames, counts, GT match; moving: block Jaccard) -> <run>/metrics_<model>.json
  gtfrc    gtfrc_run.py (density): GT-FRC, time to fidelity, sigma-gated Jaccard, ghosts, coverage (+ the v2 of --ref)
                                                                               -> <run>/gtfrc_<model>.json
Timing of every step is appended to <run>/timing.json; stdout / stderr of each step go to <run>/logs/<step>.log."""
import argparse, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('movie_dir')
    ap.add_argument('--nframes', type=int, default=0)
    ap.add_argument('--run', default=None)
    ap.add_argument('--steps', default='base,decode,v2,metrics')
    ap.add_argument('--model', default='m1809')
    ap.add_argument('--cal_kind', default='sim')
    ap.add_argument('--decode_name', default='dec_1809_2')
    ap.add_argument('--redo', default='')
    ap.add_argument('--psf', default=None, help="baselines' PSF spec (run_baselines --psf), e.g. gauss:1.3 for the preprint reproduction")
    ap.add_argument('--psf_z', default=None, help="baselines' fixed fit plane (um; run_baselines --psf_z; default = the PSF calibration's "
                    "emitter_z_median_um) or 'gt' = gen.json z_um, the median ground-truth height (the *_wsm3d sets)")
    ap.add_argument('--gt_diag', type=int, default=1, help='v2: also the GT-pattern pass-2 diagnostic (0 = skip, ~40 %% of the v2 time)')
    ap.add_argument('--tag', default='', help="suffix of the v2 output and of the metrics name, e.g. '_3d': v2 -> <run>/v2_<model>_3d, "
                    'metrics -> <run>/metrics_<model>_3d.json (a rerun of v2 with new networks next to the earlier one)')
    ap.add_argument('--name', default='', help="name of the v2 output / metrics instead of --model (needed when --model is a model folder), "
                    "e.g. --model out/psf_v2/eval/m1809v2/model_m1809 --name m1809v2 -> <run>/v2_m1809v2, metrics_m1809v2.json")
    ap.add_argument('--ref', default='', help="gtfrc: also the v2 table of this metrics name (e.g. m1809 = the earlier v2) as 'v2sig_ref'")
    a = ap.parse_args()
    sys.path.insert(0, REPO)
    from simcode.util.proctitle import auto; auto('psim-all')
    md = os.path.abspath(a.movie_dir)
    if not os.path.exists(os.path.join(md, 'movie.tif')):
        sys.exit(f'{md}/movie.tif missing: generate it first (gen_density.py --density D / gen_moving.py)')
    g = json.load(open(os.path.join(md, 'gen.json')))
    nf = a.nframes or g['nframes']
    run = os.path.abspath(a.run or os.path.join(md, 'full' if nf == g['nframes'] else f'cut{nf}'))
    os.makedirs(os.path.join(run, 'logs'), exist_ok=True)
    nm = a.name or a.model
    psf_z = (f"{float(g['z_um']):.6f}" if a.psf_z == 'gt' else a.psf_z)
    v2dir = os.path.join(run, f'v2_{nm}' + ('' if a.cal_kind == 'sim' else f'_{a.cal_kind}') + a.tag)
    mname = nm + a.tag                 # name of the metrics / hollow outputs (metrics.py reads <run>/v2_<mname>)
    py = sys.executable; S = os.path.join(HERE, '{}')
    mf = [] if nf == g['nframes'] else ['--maxframes', str(nf)]
    steps = {
        'base': ([py, S.format('run_baselines.py'), md, '--out', os.path.join(run, 'base')] + mf + (['--psf', a.psf] if a.psf else [])
                 + (['--psf_z', psf_z] if psf_z else []), os.path.join(run, 'base', 'run.json')),
        'decode': ([py, S.format('run_decode.py'), md, '--name', a.decode_name, '--out', os.path.join(run, 'decode', 'locs.npz')] + mf,
                   os.path.join(run, 'decode', 'locs.npz')),
        'decode_pre': ([py, S.format('run_decode_preprint.py'), md, '--out', os.path.join(run, 'decode_pre', 'locs.npz')] + mf, os.path.join(run, 'decode_pre', 'locs.npz')),
        'decode_sim': ([py, S.format('run_decode.py'), md, '--name', 'dec_sim1809', '--out', os.path.join(run, 'decode_sim', 'locs.npz')] + mf,
                       os.path.join(run, 'decode_sim', 'locs.npz')),
        'v1': ([py, S.format('run_simcode_v1.py'), md, '--out', os.path.join(run, 'v1', 'locs.npz')] + mf, os.path.join(run, 'v1', 'v1_info.json')),
        'v1gt': ([py, S.format('run_simcode_v1.py'), md, '--pattern', 'gt', '--out', os.path.join(run, 'v1gt', 'locs.npz')] + mf, os.path.join(run, 'v1gt', 'v1_info.json')),
        'v2': ([py, S.format('run_simcode.py'), md, '--model', a.model, '--cal_kind', a.cal_kind, '--out', v2dir, '--gt_diag', str(a.gt_diag)]
               + ([] if nf == g['nframes'] else ['--nframes', str(nf)]), os.path.join(v2dir, 'papersim_run.json')),
        'metrics': ([py, S.format('metrics.py'), run, '--movie', md, '--nframes', str(nf), '--model', mname], None),
        'gtfrc': ([py, S.format('gtfrc_run.py'), run, '--movie', md, '--nframes', str(nf), '--model', mname] + (['--ref', a.ref] if a.ref else []), None),
        'hollow': ([py, S.format('hollow_analysis.py'), run, '--movie', md, '--nframes', str(nf), '--model', mname], None),
    }
    import shutil
    def disk_ok():         # 2026-11-12: the root disk filled up once; stop cleanly instead of writing truncated files
        free = shutil.disk_usage(REPO).free / 1e9
        if free < 15:
            sys.exit(f'DISK GUARD: only {free:.1f} GB free on the output disk (< 15 GB); not starting the next step')
    tf = os.path.join(run, 'timing.json')
    TM = json.load(open(tf)) if os.path.exists(tf) else {}
    for st in a.steps.split(','):
        disk_ok()
        cmd, done = steps[st]
        if done and os.path.exists(done) and st not in a.redo.split(','):
            print(f'[{st}] done ({done})', flush=True); continue
        t0 = time.time(); print(f'[{st}] ' + ' '.join(cmd), flush=True)
        with open(os.path.join(run, 'logs', f'{st}.log'), 'a') as lf:
            r = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT)
        TM[st + ('' if st in ('base', 'decode', 'decode_pre', 'decode_sim', 'v1', 'v1gt') else f'_{mname}')] =dict(s=round(time.time() - t0, 1), rc=r.returncode, nframes=nf, when=time.ctime(),
                                                                            gpu=os.environ.get('CUDA_VISIBLE_DEVICES', ''))
        json.dump(TM, open(tf, 'w'), indent=1)
        print(f'[{st}] rc {r.returncode} {time.time() - t0:.1f} s', flush=True)
        if r.returncode: sys.exit(f'step {st} failed, see {run}/logs/{st}.log')


if __name__ == '__main__':
    main()
