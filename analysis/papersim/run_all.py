"""One command per papersim movie: all methods + metrics, resumable (a finished step is skipped; --redo <step> forces it).

  CUDA_VISIBLE_DEVICES=1 python analysis/papersim/run_all.py <movie dir> [--nframes N] [--run <run dir>]
        [--steps base,decode,v2,metrics] [--netset default] [--cal_kind sim] [--decode_name dec_1809_2] [--redo v2]
Run dir default: <movie dir>/full (all frames) or <movie dir>/cut<N> (first N frames). Steps:
  base     run_baselines.py: SMLM, SIMFLUX (estimated pattern), SIMFLUX (GT pattern)   -> <run>/base/{smlm,sf,sf_gtp}
  decode   run_decode.py                                                               -> <run>/decode/locs.npz
  v1       run_simcode_v1.py: the preprint's SIMCODE (paper defaults)                  -> <run>/v1/locs.npz
  v1gt     the same with the ground-truth pattern (diagnostic)                     -> <run>/v1gt/locs.npz
  v2       run_simcode.py (network set --netset; + the GT-pattern diagnostic)           -> <run>/v2_<netset>/locs_p2_sim.npz
  hollow   hollow_analysis.py (hollow-tube movies only): cross-section profiles, dip ratio, two-line vs one-line fit
  metrics  metrics.py (density: FRC, Jaccard vs frames, counts, GT match; moving: block Jaccard) -> <run>/metrics_<netset>.json
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
    ap.add_argument('--netset', default='default')
    ap.add_argument('--cal_kind', default='sim')
    ap.add_argument('--decode_name', default='dec_1809_2')
    ap.add_argument('--redo', default='')
    ap.add_argument('--psf', default=None, help="baselines' PSF spec (run_baselines --psf), e.g. gauss:1.3 for the preprint reproduction")
    ap.add_argument('--gt_diag', type=int, default=1, help='v2: also the GT-pattern pass-2 diagnostic (0 = skip, ~40 %% of the v2 time)')
    a = ap.parse_args()
    sys.path.insert(0, REPO)
    from simcode2.util.proctitle import auto; auto('psim-all')
    md = os.path.abspath(a.movie_dir)
    if not os.path.exists(os.path.join(md, 'movie.tif')):
        sys.exit(f'{md}/movie.tif missing: generate it first (gen_density.py --density D / gen_moving.py)')
    g = json.load(open(os.path.join(md, 'gen.json')))
    nf = a.nframes or g['nframes']
    run = os.path.abspath(a.run or os.path.join(md, 'full' if nf == g['nframes'] else f'cut{nf}'))
    os.makedirs(os.path.join(run, 'logs'), exist_ok=True)
    v2dir = os.path.join(run, f'v2_{a.netset}' + ('' if a.cal_kind == 'sim' else f'_{a.cal_kind}'))
    py = sys.executable; S = os.path.join(HERE, '{}')
    mf = [] if nf == g['nframes'] else ['--maxframes', str(nf)]
    steps = {
        'base': ([py, S.format('run_baselines.py'), md, '--out', os.path.join(run, 'base')] + mf + (['--psf', a.psf] if a.psf else []), os.path.join(run, 'base', 'run.json')),
        'decode': ([py, S.format('run_decode.py'), md, '--name', a.decode_name, '--out', os.path.join(run, 'decode', 'locs.npz')] + mf,
                   os.path.join(run, 'decode', 'locs.npz')),
        'decode_pre': ([py, S.format('run_decode_preprint.py'), md, '--out', os.path.join(run, 'decode_pre', 'locs.npz')] + mf, os.path.join(run, 'decode_pre', 'locs.npz')),
        'decode_sim': ([py, S.format('run_decode.py'), md, '--name', 'dec_sim1809', '--out', os.path.join(run, 'decode_sim', 'locs.npz')] + mf,
                       os.path.join(run, 'decode_sim', 'locs.npz')),
        'v1': ([py, S.format('run_simcode_v1.py'), md, '--out', os.path.join(run, 'v1', 'locs.npz')] + mf, os.path.join(run, 'v1', 'v1_info.json')),
        'v1gt': ([py, S.format('run_simcode_v1.py'), md, '--pattern', 'gt', '--out', os.path.join(run, 'v1gt', 'locs.npz')] + mf, os.path.join(run, 'v1gt', 'v1_info.json')),
        'v2': ([py, S.format('run_simcode.py'), md, '--netset', a.netset, '--cal_kind', a.cal_kind, '--out', v2dir, '--gt_diag', str(a.gt_diag)]
               + ([] if nf == g['nframes'] else ['--nframes', str(nf)]), os.path.join(v2dir, 'papersim_run.json')),
        'metrics': ([py, S.format('metrics.py'), run, '--movie', md, '--nframes', str(nf), '--netset', a.netset], None),
        'hollow': ([py, S.format('hollow_analysis.py'), run, '--movie', md, '--nframes', str(nf), '--netset', a.netset], None),
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
        TM[st + ('' if st in ('base', 'decode', 'decode_pre', 'decode_sim', 'v1', 'v1gt') else f'_{a.netset}')] = dict(s=round(time.time() - t0, 1), rc=r.returncode, nframes=nf, when=time.ctime(),
                                                                            gpu=os.environ.get('CUDA_VISIBLE_DEVICES', ''))
        json.dump(TM, open(tf, 'w'), indent=1)
        print(f'[{st}] rc {r.returncode} {time.time() - t0:.1f} s', flush=True)
        if r.returncode: sys.exit(f'step {st} failed, see {run}/logs/{st}.log')


if __name__ == '__main__':
    main()
