"""
SIMCODE v2 on a DMD-SIMFLUX DNA-PAINT movie, production settings.

  python -m simcode.run --movie <movie.tif> --calib <in-situ calib.json> --out <dir> [--model m1809]
  python -m simcode.run --dataset 1809_2 --out <dir>          (a paper dataset: movie, calibration, model)

Stages (each skipped if its outputs exist, unless --redo):
  1  pass 1      detector + head on every 6-frame window (start every 3 frames): network per-frame intensities
                 with covariance, bg, pattern-free position prior; candidates with detector p >= 0.3 are stored
                 (simcode/pipeline/common.CAND_PMIN)
  2  pattern     illumination pattern (k, per-step phase / depth / relative intensity, per-angle phase drift) estimated from
                 the pass-1 network intensities
  3  solve       explicit intensity-domain pattern solve of the pass-1 rows (fringe posterior, kinetics, calibrated sigma)
                 with the model's fixed-c solve calibration
  4  pass 2      pattern-aware pass-2 detector + head (p2L) fed the estimated pattern, then the same solve
  5  table       localization tables locs_p1.npz and locs_p2.npz with the selections (simcode/strict.py): main = the
                 paper's global precision rule max(sx, sy) <= sigma_max nm (--sigma_max), u = the earlier per-tile
                 "u <= 0.45" (--u_max), sel (phase-calibrated), strict (per-10-px-tile sigma quantile, q from the
                 row density); the model's selection-aware sigma rescale (--sig_u); the selections as small files
                 (--subset); summary.json
Networks and calibrations come from a model (--model, simcode/models.json; default m1809; --netset is the older name);
--det / --head / --cal / --p2_det / --p2_head / --p2_cal override single files. Paths are relative to the models directory
(simcode/config.py). --dataset <name> takes movie, calibration and model of a paper dataset from the '_production' entry of
models.json (what the paper tables are made with).
Run record: <out>/run_config.json (+ run_config.yaml), written at the start (status 'running') and completed at the end
('done'): the fully resolved settings, sha256 of the movie / calibration / every model file, code commit, environment,
pattern and selection outcome (simcode/util/runconfig.py; README "Run record").
  python -m simcode.run --config <run_config.json> --out <new dir>     re-run with exactly the settings of a record
Code path: queued batching, the exact fast solve, torch.compile of the solve's LM step when available.
Set CUDA_VISIBLE_DEVICES to choose the GPU; several GPUs: the same command once per GPU with --claim 1
(python -m simcode.fileio.launch --gpus 1,2 -- python -m simcode.run ... --claim 1): the chunks of pass 1 and pass 2 are
shared between the copies, the stages in between run in one of them.
A 6000-frame 0.5 nM chunk takes ~0.5 min (pass 1 + solve) on an idle RTX 3090.
Options removed on 2026-11-14 (their production value is now the only behaviour) are accepted only with that value:
--pass2, --autoc 0, --autoc_fast 0, --cand_pmin 0.3, --fast 1, --compile auto, --compile_head off (simcode/util/retired.py).
"""
import argparse, json, os, sys, time
from simcode import config

PROD = config.production()
SUBSETS = 'main,u,strict'           # selections written as small files (strict.write_subset)
RETIRED = {'--pass2': None, '--autoc': '0', '--autoc_fast': '0', '--cand_pmin': '0.3', '--fast': '1', '--compile': 'auto',
           '--compile_head': 'off'}


def _uncal(args):
    from simcode.pipeline import run_uncalibrated
    run_uncalibrated.main(args)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1].strip(), formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='', help='run_config.json of an earlier run (or its directory): re-run with exactly its settings '
                    '(movie, calibration, model files, crop, chunking, selection); arguments given after it override single settings; '
                    'a warning is printed for every input file, git commit or fixed value that differs from the record')
    ap.add_argument('--dataset', default='', help="a paper dataset (models.json '_production': " + ', '.join(PROD['datasets']) + '): '
                    'sets --movie, --calib, --model')
    ap.add_argument('--movie', default='', help='movie .tif (further parts movie_X1.tif ... are found automatically)')
    ap.add_argument('--calib', default='', help='in-situ PSF / camera calibration json (e.g. insitu/calib_1809_2_psf3d.json)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', '--netset', dest='model', default='', help='model of simcode/models.json (default: its _default)')
    ap.add_argument('--cal_kind', default=PROD['cal_kind'], choices=['real', 'sim'], help="solve calibrations of the model: 'real' (real movies; default) or 'sim' (simulated movies)")
    ap.add_argument('--prefetch', type=int, default=1, help='read the next chunk in a background thread (identical data); 0 = off')
    ap.add_argument('--claim', type=int, default=0, help='1: several copies of this command share the chunks (one copy per GPU)')
    ap.add_argument('--min_free_gb', type=float, default=2.0, help='a chunk is not started with less free space on the output file system')
    ap.add_argument('--crop', default='5,5,110', help='output region x0,y0,size (px)')
    ap.add_argument('--chunk', type=int, default=6000)
    ap.add_argument('--chunk_list', default='', help="process only these chunk starts, e.g. '12000'")
    ap.add_argument('--nframes', type=int, default=0)
    ap.add_argument('--det', default='', help='pass-1 detector (default: the model)'); ap.add_argument('--head', default='', help='pass-1 head')
    ap.add_argument('--cal', default='', help="pass-1 solve calibration (default: the model's)")
    ap.add_argument('--p2_det', default=''); ap.add_argument('--p2_head', default='')
    ap.add_argument('--p2_cal', default='', help="pass-2 solve calibration (default: the model's)")
    ap.add_argument('--drift', default='', help='optional drift [F,2] px for the strict-product coverage (e.g. DME output)')
    ap.add_argument('--strict_q', type=float, default=None, help='fixed per-tile quantile (default: from the row density)')
    ap.add_argument('--sigma_max', type=float, default=None, help='nm; threshold of the main selection (max(sx, sy) <= sigma_max on the rows with a rank; stored in the table; default strict.SIGMA_MAX_NM)')
    ap.add_argument('--u_max', type=float, default=None, help="threshold of the per-tile rule 'u' (strict_u <= u_max, stored in the table; default 0.45)")
    ap.add_argument('--subset', default=SUBSETS, help=f"selections also written as small files (strict.write_subset; default {SUBSETS!r}; '' = none)")
    ap.add_argument('--format', default='npz', help='table formats: npz,csv,hdf5')
    ap.add_argument('--sig_u', default='set', help='selection-aware sigma rescale json per pass, "<pass1>,<pass2>" (strict.build_table); '
                    "'set' (default) = the rescale of the model; '' = none")
    ap.add_argument('--redo', type=int, default=0)
    return ap


def resolve(ap, argv):
    """arguments -> (a, model dict, pass-1 solve calibration): the dataset, the model's files and the defaults filled in"""
    from simcode.util import retired
    a = ap.parse_args(retired.strip(argv, RETIRED, 'simcode.run'))
    if a.dataset:
        D = PROD['datasets'].get(a.dataset)
        if D is None: ap.error(f'unknown dataset {a.dataset!r} ({", ".join(PROD["datasets"])})')
        if not (a.model or D['model']): ap.error(f'dataset {a.dataset}: no model is defined for it; give --model')
        a.movie = a.movie or D['movie']; a.calib = a.calib or D['calib']; a.model = a.model or D['model']
    if not a.movie or not a.calib: ap.error('--movie and --calib are required (or --dataset)')
    N = config.model_set(a.model or None)
    a.det = a.det or N['det']; a.head = a.head or N['head']; a.p2_det = a.p2_det or N['p2_det']; a.p2_head = a.p2_head or N['p2_head']
    a.p2_cal = a.p2_cal or N[f'p2_cal_{a.cal_kind}']
    if a.sig_u == 'set': a.sig_u = N['sig_u']
    # (2026-11-11) with an explicit --cal (e.g. a retrained head's calibration) its prior_off also feeds the pattern stage
    cal = config.resolve(a.cal) if a.cal else N[f'cal_{a.cal_kind}']
    return a, N, cal


def settings_of(a, cal):
    """the fully resolved settings of a run (run_config.json 'settings'): every argument after the defaults, files as absolute paths"""
    from simcode import strict as SP
    r = config.resolve
    return dict(dataset=a.dataset, movie=r(a.movie, 'data'), calib=r(a.calib), out=os.path.abspath(a.out), model=getattr(a, 'model_name', a.model), cal_kind=a.cal_kind,
                det=r(a.det), head=r(a.head), cal=r(cal), p2_det=r(a.p2_det), p2_head=r(a.p2_head), p2_cal=r(a.p2_cal),
                sig_u=','.join(r(s) if s else '' for s in a.sig_u.split(',')) if a.sig_u else '',
                crop=a.crop, chunk=a.chunk, chunk_list=a.chunk_list, nframes=a.nframes,
                sigma_max=SP.SIGMA_MAX_NM if a.sigma_max is None else a.sigma_max, u_max=SP.U_MAX if a.u_max is None else a.u_max,
                strict_q=a.strict_q, drift=os.path.abspath(a.drift) if a.drift else '', subset=a.subset, format=a.format,
                prefetch=a.prefetch, claim=a.claim, min_free_gb=a.min_free_gb, redo=a.redo)


def fixed_values():
    """what a run does that is not an option (run_config.json 'fixed'), read from the code so that an in-memory override shows"""
    from simcode.pipeline import run_uncalibrated as RU, pattern_est as PE
    from simcode.pipeline.common import CAND_PMIN
    from simcode import strict as SP
    d = vars(RU.build_parser().parse_args(['--movie', 'x', '--calib', 'x', '--out', 'x']))
    return dict(stages='pass 1, pattern estimation, pattern solve of the pass-1 rows, pass 2 (pattern-aware) + solve, tables', pass2=True,
                window_frames=6, window_step=3, cand_pmin=CAND_PMIN, p_good=SP.P_GOOD, autoc=0,
                solve_c='fixed c: cmult / cmult_ang of the solve calibrations (model.files.cal / p2_cal, params)',
                code_path='fast (queued batching, exact fast solve)', compile=RU.COMPILE, compile_head='off',
                margin=d['margin'], nn_min=d['nn_min'], min_phot=d['min_phot'], det_batch=d['det_batch'], head_batch=d['head_batch'],
                solve_batch=d['solve_batch'], pattern_pitch_search_px=list(PE.estimate_k.__defaults__[0]),
                strict_crop_default=list(SP.CROP), strict_q_density_anchors={k: [list(v) for v in a] for k, a in SP.Q_DENSITY.items()},
                table_files='locs_p1 = tag est (sig_u pass 1), locs_p2 = tag est2 (sig_u pass 2)')


def _from_config(path, rest):
    """--config: (argv that reproduces the record + the arguments given after it, the record, its in-memory overrides)"""
    from simcode.util import runconfig as RC
    REC = RC.load(path)
    if not REC.get('settings'): raise SystemExit(f'--config {path}: no settings in this record (status {REC.get("status")!r})')
    M = REC.get('model') or {}
    if M.get('name') and M.get('definition') and M['name'] not in config.model_sets():        # e.g. the model of an automatic-pipeline dataset
        config.model_sets()[M['name']] = {k: v for k, v in M['definition'].items() if k != 'name'}
    W = REC.get('overrides')
    if W and not os.environ.get('AUTO_WRAP'):
        from simcode.auto import runwrap
        runwrap.apply({k: v for k, v in W.items() if k != 'title'})
        print('simcode.run --config: applied the in-memory overrides of the record:', ', '.join(k for k in W if k != 'title'), flush=True)
    return RC.rerun_argv(REC) + rest, REC, W


def main(argv=None):
    from simcode.util import runconfig as RC
    argv = list(sys.argv[1:] if argv is None else argv); argv0 = list(argv)
    ap = build_parser()
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False); pre.add_argument('--config', default='')
    c, rest = pre.parse_known_args(argv)
    REC = None; W = json.load(open(os.environ['AUTO_WRAP'])) if os.environ.get('AUTO_WRAP') else None
    if c.config:
        argv, REC, W0 = _from_config(c.config, rest); W = W or W0
    a, N, cal = resolve(ap, argv)
    a.model_name = N['name']
    from simcode.util.proctitle import auto; auto('run')     # recognizable name in nvidia-smi / ps
    from simcode.fileio.work import Units, atomic_json
    os.makedirs(a.out, exist_ok=True)
    # ---- the run record (run_config.json): written now with status 'running', completed at the end
    S = settings_of(a, cal); FX = fixed_values()
    su = (a.sig_u.split(',') + [''])[:2] if a.sig_u else ['', '']
    files = dict(det=S['det'], head=S['head'], cal=S['cal'], p2_det=S['p2_det'], p2_head=S['p2_head'], p2_cal=S['p2_cal'],
                 sig_u_p1=config.resolve(su[0]) if su[0] else '', sig_u_p2=config.resolve(su[1]) if su[1] else '')
    rerun_of = None
    if REC is not None:
        warn = RC.verify(REC, fixed=FX)
        for w in warn: print('simcode.run --config: WARNING', w, file=sys.stderr, flush=True)
        cf = c.config if os.path.isfile(c.config) else os.path.join(c.config, RC.NAME)
        rerun_of = dict(config=os.path.abspath(cf), sha256=RC.sha256(cf), status_of_source=REC.get('status'), arguments_after_config=rest, differences=warn)
    RC.log_command(a.out, 'simcode.run', argv0)
    RC.start(a.out, S, FX, RC.model_record(N['name'], N, files), RC.calib_record(a.calib), RC.movie_record(a.movie, a.nframes, a.chunk, a.chunk_list),
             overrides=W, rerun_of=rerun_of, argv=argv0)
    try:
        ST = _stages(a, N, cal, Units, atomic_json)
    except BaseException as e:
        RC.finish(a.out, error=f'{type(e).__name__}: {e}')
        raise
    from simcode import fastutil
    RC.finish(a.out, timing=ST, extra=dict(torch_compile_used=fastutil._STATE.get('ok')))
    print('done', json.dumps(ST), flush=True)


def _stages(a, N, cal, Units, atomic_json):
    t00 = time.time(); ST = {}
    common = ['--movie', a.movie, '--calib', a.calib, '--out', a.out, '--crop', a.crop, '--chunk', str(a.chunk),
              '--prefetch', str(a.prefetch), '--redo', str(a.redo),
              '--claim', str(a.claim), '--min_free_gb', str(a.min_free_gb), '--sc_det', a.det, '--sc_head', a.head]
    if a.chunk_list: common += ['--chunk_list', a.chunk_list]
    if a.nframes: common += ['--nframes', str(a.nframes)]
    # 1-2: pass 1 + pattern (the pattern stage only uses the calibration's prior_off, identical in all pass-1 calibrations)
    t0 = time.time()
    need_pat = a.redo or not os.path.exists(f'{a.out}/pattern_est.pt')
    _uncal(common + ['--frontend', 'sc', '--sc_cal', cal, '--stages', 'pass1' + (',pattern' if need_pat else '')])
    ST['pass1_pattern_s'] = time.time() - t0
    # 3: pattern solve of the pass-1 rows
    t0 = time.time()
    _uncal(common + ['--frontend', 'sc', '--sc_cal', cal, '--stages', 'pass2'])
    ST['solve_s'] = time.time() - t0
    # 4: pass 2
    t0 = time.time()
    _uncal(common + ['--frontend', 'sc2p', '--sc_cal', cal, '--p2_det', a.p2_det, '--p2_head', a.p2_head,
                     '--p2_cal', a.p2_cal, '--stages', 'pass2'])
    ST['pass2_s'] = time.time() - t0
    # 5: tables + selections (with --claim 1 the tables are written by one of the copies; the others skip them)
    from simcode import strict as SP
    px = json.load(open(config.resolve(a.calib)))['pixelsize_um'] * 1e3
    crop = tuple(float(v) for v in a.crop.split(','))
    u_max = SP.U_MAX if a.u_max is None else a.u_max
    sigma_max = SP.SIGMA_MAX_NM if a.sigma_max is None else a.sigma_max
    from simcode.pipeline.common import CAND_PMIN
    summ = dict(movie=a.movie, calib=a.calib, model=N['name'], cal_pass1=cal, cal_pass2=a.p2_cal, cand_pmin=CAND_PMIN,
                run_config='run_config.json (the complete, resolved settings of this run)')
    tags = [('est', 'locs_p1'), ('est2', 'locs_p2')]
    su = (a.sig_u.split(',') + [''])[:2] if a.sig_u else ['', '']
    wrote = False
    for (tag, name), su_ in zip(tags, su):
        for _ in Units(a.out, 'table', pattern=name + '.npz', claim=a.claim, redo=not a.claim, title=False).todo(['']):
            T, info = SP.build_table(a.out, tag, px, a.drift or None, a.strict_q, sig_u=config.resolve(su_) if su_ else None,
                                     crop=crop, u_max=u_max, sigma_max=sigma_max)
            files = SP.write_table(T, info, os.path.join(a.out, name), a.format.split(','))
            files += [SP.write_subset(T, os.path.join(a.out, name), r, u_max, sigma_max, px) for r in a.subset.split(',') if r]
            atomic_json(os.path.join(a.out, name + '_info.json'), info, indent=1, default=str)
            summ[name] = {k: v for k, v in info.items() if k != 'columns'}; summ[name]['files'] = files
            summ['columns'] = info['columns']; wrote = True
            print(f'{name}: {info["n_rows"]} rows, main (max sigma <= {sigma_max:g} nm) {info["n_main"]}, u <= {u_max:g} {info["n_u"]}, {info["n_sel"]} selected, strict {info["n_strict"]} (q {info["q"]:.3f}, '
                  f'{info["coverage_bins"]} / {info["coverage_bins_good"]} occupied 20-nm bins) -> {files}', flush=True)
    ST['total_s'] = time.time() - t00
    summ['timing'] = ST
    if wrote or not os.path.exists(os.path.join(a.out, 'summary.json')):
        atomic_json(os.path.join(a.out, 'summary.json'), summ, indent=1, default=str)
    return ST


if __name__ == '__main__':
    main(sys.argv[1:])
