"""
SIMCODE v2 on a DMD-SIMFLUX DNA-PAINT movie, production settings.

  python -m simcode.run --movie <movie.tif> --calib <in-situ calib.json> --model m1809 --out <dir>
  python -m simcode.run --dataset 1809_2 --data-root /mnt/data/dmdsf/bn_dnapaint --out <dir>
                                                    (a paper dataset: movie under --data-root, calibration, model)

Stages (each skipped if its outputs exist, unless --redo):
  1  pass 1      detector + head on every 6-frame window (start every 3 frames): network per-frame intensities
                 with covariance, bg, pattern-free position prior; candidates with detector p >= 0.3 are stored
                 (simcode/processing/common.CAND_PMIN)
  2  pattern     illumination pattern (k, per-step phase / depth / relative intensity, per-angle phase drift; with
                 --intensity_map N the per-orientation intensity map over the field of view) estimated from the pass-1 network
                 intensities, and the dataset's brightness noise c, c_a (the solve's relative intensity scatter per frame / per
                 orientation) from the same clean emitter-windows
  3  solve       explicit intensity-domain pattern solve of the pass-1 rows (posterior over the fringe starts and the
                 non-kinetic / kinetic models, posterior sd) with the model's solve calibration (cal_sim: prior_off from
                 simulated gate scenes) and the estimated c, c_a
  4  pass 2      pattern-aware pass-2 detector + emitter network fed the estimated pattern, then the same solve
  5  tables      the four localization tables (simcode/table.py; the torch_psf Dataset format): locs_p1_net, locs_p1_solve,
                 locs_p2_net (THE RESULT: the pass-2 emitter network's position and sd), locs_p2_solve; summary.json. The
                 paper's selection (max(sx, sy) <= 12 nm on the rows with p >= 0.63 inside the crop) is applied when a table
                 is loaded: simcode.dataset.Dataset.select.
Output region: --crop x0,y0,size (px); default the whole frame (square sensors; else the largest top-left square). The
paper datasets (--dataset) carry their crop in models.json '_production'.
Networks and calibrations come from the model (--model, required): a name of simcode/models.json (m1809, m25, g13;
file names relative to the repository's models folder, simcode/config.py), or the path of a model folder with a model.json (as written
by the automatic pipeline's training stage / simcode.pipeline.training.export_model). There are no single-file overrides: a model
with other files is a model folder. --dataset <name> takes movie, calibration and model of a paper dataset from the
'_production' entry of models.json (what the paper tables are made with); its movie path is relative to the folder given
with --data-root (the root of the DMD-SIMFLUX DNA-PAINT data).
Run record: <out>/run_config.json (+ run_config.yaml), written at the start (status 'running') and completed at the end
('done'): the fully resolved settings, sha256 of the movie / calibration / every model file, code commit, environment,
pattern and selection outcome (simcode/util/runconfig.py; README "Run record").
  python -m simcode.run --config <run_config.json> --out <new dir>     re-run with exactly the settings of a record
Code path: queued batching, the exact fast solve, torch.compile of the solve's LM step when available.
Set CUDA_VISIBLE_DEVICES to choose the GPU; several GPUs: the same command once per GPU with --claim 1
(python -m simcode.io.launch --gpus 1,2 -- python -m simcode.run ... --claim 1): the chunks of pass 1 and pass 2 are
shared between the copies, the stages in between run in one of them.
A 6000-frame 0.5 nM chunk takes ~0.5 min (pass 1 + solve) on an idle RTX 3090.
"""
import argparse, json, os, sys, time
from simcode import config

PROD = config.production()


def _uncal(args):
    from simcode.processing import run_uncalibrated
    run_uncalibrated.main(args)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1].strip(), formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='', help='run_config.json of an earlier run (or its directory): re-run with exactly its settings '
                    '(movie, calibration, model files, crop, chunking, selection); arguments given after it override single settings; '
                    'a warning is printed for every input file, git commit or fixed value that differs from the record')
    ap.add_argument('--dataset', default='', help="a paper dataset (models.json '_production': " + ', '.join(PROD['datasets']) + '): '
                    'sets --movie (under --data-root), --calib, --model')
    ap.add_argument('--data-root', default='', help="with --dataset: the folder the dataset's movie path is relative to "
                    '(e.g. /mnt/data/dmdsf/bn_dnapaint)')
    ap.add_argument('--movie', default='', help='movie .tif (further parts movie_X1.tif ... are found automatically)')
    ap.add_argument('--calib', default='', help='in-situ PSF / camera calibration json (e.g. insitu/calib_1809_2_psf_v2.json)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='', help='the model (required, or from --dataset / --config): a name of simcode/models.json '
                    '(m1809, m25, g13) or the path of a model folder with a model.json')
    ap.add_argument('--claim', type=int, default=0, help='1: several copies of this command share the chunks (one copy per GPU)')
    ap.add_argument('--min_free_gb', type=float, default=2.0, help='a chunk is not started with less free space on the output file system')
    ap.add_argument('--crop', default='', help='output region x0,y0,size (px); default: the whole frame (a --dataset: its crop)')
    ap.add_argument('--chunk', type=int, default=6000)
    ap.add_argument('--chunk_list', default='', help="process only these chunk starts, e.g. '12000'")
    ap.add_argument('--nframes', type=int, default=0)
    ap.add_argument('--drift', default='', help='optional drift [F,2] px (e.g. DME output): the tables get x_dc, y_dc and the '
                    'selection region uses the corrected positions')
    ap.add_argument('--intensity_map', type=int, default=0, help='pattern stage: 0 (default) = one relative intensity per pattern step; '
                    'N > 0 = also the per-orientation intensity map over the field of view (Legendre degree N, 4 recommended; '
                    'processing/pattern_estimation.estimate_relmap), used by every solve of the run')
    ap.add_argument('--redo', type=int, default=0)
    return ap


def resolve(ap, argv):
    """arguments -> (a, model dict, pass-1 solve calibration): the dataset, the model's files and the defaults filled in"""
    a = ap.parse_args(argv)
    if a.dataset:
        D = PROD['datasets'].get(a.dataset)
        if D is None: ap.error(f'unknown dataset {a.dataset!r} ({", ".join(PROD["datasets"])})')
        if not (a.model or D['model']): ap.error(f'dataset {a.dataset}: no model is defined for it; give --model')
        if not a.movie and not a.data_root: ap.error(f'--dataset {a.dataset}: give --data-root (the folder its movie {D["movie"]} is in) or --movie')
        a.movie = a.movie or os.path.join(a.data_root, D['movie']); a.calib = a.calib or D['calib']; a.model = a.model or D['model']; a.crop = a.crop or D.get('crop', '')
    if not a.movie or not a.calib: ap.error('--movie and --calib are required (or --dataset)')
    if not a.model: ap.error('--model is required (a name of simcode/models.json or a model folder), or --dataset')
    try:
        N = config.model_set(a.model)
    except (KeyError, ValueError, OSError) as e:
        ap.error(str(e))
    a.detector, a.emitter, a.detector_pass2, a.emitter_pass2 = N['detector'], N['emitter'], N['detector_pass2'], N['emitter_pass2']
    a.cal_pass2 = N['cal_sim_pass2']
    cal = N['cal_sim']
    return a, N, cal


def settings_of(a, cal):
    """the fully resolved settings of a run (run_config.json 'settings'): every argument after the defaults, files as absolute paths"""
    r = config.resolve
    return dict(dataset=a.dataset, movie=os.path.abspath(os.path.expanduser(a.movie)), calib=r(a.calib), out=os.path.abspath(a.out), model=getattr(a, 'model_name', a.model),
                detector=r(a.detector), emitter=r(a.emitter), cal=r(cal), detector_pass2=r(a.detector_pass2), emitter_pass2=r(a.emitter_pass2),
                cal_pass2=r(a.cal_pass2),
                crop=a.crop, chunk=a.chunk, chunk_list=a.chunk_list, nframes=a.nframes,
                drift=os.path.abspath(a.drift) if a.drift else '', claim=a.claim, min_free_gb=a.min_free_gb, redo=a.redo,
                intensity_map=a.intensity_map)


def fixed_values():
    """what a run does that is not an option (run_config.json 'fixed'), read from the code so that an in-memory override shows"""
    from simcode.processing import run_uncalibrated as RU, pattern_estimation as PE
    from simcode.processing.common import CAND_PMIN
    from simcode import table as SP
    from simcode.solve import solve as SV
    d = vars(RU.build_parser().parse_args(['--movie', 'x', '--calib', 'x', '--out', 'x']))
    return dict(stages='pass 1, pattern estimation (+ brightness noise), pattern solve of the pass-1 rows, pass 2 (pattern-aware) + solve, tables', pass2=True,
                window_frames=6, window_step=3, cand_pmin=CAND_PMIN, p_good=SP.P_GOOD,
                solve_c='estimated per run with the pattern (pattern_estimation.estimate_brightness_noise; pattern_est.pt cmult / cmult_ang; run_config pattern.brightness_noise)',
                kinetic_model=f'posterior average of the non-kinetic and kinetic fits for every row, BIC penalty {SV.KIN_PENALTY:.3f} = 2 ln 6 (solve.model_average)',
                brightness_noise_rows=PE.BN_NMAX,
                code_path='fast (queued batching, exact fast solve)', compile=RU.COMPILE, compile_emitter='off',
                margin=d['margin'], nn_min=d['nn_min'], min_phot=d['min_phot'], det_batch=d['det_batch'], emitter_batch=d['emitter_batch'],
                solve_batch=d['solve_batch'], pattern_pitch_search_px=list(PE.estimate_k.__defaults__[0]),
                crop_default='the whole frame (square sensors)',
                tables={k: f'{v} (tag {SP.TABLES[k][0]}, {SP.TABLES[k][1]})' for k, v in SP.TABLE_DOC.items()}, default_table=SP.DEFAULT_TABLE,
                table_format='torch_psf Dataset HDF5',
                selection='applied at load (Dataset.select: max(sx, sy) <= 12 nm, p >= 0.63, inside the crop)', prefetch=True)


def _from_config(path, rest):
    """--config: (argv that reproduces the record + the arguments given after it, the record, its in-memory overrides)"""
    from simcode.util import runconfig as RC
    REC = RC.load(path)
    if not REC.get('settings'): raise SystemExit(f'--config {path}: no settings in this record (status {REC.get("status")!r})')
    M = REC.get('model') or {}
    if M.get('name') and M.get('definition') and M['name'] not in config.model_sets():        # e.g. the model of an automatic-pipeline dataset
        config.model_sets()[M['name']] = {k: v for k, v in M['definition'].items() if k not in ('name', 'folder') + config.REMOVED_KEYS}
    if (REC.get('settings') or {}).get('cal_kind') == 'real':
        print(f'simcode.run --config: WARNING {path} was made with the real-data sigma calibrations (cal_kind real, removed '
              '2026-11-17); this run uses the simulation-fitted ones (cal_sim), so its sigma columns (and the sigma selection) differ',
              file=sys.stderr, flush=True)
    W = REC.get('overrides')
    if W and not os.environ.get('AUTO_WRAP'):
        from simcode.pipeline import run_with_overrides
        run_with_overrides.apply({k: v for k, v in W.items() if k != 'title'})
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
    from simcode.util.proctitle import set_title; set_title('run')     # recognizable name in nvidia-smi / ps
    from simcode.io.work import Units, atomic_json
    os.makedirs(a.out, exist_ok=True)
    # ---- the run record (run_config.json): written now with status 'running', completed at the end
    S = settings_of(a, cal); FX = fixed_values()
    files = dict(detector=S['detector'], emitter=S['emitter'], cal=S['cal'], detector_pass2=S['detector_pass2'], emitter_pass2=S['emitter_pass2'],
                 cal_pass2=S['cal_pass2'])
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
    from simcode.util import compile as torch_compile
    RC.finish(a.out, timing=ST, extra=dict(torch_compile_used=torch_compile._STATE.get('ok')))
    print('done', json.dumps(ST), flush=True)


def _stages(a, N, cal, Units, atomic_json):
    t00 = time.time(); ST = {}
    common = ['--movie', a.movie, '--calib', a.calib, '--out', a.out, '--chunk', str(a.chunk), '--redo', str(a.redo),
              '--claim', str(a.claim), '--min_free_gb', str(a.min_free_gb), '--detector', a.detector, '--emitter', a.emitter,
              '--cal', cal]
    if a.crop: common += ['--crop', a.crop]
    if a.chunk_list: common += ['--chunk_list', a.chunk_list]
    if a.nframes: common += ['--nframes', str(a.nframes)]
    # 1-2: pass 1 + pattern (the pattern stage only uses the calibration's prior_off, identical in all pass-1 calibrations)
    t0 = time.time()
    need_pat = a.redo or not os.path.exists(f'{a.out}/pattern_est.pt')
    _uncal(common + ['--stages', 'pass1' + (',pattern' if need_pat else '')] + (['--intensity_map', str(a.intensity_map)] if a.intensity_map else []))
    ST['pass1_pattern_s'] = time.time() - t0
    # 3: pattern solve of the pass-1 rows
    t0 = time.time()
    _uncal(common + ['--stages', 'pass1_solve'])
    ST['solve_s'] = time.time() - t0
    # 4: pass 2
    t0 = time.time()
    _uncal(common + ['--detector_pass2', a.detector_pass2, '--emitter_pass2', a.emitter_pass2, '--cal_pass2', a.cal_pass2, '--stages', 'pass2'])
    ST['pass2_s'] = time.time() - t0
    # 5: tables (with --claim 1 the tables are written by one of the copies; the others skip them)
    from simcode import table as SP
    from simcode import psf as spsf
    px = spsf.read_calib(config.resolve(a.calib))['pixelsize_um'] * 1e3
    crop = tuple(float(v) for v in a.crop.split(',')) if a.crop else SP.run_crop(a.out)
    from simcode.processing.common import CAND_PMIN
    summ = dict(movie=a.movie, calib=a.calib, model=N['name'], cal_pass1=cal, cal_pass2=a.cal_pass2, cand_pmin=CAND_PMIN,
                run_config='run_config.json (the complete, resolved settings of this run)')
    wrote = False
    for _ in Units(a.out, 'table', pattern=SP.DEFAULT_TABLE + '.hdf5', claim=a.claim, redo=not a.claim, title=False).todo(['']):
        for name, (info, files) in SP.write_tables(a.out, px, a.drift or None, crop).items():
            summ[name] = {k: v for k, v in info.items() if k != 'columns'}; summ[name]['files'] = files
            summ.setdefault('columns', {})[name] = info['columns']; wrote = True
            print(f'{name}: {info["n_rows"]} rows (the paper\'s selection: {info["n_main"]}, median sigma {info["median_sigma_nm"]:.2f} nm), '
                  f'sel {info["n_sel"]} -> {files}', flush=True)
        summ['default_table'] = SP.DEFAULT_TABLE
    ST['total_s'] = time.time() - t00
    summ['timing'] = ST
    if wrote or not os.path.exists(os.path.join(a.out, 'summary.json')):
        atomic_json(os.path.join(a.out, 'summary.json'), summ, indent=1, default=str)
    return ST


if __name__ == '__main__':
    main(sys.argv[1:])
