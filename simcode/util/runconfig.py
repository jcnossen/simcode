"""
The run record: <out>/run_config.json (+ run_config.yaml, the same content for reading) = how the files of an output
directory were calculated. Written by simcode.run at the start (status 'running') and completed at the end ('done').

  format        'simcode.run_config/1'
  status        'running' | 'done' | 'failed' | 'reconstructed' (written afterwards for a run made before this record existed)
  settings      every argument of simcode.run after the defaults (movie, calibration, model, single files, crop, chunking,
                selection thresholds); file arguments as absolute paths
  fixed         what is not an option: pass 2, window 6 / step 3, stored-candidate cut, p_good, the tables, the kinetic model
                averaging, the brightness-noise estimate, code path, compile
                mode, batch sizes, margin, pattern search range, strict crop / density anchors
  overrides     the in-memory overrides of simcode.pipeline.run_with_overrides (AUTO_WRAP), if any
  model         name, its definition (models.json entry), and per file: absolute path, size, sha256, modification time; for
                network checkpoints the trainer arguments and step stored in the file; for calibrations the scalar parameters
  calib         in-situ PSF / camera calibration: path, sha256, camera-map file + sha256, the key PSF and camera parameters
  movie         path, parts (size, sha256), frame count, frame size, frames and chunks processed, frame order / pattern scheme
  code          package version, git commit, dirty flag, modified files, DME version / commit
  environment   python / numpy / torch / CUDA versions, GPU, host, platform (no user name, no e-mail)
  invocations   one entry per process that worked on the directory (resumed runs, one copy per GPU with --claim 1)
  pattern       summary of the estimated pattern (pitch, angle, depth, phases, drift range, the brightness noise c / c_a used by
                every solve of the run) + sha256 of pattern_est.pt
  tables        the four tables of the run (simcode/table.py TABLES): name -> what it is; 'default' = the result table
  selection     per table (locs_p1_net, locs_p1_solve, locs_p2_net, locs_p2_solve): the rule and the counts per rule
  counts, timing
  outputs       table files: size, arrays_sha256 (hash of the array contents: an .npz holds zip time stamps, so
                the file hash differs between identical runs), written_by ('run' or 'steps[i]')
  steps         later work on the directory, appended by its tool (simcode.table: table / selection rebuild with its settings)

Re-run with exactly the settings of a record: python -m simcode.run --config <run_config.json> --out <new dir>
(rerun_argv; verify() warns when a file hash, the git commit or a fixed value differs from the record).

  python -m simcode.util.runconfig show <dir>            one-page summary
  python -m simcode.util.runconfig verify <dir>          compare the recorded file hashes / commit with the current ones
"""
import contextlib, hashlib, json, os, platform, shlex, socket, subprocess, sys, time
import numpy as np
from simcode import config
from simcode.util.legacy_names import RUN_RECORD_FORMAT1_KEYS, RUN_RECORD_FORMAT1_FIXED

FORMAT = 'simcode.run_config/2'
# format 1 (records written before 2026-11-15; key names in util/legacy_names.py): load / update read it and convert the names (upgrade)
NAME = 'run_config.json'
REPO = str(config.REPO)
FRAME_ORDER = ('blocked: frame f = pattern step f % 6 = orientation (f % 6) // 3, phase step f % 3 (2 orientations x 3 steps); '
               'windows of 6 frames starting every 3 frames')
# arguments of simcode.run that define the results (the others: out, claim, redo, config)
RERUN_KEYS = ('movie', 'calib', 'model', 'crop', 'chunk', 'chunk_list', 'nframes', 'drift', 'min_free_gb')


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


# ---- hashes
_CACHE = os.path.join(os.path.expanduser('~'), '.cache', 'simcode', 'sha256.json')


def sha256(path, cache=True):
    """sha256 of a file; cached per (path, size, modification time) in ~/.cache/simcode (movies are several GB)"""
    path = os.path.abspath(path); st = os.stat(path); key = f'{path}|{st.st_size}|{st.st_mtime_ns}'
    C = {}
    if cache and st.st_size > 1 << 24:
        try: C = json.load(open(_CACHE))
        except (OSError, ValueError): C = {}
        if key in C: return C[key]
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    d = h.hexdigest()
    if cache and st.st_size > 1 << 24:
        try:
            os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
            try: C = json.load(open(_CACHE))
            except (OSError, ValueError): C = {}
            C[key] = d
            tmp = f'{_CACHE}.{os.getpid()}.tmp'; json.dump(C, open(tmp, 'w')); os.replace(tmp, _CACHE)
        except OSError:
            pass
    return d


def arrays_sha256(path):
    """hash of the contents of an .npz, or of the /localizations fields of a table .hdf5 (keys sorted; name, dtype, shape,
    bytes): equal for identical tables, unlike the file hash"""
    h = hashlib.sha256()
    if path.endswith('.hdf5'):
        import h5py
        with h5py.File(path, 'r') as f:
            for k in sorted(f['localizations']):
                a = np.ascontiguousarray(f['localizations'][k][...]); h.update(f'{k}|{a.dtype.str}|{a.shape}|'.encode()); h.update(a.data)
        return h.hexdigest()
    with np.load(path, allow_pickle=False) as d:
        for k in sorted(d.files):
            a = np.ascontiguousarray(d[k]); h.update(f'{k}|{a.dtype.str}|{a.shape}|'.encode()); h.update(a.data)
    return h.hexdigest()


def file_record(path, hash=True):
    """{path (absolute), size, sha256, mtime}; {path, missing: True} for a file that does not exist"""
    if not path: return None
    p = os.path.abspath(path)
    if not os.path.isfile(p): return dict(path=p, missing=True)
    st = os.stat(p)
    r = dict(path=p, size=st.st_size, mtime=time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(st.st_mtime)))
    if hash: r['sha256'] = sha256(p)
    return r


def output_record(path, written_by='run'):
    tab = path.endswith(('.npz', '.hdf5'))
    r = file_record(path, hash=not tab)
    if tab and not r.get('missing'): r['arrays_sha256'] = arrays_sha256(path)
    r['written_by'] = written_by
    return r


# ---- code and environment
def _git(d, *a):
    try:
        r = subprocess.run(['git', '-C', d, *a], capture_output=True, text=True, timeout=20)
        return r.stdout.strip('\n') if r.returncode == 0 else None
    except Exception:
        return None


def dme_record():
    """version / path / commit of the installed DME drift-estimation package that simcode.util.drift uses (not imported here)"""
    r = {}
    try:
        from importlib import metadata, util
        r = dict(version=metadata.version('dme'))
        sp = util.find_spec('dme')
        if sp and sp.origin:
            p = os.path.dirname(os.path.dirname(os.path.abspath(sp.origin)))
            c = _git(p, 'rev-parse', 'HEAD')
            r.update(path=p, git_commit=c, git_dirty=bool(_git(p, 'status', '--porcelain', '--untracked-files=no')) if c else None)
    except Exception as e:
        r = dict(error=f'{type(e).__name__}: {e}')
    r['note'] = 'DME v2 (github.com/jcnossen/drift-estimation); used for drift estimation of the tables, not by simcode.run itself'
    return r


def code_record(dme=True):
    import simcode
    st = _git(REPO, 'status', '--porcelain')
    mod = [l[3:] for l in (st or '').splitlines() if not l.startswith('??')]
    unt = [l[3:] for l in (st or '').splitlines() if l.startswith('??') and l[3:].startswith('simcode/')]
    r = dict(package='simcode', version=simcode.__version__, repo=REPO, git_commit=_git(REPO, 'rev-parse', 'HEAD'),
             git_commit_time=_git(REPO, 'log', '-1', '--format=%cI'), git_branch=_git(REPO, 'rev-parse', '--abbrev-ref', 'HEAD'),
             git_dirty=bool(mod or unt) if st is not None else None, modified_files=mod[:200], untracked_package_files=unt[:200],
             models_file=file_record(str(config.MODELS_FILE)))
    if dme: r['dme'] = dme_record()
    return r


def env_record(gpu=True):
    r = dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform(), host=socket.gethostname(),
             cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    try:
        import torch
        r.update(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
        if gpu and torch.cuda.is_available():
            r.update(gpu=torch.cuda.get_device_name(0), gpu_memory_gb=round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1))
    except Exception as e:
        r['torch_error'] = f'{type(e).__name__}: {e}'
    try:
        import triton; r['triton'] = triton.__version__
    except Exception:
        r['triton'] = None
    return r


def invocation(argv=None, prog='simcode.run'):
    return dict(command='python -m ' + prog + ' ' + shlex.join(sys.argv[1:] if argv is None else argv), cwd=os.getcwd(), host=socket.gethostname(),
                pid=os.getpid(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), started=now(), finished=None,
                git_commit=_git(REPO, 'rev-parse', 'HEAD'))


def log_command(out, prog, argv=None):
    """command.txt of an output directory: one plain line per command (the settings are in run_config.json)"""
    with open(os.path.join(out, 'command.txt'), 'a') as f:
        f.write(f'{now()} python -m {prog} ' + shlex.join(sys.argv[1:] if argv is None else argv) + '\n')


# ---- inputs
def checkpoint_info(path):
    """trainer arguments and step stored in a network checkpoint"""
    try:
        import torch
        c = torch.load(path, map_location='cpu', weights_only=False)
        if not isinstance(c, dict): return {}
        r = {k: c[k] for k in ('step', 'args') if k in c}
        if 'args' in r and not isinstance(r['args'], dict): r['args'] = vars(r['args'])
        return json.loads(json.dumps(r, default=str))
    except Exception as e:
        return dict(error=f'{type(e).__name__}: {e}')


def json_params(path, maxlen=8):
    """scalar / short entries of a calibration json (cmult = the fixed c, prior_off, kappa, sigma model, notes) or of a
    PSF calibration psf.zip (simcode.psf.read_calib)"""
    try:
        if path.endswith('.zip'):
            from simcode import psf as spsf
            J = spsf.read_calib(path)
        else:
            J = json.load(open(path))
    except Exception as e:
        return dict(error=f'{type(e).__name__}: {e}')
    return {k: v for k, v in J.items() if isinstance(v, (int, float, str, bool)) or (isinstance(v, list) and len(v) <= maxlen
                                                                                   and all(isinstance(x, (int, float, str)) for x in v))}


def model_file(path):
    r = file_record(path)
    if r.get('missing'): return r
    if path.endswith('.pt'): r['checkpoint'] = checkpoint_info(path)
    elif path.endswith(('.json', '.zip')): r['params'] = json_params(path)
    return r


def model_record(name, definition, files):
    """files: {role: absolute path} of the files actually used (detector, emitter, cal, detector_pass2, emitter_pass2, cal_pass2,
    )"""
    F = {k: model_file(v) for k, v in files.items() if v}
    d = {k: config.resolve(v) for k, v in (definition or {}).items() if isinstance(v, str) and k not in ('name', 'psf')}
    used = ('detector', 'emitter', 'detector_pass2', 'emitter_pass2')
    same = all(os.path.abspath(files[r]) == os.path.abspath(d[r]) for r in used if r in files and r in d)
    r = dict(name=name, definition=definition, files_are_the_models=bool(same), files=F)
    if not same: r['note'] = 'the network files given on the command line are not those of the named model: the files below are what was used'
    if definition and definition.get('psf'): r['training_psf'] = file_record(config.resolve(definition['psf']))
    return r


def calib_record(path):
    """in-situ PSF / camera calibration (calib json or psf.zip): file + camera maps + the key parameters"""
    p = config.resolve(path); r = file_record(p)
    if r.get('missing'): return r
    try:
        from simcode import psf as spsf
        c = spsf.read_calib(p)                     # calib json or psf.zip
        r['format'] = 'psf.zip' if spsf.is_psf_zip(p) else 'calib json'
        mp = spsf.camera_maps_path(p, c)
        r['camera_maps'] = (dict(path=os.path.abspath(p), inside='psf.zip camera arrays') if spsf.is_psf_zip(p) and mp
                            else file_record(mp))
        z = np.asarray(c.get('zernike_waves', []), float)
        r['psf'] = dict(model='vectorial (torch_psf.VectorialPSF, 11 px, oversample 3; LUT z range +-0.2 um)',
                        NA=c.get('NA'), wavelength_um=c.get('wavelength_um'), n_imm=c.get('n_imm'), n_med=c.get('n_med'),
                        pixelsize_um=c.get('pixelsize_um'), focus_um=c.get('focus_um'), zernike_noll=c.get('zernike_noll'),
                        zernike_waves=c.get('zernike_waves'), zernike_rms_waves=float(np.sqrt((z ** 2).sum())) if z.size else None,
                        apod=c.get('apod'), tirf_depth_um=c.get('tirf_depth_um'), emitter_z_median_um=c.get('emitter_z_median_um'),
                        extra_terms=sorted(k for k in c if k.startswith('psf_')), psf_blur_px2=c.get('psf_blur_px2'),
                        psf_gauss_sigma_um=c.get('psf_gauss_sigma_um'))
        cam = c.get('camera', {})
        r['camera'] = {k: v for k, v in cam.items() if isinstance(v, (int, float, str))}
        r['source'] = c.get('source')
        r['note'] = "only the camera and PSF entries are used by the pipeline; the calibration's 'pattern' entry is never read"
    except Exception as e:
        r['error'] = f'{type(e).__name__}: {e}'
    return r


def movie_record(path, nframes=0, chunk=6000, chunk_list=''):
    p = os.path.abspath(os.path.expanduser(path))
    try:
        from simcode.io.movie import open_movie
        m = open_movie(p); nf = min(m.n, nframes) if nframes else m.n
        chunks = [int(v) for v in chunk_list.split(',')] if chunk_list else list(range(0, nf, chunk))
        return dict(path=p, parts=[file_record(q) for q in m.parts], n_frames=int(m.n), frame_shape=[int(v) for v in m.shape[-2:]],
                    frames_processed=int(nf), chunk=int(chunk), chunks=chunks, frame_order=FRAME_ORDER)
    except Exception as e:
        return dict(path=p, error=f'{type(e).__name__}: {e}')


# ---- results
def pattern_summary(out):
    """summary of pattern_est.json / pattern_est.pt of a run directory"""
    r = {}
    try:
        J = json.load(open(os.path.join(out, 'pattern_est.json')))
        r['angles'] = {a: dict(pitch_px=J['k'][a].get('pitch_px'), angle_deg=J['k'][a].get('angle_deg'), n=J['k'][a].get('n'),
                               depth=J.get('depth', {}).get(a, {}).get('m'), depth_se=J.get('depth', {}).get(a, {}).get('se')) for a in J.get('k', {})}
        r.update({k: J[k] for k in ('n', 'n_on', 'n_clean') if k in J})
        if 'brightness_noise' in J:
            B = J['brightness_noise']
            r['brightness_noise'] = dict(c=B.get('c'), c_a=B.get('c_a'), n_rows=B.get('n_rows'), rule=B.get('rule'))
    except Exception as e:
        r['error'] = f'pattern_est.json: {type(e).__name__}: {e}'
    pt = os.path.join(out, 'pattern_est.pt')
    if os.path.exists(pt):
        r['file'] = file_record(pt)
        try:
            import torch
            P = torch.load(pt, map_location='cpu', weights_only=False)
            g = lambda k: np.asarray(P[k], float)
            r.update(k_rad_per_px=g('k').tolist(), phase_rad=g('phi').tolist(), depth_per_step=g('m').tolist(),
                     rel_intensity=np.exp(g('logrel') - g('logrel').mean()).tolist(), knot_frames=float(P['knot_frames']),
                     phase_drift_rad=dict(knots=int(g('dphi').shape[1]), min=g('dphi').min(1).tolist(), max=g('dphi').max(1).tolist()))
        except Exception as e:
            r['pt_error'] = f'{type(e).__name__}: {e}'
    return r


def selection_summary(info):
    """locs_p*_info.json -> the selection outcome: rule + counts per rule"""
    g = info.get
    return dict(base='rows with score p >= 0.63 inside the crop', crop=g('crop'), px_nm=g('px_nm'),
                main=dict(rule='p >= 0.63, inside the crop, max(sx, sy) * px_nm <= 12 nm (applied at load: Dataset.select)',
                          sigma_max_nm=g('sigma_max', 12.0), n=g('n_main')),
                sel=dict(rule='phase-calibrated detection selection', n=g('n_sel')),
                counts=dict(n_rows=g('n_rows'), n_good=g('n_good'), n_good_crop=g('n_good_crop')), median_sigma_nm=g('median_sigma_nm'),
                drift=g('drift') or '', tag=g('tag'), kind=g('kind'))


def results(out, written_by='run'):
    """pattern, selection, counts and outputs of a finished run directory (from the files on disk)"""
    from simcode import table as SP
    R = dict(pattern=pattern_summary(out), tables=dict(SP.TABLE_DOC, default=SP.DEFAULT_TABLE), selection={}, counts={}, outputs={})
    import glob
    for name in SP.TABLES:
        f = os.path.join(out, name + '_info.json')
        if os.path.exists(f): R['selection'][name] = selection_summary(json.load(open(f)))
        for q in sorted(glob.glob(os.path.join(out, name + '.hdf5'))):
            R['outputs'][os.path.basename(q)] = output_record(q, written_by)
    try:
        RT = json.load(open(os.path.join(out, 'runtime.json')))
        p1 = RT.get('pass1', {})
        R['counts'] = dict(chunks_pass1=len(p1), frames=sum(v.get('frames', 0) for v in p1.values()), pass1_detections=sum(v.get('n_det', 0) for v in p1.values()),
                           pass1_localizations=sum(v.get('n_out', 0) for v in p1.values()), chunks_solve=len(RT.get('pass2_est', {})),
                           chunks_pass2=len(RT.get('pass2_est2', {})))
        R['chunk_seconds'] = {k: float(sum(v.get('total', 0) for v in RT.get(k, {}).values())) for k in ('pass1', 'pass2_est', 'pass2_est2')}
    except Exception:
        pass
    for name in SP.TABLES:
        if name in R['selection']: R['counts'][name + '_rows'] = R['selection'][name]['counts']['n_rows']
    return R


# ---- the file
@contextlib.contextmanager
def locked(out):
    import fcntl
    with open(os.path.join(out, NAME + '.lock'), 'w') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(lf, fcntl.LOCK_UN)


def upgrade(R):
    """a record of format 1 -> the present format (file keys renamed, legacy_names.RUN_RECORD_FORMAT1_*); other records unchanged"""
    if not R or R.get('format') != 'simcode.run_config/1': return R
    ren = lambda d: {RUN_RECORD_FORMAT1_KEYS.get(k, k): v for k, v in d.items()} if isinstance(d, dict) else d
    R = dict(R, format=FORMAT, converted_from='simcode.run_config/1')
    if 'settings' in R: R['settings'] = ren(R['settings'])
    if isinstance(R.get('fixed'), dict):
        R['fixed'] = {RUN_RECORD_FORMAT1_FIXED.get(k, k): v for k, v in R['fixed'].items()}
        if isinstance(R['fixed'].get('solve_c'), str): R['fixed']['solve_c'] = R['fixed']['solve_c'].replace('/ p2_cal,', '/ cal_pass2,')
    if isinstance(R.get('model'), dict):
        R['model'] = dict(R['model'], definition=ren(R['model'].get('definition')), files=ren(R['model'].get('files') or {}))
    if R.get('earlier_settings'):
        R['earlier_settings'] = [dict(e, settings=ren(e.get('settings')),
                                      model=dict(e['model'], definition=ren(e['model'].get('definition')), files=ren(e['model'].get('files') or {}))
                                      if isinstance(e.get('model'), dict) else e.get('model')) for e in R['earlier_settings']]
    return R


def load(path):
    """a run_config.json, or the one of a run directory (a format-1 record converted, upgrade)"""
    if os.path.isdir(path): path = os.path.join(path, NAME)
    return upgrade(json.load(open(path)))


def _yaml_text(R):
    try:
        import yaml
    except ImportError:
        return None
    return ('# SIMCODE run record, human-readable copy of run_config.json (the json is the one that is read back).\n'
            '# Re-run with these settings: python -m simcode.run --config run_config.json --out <new dir>\n'
            + yaml.safe_dump(json.loads(json.dumps(R, default=str)), sort_keys=False, default_flow_style=None, width=140))


def write(out, R):
    from simcode.io.work import atomic_json, atomic_text
    atomic_json(os.path.join(out, NAME), R, indent=1, default=str)
    y = _yaml_text(R)
    if y: atomic_text(os.path.join(out, 'run_config.yaml'), y)


def update(out, fn):
    """locked read-modify-write of <out>/run_config.json; fn(record or None) -> record"""
    with locked(out):
        p = os.path.join(out, NAME)
        try: R = upgrade(json.load(open(p)))
        except (OSError, ValueError): R = None
        R = fn(R); write(out, R)
    return R


def _inputs_key(R):
    """what defines the results of a record: result settings, fixed values, overrides, file hashes"""
    S = {k: R.get('settings', {}).get(k) for k in RERUN_KEYS if k != 'min_free_gb'}
    H = {k: v.get('sha256') for k, v in R.get('model', {}).get('files', {}).items()}
    H['calib'] = R.get('calib', {}).get('sha256'); H['maps'] = (R.get('calib', {}).get('camera_maps') or {}).get('sha256')
    return json.dumps([S, R.get('fixed'), R.get('overrides'), H], sort_keys=True, default=str)


def start(out, settings, fixed, model, calib, movie, overrides=None, rerun_of=None, argv=None):
    """write the record at the start of simcode.run (status 'running'). A directory that already has a record (resumed run,
    another copy with --claim 1) keeps its invocations; if the settings or input files differ from the recorded ones the earlier
    record is kept under 'earlier_settings' with a warning (chunk files of the directory may come from it)."""
    inv = invocation(argv)
    new = dict(format=FORMAT, status='running', started=inv['started'], finished=None,
               reproduce='python -m simcode.run --config <this file> --out <new dir>', settings=settings, fixed=fixed,
               overrides=overrides or None, model=model, calib=calib, movie=movie, code=code_record(), environment=env_record(),
               invocations=[inv], steps=[])
    if rerun_of: new['rerun_of'] = rerun_of

    def fn(old):
        if old and old.get('format') == FORMAT:
            new['invocations'] = old.get('invocations', []) + [inv]
            new['steps'] = old.get('steps', [])
            new['started'] = old.get('started') or new['started']
            if 'earlier_settings' in old: new['earlier_settings'] = old['earlier_settings']
            if old.get('settings') and _inputs_key(old) != _inputs_key(new):
                new.setdefault('earlier_settings', []).append(dict(
                    replaced=inv['started'], warning='the settings or input files of this directory changed between invocations: chunk files '
                    'that existed before this invocation (and were not redone) come from the settings below',
                    settings=old.get('settings'), fixed=old.get('fixed'), overrides=old.get('overrides'), model=old.get('model'),
                    calib=old.get('calib'), code=old.get('code')))
                print(f'run_config: WARNING the settings / input files differ from the earlier record of {out}; existing chunk files are '
                      'reused unless --redo 1 (see earlier_settings in run_config.json)', file=sys.stderr, flush=True)
        return new
    return update(out, fn)


def finish(out, timing=None, error=None, extra=None):
    """complete the record: this invocation's end time; with the tables on disk: pattern, selection, counts, outputs, 'done'"""
    res = None
    if error is None and os.path.exists(os.path.join(out, 'locs_p2_net_info.json')): res = results(out)

    def fn(R):
        R = R or dict(format=FORMAT, invocations=[], steps=[])
        for inv in R.get('invocations', []):
            if inv.get('pid') == os.getpid() and inv.get('host') == socket.gethostname() and not inv.get('finished'):
                inv['finished'] = now()
                if timing: inv['timing'] = timing
                if error: inv['error'] = error
                if extra: inv.update(extra)
        if error:
            R['status'] = 'failed'; R['error'] = error
        elif res is not None:
            R.update(res); R['status'] = 'done'; R['finished'] = now(); R.pop('error', None)
            if timing: R['timing'] = timing
        return R
    return update(out, fn)


def append_step(out, step, outputs=None, selection=None):
    """append a later step (table rebuild, ...) to the 'steps' list of the run directory's record; its outputs and selection
    replace the entries of the same files / tables (written_by 'steps[i]'). A directory without a record gets a minimal one."""
    def fn(R):
        if not R or R.get('format') != FORMAT:
            R = dict(format=FORMAT, status='no run record', note='this directory was made before run_config.json existed, or by another tool; '
                     'only later steps are recorded', steps=[])
        R.setdefault('steps', []).append(step); i = len(R['steps']) - 1
        for f in outputs or []:
            if os.path.dirname(os.path.abspath(f)) == os.path.abspath(out):
                R.setdefault('outputs', {})[os.path.basename(f)] = output_record(f, f'steps[{i}]')
        for name, s in (selection or {}).items():
            R.setdefault('selection', {})[name] = dict(s, written_by=f'steps[{i}]')
        return R
    return update(out, fn)


# ---- re-run
def rerun_argv(R):
    """the simcode.run arguments that reproduce a record (without --out, --claim, --redo)"""
    S = dict(R['settings']); a = []
    # settings of earlier records that are no longer options (2026-11-15: format, subset, sigma_max, prefetch, sig_u = the
    # model's; 2026-11-17: cal_kind, see simcode.run._from_config) are not passed; an earlier record's crop is (its default
    # 5,5,110 was written into the record)
    for k in RERUN_KEYS:
        v = S.get(k)
        if v is None or v == '' or (k in ('nframes',) and not v): continue
        a += ['--' + k, str(v)]
    return a


def verify(R, check_movie=True, fixed=None, overrides=None):
    """compare a record with the present state -> list of warnings (file hashes, git commit, dirty tree, versions, fixed values)"""
    W = []
    def chk(what, rec):
        if not rec or not rec.get('path'): return
        if not os.path.isfile(rec['path']): W.append(f'{what}: {rec["path"]} is missing'); return
        if rec.get('sha256') and sha256(rec['path']) != rec['sha256']:
            W.append(f'{what}: {rec["path"]} differs from the record (sha256 {sha256(rec["path"])[:12]} now, {rec["sha256"][:12]} recorded)')
    for k, v in R.get('model', {}).get('files', {}).items(): chk(f'model file {k}', v)
    chk('calibration', R.get('calib')); chk('camera maps', (R.get('calib') or {}).get('camera_maps'))
    if check_movie:
        for v in (R.get('movie') or {}).get('parts', []): chk('movie', v)
    c0 = R.get('code', {}); c1 = code_record(dme=False)
    if c0.get('git_commit') and c0['git_commit'] != c1['git_commit']:
        W.append(f'git commit {str(c1["git_commit"])[:10]} now, {c0["git_commit"][:10]} recorded')
    if c0.get('git_dirty'): W.append(f'the record was made from a modified working tree ({len(c0.get("modified_files", []))} modified files): the commit alone does not define its code')
    if c1.get('git_dirty'): W.append(f'the working tree is modified now ({", ".join((c1["modified_files"] + c1["untracked_package_files"])[:8])} ...)')
    e0 = R.get('environment', {}); e1 = env_record(gpu=False)
    for k in ('torch', 'cuda', 'numpy'):
        if e0.get(k) and e0[k] != e1.get(k): W.append(f'{k} {e1.get(k)} now, {e0[k]} recorded')
    if fixed is not None and R.get('fixed'):
        for k, v in R['fixed'].items():
            if k in fixed and json.dumps(fixed[k], default=str) != json.dumps(v, default=str):
                W.append(f'fixed value {k}: {fixed[k]!r} now, {v!r} recorded')
    return W


def show(R):
    S = R.get('settings', {}); M = R.get('model', {}); c = R.get('code', {}); e = R.get('environment', {})
    L = [f'status {R.get("status")}   started {R.get("started")}   finished {R.get("finished")}',
         f'movie   {(R.get("movie") or {}).get("path")}  ({(R.get("movie") or {}).get("frames_processed")} of {(R.get("movie") or {}).get("n_frames")} frames)',
         f'calib   {(R.get("calib") or {}).get("path")}  sha256 {str((R.get("calib") or {}).get("sha256"))[:12]}',
         f'model   {M.get("name")}' + ('' if M.get('files_are_the_models', True) else '  (files overridden)')]
    L += [f'  {k:9s} {v.get("path")}  sha256 {str(v.get("sha256"))[:12]}' for k, v in M.get('files', {}).items()]
    L += [f'settings crop {S.get("crop")} chunk {S.get("chunk")}',
          f'code    {c.get("package")} {c.get("version")} git {str(c.get("git_commit") or c.get("git_commit_inferred"))[:10]} dirty {c.get("git_dirty")}; '
          f'torch {e.get("torch")} cuda {e.get("cuda")} gpu {e.get("gpu")} host {e.get("host")}']
    for n, s in (R.get('selection') or {}).items():
        L.append(f'{n}: rows {s["counts"]["n_rows"]}, selection (sigma <= {s["main"]["sigma_max_nm"]} nm) {s["main"]["n"]}, sel {s["sel"]["n"]}')
    L += [f'step {i}: {s.get("step")} {s.get("time")} {s.get("command", "")}' for i, s in enumerate(R.get('steps', []))]
    return '\n'.join(L)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='run record (run_config.json) of simcode output directories')
    ap.add_argument('action', choices=['show', 'verify'])
    ap.add_argument('dirs', nargs='+')
    a = ap.parse_args(argv)
    for d in a.dirs:
        print(f'== {d}')
        if a.action == 'show':
            print(show(load(d)))
        else:
            W = verify(load(d))
            print('\n'.join('  DIFFERS: ' + w for w in W) if W else '  input files, commit and versions match the record')


if __name__ == '__main__':
    main()
