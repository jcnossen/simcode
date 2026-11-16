"""
Paths of a SIMCODE v2 installation and the models (a "model" = one trained set of networks + their solve calibrations). Nothing in the package hard-codes an absolute path; everything goes
through here.

Roots, each set by (in order of precedence) an environment variable, a key in a json config file, or a default:
  models   trained networks (*.pt) and calibrations (calib/, insitu/)   SIMCODE_MODELS     default <repo>/models
  data     raw movies (the DNA-PAINT TIFF stacks)                         SIMCODE_DATA       default: none (pass full paths)
  out      outputs of the benchmark / analysis scripts                    SIMCODE_OUT        default ./simcode_out
  fourpi   work directory of the 4Pi pipelines (models, PSF, registration, SIMCODE_4PI        default <out>/4pi
           tables)
  data4pi  raw 4Pi movies (dcimg)                                         SIMCODE_4PI_DATA   default: none (the 4Pi acquisition
           configs, simcode/pipeline/configs/fourpi.py, then use their own data path)
One optional path, set the same way: dme = a checkout of the DME drift-estimation package (DME_PATH; default: the installed
package; see simcode/util/drift.py).
Config file: $SIMCODE_CONFIG, else <repo>/simcode.json, else ~/.config/simcode.json; keys "models", "data", "out", "fourpi",
"data4pi", "dme" (see simcode.example.json). Relative paths in the config file are relative to the file's directory.

resolve(p, kind) turns a user path into an absolute one: absolute paths and paths that exist relative to the current directory
are kept; otherwise the path is looked up under the given root (models: also under models/calib and models/insitu).
model_set(name) returns one model: a name of simcode/models.json (the definition of the shipped models), or the path of a
model folder (model.json + the network and calibration files, as written by the automatic pipeline's training stage /
simcode.pipeline.training.export_model), or of such a model.json.
"""
import json, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_ENV = {'models': 'MODELS', 'data': 'DATA', 'out': 'OUT', 'fourpi': '4PI', 'data4pi': '4PI_DATA'}
_cfg = None


def env(name, default=None):
    """environment variable SIMCODE_<name>, else default"""
    return os.environ.get('SIMCODE_' + name) or default


def _config():
    global _cfg
    if _cfg is None:
        _cfg = {}
        for p in (env('CONFIG'), REPO / 'simcode.json', Path.home() / '.config' / 'simcode.json'):
            if p and Path(p).is_file():
                c = json.load(open(p)); base = Path(p).resolve().parent
                ex = lambda v: os.path.expanduser(v) if isinstance(v, str) else v
                _cfg = {k: (str((base / ex(v)).resolve()) if isinstance(v, str) and v and not k.startswith('_') and not os.path.isabs(ex(v))
                            else ex(v)) for k, v in c.items()}
                _cfg['_file'] = str(p)
                break
    return _cfg


def root(kind):
    """'models' | 'data' | 'out' | 'fourpi' | 'data4pi' | 'dme' -> directory (str; '' if unset)"""
    if kind == 'dme': return os.environ.get('DME_PATH') or _config().get('dme') or ''
    v = env(_ENV[kind]) or _config().get(kind)
    if v: return v
    if kind == 'models': return str(REPO / 'models')
    if kind == 'out': return 'simcode_out'
    if kind == 'fourpi': return os.path.join(root('out'), '4pi')
    return ''


def models_dir():
    return root('models')


def out_dir(*sub, make=True):
    p = os.path.join(root('out'), *sub)
    if make: os.makedirs(p, exist_ok=True)
    return p


def resolve(p, kind='models'):
    """user path -> absolute path (see the module doc); returns the input unchanged if nothing is found"""
    if not p: return p
    p = os.path.expanduser(str(p))
    if os.path.isabs(p) or os.path.exists(p): return os.path.abspath(p)
    r = root(kind)
    cands = [os.path.join(r, p)]
    if kind == 'models': cands += [os.path.join(r, 'calib', p), os.path.join(r, 'insitu', p)]
    for c in cands:
        if os.path.exists(c): return c
    return cands[0] if r else p


def model(name):
    """network / calibration file shipped in the models directory, e.g. model('pretrain_detector_pass1.pt'), model('calib/x.json')"""
    return resolve(name, 'models')


# ---- models: the shipped ones are defined in simcode/models.json (next to this file); any other model is a folder
MODELS_FILE = Path(__file__).with_name('models.json')
MODEL_JSON = 'model.json'
_nets = None


def model_sets():
    """the whole models.json (models, '_default', '_production')"""
    global _nets
    if _nets is None: _nets = json.load(open(MODELS_FILE))
    return _nets


# file keys of a model definition before 2026-11-15 (det / head / p2_*): refused with the name of the migration script
OLD_KEYS = ('det', 'head', 'p2_det', 'p2_head', 'p2_cal_sim', 'p2_cal_real')


def _check_keys(name, M, where):
    old = [k for k in M if k in OLD_KEYS]
    if old: raise ValueError(f'{where}: model {name!r} uses the key names before 2026-11-15 {old}; rewrite the file with '
                             'analysis/maintenance/rename_model_keys.py')


def model_folder(path):
    """a model folder (or its model.json) -> model dict with absolute file paths, 'name' and 'folder'. model.json holds one
    entry {name: {detector, emitter, cal_sim, cal_real, detector_pass2, emitter_pass2, cal_sim_pass2, cal_real_pass2, sig_u, psf}}
    with paths relative to
    the folder (absolute paths are kept), plus '_recipe' / '_files' (training record, hashes)."""
    f = os.path.join(path, MODEL_JSON) if os.path.isdir(path) else path
    S = json.load(open(f)); base = os.path.dirname(os.path.abspath(f))
    ent = [(k, v) for k, v in S.items() if not k.startswith('_')]
    if len(ent) != 1: raise ValueError(f'{f}: a model.json defines exactly one model, found {[k for k, _ in ent]}')
    name, M = ent[0]
    _check_keys(name, M, f)
    ab = lambda v: v if os.path.isabs(v) else os.path.join(base, v)
    d = {}
    for k, v in M.items():
        if k.startswith('_'): continue
        if k == 'sig_u' and isinstance(v, str): d[k] = ','.join(ab(s) if s else '' for s in v.split(','))
        else: d[k] = ab(v) if isinstance(v, str) and v else v
    d['name'] = name; d['folder'] = base
    return d


def is_model_folder(p):
    """True for a directory with a model.json, or a model.json file"""
    p = os.path.expanduser(str(p))
    return os.path.isfile(os.path.join(p, MODEL_JSON)) or (os.path.isfile(p) and p.endswith('.json'))


def model_set(name=None):
    """model -> dict with the files detector, emitter (pass-1 networks), cal_sim, cal_real (their solve calibrations),
    detector_pass2, emitter_pass2, cal_sim_pass2, cal_real_pass2 (pass 2), sig_u ('<pass 1 json>,<pass 2 json>'), psf, and name. `name`: a model of models.json (file names relative to the models
    directory), or the path of a model folder / model.json (absolute file names; also 'folder'). None: env SIMCODE_MODEL,
    else '_default' of models.json (the default is for the package's own gate scenes and trainers; simcode.run and the
    pipeline require an explicit model)."""
    S = model_sets()
    name = name or env('MODEL') or S['_default']
    if not name.startswith('_') and name in S:
        d = {k: v for k, v in S[name].items() if not k.startswith('_')}
        _check_keys(name, d, MODELS_FILE)
        d['name'] = name
        return d
    if is_model_folder(name): return model_folder(os.path.expanduser(name))
    raise KeyError(f'model {name!r} is neither a model of {MODELS_FILE} ({[k for k in S if not k.startswith("_")]}) nor a folder with a {MODEL_JSON}')


def production():
    """the paper's production recipe ('_production' of models.json): calibration kind, selection, datasets (movie, calib, model)"""
    return model_sets()['_production']


# files of the default model (models directory, relative names)
_S = model_set()
DETECTOR = _S['detector']                                      # pass-1 detector
EMITTER = _S['emitter']                                        # pass-1 emitter network
DETECTOR_PASS2 = _S['detector_pass2']                          # pass-2 (pattern-aware) detector
EMITTER_PASS2 = _S['emitter_pass2']                            # pass-2 emitter network
CAL_REAL = _S['cal_real']                                      # pass-1 solve calibration, real-data sigma (fixed c)
CAL_SIM = _S['cal_sim']                                        # same, sim sigma table (gate scenes)
CAL_REAL_PASS2 = _S['cal_real_pass2']                          # pass-2 solve calibration, real-data sigma
CAL_SIM_PASS2 = _S['cal_sim_pass2']                            # pass-2, sim sigma (gate scenes)
# calibrations of the pretraining networks (pretrain_detector_pass1 + pretrain_emitter_pass1): the sigma table of the opt-in automatic
# (s, c, c_a) fit (solve/brightness_noise_fit.py) and its start; the templates from which the calibration of a new model is fitted
# (analysis/psfmatch/fit_cal_set.py) are TEMPLATE_CAL_*
CAL_BRIGHTNESS_NOISE_FIT = 'calib/template_calib_emitter_pass1_brightness_noise_fit.json'
TEMPLATE_CAL_SIM = 'calib/template_calib_emitter_pass1_sim.json'           # pass 1, sim sigma
TEMPLATE_CAL_REAL = 'calib/template_calib_emitter_pass1_real.json'         # pass 1, real-data sigma multiplier
TEMPLATE_CAL_SIM_PASS2 = 'calib/template_calib_emitter_pass2_sim.json'     # pass 2, sim sigma
TEMPLATE_CAL_REAL_PASS2 = 'calib/template_calib_emitter_pass2_real.json'   # pass 2, real-data sigma multiplier
PRETRAIN_DETECTOR = 'pretrain_detector_pass1.pt'                         # start of a new pass-1 detector (4Pi) / default detector of the emitter trainer
PRETRAIN_EMITTER = 'pretrain_emitter_pass1.pt'                           # start of a new pass-1 emitter network
CAL_BRIGHTNESS_NOISE_FIT_START = TEMPLATE_CAL_SIM
# file names inside a model folder (model.json; simcode.pipeline.training writes them)
MODEL_FILES = dict(detector='detector_pass1.pt', emitter='emitter_pass1.pt', detector_pass2='detector_pass2.pt', emitter_pass2='emitter_pass2.pt',
                   cal_sim='calib/calib_emitter_pass1_sim.json', cal_real='calib/calib_emitter_pass1_real.json',
                   cal_sim_pass2='calib/calib_emitter_pass2_sim.json', cal_real_pass2='calib/calib_emitter_pass2_real.json',
                   sig_u='calib/sigma_rescale_pass1.json,calib/sigma_rescale_pass2.json')
INSITU_05 = 'insitu/calib_0.5nM.json'                          # in-situ PSF + camera calibration, 0.5 nM movie (default PSF of the gate scenes)
