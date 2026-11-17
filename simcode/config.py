"""
The models of SIMCODE v2 (a "model" = one trained set of networks + their solve calibrations) and where their files are.
There are no path settings (no config file, no environment variables): every function takes its paths as arguments and
every tool takes them on the command line (--movie, --out, --model, ...). The one default location is the repository's own
models folder, MODELS = <repo>/models, where the names of shipped networks and calibrations are looked up.

resolve(p) turns a model / calibration name or path into an absolute path: absolute paths and paths that exist relative to
the current directory are kept; otherwise the name is looked up under MODELS (also MODELS/calib and MODELS/insitu).
model_set(name) returns one model: a name of simcode/models.json (the definition of the shipped models), or the path of a
model folder (model.json + the network and calibration files, as written by the automatic pipeline's training stage /
simcode.pipeline.training.export_model), or of such a model.json.
"""
import json, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = str(REPO / 'models')            # the repository's models folder (shipped networks and calibrations)


def resolve(p):
    """model / calibration name or path -> absolute path (see the module doc); the input unchanged if empty"""
    if not p: return p
    p = os.path.expanduser(str(p))
    if os.path.isabs(p) or os.path.exists(p): return os.path.abspath(p)
    cands = [os.path.join(MODELS, p), os.path.join(MODELS, 'calib', p), os.path.join(MODELS, 'insitu', p)]
    for c in cands:
        if os.path.exists(c): return c
    return cands[0]


def model(name):
    """network / calibration file shipped in the models folder, e.g. model('pretrain_detector_pass1.pt'), model('calib/x.json')"""
    return resolve(name)


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
# calibrations with the real-data sigma multiplier (sig_ph_real_mult), removed 2026-11-17: a model has one solve calibration per
# pass (cal_sim, cal_sim_pass2), fitted on simulation only; a definition that still names cal_real / cal_real_pass2 is refused.
# The selection-aware sigma rescale sig_u (sigma x g(u), u = per-tile sigma rank) was removed on 2026-11-17 as well: the reported
# sigma is the emitter network's own (joint covariance) or the solve's posterior sd, with no factor; a definition naming sig_u is refused
REMOVED_KEYS = ('cal_real', 'cal_real_pass2', 'sig_u')


def _check_keys(name, M, where):
    old = [k for k in M if k in OLD_KEYS]
    if old: raise ValueError(f'{where}: model {name!r} uses the key names before 2026-11-15 {old}; rewrite the file with '
                             'analysis/maintenance/rename_model_keys.py')
    rem = [k for k in M if k in REMOVED_KEYS]
    if rem: raise ValueError(f'{where}: model {name!r} names {rem}, removed on 2026-11-17 (cal_real*: the real-data sigma '
                             'calibrations, real movies use the simulation-fitted cal_sim / cal_sim_pass2; sig_u: the selection-aware '
                             'sigma rescale, no sigma factor is applied any more); delete these keys')


def model_folder(path):
    """a model folder (or its model.json) -> model dict with absolute file paths, 'name' and 'folder'. model.json holds one
    entry {name: {detector, emitter, cal_sim, detector_pass2, emitter_pass2, cal_sim_pass2, psf}}
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
        d[k] = ab(v) if isinstance(v, str) and v else v
    d['name'] = name; d['folder'] = base
    return d


def is_model_folder(p):
    """True for a directory with a model.json, or a model.json file"""
    p = os.path.expanduser(str(p))
    return os.path.isfile(os.path.join(p, MODEL_JSON)) or (os.path.isfile(p) and p.endswith('.json'))


def model_set(name=None):
    """model -> dict with the files detector, emitter (pass-1 networks), cal_sim (their solve calibration, fitted on simulation),
    detector_pass2, emitter_pass2, cal_sim_pass2 (pass 2), psf, and name. `name`: a model of models.json (file names relative to the models
    folder), or the path of a model folder / model.json (absolute file names; also 'folder'). None: '_default' of models.json (the default is for the package's own gate scenes and trainers; simcode.run and the
    pipeline require an explicit model)."""
    S = model_sets()
    name = name or S['_default']
    if not name.startswith('_') and name in S:
        d = {k: v for k, v in S[name].items() if not k.startswith('_')}
        _check_keys(name, d, MODELS_FILE)
        d['name'] = name
        return d
    if is_model_folder(name): return model_folder(os.path.expanduser(name))
    raise KeyError(f'model {name!r} is neither a model of {MODELS_FILE} ({[k for k in S if not k.startswith("_")]}) nor a folder with a {MODEL_JSON}')


def production():
    """the paper's production recipe ('_production' of models.json): selection, datasets (movie, calib, model)"""
    return model_sets()['_production']


# files of the default model (models directory, relative names)
_S = model_set()
DETECTOR = _S['detector']                                      # pass-1 detector
EMITTER = _S['emitter']                                        # pass-1 emitter network
DETECTOR_PASS2 = _S['detector_pass2']                          # pass-2 (pattern-aware) detector
EMITTER_PASS2 = _S['emitter_pass2']                            # pass-2 emitter network
CAL_SIM = _S['cal_sim']                                        # pass-1 solve calibration (fixed c, prior_off from the gate scenes)
CAL_SIM_PASS2 = _S['cal_sim_pass2']                            # pass-2 solve calibration (the same for pass 2)
# calibrations of the pretraining networks (pretrain_detector_pass1 + pretrain_emitter_pass1): the templates from which the calibration
# of a new model is fitted (simcode.calibration.fit_cal_set) are TEMPLATE_CAL_*
TEMPLATE_CAL_SIM = 'calib/template_calib_emitter_pass1_sim.json'           # pass 1
TEMPLATE_CAL_SIM_PASS2 = 'calib/template_calib_emitter_pass2_sim.json'     # pass 2
PRETRAIN_DETECTOR = 'pretrain_detector_pass1.pt'                         # start of a new pass-1 detector (4Pi) / default detector of the emitter trainer
PRETRAIN_EMITTER = 'pretrain_emitter_pass1.pt'                           # start of a new pass-1 emitter network
# file names inside a model folder (model.json; simcode.pipeline.training writes them)
MODEL_FILES = dict(detector='detector_pass1.pt', emitter='emitter_pass1.pt', detector_pass2='detector_pass2.pt', emitter_pass2='emitter_pass2.pt',
                   cal_sim='calib/calib_emitter_pass1_sim.json', cal_sim_pass2='calib/calib_emitter_pass2_sim.json')
INSITU_05 = 'insitu/calib_0.5nM.json'                          # in-situ PSF + camera calibration, 0.5 nM movie (default PSF of the gate scenes)
