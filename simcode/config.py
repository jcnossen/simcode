"""
Paths of a SIMCODE v2 installation and the network sets. Nothing in the package hard-codes an absolute path; everything goes
through here.

Roots, each set by (in order of precedence) an environment variable, a key in a json config file, or a default:
  models   trained networks (*.pt) and calibrations (calib/, insitu/)   SIMCODE_MODELS     default <repo>/models
  data     raw movies (the DNA-PAINT TIFF stacks)                         SIMCODE_DATA       default: none (pass full paths)
  out      outputs of the benchmark / analysis scripts                    SIMCODE_OUT        default ./simcode_out
  fourpi   work directory of the 4Pi pipelines (models, PSF, registration, SIMCODE_4PI        default <out>/4pi
           tables)
  data4pi  raw 4Pi movies (dcimg)                                         SIMCODE_4PI_DATA   default: 'root' of simcode/fourpi/datasets.json
One optional path, set the same way: dme = a checkout of the DME drift-estimation package (DME_PATH; default: the installed
package; see simcode/util/drift.py).
Config file: $SIMCODE_CONFIG, else <repo>/simcode.json, else ~/.config/simcode.json; keys "models", "data", "out", "fourpi",
"data4pi", "dme" (see simcode.example.json). Relative paths in the config file are relative to the file's directory.
The names used before the package was renamed (simcode2 -> simcode, 2026-11) are still accepted: SIMCODE2_* environment
variables, <repo>/simcode2.json, ~/.config/simcode2.json (the new name wins when both exist).

resolve(p, kind) turns a user path into an absolute one: absolute paths and paths that exist relative to the current directory
are kept; otherwise the path is looked up under the given root (models: also under models/calib and models/insitu).
netset(name) returns one network set of simcode/netsets.json (the single definition of the sets).
"""
import json, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_ENV = {'models': 'MODELS', 'data': 'DATA', 'out': 'OUT', 'fourpi': '4PI', 'data4pi': '4PI_DATA'}
_cfg = None


def env(name, default=None):
    """environment variable SIMCODE_<name>, else the pre-rename SIMCODE2_<name>, else default"""
    return os.environ.get('SIMCODE_' + name) or os.environ.get('SIMCODE2_' + name) or default


def _config():
    global _cfg
    if _cfg is None:
        _cfg = {}
        for p in (env('CONFIG'), REPO / 'simcode.json', REPO / 'simcode2.json', Path.home() / '.config' / 'simcode.json',
                  Path.home() / '.config' / 'simcode2.json'):
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
    if kind == 'out': return 'simcode2_out' if os.path.isdir('simcode2_out') and not os.path.isdir('simcode_out') else 'simcode_out'
    if kind == 'fourpi': return os.path.join(root('out'), '4pi')
    if kind == 'data4pi': return json.load(open(Path(__file__).with_name('fourpi') / 'datasets.json'))['root']
    return ''


def models_dir():
    return root('models')


def data_root():
    return root('data')


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
    """network / calibration file shipped in the models directory, e.g. model('det_n1.pt'), model('calib/x.json')"""
    return resolve(name, 'models')


# ---- network sets: the single definition is simcode/netsets.json (next to this file)
NETSETS_FILE = Path(__file__).with_name('netsets.json')
_nets = None


def netsets():
    """the whole netsets.json (network sets, '_default', '_production')"""
    global _nets
    if _nets is None: _nets = json.load(open(NETSETS_FILE))
    return _nets


def netset(name=None):
    """network set by name (default: env SIMCODE_NETSET, else '_default' of netsets.json) -> dict with the file names relative
    to the models directory: det, head, cal_sim, cal_real, p2_det, p2_head, p2_cal_sim, p2_cal_real, sig_u ('<pass 1 json>,<pass 2 json>'),
    psf, name"""
    S = netsets()
    name = name or env('NETSET') or S['_default']
    if name.startswith('_') or name not in S:
        raise KeyError(f'network set {name!r} is not defined in {NETSETS_FILE} (sets: {[k for k in S if not k.startswith("_")]})')
    d = {k: v for k, v in S[name].items() if not k.startswith('_')}
    d['name'] = name
    return d


def production():
    """the paper's production recipe ('_production' of netsets.json): flags, selection, datasets (movie, calib, netset)"""
    return netsets()['_production']


# files of the default network set (models directory, relative names)
_S = netset()
DET = _S['det']                                                # pass-1 detector
HEAD = _S['head']                                              # pass-1 head (SCHead4, neighbour-aware)
P2_DET = _S['p2_det']                                          # pass-2 (pattern-aware) detector
P2_HEAD = _S['p2_head']                                        # pass-2 head (p2L)
CAL_P1_REAL = _S['cal_real']                                   # pass-1 solve calibration, real-data sigma (fixed c)
CAL_P1_SIM = _S['cal_sim']                                     # same, sim sigma table (gate scenes)
CAL_P2_REAL = _S['p2_cal_real']                                # pass-2 solve calibration, real-data sigma
CAL_P2_SIM = _S['p2_cal_sim']                                  # pass-2, sim sigma (gate scenes)
CAL_P2_GATE = CAL_P2_SIM
# calibrations that exist only for the earlier default head (hd_sc5nb): the sigma table of the opt-in automatic-(c, c_a)
# solve (run --autoc 1), and the templates from which the calibration of a new set is fitted (analysis/psfmatch/fit_cal_set.py)
CAL_P1_AUTOC = 'calib/calib_sc_hd_sc5nbb15_pm_autoc1.json'
CAL_P1_AUTOC_START = 'calib/calib_sc_hd_sc5nbb15_pm_bright.json'
INSITU_05 = 'insitu/calib_0.5nM.json'                          # in-situ PSF + camera calibration, 0.5 nM movie (default PSF of the gate scenes)
