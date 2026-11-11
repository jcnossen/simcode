"""
Paths of a SIMCODE v2 installation. Nothing in the package hard-codes an absolute path; everything goes through here.

Three roots, each set by (in order of precedence) an environment variable, a key in a json config file, or a default:
  models   trained networks (*.pt) and calibrations (calib/, insitu/)   SIMCODE2_MODELS   default <repo>/models
  data     raw movies (the DNA-PAINT TIFF stacks)                         SIMCODE2_DATA     default: none (pass full paths)
  out      outputs of the benchmark / analysis scripts                    SIMCODE2_OUT      default ./simcode2_out
One optional path, set the same way: dme = a checkout of the DME drift-estimation package (DME_PATH; default: the installed
package; see simcode2/util/drift.py).
Config file: $SIMCODE2_CONFIG, else <repo>/simcode2.json, else ~/.config/simcode2.json; keys "models", "data", "out"
(see simcode2.example.json). Relative paths in the config file are relative to the file's directory.

resolve(p, kind) turns a user path into an absolute one: absolute paths and paths that exist relative to the current directory
are kept; otherwise the path is looked up under the given root (models: also under models/calib and models/insitu).
"""
import json, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_ENV = {'models': 'SIMCODE2_MODELS', 'data': 'SIMCODE2_DATA', 'out': 'SIMCODE2_OUT', 'dme': 'DME_PATH'}
_DEF = {'models': str(REPO / 'models'), 'data': '', 'out': 'simcode2_out', 'dme': ''}
_cfg = None


def _config():
    global _cfg
    if _cfg is None:
        _cfg = {}
        for p in (os.environ.get('SIMCODE2_CONFIG'), REPO / 'simcode2.json', Path.home() / '.config' / 'simcode2.json'):
            if p and Path(p).is_file():
                c = json.load(open(p)); base = Path(p).resolve().parent
                ex = lambda v: os.path.expanduser(v) if isinstance(v, str) else v
                _cfg = {k: (str((base / ex(v)).resolve()) if isinstance(v, str) and v and not k.startswith('_') and not os.path.isabs(ex(v))
                            else ex(v)) for k, v in c.items()}
                _cfg['_file'] = str(p)
                break
    return _cfg


def root(kind):
    """'models' | 'data' | 'out' | 'dme' -> directory (str; '' if unset)"""
    return os.environ.get(_ENV[kind]) or _config().get(kind) or _DEF[kind]


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


# production files (models directory, relative names)
DET = 'det_n1.pt'                                              # pass-1 detector
HEAD = 'hd_sc5nb.pt'                                           # pass-1 head (SCHead4, neighbour-aware)
P2_DET = 'p2_det.pt'                                           # pass-2 (pattern-aware) detector
P2_HEAD = 'p2_head_long.pt'                                    # pass-2 head (p2L)
BGNET_HEAD = 'hd_sc7bg.pt'                                     # optional: sc5nb + dense BgNet background
CAL_P1_REAL = 'calib/calib_sc_hd_sc5nbb15_pm_bright_real.json'  # pass-1 solve calibration, real-data sigma (fixed c 0.10 + 0.10)
CAL_P1_SIM = 'calib/calib_sc_hd_sc5nbb15_pm_bright.json'       # same, sim sigma table (gate scenes; start of the automatic c)
CAL_P1_AUTOC = 'calib/calib_sc_hd_sc5nbb15_pm_autoc1.json'     # sigma table for the automatic-(c, c_a) solve
CAL_P2_REAL = 'calib/calib_p2L_b10_real.json'                  # pass-2 solve calibration, real-data sigma
CAL_P2_SIM = 'calib/calib_p2L_movie_b10.json'                  # pass-2, sim sigma (movie benchmark)
CAL_P2_GATE = 'calib/calib_p2L.json'                           # pass-2, gate calibration
INSITU_05 = 'insitu/calib_0.5nM.json'                          # in-situ PSF + camera calibration, 0.5 nM movie (default PSF for sims)
