"""The 4Pi-SIMFLUX set-up in one place: paths and the constants of the data set (simcode/fourpi/datasets.json).

  P4        work directory of the 4Pi pipelines (config root 'fourpi', default <out>/4pi): models/, psf4pi/, smlm/ (camera
            maps, registration, pattern), net/ (network tables), classical/ (classical tables), sim*/ ...
  DATA4     directory of the raw dcimg files (config root 'data4pi')
  LAYOUT    channel crop windows of one camera frame: dict(x0, cw, ch, mirror)
  GAIN      camera gain (e / ADU);  PX_NM pixel size (nm);  FRAME_DT_S frame time (s)
  P         candidate patch size (px);  XLIM illuminated columns [x0, x1) of a channel crop
  DELTA     per cell: 4Pi phase offset of the second pattern angle (rad)
movie_files(cell), zstack_files(cell), bead_align_file(cell), file_index(path): the files of a cell.
"""
import glob, json, math, os
from pathlib import Path
from simcode import config

_D = json.load(open(Path(__file__).with_name('datasets.json')))
P4 = config.root('fourpi')
DATA4 = config.root('data4pi')
LAYOUT = dict(_D['layout'], x0=tuple(_D['layout']['x0']), mirror=tuple(_D['layout']['mirror']))
GAIN = _D['gain_e_per_adu']
PX_NM = _D['px_nm']
FRAME_DT_S = _D['frame_dt_s']
P = _D['patch']
XLIM = tuple(_D['xlim'])
DELTA = {c: math.radians(v['delta_deg']) for c, v in _D['cells'].items()}
CELLS = tuple(_D['cells'])


def _cell(cell):
    if cell not in _D['cells']: raise ValueError(cell)
    return _D['cells'][cell]


def movie_files(cell):
    return sorted(glob.glob(os.path.join(DATA4, _cell(cell)['movies'])))


def zstack_files(cell):
    return sorted(glob.glob(os.path.join(DATA4, _cell(cell)['zstack'])))


def bead_align_file(cell):
    return os.path.join(DATA4, _cell(cell)['bead_align'])


def file_index(fn):
    return int(os.path.basename(fn).split('_')[-1].split('.')[0])


def movie_file(cell, fi):
    """path of file number fi of a cell's movie"""
    fs = [f for f in movie_files(cell) if file_index(f) == fi]
    if not fs: raise FileNotFoundError(f'{cell}: no movie file with index {fi} under {DATA4}')
    return fs[0]
