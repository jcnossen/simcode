"""4Pi-SIMFLUX raw data I/O for the analysis scripts: the dcimg reader (simcode/io/movie.DcimgMovie), the channel
crops of the data set and its file lists (the 4Pi acquisition configs simcode/pipeline/configs/fourpi.py through
simcode/setups/fourpi.py).

Channels: 4 crop windows of 360 x 200 px at x0 = 217, 652, 1420, 1855 (the stored crop windows of the data set);
channels 2 and 3 (0-based) are mirrored left-right so that all four overlay. Channel c pixel (row y, col x) in the
mirrored crop = raw column X0[c] + x (c = 0, 1) or X0[c] + 359 - x (c = 2, 3).
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
import numpy as np
import glob
from simcode.io.movie import DcimgMovie as Dcimg
from simcode.setups.fourpi import FourPiSetup, acquisition, file_index  # noqa

_S = FourPiSetup()
_LAYOUT = _S.layout(acquisition('MT'))                      # the same channel layout for MT and NPC
X0 = [c.crop[0] for c in _LAYOUT.channels]
CW, CH = _LAYOUT.W, _LAYOUT.H
MIRROR = [c.mirror != 'none' for c in _LAYOUT.channels]
FRAME_DT_S = acquisition('MT')['fourpi']['frame_dt_s']


def movie_files(cell):
    return _S.movie_files(acquisition(cell))


def zstack_files(cell):
    return sorted(glob.glob(acquisition(cell)['fourpi']['zstack']))


def bead_align_file(cell):
    return acquisition(cell)['fourpi']['bead_align']


def crops(a):
    """raw frames [n, 200, 2304] -> [n, 4, 200, 360] (float32, channel 2/3 mirrored)"""
    return _LAYOUT.split(a)


def zstack(cell):
    """bead z-stack -> [61, 4, 200, 360] ADU (one frame per stage step, 20 nm steps; stage z = (k - 30) * 0.02 um)"""
    fs = zstack_files(cell)
    return np.stack([crops(Dcimg(f).raw(0, 1)[0]) for f in fs])
