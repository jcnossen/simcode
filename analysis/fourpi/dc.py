"""4Pi-SIMFLUX raw data I/O for the analysis scripts: the dcimg reader (simcode/io/movie.DcimgMovie), the channel
crops of the data set and its file lists (the 4Pi acquisition configs simcode/pipeline/configs/fourpi.py through
simcode/setups/fourpi.py; the file-list helpers take work = the 4Pi work folder and data = the raw 4Pi movie folder).

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
from simcode.setups.fourpi import FourPiSetup, DEFAULTS, acquisition, file_index  # noqa
from simcode.pipeline.configs import fourpi as _CF
from simcode.channels.layout import ChannelLayout

_S = FourPiSetup()
# the same channel layout for MT and NPC: the 'channels' entry of simcode/pipeline/configs/fourpi.py (path-free)
_LAYOUT = ChannelLayout.from_config(dict(channels=[dict(name=f'c{i}', crop=[x, 0, 360, 200], mirror=m)
                                                   for i, (x, m) in enumerate(zip(_CF.X0, _CF.MIRROR))]))
X0 = [c.crop[0] for c in _LAYOUT.channels]
CW, CH = _LAYOUT.W, _LAYOUT.H
MIRROR = [c.mirror != 'none' for c in _LAYOUT.channels]
FRAME_DT_S = DEFAULTS['frame_dt_s']


def movie_files(cell, work, data):
    return _S.movie_files(acquisition(cell, work, data))


def zstack_files(cell, work, data):
    return sorted(glob.glob(acquisition(cell, work, data)['fourpi']['zstack']))


def bead_align_file(cell, work, data):
    return acquisition(cell, work, data)['fourpi']['bead_align']


def crops(a):
    """raw frames [n, 200, 2304] -> [n, 4, 200, 360] (float32, channel 2/3 mirrored)"""
    return _LAYOUT.split(a)


def zstack(cell, work, data):
    """bead z-stack -> [61, 4, 200, 360] ADU (one frame per stage step, 20 nm steps; stage z = (k - 30) * 0.02 um)"""
    fs = zstack_files(cell, work, data)
    return np.stack([crops(Dcimg(f).raw(0, 1)[0]) for f in fs])
