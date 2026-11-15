"""4Pi-SIMFLUX raw data I/O for the analysis scripts: the dcimg reader (simcode/io/movie.DcimgMovie), the channel
crops of the data set and its file lists (simcode/fourpi/setup.py, simcode/fourpi/datasets.json).

Channels: 4 crop windows of 360 x 200 px at x0 = 217, 652, 1420, 1855 (the stored crop windows of the data set);
channels 2 and 3 (0-based) are mirrored left-right so that all four overlay. Channel c pixel (row y, col x) in the
mirrored crop = raw column X0[c] + x (c = 0, 1) or X0[c] + 359 - x (c = 2, 3).
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
import numpy as np
from simcode.io.movie import DcimgMovie as Dcimg, crops as _crops
from simcode.fourpi.setup import DATA4 as ROOT, LAYOUT, FRAME_DT_S, movie_files, file_index, zstack_files, bead_align_file  # noqa

X0 = LAYOUT['x0']
CW, CH = LAYOUT['cw'], LAYOUT['ch']
MIRROR = LAYOUT['mirror']


def crops(a):
    """raw frames [..., 200, 2304] -> [..., 4, 200, 360] (float32, channel 2/3 mirrored)"""
    return _crops(a, LAYOUT)


def zstack(cell):
    """bead z-stack -> [61, 4, 200, 360] ADU (one frame per stage step, 20 nm steps; stage z = (k - 30) * 0.02 um)"""
    fs = zstack_files(cell)
    return np.stack([crops(Dcimg(f).raw(0, 1)[0]) for f in fs])
