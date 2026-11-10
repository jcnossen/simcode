"""Zhang-Suen thinning in numpy: a drop-in for skimage.morphology.skeletonize on 2D boolean images (its default 2D method,
skimage's _fast_skeletonize), so that zoompick.py does not need scikit-image. Same algorithm and the same lookup table as
scikit-image (BSD-3-Clause; table read from scikit-image 0.26's _fast_skeletonize): 1-px zero border; per iteration two
parallel sub-passes; in each, every set pixel's 8-neighbourhood code (1 NW, 2 N, 4 NE, 8 E, 16 SE, 32 S, 64 SW, 128 W) on
the image as it was at the start of the sub-pass is looked up (1: removable in the first sub-pass, 2: in the second, 3: in
both); repeat until nothing is removed. Checked identical to skimage 0.26 on the zoompick inputs and on random images
(see analysis/figures/README.md)."""
import numpy as np

# scikit-image's Zhang-Suen lookup table (256 neighbourhood codes)
LUT = np.array([0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0,
                0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0,
                0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0, 0, 0, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1,
                3, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 1, 0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0], np.uint8)
assert LUT.shape == (256,)
_W = np.array([[1, 2, 4], [128, 0, 8], [64, 32, 16]])


def skeletonize(image):
    sk = np.zeros((image.shape[0] + 2, image.shape[1] + 2), np.uint8)
    sk[1:-1, 1:-1] = np.asarray(image).astype(bool)
    removed = True
    while removed:
        removed = False
        for first in (True, False):
            code = np.zeros(sk.shape, np.int32)
            for dy in range(3):
                for dx in range(3):
                    if _W[dy, dx]:
                        code[1:-1, 1:-1] += _W[dy, dx] * sk[dy:dy + sk.shape[0] - 2, dx:dx + sk.shape[1] - 2]
            v = LUT[code]
            rm = (sk == 1) & ((v == 3) | ((v == 1) & first) | ((v == 2) & (not first)))
            rm[0, :] = rm[-1, :] = False; rm[:, 0] = rm[:, -1] = False
            if rm.any():
                sk[rm] = 0; removed = True
    return sk[1:-1, 1:-1].astype(bool)
