"""Cache the sim4_1 movie locally in the frame order the SIMCODE v2 pipeline assumes.

The SIMFLUX setup interleaves the two pattern orientations: acquisition frame t has pattern index t % 6 with orientation
(t % 6) % 2 and phase step (t % 6) // 2 (the paper's processing: patterns [0 2 4] = axis 0, [1 3 5] = axis 1).
simcode2 assumes orientation = (t % 6) // 3. Here every 6-frame cycle is written in the order [0, 2, 4, 1, 3, 5], so that
written frame 6 c + j = acquisition frame 6 c + PERM[j]. Only windows that start at a multiple of 6 are true 6-consecutive-
frame windows afterwards. Also records per-frame Micro-Manager elapsed time and mean / median ADU (dropped-frame check).
Output: out/nanoruler/raw/sim4_1_reord.tif (BigTIFF, uint16), out/nanoruler/frames_info.npz"""
import sys, os, numpy as np, tifffile
sys.path.insert(0, '/home/jc/simcode2')
from simcode2.util.proctitle import set_title
set_title('nr-reord')
R = '/home/jc/simcode2/out/nanoruler/raw/'
PERM = [0, 2, 4, 1, 3, 5]
parts = ['sim4_1_MMStack_Pos0.ome.tif', 'sim4_1_MMStack_Pos0_1.ome.tif', 'sim4_1_MMStack_Pos0_2.ome.tif']
tifs = [tifffile.TiffFile(R + p, is_ome=False) for p in parts]
bounds = np.cumsum([0] + [len(t.pages) for t in tifs]); N = int(bounds[-1]); print('frames', N, bounds)
assert N % 6 == 0
def rd(fi):
    j = int(np.searchsorted(bounds, fi, side='right') - 1); pg = tifs[j].pages[int(fi - bounds[j])]
    tg = pg.tags.get('MicroManagerMetadata'); m = tg.value if tg is not None else {}
    return pg.asarray(), m.get('ElapsedTime-ms', np.nan), m.get('FrameIndex', -1)
el = np.zeros(N); fidx = np.zeros(N, int); mean = np.zeros(N); med = np.zeros(N)
with tifffile.TiffWriter(R + 'sim4_1_reord.tif', bigtiff=True) as tw:
    for c in range(N // 6):
        fr = []
        for j in range(6):
            a, e, fi = rd(6 * c + j); fr.append(a); el[6 * c + j] = e; fidx[6 * c + j] = fi
            mean[6 * c + j] = a.mean(); med[6 * c + j] = np.median(a)
        for j in range(6):
            tw.write(fr[PERM[j]], contiguous=True)
        if c % 500 == 0: print(c, flush=True)
np.savez(R + '../frames_info.npz', elapsed_ms=el, frameindex=fidx, mean=mean, median=med, perm=PERM)
d = np.diff(el); print('dt ms median %.3f min %.3f max %.3f; n > 1.5 median: %d' % (np.median(d), d.min(), d.max(), (d > 1.5 * np.median(d)).sum()))
print('frameindex contiguous', bool((np.diff(fidx) == 1).all()), fidx[0], fidx[-1])
for s in range(6): print('step', s, 'mean ADU %.3f median %.2f' % (mean[s::6].mean(), np.median(med[s::6])))
