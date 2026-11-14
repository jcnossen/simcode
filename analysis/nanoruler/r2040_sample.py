"""Cache the first N frames of the first part of a nanoruler movie as one TIFF (acquisition order, uint16), with the
per-frame Micro-Manager elapsed time / frame index and mean / median level. One sequential read of ~1 GB.
Output: <outdir>/sample.tif, <outdir>/frames_info.npz"""
import argparse, sys, os, numpy as np, tifffile
sys.path.insert(0, '/home/jc/simcode2')
from simcode.util.proctitle import set_title
set_title('nr-facts')
ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True); ap.add_argument('--outdir', required=True); ap.add_argument('--n', type=int, default=3000)
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
el = np.full(a.n, np.nan); fi = np.full(a.n, -1); mean = np.zeros(a.n); med = np.zeros(a.n); extra = []
with tifffile.TiffFile(a.src, is_ome=False) as t, tifffile.TiffWriter(a.outdir + '/sample.tif', bigtiff=True) as tw:
    for i in range(a.n):
        pg = t.pages[i]; im = pg.asarray()
        tg = pg.tags.get('MicroManagerMetadata'); m = tg.value if tg is not None else {}
        el[i] = m.get('ElapsedTime-ms', np.nan); fi[i] = m.get('FrameIndex', m.get('Frame', -1))
        mean[i] = im.mean(); med[i] = np.median(im)
        tw.write(im, contiguous=True)
        if i % 500 == 0: print(i, im.shape, im.dtype, el[i], fi[i], flush=True)
np.savez(a.outdir + '/frames_info.npz', elapsed_ms=el, frameindex=fi, mean=mean, median=med, src=a.src)
d = np.diff(el); print('dt ms median %.3f min %.3f max %.3f; n > 1.5 median: %d' % (np.nanmedian(d), np.nanmin(d), np.nanmax(d), (d > 1.5 * np.nanmedian(d)).sum()))
print('frameindex contiguous', bool((np.diff(fi) == 1).all()), fi[0], fi[-1])
