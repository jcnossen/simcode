"""Reordered local cache of a SIMFLUX-setup movie (interleaved orientations): as reorder.py, for any multi-part movie,
read at a limited rate (the source HDD is shared), written directly to the data disk. Every complete 6-frame cycle is written
in the order [0, 2, 4, 1, 3, 5]; an incomplete last cycle is dropped. Also stores per-frame Micro-Manager elapsed time, frame
index and mean level (dropped-frame check over the WHOLE movie).
usage: ionice -c3 nice -n 19 python reorder_any.py <first part .ome.tif> <out dir> [MB/s limit, default 40]"""
import sys, os, re, glob, time, numpy as np, tifffile
sys.path.insert(0, '/home/jc/simcode2')
from simcode.util.proctitle import set_title
set_title('nr-reord')
first, out = sys.argv[1:3]; lim = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
os.makedirs(out, exist_ok=True)
base = first[:-len('.ome.tif')]
parts = [first] + sorted(glob.glob(base + '_*.ome.tif'), key=lambda p: int(re.search(r'_(\d+)\.ome\.tif$', p).group(1)))
tifs = [tifffile.TiffFile(p, is_ome=False) for p in parts]
bounds = np.cumsum([0] + [len(t.pages) for t in tifs]); N = int(bounds[-1]); NC = N // 6
print('parts', [os.path.basename(p) for p in parts], 'frames', N, bounds.tolist(), 'cycles', NC, 'dropped tail frames', N - 6 * NC, flush=True)
PERM = [0, 2, 4, 1, 3, 5]
def rd(fi):
    j = int(np.searchsorted(bounds, fi, side='right') - 1); pg = tifs[j].pages[int(fi - bounds[j])]
    tg = pg.tags.get('MicroManagerMetadata'); m = tg.value if tg is not None else {}
    return pg.asarray(), m.get('ElapsedTime-ms', np.nan), m.get('FrameIndex', -1)
el = np.full(N, np.nan); fidx = -np.ones(N, int); mean = np.zeros(N); t0 = time.time(); nbytes = 0
tmp = os.path.join(out, 'movie_reord.tif.part')
with tifffile.TiffWriter(tmp, bigtiff=True) as tw:
    for c in range(NC):
        fr = []
        for j in range(6):
            a, e, fi = rd(6 * c + j); fr.append(a); el[6 * c + j] = e; fidx[6 * c + j] = fi; mean[6 * c + j] = a.mean(); nbytes += a.nbytes
        for j in range(6): tw.write(fr[PERM[j]], contiguous=True)
        w = nbytes / (lim * 1e6) - (time.time() - t0)
        if w > 0: time.sleep(w)
        if c % 1000 == 0: print(c, '%.0f s' % (time.time() - t0), flush=True)
os.rename(tmp, os.path.join(out, 'movie_reord.tif'))
np.savez(os.path.join(out, 'frames_info.npz'), elapsed_ms=el, frameindex=fidx, mean=mean, perm=PERM, parts=parts, bounds=bounds)
k = 6 * NC; d = np.diff(el[:k])
print('frameindex contiguous', bool((np.diff(fidx[:k]) == 1).all()), 'dt median %.3f max %.1f, n > 1.5 median %d' % (np.median(d), d.max(), (d > 1.5 * np.median(d)).sum()))
for s in range(6): print('acq step', s, 'mean ADU %.3f' % mean[:k][s::6].mean())
print('done', time.time() - t0)
