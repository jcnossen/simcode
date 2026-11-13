"""Stage 'order': movies whose frame order is not the blocked 6-frame cycle [a0s0 a0s1 a0s2 a1s0 a1s1 a1s2] are written once as
a blocked-order cache (every complete cycle permuted; an incomplete last cycle dropped), the input of every later stage
(docs/AUTO_PIPELINE.md section 7, assumption 1). As analysis/nanoruler/reorder_any.py, for Andor multi-part TIFF
(movie.tif, movie_X1.tif, ...) and Micro-Manager multi-part OME-TIFF (name.ome.tif, name_1.ome.tif, ...)."""
import glob, os, re, time
import numpy as np


class _Src:
    def __init__(self, path):
        import tifffile
        if path.endswith('.ome.tif'):
            base = path[:-len('.ome.tif')]
            parts = [path] + sorted(glob.glob(base + '_*.ome.tif'), key=lambda p: int(re.search(r'_(\d+)\.ome\.tif$', p).group(1)))
            self.tifs = [tifffile.TiffFile(p, is_ome=False) for p in parts]
        else:
            from simcode.fileio.movie import TiffMovie
            parts = TiffMovie(path).parts
            self.tifs = [tifffile.TiffFile(p) for p in parts]
        self.parts = parts
        self.bounds = np.cumsum([0] + [len(t.pages) for t in self.tifs]); self.n = int(self.bounds[-1])

    def frame(self, fi):
        j = int(np.searchsorted(self.bounds, fi, side='right') - 1)
        return self.tifs[j].pages[int(fi - self.bounds[j])].asarray()


def write_blocked(src, cache, inv, mb_s=60.0, log=print):
    import tifffile
    S = _Src(src); NC = S.n // 6
    os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
    tmp = cache + '.part'; t0 = time.time(); nb = 0
    with tifffile.TiffWriter(tmp, bigtiff=True) as tw:
        for c in range(NC):
            fr = [S.frame(6 * c + j) for j in range(6)]; nb += sum(a.nbytes for a in fr)
            for s in range(6): tw.write(fr[inv[s]], contiguous=True)
            w = nb / (mb_s * 1e6) - (time.time() - t0)
            if w > 0: time.sleep(w)
            if c % 2000 == 0: log(f'order: cycle {c} / {NC}')
    os.replace(tmp, cache)


def verify(src, cache, inv, ncheck=4, log=print):
    from simcode.fileio.movie import open_movie
    S = _Src(src); M = open_movie(cache); NC = S.n // 6
    cyc = sorted({0, NC // 3, 2 * NC // 3, NC - 1})[:ncheck]
    ok = M.n == 6 * NC
    same = True
    for c in cyc:
        a = M.read(6 * c, 6 * c + 6)
        for s in range(6): same &= bool(np.array_equal(a[s], S.frame(6 * c + inv[s]).astype(np.float32)))
    lv = M.read(0, min(M.n, 600)).reshape(-1, 6, *M.shape[-2:]).mean((0, 2, 3))
    info = dict(n_source=S.n, n_cache=M.n, cycles=NC, dropped_tail=S.n - 6 * NC, checked_cycles=cyc, slot_mean_adu=lv.tolist())
    log(f'order: cache {cache}: {M.n} frames ({NC} cycles; source {S.n}), permutation verified on cycles {cyc}: {same}; slot means {np.round(lv, 2).tolist()}')
    return dict(info=info, checks=[dict(name='cache frames = 6 x complete source cycles', value=M.n, ok=ok),
                                   dict(name='cache = source frames in the blocked order (sampled cycles)', value=same, ok=same)])
