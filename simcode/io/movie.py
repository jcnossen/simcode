"""
One movie-reader interface for both pipelines (2D SIMCODE and 4Pi). The interface and the TIFF / array readers are
torch_psf.movie (TiffMovie, ArrayMovie, open_movie); this module adds the Hamamatsu dcimg reader of the 4Pi data and the
prefetching wrapper.

  reader.n          number of frames
  reader.shape      shape of one frame as returned by read() without a crop: (H, W)
  reader.parts      the source files
  reader.read(f0, f1, y0=0, x0=0, S=None) -> float32 [f1 - f0, ...], camera ADU, absolute frame index;
                    (y0, x0, S) = a square crop of the last two axes (S None = the whole frame)

  torch_psf.movie.TiffMovie     multi-part Andor SOLIS TIFF: movie.tif + movie_X1.tif + ... (frame index runs over the parts)
  DcimgMovie(path)              Hamamatsu dcimg (memory-mapped header, large sequential reads): read() returns the frames
                                (float32), raw(f0, n) the uint16 frames; footer(i) / time_s(i) the per-frame footer (camera
                                frame counter, time stamp). The detection channels of a multi-channel camera frame:
                                simcode.channels.layout.ChannelReader(DcimgMovie(path), layout).
  torch_psf.movie.ArrayMovie    an in-memory or memory-mapped stack (tests, simulated movies)
  open_movie(path, **kw)        by extension (.dcimg here; .tif / .tiff, .npy: torch_psf.open_movie)
  Prefetch(reader, plan)        wraps a reader: the next expected read() runs in a background thread while the caller works
                                on the current one. plan = the expected read() argument tuples in order; a read that is
                                not the expected one is done synchronously (same data, only slower).

Frame record layout of dcimg (format 0x2000000, one session; verified on the 4Pi files): header word 9 = number of frames,
data start = session0 offset + offset_to_data, each frame = nrow x ncol uint16 pixels followed by a footer
(bytes_per_frame - bytes_per_image bytes): footer word 0 = camera frame counter, words 1-2 = Unix time (s, us).
"""
import os, threading, time
import numpy as np
from simcode.psf import open_movie as psf_open_movie


class DcimgMovie:
    def __init__(self, path):
        self.fn = path; self.parts = [path]
        h = np.fromfile(path, dtype='<u4', count=200)
        self.version = int(h[2]); self.nframes = int(h[9]); s0 = int(h[10])
        with open(path, 'rb') as f:
            f.seek(s0); s = np.fromfile(f, dtype='<u4', count=50)
        assert self.version in (0x1000000, 0x2000000), hex(self.version)
        self.ncol, self.nrow, self.bpr, self.bpi = int(s[18]), int(s[19]), int(s[20]), int(s[21])
        self.bpf = int(s[29]); self.off = s0 + int(s[24])
        assert self.bpr == 2 * self.ncol
        self.mm = np.memmap(path, dtype=np.uint8, mode='r', offset=self.off, shape=(self.nframes, self.bpf))
        self.n = self.nframes
        self.shape = (self.nrow, self.ncol)

    def __len__(self):
        return self.nframes

    def raw(self, f0, n):
        """[n, nrow, ncol] uint16: one large sequential pread (the data disk is a single HDD; memmap page faults with
        small readahead collapse to ~30 MB/s when two streams interleave)"""
        n = min(n, self.nframes - f0)
        buf = np.empty((n, self.bpf), np.uint8)
        fd = os.open(self.fn, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, self.off + f0 * self.bpf, n * self.bpf, os.POSIX_FADV_SEQUENTIAL)
            mv = memoryview(buf).cast('B'); pos = 0; tot = n * self.bpf
            while pos < tot:
                k = os.preadv(fd, [mv[pos:pos + (64 << 20)]], self.off + f0 * self.bpf + pos)
                if k <= 0: raise IOError('short read')
                pos += k
        finally:
            os.close(fd)
        return buf[:, :self.bpi].copy().view('<u2').reshape(n, self.nrow, self.ncol)

    def read(self, f0, f1, y0=0, x0=0, S=None):
        a = self.raw(f0, f1 - f0)
        a = a.astype(np.float32)
        return a[..., y0:y0 + S, x0:x0 + S] if S is not None else (a[..., y0:, x0:] if (y0 or x0) else a)

    def footer(self, i, nwords=5):
        return np.asarray(self.mm[i, self.bpi:self.bpi + 4 * nwords]).view('<u4').copy()

    def time_s(self, i):
        w = self.footer(i, 3)
        return w[1] + 1e-6 * w[2]


def open_movie(path, **kw):
    """reader by file extension: .dcimg -> DcimgMovie, else torch_psf.open_movie (.tif / .tiff -> TiffMovie,
    .npy -> ArrayMovie)"""
    if os.path.splitext(str(path))[1].lower() == '.dcimg': return DcimgMovie(path, **kw)
    return psf_open_movie(path)


class Prefetch:
    """Reader wrapper that performs the NEXT expected read() in a background thread while the current data are processed.
    `plan` = the expected read() calls in order, each a tuple of read() arguments; a read() that matches the prefetched
    request returns it (identical data, same reader.read), any other request is read synchronously. Attributes other than
    read() are those of the wrapped reader. stats: hits, misses, wait_s."""
    def __init__(self, reader, plan):
        self._m, self._plan, self._i = reader, [tuple(p) for p in plan], 0
        self._fut = None
        self.stats = dict(hits=0, misses=0, wait_s=0.0)
        self._start(0)

    def __getattr__(self, k):
        return getattr(self._m, k)

    def _start(self, i):
        if i >= len(self._plan): self._fut = None; return
        key = self._plan[i]; box = {}
        def work():
            try: box['a'] = self._m.read(*key)
            except Exception as e: box['e'] = e
        th = threading.Thread(target=work, daemon=True); th.start()
        self._fut = (key, th, box)

    def read(self, *key):
        key = tuple(key)
        if self._fut is not None and self._fut[0] == key:
            _, th, box = self._fut
            t = time.time(); th.join(); self.stats['wait_s'] += time.time() - t
            self._i += 1; self._start(self._i)
            if 'e' in box: raise box['e']
            self.stats['hits'] += 1
            return box['a']
        self.stats['misses'] += 1
        if self._fut is not None:          # never two readers on the file handles at once
            self._fut[1].join(); self._fut = None
        return self._m.read(*key)

    def close(self):
        """wait for a running background read (its result is dropped)"""
        if self._fut is not None: self._fut[1].join(); self._fut = None


class InterleavedTiffMovie:
    """The two-colour (multi-ROI) acquisitions: a directory with data.tif, data_X1.tif, ... = one stack of uncompressed
    uint16 pages (H x W = the ROI size; tifffile writer, page k of a part at offset 256 + k * (2 H W + 240), checked
    against the tifffile page offsets on open) and info.yaml (aoi, rois). The ROIs of one camera frame are consecutive
    pages (even pages = ROI 0, odd = ROI 1 for two ROIs). This reader works on pages:

      n               number of pages (all parts)
      read(f0, f1)    pages f0..f1 -> [f1 - f0, H, W] float32 ADU;  raw(f0, n) -> uint16 pages
      info            the parsed info.yaml

    The channels: simcode.channels.layout.ChannelReader(InterleavedTiffMovie(d), ChannelLayout([Channel(name='roi0',
    page=0), Channel(name='roi1', page=1)], frame_shape=(H, W))) -> read(f0, f1) = [n, 2, H, W] per camera frame."""
    def __init__(self, d):
        import glob, re
        import tifffile, yaml
        self.dir = d
        fs = [f for f in glob.glob(os.path.join(d, 'data*.tif')) if re.fullmatch(r'data(_X\d+)?\.tif', os.path.basename(f))]
        if not fs: raise FileNotFoundError(f'no data.tif / data_X*.tif in {d}')
        key = lambda f: 0 if os.path.basename(f) == 'data.tif' else int(re.search(r'_X(\d+)\.tif$', f).group(1))
        self.parts = sorted(fs, key=key)
        fi = os.path.join(d, 'info.yaml')
        self.info = yaml.safe_load(open(fi)) if os.path.exists(fi) else None
        with tifffile.TiffFile(self.parts[0]) as t:
            p0, p1 = t.pages[0], t.pages[1]
            self.H, self.W = p0.shape
            self.off0, self.stride = p0.dataoffsets[0], p1.dataoffsets[0] - p0.dataoffsets[0]
            if p0.dtype != np.uint16 or p0.compression != 1:
                raise ValueError(f'{self.parts[0]}: uncompressed uint16 pages expected')
        if self.stride != 2 * self.H * self.W + 240 or self.off0 != 256:
            raise ValueError(f'{self.parts[0]}: unexpected page layout (offset {self.off0}, stride {self.stride})')
        self.npages = [(os.path.getsize(f) - 16) // self.stride for f in self.parts]
        nb = 2 * self.H * self.W
        self.mm = [np.lib.stride_tricks.as_strided(np.memmap(f, np.uint8, 'r', offset=self.off0, shape=((n - 1) * self.stride + nb,)),
                                                   shape=(n, nb), strides=(self.stride, 1), writeable=False)
                   for f, n in zip(self.parts, self.npages)]
        self.cum = np.cumsum([0] + self.npages)
        self.n = int(self.cum[-1])
        self.shape = (self.H, self.W)

    def __len__(self):
        return self.n

    def raw(self, f0, n):
        """pages f0 .. f0 + n -> [n, H, W] uint16"""
        p0, p1 = int(f0), int(min(f0 + n, self.n))
        out = []
        while p0 < p1:
            i = int(np.searchsorted(self.cum, p0, 'right') - 1)
            q1 = min(p1, int(self.cum[i + 1]))
            out.append(np.asarray(self.mm[i][p0 - self.cum[i]:q1 - self.cum[i]]).view('<u2').reshape(-1, self.H, self.W))
            p0 = q1
        return np.concatenate(out) if out else np.zeros((0, self.H, self.W), np.uint16)

    def read(self, f0, f1, y0=0, x0=0, S=None):
        a = self.raw(f0, f1 - f0).astype(np.float32)
        return a[..., y0:y0 + S, x0:x0 + S] if S is not None else (a[..., y0:, x0:] if (y0 or x0) else a)


def camera_calib_pickle(fn, roi):
    """per-pixel sCMOS calibration pickle of the two-colour set-up (dict with 'gain' (e / ADU), 'baseline' (ADU), ...
    maps of the camera AOI) -> (gain [h, w], baseline [h, w]) float32 of roi = (x0, y0, w, h) in AOI px. The unpickler only
    admits numpy arrays / dtypes / scalars (no other classes)."""
    import pickle

    class _NumpyOnly(pickle.Unpickler):
        def find_class(self, m, n):
            if m.startswith('numpy') and n in ('_reconstruct', 'ndarray', 'dtype', 'scalar'): return super().find_class(m, n)
            raise pickle.UnpicklingError(f'{m}.{n} not allowed in a camera calibration pickle')

    with open(fn, 'rb') as f:
        K = _NumpyOnly(f).load()
    x0, y0, w, h = (int(v) for v in roi)
    return K['gain'][y0:y0 + h, x0:x0 + w].astype(np.float32), K['baseline'][y0:y0 + h, x0:x0 + w].astype(np.float32)
