"""
One movie-reader interface for both pipelines (2D SIMCODE and 4Pi). The interface and the TIFF / array readers are
torch_psf.movie (TiffMovie, ArrayMovie, open_movie); this module adds the Hamamatsu dcimg reader of the 4Pi data and the
prefetching wrapper.

  reader.n          number of frames
  reader.shape      shape of one frame as returned by read() without a crop: (H, W), or (4, H, W) for a dcimg with a layout
  reader.parts      the source files
  reader.read(f0, f1, y0=0, x0=0, S=None) -> float32 [f1 - f0, ...], camera ADU, absolute frame index;
                    (y0, x0, S) = a square crop of the last two axes (S None = the whole frame)

  torch_psf.movie.TiffMovie     multi-part Andor SOLIS TIFF: movie.tif + movie_X1.tif + ... (frame index runs over the parts)
  DcimgMovie(path, layout=None) Hamamatsu dcimg (memory-mapped header, large sequential reads). layout = dict(x0, cw, ch,
                                mirror): the frame is cut into len(x0) channel crops of ch x cw px at the raw columns x0,
                                mirrored channels flipped left-right -> read() returns [n, C, ch, cw]. Without a layout the
                                raw frames. raw(f0, n) returns the uint16 raw frames; footer(i) / time_s(i) the per-frame
                                footer (camera frame counter, time stamp).
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
    def __init__(self, path, layout=None):
        self.fn = path; self.parts = [path]; self.layout = layout
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
        self.shape = (self.nrow, self.ncol) if layout is None else (len(layout['x0']), layout['ch'], layout['cw'])

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
        a = crops(a, self.layout) if self.layout is not None else a.astype(np.float32)
        return a[..., y0:y0 + S, x0:x0 + S] if S is not None else (a[..., y0:, x0:] if (y0 or x0) else a)

    def footer(self, i, nwords=5):
        return np.asarray(self.mm[i, self.bpi:self.bpi + 4 * nwords]).view('<u4').copy()

    def time_s(self, i):
        w = self.footer(i, 3)
        return w[1] + 1e-6 * w[2]


def crops(a, layout):
    """raw frames [..., nrow, ncol] -> [..., C, ch, cw] float32: channel c = columns x0[c] .. x0[c] + cw, flipped left-right
    where mirror[c]"""
    out = []
    for c, x0 in enumerate(layout['x0']):
        x = a[..., x0:x0 + layout['cw']]
        out.append(x[..., ::-1] if layout['mirror'][c] else x)
    return np.stack(out, -3).astype(np.float32)


def open_movie(path, **kw):
    """reader by file extension: .dcimg -> DcimgMovie(layout=...), else torch_psf.open_movie (.tif / .tiff -> TiffMovie,
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
