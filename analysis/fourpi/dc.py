"""4Pi-SIMFLUX raw data I/O: read-only memory-mapped Hamamatsu dcimg reader, channel crops and mirroring, file lists.

Frame record layout (format 0x2000000, one session; verified on these files): header word 9 = number of frames,
data start = session0 offset + offset_to_data (= 5632 here), each frame = 200 x 2304 uint16 pixels followed by a footer
(bytes_per_frame - bytes_per_image = 512 bytes): footer word 0 = camera frame counter, words 1-2 = Unix time (s, us).

Channels: 4 crop windows of 360 x 200 px at x0 = 217, 652, 1420, 1855 (the stored crop windows of the data set);
channels 2 and 3 (0-based) are mirrored left-right so that all four overlay. Channel c pixel (row y, col x) in the
mirrored crop = raw column X0[c] + x (c = 0, 1) or X0[c] + 359 - x (c = 2, 3).
"""
import glob, os
import numpy as np

ROOT = '/mnt/data/4pi-sf'
X0 = (217, 652, 1420, 1855)
CW, CH = 360, 200
MIRROR = (False, False, True, True)
FRAME_DT_S = 5.575e-3


class Dcimg:
    def __init__(self, fn):
        self.fn = fn
        h = np.fromfile(fn, dtype='<u4', count=200)
        self.version = int(h[2]); self.nframes = int(h[9]); s0 = int(h[10])
        with open(fn, 'rb') as f:
            f.seek(s0); s = np.fromfile(f, dtype='<u4', count=50)
        assert self.version in (0x1000000, 0x2000000), hex(self.version)
        self.ncol, self.nrow, self.bpr, self.bpi = int(s[18]), int(s[19]), int(s[20]), int(s[21])
        self.bpf = int(s[29]); self.off = s0 + int(s[24])
        assert self.bpr == 2 * self.ncol
        self.mm = np.memmap(fn, dtype=np.uint8, mode='r', offset=self.off, shape=(self.nframes, self.bpf))

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

    def footer(self, i, nwords=5):
        return np.asarray(self.mm[i, self.bpi:self.bpi + 4 * nwords]).view('<u4').copy()

    def time_s(self, i):
        w = self.footer(i, 3)
        return w[1] + 1e-6 * w[2]


def crops(a):
    """raw frames [..., 200, 2304] -> [..., 4, 200, 360] (float32, channel 2/3 mirrored)"""
    out = []
    for c in range(4):
        x = a[..., X0[c]:X0[c] + CW]
        out.append(x[..., ::-1] if MIRROR[c] else x)
    return np.stack(out, -3).astype(np.float32)


def movie_files(cell):
    if cell == 'MT':
        return sorted(glob.glob(f'{ROOT}/MT/Cell03_MT/Cell03_561_000_*.dcimg'))
    if cell == 'NPC':
        return sorted(glob.glob(f'{ROOT}/NPC/movies/Cell07_561_000_*.dcimg'))
    raise ValueError(cell)


def file_index(fn):
    return int(os.path.basename(fn).split('_')[-1].split('.')[0])


def zstack_files(cell):
    if cell == 'MT':
        return sorted(glob.glob(f'{ROOT}/MT/Cell02_beads_z_stack/Cell02_561_*_000.dcimg'))
    return sorted(glob.glob(f'{ROOT}/NPC/zstack/Cell06_561_*_000.dcimg'))


def bead_align_file(cell):
    return (f'{ROOT}/MT/Cell01_beads_align/Cell01_561_000_000.dcimg' if cell == 'MT'
            else f'{ROOT}/NPC/beads_align/Cell05_561_000_000.dcimg')


def zstack(cell):
    """bead z-stack -> [61, 4, 200, 360] ADU (one frame per stage step, 20 nm steps; stage z = (k - 30) * 0.02 um)"""
    fs = zstack_files(cell)
    return np.stack([crops(Dcimg(f).raw(0, 1)[0]) for f in fs])
