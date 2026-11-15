"""simcode.evaluation.frc.frc on the GPU with optional per-row weights (bootstrap) and a fixed render grid, for the many FRCs of
the referee checks. Same estimator: histogram on px-nm pixels (rows = y, columns = x, grid floor(min)..ceil(max) camera px of
the reference set, n even), FFT, integer rings around the centre, 5-ring moving average, resolution = n px / (first ring >= 3
below 1/7). check() asserts equality with the reference implementation."""
import numpy as np, torch

PX_NM = 108.33


class FRC:
    def __init__(self, x, y, px=5.0, dev='cuda'):
        s = PX_NM / px; self.s = s; self.px = px; self.dev = dev
        self.lo = float(np.floor(min(x.min(), y.min()))); hi = float(np.ceil(max(x.max(), y.max())))
        n = int((hi - self.lo) * s); n += n % 2; self.n = n
        yy, xx = torch.meshgrid(torch.arange(n, device=dev) - n // 2, torch.arange(n, device=dev) - n // 2, indexing='ij')
        self.rr = torch.hypot(xx.double(), yy.double()).long().ravel(); self.nr = int(self.rr.max()) + 1
        self.k5 = torch.ones(1, 1, 5, dtype=torch.float64, device=dev) / 5

    def img(self, x, y, w=None):
        n = self.n; s = self.s
        # np.histogram2d(y, x, bins=n, range=[[lo, lo + n / s]] * 2): bin = floor((v - lo) * s), last edge inclusive
        bx = np.floor((x - self.lo) * s).astype(np.int64); by = np.floor((y - self.lo) * s).astype(np.int64)
        bx[bx == n] = n - 1; by[by == n] = n - 1
        ok = (bx >= 0) & (bx < n) & (by >= 0) & (by < n)
        idx = torch.as_tensor(by[ok] * n + bx[ok], device=self.dev)
        wt = torch.ones(len(idx), dtype=torch.float64, device=self.dev) if w is None else torch.as_tensor(np.asarray(w, np.float64)[ok], device=self.dev)
        h = torch.zeros(n * n, dtype=torch.float64, device=self.dev); h.index_add_(0, idx, wt)
        return h.view(n, n)

    def curve_imgs(self, ha, hb):
        A = torch.fft.fftshift(torch.fft.fft2(ha)); B = torch.fft.fftshift(torch.fft.fft2(hb))
        num = torch.bincount(self.rr, (A * B.conj()).real.ravel(), self.nr)
        da = torch.bincount(self.rr, (A.abs() ** 2).ravel(), self.nr); db = torch.bincount(self.rr, (B.abs() ** 2).ravel(), self.nr)
        c = num / torch.sqrt(da * db + 1e-12)
        c = torch.nn.functional.conv1d(c[None, None], self.k5, padding=2)[0, 0]
        return c.cpu().numpy()

    def res(self, c):
        n = self.n; below = np.nonzero(c[3:n // 2] < 1 / 7)[0]
        return float('nan') if len(below) == 0 else float(n * self.px / (below[0] + 3))

    def __call__(self, xa, ya, xb, yb, wa=None, wb=None, curve=False):
        c = self.curve_imgs(self.img(xa, ya, wa), self.img(xb, yb, wb))
        r = self.res(c)
        return (r, np.arange(self.n // 2) / (self.n * self.px), c[:self.n // 2]) if curve else r

    def halves(self, x, y, f, blk=300, w=None, curve=False, assign=None):
        """assign: optional bool per block index (True = half A) instead of the alternating blocks"""
        b = f // blk
        h = (b % 2 == 0) if assign is None else assign[b]
        wa = None if w is None else w[h]; wb = None if w is None else w[~h]
        return self(x[h], y[h], x[~h], y[~h], wa, wb, curve=curve)


def check(x, y, f, fr=None):
    from simcode.evaluation.frc import frc_halves
    fr = fr or FRC(x, y)
    a = fr.halves(x, y, f); b = frc_halves(x, y, f)
    assert abs(a - b) < 1e-9, (a, b)
    return a
