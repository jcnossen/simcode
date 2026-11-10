"""Simple RCC drift estimate (time blocks, rendered at 20 nm, FFT cross-correlation vs the all-frame image, parabolic
sub-pixel peak, iterated twice); returns drift per frame in px."""
import numpy as np
def render(x, y, lim, px):
    a, b, c, d = lim
    H, _, _ = np.histogram2d(y, x, bins=[int((d - c) / px), int((b - a) / px)], range=[[c, d], [a, b]]); return H
def xcorr_shift(A, B):
    F = np.fft.fft2(A - A.mean()) * np.conj(np.fft.fft2(B - B.mean())); C = np.fft.fftshift(np.real(np.fft.ifft2(F)))
    iy, ix = np.unravel_index(np.argmax(C), C.shape)
    def sub(c_m, c_0, c_p): d = c_m - 2 * c_0 + c_p; return 0.0 if d == 0 else 0.5 * (c_m - c_p) / d
    dy = sub(C[iy - 1, ix], C[iy, ix], C[iy + 1, ix]); dx = sub(C[iy, ix - 1], C[iy, ix], C[iy, ix + 1])
    return (iy + dy - C.shape[0] // 2, ix + dx - C.shape[1] // 2)
def rcc(x, y, fr, nframes, nblk=15, lim=(5, 115, 5, 115), px=0.2, iters=2):
    dxf = np.zeros(nframes); dyf = np.zeros(nframes)
    edges = np.linspace(0, nframes, nblk + 1); cen = 0.5 * (edges[1:] + edges[:-1])
    for _ in range(iters):
        xc = x - dxf[fr]; yc = y - dyf[fr]
        ref = render(xc, yc, lim, px); sh = []
        for i in range(nblk):
            m = (fr >= edges[i]) & (fr < edges[i + 1])
            B = render(x[m], y[m], lim, px); sy, sx = xcorr_shift(B, ref); sh.append((sx * px, sy * px))
        sh = np.array(sh); sh -= sh.mean(0)
        dxf = np.interp(np.arange(nframes), cen, sh[:, 0]); dyf = np.interp(np.arange(nframes), cen, sh[:, 1])
    return dxf, dyf, cen, sh
