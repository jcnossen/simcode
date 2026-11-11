"""Step 5, 4Pi: per-emitter interference phase from the 4 channel amplitudes.
f_c = A_c / sum_c A_c; with f_c = tau_c (1 + V cos(Phi + phi_c)) and phi_c = (0, 270, 180, 90) deg (sum exp(-2i phi_c) = 0):
Q = 1/2 sum_c (f_c / tau_c - 1) exp(-i phi_c) = V exp(i Phi). tau_c = population mean fractions, phi_c from the bead z-stack.
"""
import numpy as np
from pattern import wrap, cmean


def chan_phase(A, tau, phc):
    """A [..., 4] amplitudes -> Phi, V"""
    f = A / A.sum(-1, keepdims=True)
    Q = 0.5 * ((f / tau - 1) * np.exp(-1j * phc)).sum(-1)
    return np.angle(Q), np.abs(Q), Q


def chan_phase_err(A, sA, tau, phc, nsim=20, seed=0):
    rng = np.random.default_rng(seed)
    p0, _, _ = chan_phase(A, tau, phc)
    d = [wrap(chan_phase(A + rng.normal(size=A.shape) * sA, tau, phc)[0] - p0) for _ in range(nsim)]
    return np.sqrt(np.mean(np.square(d), 0))


def fit_period(Phi, z, Lams=np.arange(0.15, 0.40, 0.001), w=None):
    """maximise |<exp(i (Phi - s 2 pi z / Lam))>| over Lam and s = +-1. Returns Lam, s, Phi0, R, curve"""
    best = (0, None, None, None); curve = []
    for s in (1, -1):
        for L in Lams:
            c = cmean(Phi - s * 2 * np.pi * z / L, w)
            curve.append((s, L, abs(c)))
            if abs(c) > best[0]: best = (abs(c), L, s, np.angle(c))
    return best[1], best[2], best[3], best[0], np.array(curve)


def bead_channel_fit(z, I):
    """bead z-stack photometry I [nz,4] -> common period, phases, visibility envelope (Gaussian), fractions"""
    from scipy.optimize import least_squares
    f = I / I.sum(1, keepdims=True)
    def model(p):
        L, z0, w, tz = p[:4]; tau = p[4:8]; ph = p[8:12]; V = p[12]
        env = V * np.exp(-0.5 * ((z - tz) / w) ** 2)
        return tau * (1 + env[:, None] * np.cos(2 * np.pi * z[:, None] / L + ph))
    best = None
    for L0 in np.arange(0.18, 0.32, 0.01):
        for p00 in (0, 1.5, 3, 4.5):
            p0 = [L0, 0, 0.3, 0] + [0.25] * 4 + [p00, p00 + 3 * np.pi / 2, p00 + np.pi, p00 + np.pi / 2] + [0.6]
            r = least_squares(lambda p: (model(p) - f).ravel(), p0,
                              bounds=([0.1, -1, 0.05, -0.5] + [0.05] * 4 + [-20] * 4 + [0], [0.5, 1, 2, 0.5] + [0.6] * 4 + [20] * 4 + [1]))
            if best is None or r.cost < best.cost: best = r
    p = best.x
    ph = np.mod(p[8:12] - p[8], 2 * np.pi)
    return dict(Lam=p[0], env_w=p[2], env_z=p[3], tau=p[4:8].tolist(), phc=ph.tolist(), phc_abs=p[8:12].tolist(), V=p[12],
                rms=float(np.sqrt(np.mean(best.fun ** 2))))
