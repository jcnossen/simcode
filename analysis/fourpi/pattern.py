"""Step 5, pattern: per-emitter modulation depth and phase from the 6 per-frame (channel-summed) amplitudes, global
k / step phases per angle, coherence against a plane-wave pattern, and the tests of WHY the coherence is low.

Per emitter e, angle a (frames 3a .. 3a+2 of the window): I_j = B + C cos psi_j + S sin psi_j (3 equations, exact),
phi_e = atan2(S, C) = the pattern phase seen by the emitter, m_e = sqrt(C^2 + S^2) / B.
Plane-wave model: phi_e = k_a . r_e (the global offset is absorbed in psi_j). Residual delta_e = wrap(phi_e - k_a . r_e).
Coherence R = |mean exp(i delta_e)|. Noise-limited R_noise: same emitters, simulated I_j with the fitted depth and the
amplitude-fit errors (pattern exactly a plane wave), same estimator. R / R_noise = coherence of the pattern model itself.
Global k, psi_j: maximise sum_j |Z_j|^2, Z_j = <(I_j / mean - 1) exp(-i k . r)>, psi_j = -arg Z_j.
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
import numpy as np

TWO_PI = 2 * np.pi


def wrap(a):
    return (a + np.pi) % TWO_PI - np.pi


def cmean(a, w=None):
    e = np.exp(1j * a)
    return np.average(e, weights=w) if w is not None else e.mean()


from simcode.processing.fourpi_pattern_phase import slot_z, refine_k, emitter_phase  # noqa (the estimator lives in the package)


def emitter_phase_err(I3, sI3, psi, nsim=1, rng=None):
    """phase error sd by propagation (numerical): returns sd of phi from nsim noise draws"""
    rng = rng or np.random.default_rng(0)
    phi0, _, _ = emitter_phase(I3, psi)
    d = []
    for _ in range(nsim):
        p, _, _ = emitter_phase(I3 + rng.normal(size=I3.shape) * sI3, psi)
        d.append(wrap(p - phi0))
    return np.sqrt(np.mean(np.square(d), 0))


class AnglePattern:
    def __init__(self, I3, sI3, X, k0):
        self.k = refine_k(I3, X, k0)
        Z = slot_z(I3, X, self.k)
        self.Z = Z; self.psi = -np.angle(Z); self.m_slot = 2 * np.abs(Z) / 2      # |Z_j| = m * R / 1 (see docstring)
        self.I3, self.sI3, self.X = I3, sI3, X
        self.phi, self.m, self.B = emitter_phase(I3, self.psi)
        self.delta = wrap(self.phi - X @ self.k)

    def coherence(self, sel=None, delta=None):
        d = self.delta if delta is None else delta
        s = np.ones(len(d), bool) if sel is None else sel
        return np.abs(cmean(d[s]))

    def depth(self, sel=None):
        """depth from projected modulation: m_eff = < m_e cos delta_e > / R_noise-free"""
        s = np.ones(len(self.m), bool) if sel is None else sel
        return np.nanmedian(self.m[s]), np.nanmean(self.m[s] * np.cos(self.delta[s]))

    def noise_R(self, m_true, sel=None, nsim=3, seed=0):
        rng = np.random.default_rng(seed)
        s = np.ones(len(self.m), bool) if sel is None else sel
        B = self.I3[s].mean(1)
        phi_true = self.X[s] @ self.k
        Rs = []
        for _ in range(nsim):
            I = B[:, None] * (1 + m_true * np.cos(phi_true[:, None] - self.psi[None])) + rng.normal(size=(s.sum(), 3)) * self.sI3[s]
            p, _, _ = emitter_phase(I, self.psi)
            Rs.append(np.abs(cmean(wrap(p - phi_true))))
        return float(np.mean(Rs))
