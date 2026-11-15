"""Per-file SIM pattern and 4Pi phase field of the 4Pi pipelines (network and classical), estimated in situ from the
candidates themselves (no calibration sample):

  SIM pattern   slot_z / refine_k / emitter_phase (per-emitter 3-step solve; global k and step phases from the slot DFT),
                repattern_xy (pattern of a file from per-frame intensities at given positions),
                pattern_psi (per-candidate, per-frame pattern phase and depth in local patch coordinates)
  4Pi phase     phase_field: Phi0(x, y, t) = tile map M(x, y) + block drift d(t), by alternating circular means

Per emitter e, angle a (frames 3a .. 3a+2 of the window): I_j = B + C cos psi_j + S sin psi_j (3 equations, exact),
phi_e = atan2(S, C) = the pattern phase seen by the emitter, m_e = sqrt(C^2 + S^2) / B. Plane-wave model:
phi_e = k_a . r_e. Global k, psi_j: maximise sum_j |Z_j|^2, Z_j = <(I_j / mean - 1) exp(-i k . r)>, psi_j = -arg Z_j.
"""
import numpy as np


def slot_z(I3, X, k):
    m = I3.mean(1, keepdims=True)
    r = I3 / m - 1
    return 2 * (r * np.exp(-1j * (X @ k))[:, None]).mean(0)


def refine_k(I3, X, k0, span=0.05, n=41, rounds=2):
    k = np.array(k0, float)
    for rd in range(rounds):
        sp = span / (8 ** rd)
        best = (-1, k)
        for dx in np.linspace(-sp, sp, n):
            for dy in np.linspace(-sp, sp, n):
                kk = k + [dx, dy]
                s = (np.abs(slot_z(I3, X, kk)) ** 2).sum()
                if s > best[0]: best = (s, kk)
        k = best[1]
    return k


def emitter_phase(I3, psi):
    """I3 [n,3], psi [3] -> phi [n], m [n], B [n] (exact 3-step solve)"""
    M = np.stack([np.ones(3), np.cos(psi), np.sin(psi)], 1)       # [3,3]
    sol = I3 @ np.linalg.inv(M).T                                   # [n,3]: B, C, S
    B, C, S = sol.T
    return np.arctan2(S, C), np.hypot(C, S) / np.where(np.abs(B) > 1e-6, B, np.nan), B


def repattern_xy(I, X, ref, sel):
    """slot-DFT pattern estimate of a file from per-frame intensities I [n,6] at positions X [n,2] (rows sel): k refined
    around the reference ref['angle<a>']['k'], step phases psi from the slot DFT, depth = median per-emitter 3-step
    modulation, R = coherence of the per-emitter phases against the plane wave"""
    out = dict(n=int(sel.sum()))
    for a in range(2):
        I3 = I[sel, 3 * a:3 * a + 3]
        k = refine_k(I3, X[sel], np.array(ref[f'angle{a}']['k']), span=0.02, n=21)
        psi = -np.angle(slot_z(I3, X[sel], k)); ph, mm, _ = emitter_phase(I3, psi)
        out[f'angle{a}'] = dict(k=k.tolist(), psi_deg=np.degrees(psi).tolist(), m_median=float(np.nanmedian(np.clip(mm, 0, 1.5))),
                                 R=float(np.abs(np.mean(np.exp(1j * (ph - X[sel] @ k))))))
    return out


def pattern_psi(pat, cx, cy, dev):
    """per-candidate per-frame phases psi_eff [n,6] for local coordinates, depth m [n,6], main-axis k (2)"""
    import torch
    psi, m, kk = [], [], []
    for a in range(2):
        A = pat[f'angle{a}']
        k = np.array(A['k']); ps = np.radians(A['psi_deg'])
        kr = k[0] * cx + k[1] * cy
        for s in range(3):
            psi.append(ps[s] - kr); m.append(np.full(len(cx), A['m_median']))
        kk.append(k[a])
    return (torch.tensor(np.stack(psi, 1), device=dev, dtype=torch.float32),
            torch.tensor(np.stack(m, 1), device=dev, dtype=torch.float32), tuple(kk))


def phase_field(X, Y, W, phi, wgt, tile=12.0, msmooth=1.0, tblk=50, nit=4, H=200, Wd=360):
    """Phi0 = M(x, y) + d(t) by alternating circular means. Returns a function (x, y, w) -> Phi0 and diagnostics"""
    from scipy import ndimage as ndi
    tx = np.clip((X / tile).astype(int), 0, int(Wd / tile)); ty = np.clip((Y / tile).astype(int), 0, int(H / tile))
    nx, ny = int(Wd / tile) + 1, int(H / tile) + 1
    tb = (W // tblk).astype(int); nb = tb.max() + 1
    d = np.zeros(nb); M = np.zeros((ny, nx))
    z = np.exp(1j * phi) * wgt
    for _ in range(nit):
        r = z * np.exp(-1j * d[tb])
        G = np.zeros((ny, nx), complex); np.add.at(G, (ty, tx), r)
        G = ndi.gaussian_filter(G.real, msmooth) + 1j * ndi.gaussian_filter(G.imag, msmooth)
        M = np.angle(G)
        r = z * np.exp(-1j * M[ty, tx])
        B = np.zeros(nb, complex); np.add.at(B, tb, r)
        Bs = ndi.gaussian_filter1d(B.real, 1.0) + 1j * ndi.gaussian_filter1d(B.imag, 1.0)
        d = np.angle(Bs)
    res = np.angle(np.exp(1j * (phi - M[ty, tx] - d[tb])))
    R = float(np.abs(np.sum(wgt * np.exp(1j * res))) / np.sum(wgt))

    def f(x, y, w):
        ix = np.clip((x / tile).astype(int), 0, nx - 1); iy = np.clip((y / tile).astype(int), 0, ny - 1)
        return M[iy, ix] + d[np.clip(w // tblk, 0, nb - 1).astype(int)]
    return f, dict(R=R, M=M, d=d, tile=tile, tblk=tblk, res=res)
