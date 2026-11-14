"""S0 CRLB check for ZIMFLUX (DESIGN.md section 6), isolated emitter, vectorial astigmatic PSF (simcode.psf.vectorial).

Bounds per emitter (x, y, z, N, b; b = background per pixel per frame, shared):
 (a) astig  : astigmatic SMLM on the summed K-frame image, mu_j = N H_j + K b                       (paper eq. 28)
 (b) zimflux: all pixels of the K frames, mu_kj = N w_k(r) H_j + b,
              w_k = eta_k (1 + m cos(q_k . r_xy + k_z z_e - psi_k)), sum eta = 1                    (paper eq. 1, 13, 23)
 (c) fact   : the network pipeline. Pass 1 = the pattern-free model mu_kj = N_k H_j + b (per-frame amplitudes free)
              -> T = (x, y, z, N_1..N_K) with covariance C (b marginalised). The explicit solve fits
              (x, y, z, N) to T with W = blockdiag(C_rr^-1, C_NN^-1) (3D prior and intensities treated as independent, as
              DESIGN 4.2); its covariance is the sandwich A^-1 J' W C W J A^-1, A = J' W J. fact_joint uses W = C^-1 (the
              exact factorisation, must equal (b): a check).
H is normalised to sum 1 over the 16 x 16 ROI (N = photons in the ROI). z_e = emitter height above the coverslip; the
focal plane is at the Strehl maximum of an aberration-free emitter at the imaging depth z_d (as zimflux config.set_saffocus);
z (paper) = z_e - z_d. CRLB values are averaged over the pattern phase (uniform grid) as sqrt(mean variance).

usage: python crlb_s0.py [gpu]  -> out/zimflux/crlb/crlb_s0.json + crlb_s0.txt"""
import sys, json, math, os, numpy as np, torch
sys.path.insert(0, '/home/jc/simcode2')
from simcode.psf.vectorial import VectorialPSF, noll_to_nm
dev = torch.device(f'cuda:{sys.argv[1]}' if len(sys.argv) > 1 and sys.argv[1].isdigit() else 'cpu')
OUT = '/home/jc/simcode2/out/zimflux/crlb/'; os.makedirs(OUT, exist_ok=True)
PX, LAM, ROI = 0.065, 0.715, 16
NOLL = {noll_to_nm(j): j for j in range(1, 40)}
NPHASE = 12
NORM = os.environ.get('ZF_NORM', 'roi')   # 'roi': H sums to 1 over the ROI; 'native': simcode normalisation (1 = whole plane, aberration-free emitter at the coverslip)


def make_psf(zern):
    """zern: dict noll -> waves rms."""
    idx = sorted(zern) if zern else [6]
    p = VectorialPSF(ROI, pixelsize=PX, NA=1.49, n_imm=1.52, n_med=1.33, wavelength=LAM, zernike_indices=idx,
                     npupil=64, oversample=3, device=dev)
    with torch.no_grad():
        p.zcoef[:] = torch.tensor([zern.get(j, 0.0) for j in idx], device=dev)
    return p


def focus_for_depth(zd):
    p0 = make_psf({})
    fs = torch.linspace(zd - 0.2, zd * 1.6 + 0.4, 241, device=dev)
    with torch.no_grad():
        I = p0(torch.tensor([[0.0, 0.0, zd]], device=dev).expand(len(fs), 3).contiguous(), fs)
    pk = I.reshape(len(fs), -1).max(1).values
    return float(fs[pk.argmax()])


def psf_derivs(psf, ze, focus, eps=(0.01, 0.01, 0.001)):
    """H [B,J] normalised over the ROI and dH/d(x, y, z) [B,3,J], x / y in um, z in um (central differences)."""
    B = len(ze)
    pos = torch.zeros(B, 3, device=dev); pos[:, 2] = torch.as_tensor(ze, device=dev, dtype=torch.float32)
    fo = torch.full((B,), focus, device=dev)

    def H(p):
        with torch.no_grad():
            h = psf(p, fo).reshape(B, -1).double()
        return h / h.sum(1, keepdim=True) if NORM == 'roi' else h
    h0 = H(pos); d = []
    for i, e in enumerate(eps):
        dp = torch.zeros_like(pos); dp[:, i] = e
        g = (H(pos + dp) - H(pos - dp)) / (2 * e)
        d.append(g / PX if i < 2 else g)        # x, y: px -> um
    return h0.cpu().numpy(), torch.stack(d, 1).cpu().numpy()


def bounds(h, dh, ze, N, b, frames):
    """h [J], dh [3,J]; frames: list of (qx, qy, kz [rad/um], psi, eta, m). Returns dict of CRLB std (nm) for x, y, z."""
    K = len(frames)
    r = np.array([0.0, 0.0, ze])
    w = np.array([e * (1 + m * np.cos(qx * r[0] + qy * r[1] + kz * r[2] - ps)) for qx, qy, kz, ps, e, m in frames])
    dw = np.array([[-e * m * np.sin(qx * r[0] + qy * r[1] + kz * r[2] - ps) * c for c in (qx, qy, kz)]
                   for qx, qy, kz, ps, e, m in frames])                      # [K,3]
    res = {}
    # (a) summed image
    mu = N * h + K * b
    D = np.concatenate([N * dh, h[None], np.full((1, len(h)), K)], 0)       # [5,J]
    F = (D / mu) @ D.T
    res['astig'] = np.linalg.inv(F)
    # (b) full pixels
    F = np.zeros((5, 5))
    for k in range(K):
        mu = N * w[k] * h + b
        D = np.concatenate([N * (dw[k][:, None] * h[None] + w[k] * dh), (w[k] * h)[None], np.ones((1, len(h)))], 0)
        F += (D / mu) @ D.T
    res['zimflux'] = np.linalg.inv(F)
    # (c) pattern-free per-frame model theta' = (x, y, z, N_1..N_K, b)
    P = 3 + K + 1; Fp = np.zeros((P, P))
    Nk = N * w
    for k in range(K):
        mu = Nk[k] * h + b
        D = np.zeros((P, len(h))); D[:3] = Nk[k] * dh; D[3 + k] = h; D[-1] = 1
        Fp += (D / mu) @ D.T
    C = np.linalg.inv(Fp)[:3 + K, :3 + K]                                   # b marginalised
    J = np.zeros((3 + K, 4)); J[:3, :3] = np.eye(3)
    J[3:, :3] = N * dw; J[3:, 3] = w
    Wj = np.linalg.inv(C)
    res['fact_joint'] = np.linalg.inv(J.T @ Wj @ J)
    W = np.zeros_like(C); W[:3, :3] = np.linalg.inv(C[:3, :3]); W[3:, 3:] = np.linalg.inv(C[3:, 3:])
    Ai = np.linalg.inv(J.T @ W @ J)
    res['fact'] = Ai @ J.T @ W @ C @ W @ J @ Ai
    res['prior_only'] = C[:3, :3]
    return {k: np.diag(v)[:3] for k, v in res.items()}                       # variances (um^2)


def run(zern, zd, zrel_nm, N, b, pattern, label):
    """pattern(psi0) -> frames. Returns per z: sqrt(mean over phases of variance) in nm for each bound."""
    psf = make_psf(zern); f = focus_for_depth(zd)
    ze = zd + np.asarray(zrel_nm) / 1000
    h, dh = psf_derivs(psf, ze, f)
    out = {}
    for i, z in enumerate(ze):
        acc = {}
        for ps in np.arange(NPHASE) * 2 * np.pi / NPHASE:
            v = bounds(h[i], dh[i], z, N, b, pattern(ps))
            for k, a in v.items(): acc.setdefault(k, []).append(a)
        for k, a in acc.items(): out.setdefault(k, []).append((np.sqrt(np.mean(a, 0)) * 1000).tolist())
    return {'label': label, 'focus_um': f, 'zrel_nm': list(map(float, zrel_nm)), 'N': N, 'b_per_frame': b,
            'crlb_nm_xyz': out}


def one_orient(plat, pax, ang_deg, m, K=3, eta=None):
    q = 2 * np.pi / plat; kz = 2 * np.pi / pax; a = np.deg2rad(ang_deg)
    eta = eta or [1 / K] * K
    return lambda p0: [(q * np.cos(a), q * np.sin(a), kz, p0 + 2 * np.pi * k / K, eta[k], m) for k in range(K)]


def mean_over(r):   # mean CRLB over the z grid (paper Fig. 6: emitters uniform over the z range), per bound and axis
    return {k: np.mean(v, 0).round(2).tolist() for k, v in r['crlb_nm_xyz'].items()}


if __name__ == '__main__':
    R = {}
    zr = np.arange(-300, 301, 50)
    # 1. paper Fig. 5b/c: z_d 300 nm, astigmatism only, N 2800, b 8 total over 3 frames, m 0.85, p 452 / 492
    for ast in (-30, -60, -90):
        R[f'fig5_{ast}'] = run({6: ast / 1000}, 0.3, zr, 2800, 8 / 3, one_orient(0.452, 0.492, 42, 0.85), f'Fig5 Z22 {ast} mlambda')
        print(R[f'fig5_{ast}']['label'], {k: np.array(v)[:, 2].round(1).tolist() for k, v in R[f'fig5_{ast}']['crlb_nm_xyz'].items()}, flush=True)
    # 2. paper Fig. 6b/d: mean over z in [-300, 300], p_ax 500..1500 with a1 = 0 geometry (p_lat = l0 / (n0 sin b1))
    zr6 = np.arange(-300, 301, 25)
    for ast in (-30, -60, -90):
        for pax in (0.5, 0.75, 1.0, 1.25, 1.5):
            cb = 1 - 0.642 / 1.33 / pax; plat = 0.642 / 1.33 / math.sqrt(1 - cb ** 2)
            r = run({6: ast / 1000}, 0.3, zr6, 2800, 8 / 3, one_orient(plat, pax, 42, 0.85), f'Fig6 Z22 {ast} pax {pax}')
            r['plat'] = plat; r['mean'] = mean_over(r)
            r['improvement_z'] = r['mean']['astig'][2] / r['mean']['zimflux'][2]
            R[f'fig6_{ast}_{pax}'] = r
            print(r['label'], 'plat %.3f' % plat, {k: v[2] for k, v in r['mean'].items()}, 'impr %.2f' % r['improvement_z'], flush=True)
    json.dump(R, open(OUT + 'crlb_paperfigs.json', 'w'), indent=1)
