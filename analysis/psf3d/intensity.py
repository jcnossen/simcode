"""(2026-11-11 in-situ 3D PSF) Intensity side of the per-axis PSF: what fitting with the AVERAGED PSF does when the data have the
per-axis (per pattern angle) PSF, for per-frame intensities and for a SIMFLUX joint fit.  Expected (noise-free) data, Poisson MLE.
 1. per-frame intensity CRLB (position known; I_t and bg free) on 8x8 and 11x11 ROIs, per angle: per-axis truth vs average
 2. per-frame intensity bias of the averaged-PSF fit (position fixed at truth, I and bg free), per angle, over sub-pixel position,
    z and ROI; split into the common per-angle factor (mean over the grid; absorbed by a per-angle relint) and the residual;
    residual expressed as a modulation error: phase (-> nm via the pitch) and depth from the 3 frames of an angle
 3. SIMFLUX (joint 6-frame, known pattern: k, phases, depth 0.925 / 0.955, relint from the calib): CRLB(x, y) per-axis vs average;
    position bias of the averaged-PSF fit to per-axis data, with one amplitude I (relint fixed) or one amplitude per angle (A_a)
usage: python analysis/psf3d/intensity.py <fit _psf.pt (blps3)> <calib json (pattern)> <out prefix>"""
import sys, json, math
import numpy as np, torch
from simcode.psf.vectorial import VectorialPSF

import os
dev = os.environ.get('PSF3D_DEV', 'cuda')
fit, calib, outp = sys.argv[1:4]
S = torch.load(fit, weights_only=False); ex = S['extras']; focus = float(S['focus'])
P = json.load(open(calib))['pattern']
K = np.array(P['k_rad_per_px']); PHI = np.array(P['phase']); DEP = np.array(P['depth']); REL = np.array(P['relint'])
KW = [dict(dipM=torch.tensor(ex['dipole_M'][a], device=dev), blur=torch.tensor(ex['blur_px2'][0], device=dev)) for a in range(2)]
KWM = dict(dipM=torch.tensor(ex['dipole_M'], device=dev).mean(0), blur=torch.tensor(ex['blur_px2'][0], device=dev))
MODELS = {}


def model(R):
    if R not in MODELS:
        m = VectorialPSF(R, oversample=3, NA=float(S['NA']), wavelength=0.70, n_med=1.33, device=dev)
        with torch.no_grad(): m.zcoef.copy_(S['zcoef'].to(dev)); m.apod.copy_(S['apod'].to(dev))
        for p_ in m.parameters(): p_.requires_grad_(False)
        MODELS[R] = m
    return MODELS[R]


def psf(R, xy, z, kw, h=1e-3):
    """[3, R*R] float64: value, d/dx, d/dy at offset xy (px from the ROI centre)"""
    m = model(R)
    pos = torch.tensor([[xy[0], xy[1], z], [xy[0] + h, xy[1], z], [xy[0] - h, xy[1], z], [xy[0], xy[1] + h, z], [xy[0], xy[1] - h, z]], device=dev)
    with torch.no_grad():
        im = m(pos, focus, **kw).reshape(5, -1).double().cpu().numpy()
    return np.stack([im[0], (im[1] - im[2]) / (2 * h), (im[3] - im[4]) / (2 * h)])


SUB = [(0.0, 0.0), (-0.4, -0.4), (0.25, -0.1), (0.5, 0.5), (0.45, 0.0), (0.0, 0.45)]
ZS = [0.02, 0.06, 0.12]
ROIS = [8, 11]
off = {8: 0.5, 11: 0.0}            # 8 x 8: emitter pixel = the one right / below the geometric centre (SMLM ROI convention)
BG = 10.0
res = dict(fit=fit, calib=calib)


def mle_Ibg(d, g, it=50):
    """Poisson MLE of (I, bg) for expected data d with shape g (position fixed). Newton on (I, bg)."""
    I, b = max(d.sum() - BG * len(d), 1.0), BG
    for _ in range(it):
        m = I * g + b
        r = d / m - 1
        gr = np.array([(r * g).sum(), r.sum()])
        w = d / m ** 2
        H = np.array([[(w * g * g).sum(), (w * g).sum()], [(w * g).sum(), w.sum()]])
        st = np.linalg.solve(H, gr); I, b = I + st[0], b + st[1]
    return I, b


# ---------------------------------------------------------------- 1 + 2: per-frame intensity
rows = []
for R in ROIS:
    for z in ZS:
        for sx, sy in SUB:
            xy = (sx + off[R], sy + off[R])
            ga = [psf(R, xy, z, KW[a])[0] for a in range(2)]; gm = psf(R, xy, z, KWM)[0]
            for a in range(2):
                for I0 in (1000.0, 300.0):
                    d = I0 * ga[a] + BG
                    # CRLB of I (I, bg free, position known): truth per-axis vs truth = average
                    def crlb(g):
                        mu = I0 * g + BG; J = np.stack([g, np.ones_like(g)], 1)
                        return math.sqrt(np.linalg.inv(J.T @ (J / mu[:, None]))[0, 0])
                    Ih, bh = mle_Ibg(d, gm)
                    rows.append(dict(R=R, z=z, sx=sx, sy=sy, angle=a, I0=I0, crlb_perax=crlb(ga[a]), crlb_avg=crlb(gm),
                                     ratio=Ih / I0, bg_hat=bh))
res['frame'] = rows
print('== 1/2: per-frame intensity (position known, I + bg free), expected data')
for R in ROIS:
    for I0 in (1000.0, 300.0):
        for a in range(2):
            q = [r for r in rows if r['R'] == R and r['I0'] == I0 and r['angle'] == a]
            cr = np.array([r['crlb_perax'] / r['crlb_avg'] for r in q]); rt = np.array([r['ratio'] for r in q])
            print(f'ROI {R:2d} I {I0:5.0f} angle {a}: CRLB(I) per-axis / avg {cr.mean():.5f} [{cr.min():.5f}, {cr.max():.5f}] | '
                  f'I_hat / I (avg-PSF fit): common {rt.mean():.5f}, residual range [{(rt - rt.mean()).min():+.5f}, {(rt - rt.mean()).max():+.5f}] sd {rt.std():.5f}')

# modulation error from the 3 frames of an angle: frame intensities I_k = I0 rel_k (1 + m cos(theta - 2 pi k / 3)) (k = step);
# each frame fitted alone with the averaged PSF (bias depends on I_k through the Poisson weights); then (A, m, theta) from the 3 Ihat / rel
mod = []
for R in ROIS:
    for z in ZS:
        for sx, sy in SUB:
            xy = (sx + off[R], sy + off[R])
            ga = [psf(R, xy, z, KW[a])[0] for a in range(2)]; gm = psf(R, xy, z, KWM)[0]
            for a in range(2):
                for I0 in (1000.0, 300.0):
                    for th in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                        st = [3 * a + k for k in range(3)]
                        ph = th - PHI[st] + PHI[st[0]]                       # pattern phase of each step at the emitter
                        It = I0 * REL[st] * (1 + DEP[st] * np.cos(ph))
                        Ih = np.array([mle_Ibg(It[k] * ga[a] + BG, gm)[0] for k in range(3)])
                        # linear 3-step demodulation (relint known): y_k = Ih_k / rel_k = c0 + c1 cos(ph_k) + c2 sin(ph_k)
                        M = np.stack([np.ones(3), np.cos(ph - th), -np.sin(ph - th)], 1)       # in the frame of th
                        c = np.linalg.solve(M, Ih / REL[st]); c_true = np.linalg.solve(M, It / REL[st])
                        dphi = (math.atan2(c[2], c[1]) - math.atan2(c_true[2], c_true[1]) + math.pi) % (2 * math.pi) - math.pi
                        dm = math.hypot(c[1], c[2]) / c[0] - math.hypot(c_true[1], c_true[2]) / c_true[0]
                        mod.append(dict(R=R, z=z, sx=sx, sy=sy, angle=a, I0=I0, theta=float(th), dphi=dphi,
                                        dpos_nm=dphi / np.linalg.norm(K[a]) * 108.33, ddepth=dm, amp_ratio=c[0] / c_true[0]))
res['modulation'] = mod
print('== 2b: modulation error of the 3-frame demodulation (averaged-PSF intensities, relint known)')
for R in ROIS:
    for I0 in (1000.0, 300.0):
        q = [r for r in mod if r['R'] == R and r['I0'] == I0]
        dp = np.array([r['dpos_nm'] for r in q]); dd = np.array([r['ddepth'] for r in q]); ar = np.array([r['amp_ratio'] for r in q])
        print(f'ROI {R:2d} I {I0:5.0f}: phase error as position: max |{np.abs(dp).max():.4f}| nm, rms {np.sqrt((dp ** 2).mean()):.4f} nm | '
              f'depth error max |{np.abs(dd).max():.5f}| | amplitude ratio range [{ar.min():.5f}, {ar.max():.5f}]')

# ---------------------------------------------------------------- 3: SIMFLUX joint fit
def simflux_model(R, r0, xy, z, kw_frames, A, bg, I0):
    """expected 6 frames [6, R*R] and Jacobian wrt (x, y, A..., bg0, bg1); A = [I] or [A0, A1]"""
    out, J = [], []
    for t in range(6):
        a = t // 3
        p = psf(R, xy, z, kw_frames[a])
        X = r0 + np.array(xy)                                                   # emitter position in the pattern frame (px)
        arg = K[a] @ X - PHI[t]
        il = REL[t] * (1 + DEP[t] * math.cos(arg)); dil = -REL[t] * DEP[t] * math.sin(arg) * K[a]
        Aa = A[0] if len(A) == 1 else A[a]
        mu = Aa * il * p[0] + bg[a]
        jx = Aa * (dil[0] * p[0] + il * p[1]); jy = Aa * (dil[1] * p[0] + il * p[2])
        jA = [il * p[0] if (len(A) == 1 or k == a) else np.zeros_like(p[0]) for k in range(len(A))]
        jb = [np.ones_like(p[0]) if k == a else np.zeros_like(p[0]) for k in range(2)]
        out.append(mu); J.append(np.stack([jx, jy] + jA + jb, 1))
    return np.concatenate(out), np.concatenate(J)


def sf_crlb(R, r0, xy, z, kws, I0, nA):
    mu, J = simflux_model(R, r0, xy, z, kws, [I0] * nA, [BG, BG], I0)
    return np.sqrt(np.diag(np.linalg.inv(J.T @ (J / mu[:, None])))[:2]) * 108.33


def sf_fit(R, r0, xy, z, d, nA, I0, it=30):
    p = np.array(list(xy) + [I0] * nA + [BG, BG], float)
    for _ in range(it):
        mu, J = simflux_model(R, r0, (p[0], p[1]), z, [KWM, KWM], p[2:2 + nA], p[2 + nA:], I0)
        r = d / mu - 1; w = d / mu ** 2
        st = np.linalg.solve(J.T @ (J * w[:, None]), J.T @ r); p = p + st
        if np.abs(st[:2]).max() < 1e-9: break
    return p


sf = []
for R in ROIS:
    for z in ZS:
        for sx, sy in SUB:
            xy = (sx + off[R], sy + off[R])
            for th in np.linspace(0, 1, 4, endpoint=False):                     # pattern offsets: emitter shifted by a fraction of the period
                r0 = th * 2 * np.pi * (K[0] + K[1]) / (np.linalg.norm(K[0] + K[1]) ** 2)
                for I0 in (1000.0, 300.0):
                    d, _ = simflux_model(R, r0, xy, z, KW, [I0], [BG, BG], I0)
                    row = dict(R=R, z=z, sx=sx, sy=sy, offset=float(th), I0=I0)
                    row['crlb_perax_nm'] = sf_crlb(R, r0, xy, z, KW, I0, 1).tolist(); row['crlb_avg_nm'] = sf_crlb(R, r0, xy, z, [KWM, KWM], I0, 1).tolist()
                    for nA in (1, 2):
                        p = sf_fit(R, r0, xy, z, d, nA, I0)
                        row[f'bias_nA{nA}_nm'] = ((p[:2] - np.array(xy)) * 108.33).tolist()
                        if nA == 2: row['A_ratio'] = (p[3] / p[2])
                    sf.append(row)
res['simflux'] = sf
print('== 3: SIMFLUX (joint 6-frame, known pattern)')
for R in ROIS:
    for I0 in (1000.0, 300.0):
        q = [r for r in sf if r['R'] == R and r['I0'] == I0]
        cr = np.array([np.array(r['crlb_perax_nm']) / np.array(r['crlb_avg_nm']) for r in q])
        b1 = np.array([r['bias_nA1_nm'] for r in q]); b2 = np.array([r['bias_nA2_nm'] for r in q]); c0 = np.array([r['crlb_perax_nm'] for r in q])
        print(f'ROI {R:2d} I {I0:5.0f}: CRLB x/y (per-axis) median {np.median(c0, 0).round(3)} nm; per-axis / avg {cr.mean(0).round(5)} '
              f'[{cr.min():.5f}, {cr.max():.5f}] | bias avg-PSF fit, one amplitude: max |x| {np.abs(b1[:, 0]).max():.3f} |y| {np.abs(b1[:, 1]).max():.3f} nm '
              f'(rms {np.sqrt((b1 ** 2).mean(0)).round(3)}) | per-angle amplitude: max |x| {np.abs(b2[:, 0]).max():.3f} |y| {np.abs(b2[:, 1]).max():.3f} nm '
              f'(rms {np.sqrt((b2 ** 2).mean(0)).round(3)}), A1/A0 {np.mean([r["A_ratio"] for r in q]):.4f}')
json.dump(res, open(outp + '.json', 'w'), indent=1, default=float)
