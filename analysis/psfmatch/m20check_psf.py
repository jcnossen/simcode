"""m20check part 1 (2026-11-12): is the in-situ PSF of 20nM_2 the same as that of 2.5nM_2 (the PSF the m25 network set was trained on)?
PSFs = models/insitu/calib_<movie>_psf3d_avg.json (the angle-averaged in-situ PSF; m25 / m1809 were trained on these, runs/psfmatch/env.sh),
1809_2 as the third column for scale.  CPU only (no GPU use).
  A  model parameters side by side
  B  PSF images (15 x 15 px, unit sum, emitter centred, z = 0.055 um = emitter_z_median, and the TIRF-weighted z average), differences,
     rms pixel difference, widths per axis (elliptical-Gaussian fit), centroid
  C  CRLB of an isolated emitter, 6-frame window, parameters (x, y, N, bg), Poisson + read noise (mu + rn^2):
       SMLM    = the 6 frames summed (no pattern information)
       SIMFLUX = 6 frames, known pattern (pitch / depth of the calib pattern block, 3 phase steps per angle, random pattern phase per emitter)
  D  mismatch cost: data generated with the 20 nM PSF, maximum-likelihood fit (Levenberg-Marquardt, Poisson) with the 20 nM (control),
     2.5 nM and 1809 PSF models: bias, sd and rms error vs the CRLB of the true model.  SIMFLUX fits: with the true pattern phase, and
     ('gauge') with the pattern phase shifted by k . (SMLM bias of that PSF model), i.e. the phase a pattern estimate made from localizations
     of the same wrong PSF model would have (the pipeline estimates the pattern from its own localizations).
usage: python analysis/psfmatch/m20check_psf.py [n_per_level]  -> out/psfmatch/m20check/psf_*.json / .png / .txt"""
import json, math, os, sys, time
import numpy as np, torch
from scipy.optimize import least_squares
from simcode2 import config
from simcode2.util import proctitle
from simcode2.psf.vectorial import VectorialPSF, set_psf_extras

proctitle.set_title('m20check')
torch.set_num_threads(8)
OUT = os.path.join(config.root('out'), 'psfmatch', 'm20check'); os.makedirs(OUT, exist_ok=True)
MOV = ['20nM_m2', '2p5nM_m2', '1809_2']
LAB = {'20nM_m2': '20nM_2', '2p5nM_m2': '2.5nM_2 (m25)', '1809_2': '1809_2 (m1809)'}
PX = 108.33; R = 15; Z0 = 0.055
PHOT = [500., 1000., 2000., 4000., 8000.]     # photons per 6-frame window (20 nM v2 rows: median 870 all / 1750 sel / 3000 strict)
BG = 15.0                                     # photons / px / frame (20 nM SMLM baseline: median bg 88 per 6-frame sum)
dev = 'cpu'


def load(m):
    c = json.load(open(config.resolve(f'insitu/calib_{m}_psf3d_avg.json')))
    p = VectorialPSF(R, pixelsize=c['pixelsize_um'], NA=c['NA'], n_imm=c['n_imm'], n_med=c['n_med'], wavelength=c['wavelength_um'],
                     oversample=3, device=dev)
    with torch.no_grad():
        p.zcoef.copy_(torch.tensor(c['zernike_waves'])); p.apod.copy_(torch.tensor(c['apod']))
    set_psf_extras(p, c)
    return c, p


@torch.no_grad()
def render(p, c, x, y, z):
    """[n, R, R] PSF (float64) for emitters at (x, y) px from the ROI centre, height z um"""
    out = []
    pos = torch.stack([x, y, z], 1).float()
    for q in pos.split(500):
        out.append(p(q, torch.full((len(q),), c['focus_um'])).double())
    return torch.cat(out)


def psf_d(p, c, x, y, z, h=1e-3):
    """PSF and central-difference d/dx, d/dy"""
    f = render(p, c, x, y, z)
    fx = (render(p, c, x + h, y, z) - render(p, c, x - h, y, z)) / (2 * h)
    fy = (render(p, c, x, y + h, z) - render(p, c, x, y - h, z)) / (2 * h)
    return f, fx, fy


def weights(K, dep, psi, x, y):
    """pattern weights w [n, 6] (sum 1) and d/dx, d/dy; frame f: angle f // 3, phase psi[angle] + 2 pi / 3 * (f % 3)"""
    w, wx, wy = [], [], []
    for f in range(6):
        a = f // 3
        arg = K[a, 0] * x + K[a, 1] * y - psi[:, a] - 2 * math.pi / 3 * (f % 3)
        w.append((1 + dep[f] * torch.cos(arg)) / 6); wx.append(-dep[f] * torch.sin(arg) * K[a, 0] / 6); wy.append(-dep[f] * torch.sin(arg) * K[a, 1] / 6)
    return torch.stack(w, 1), torch.stack(wx, 1), torch.stack(wy, 1)


def model(p, c, th, z, mode, pat):
    """expected photons mu [n, F, R*R] (without read noise) and Jacobian [n, F, R*R, 4] for th = (x, y, N, b)"""
    x, y, N, b = th.unbind(1)
    f, fx, fy = psf_d(p, c, x, y, z)
    n = len(x); f, fx, fy = f.reshape(n, 1, -1), fx.reshape(n, 1, -1), fy.reshape(n, 1, -1)
    if mode == 'smlm':
        mu = N[:, None, None] * f + 6 * b[:, None, None]
        J = torch.stack([N[:, None, None] * fx, N[:, None, None] * fy, f, torch.full_like(f, 6.0)], -1)
    else:
        w, wx, wy = weights(pat['K'], pat['dep'], pat['psi'], x, y)
        w, wx, wy = w[:, :, None], wx[:, :, None], wy[:, :, None]
        NN = N[:, None, None]
        mu = NN * w * f + b[:, None, None]
        J = torch.stack([NN * (wx * f + w * fx), NN * (wy * f + w * fy), w * f, torch.ones_like(mu)], -1)
    return mu, J


def crlb(p, c, th, z, mode, pat, rn2):
    mu, J = model(p, c, th, z, mode, pat)
    v = mu + (6 * rn2 if mode == 'smlm' else rn2)
    FI = torch.einsum('nfpi,nfpj->nij', J / v[..., None], J)
    return torch.sqrt(torch.diagonal(torch.linalg.inv(FI), dim1=1, dim2=2))      # [n, 4]


def fit(p, c, d, th0, z, mode, pat, rn2, iters=30):
    """Poisson MLE (read noise as mu + rn^2), Levenberg-Marquardt with per-emitter damping"""
    r2 = 6 * rn2 if mode == 'smlm' else rn2
    th = th0.clone(); lam = torch.full((len(th),), 1.0, dtype=torch.float64)
    dd = d + r2

    def nll(th):
        mu, J = model(p, c, th, z, mode, pat); v = (mu + r2).clamp(min=1e-6)
        return (v - dd.clamp(min=0) * torch.log(v)).sum((1, 2)), v, J
    L, v, J = nll(th)
    for _ in range(iters):
        g = torch.einsum('nfpi,nfp->ni', J, (dd - v) / v)
        H = torch.einsum('nfpi,nfpj->nij', J / v[..., None], J)
        Hd = H + lam[:, None, None] * torch.diag_embed(torch.diagonal(H, dim1=1, dim2=2))
        st = torch.linalg.solve(Hd, g[..., None])[..., 0]
        st[:, :2] = st[:, :2].clamp(-0.5, 0.5)
        tn = th + st; tn[:, 2] = tn[:, 2].clamp(min=10.0); tn[:, 3] = tn[:, 3].clamp(min=0.05); tn[:, :2] = tn[:, :2].clamp(-2.5, 2.5)
        Ln, vn, Jn = nll(tn)
        ok = Ln < L
        th = torch.where(ok[:, None], tn, th); L = torch.where(ok, Ln, L)
        v = torch.where(ok[:, None, None], vn, v); J = torch.where(ok[:, None, None, None], Jn, J)
        lam = torch.where(ok, lam * 0.3, lam * 5.0).clamp(1e-4, 1e6)
    return th


def gfit(img):
    """elliptical Gaussian (pixel-sampled) least-squares fit: sigma along x, y (px, marginal sd of the fitted covariance), major / minor
    sd and orientation"""
    yy, xx = np.mgrid[:R, :R] - R // 2

    def f(q):
        A, x0, y0, a, b, cxy, o = q
        return (A * np.exp(-0.5 * (a * (xx - x0) ** 2 + b * (yy - y0) ** 2 + 2 * cxy * (xx - x0) * (yy - y0))) + o - img).ravel()
    q = least_squares(f, [img.max(), 0, 0, 0.5, 0.5, 0, 0]).x
    C = np.linalg.inv(np.array([[q[3], q[5]], [q[5], q[4]]]))
    ev, evec = np.linalg.eigh(C)
    return dict(sx=float(np.sqrt(C[0, 0])), sy=float(np.sqrt(C[1, 1])), smaj=float(np.sqrt(ev[1])), smin=float(np.sqrt(ev[0])),
                ang_deg=float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    t0 = time.time()
    C, Pm = {}, {}
    for m in MOV: C[m], Pm[m] = load(m)
    res = dict(photons=PHOT, bg_per_px_frame=BG, n_per_level=n, z_um=Z0)
    L = []
    # ---------------------------------------------------------------- A parameters
    par = {}
    for m in MOV:
        c = C[m]; b = c['psf_blur_px2']; Bm = np.array([[b[0], b[2]], [b[2], b[1]]]); ev, evec = np.linalg.eigh(Bm)
        ca = json.load(open(config.resolve(f'insitu/calib_{m}_psf3d.json')))
        par[m] = dict(zernike={int(j): float(v) for j, v in zip(c['zernike_noll'], c['zernike_waves'])},
                      zernike_rms_waves=float(np.sqrt(np.sum(np.square(c['zernike_waves'])))),
                      focus_um=c['focus_um'], blur_px2=b, blur_major_sd_px=float(np.sqrt(max(ev[1], 0))), blur_minor_sd_px=float(np.sqrt(max(ev[0], 0))),
                      blur_major_angle_deg=float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))),
                      dipole_M_diag=[c['psf_dipole_M'][i][i] for i in range(3)], gamma=c['psf3d']['gamma'], zeta=c['psf3d']['zeta'],
                      dipole_M_angle_diag=[[Ma[i][i] for i in range(3)] for Ma in ca['psf_dipole_M_angle']],
                      resid_var=c['psf3d']['resid_var'], NA=c['NA'], wavelength_um=c['wavelength_um'], n_med=c['n_med'],
                      readnoise_e=c['camera']['readnoise_e'], tirf_depth_um=c['tirf_depth_um'])
    res['params'] = par
    dz = np.array([par[MOV[0]]['zernike'][j] - par[MOV[1]]['zernike'][j] for j in par[MOV[0]]['zernike']])
    dz2 = np.array([par[MOV[0]]['zernike'][j] - par[MOV[2]]['zernike'][j] for j in par[MOV[0]]['zernike']])
    res['zernike_diff_rms_waves'] = {'20nM-2p5nM': float(np.sqrt((dz ** 2).sum())), '20nM-1809': float(np.sqrt((dz2 ** 2).sum()))}
    # ---------------------------------------------------------------- B images
    zz = torch.linspace(0, 0.2, 21, dtype=torch.float64); wz = torch.exp(-zz / 0.082); wz /= wz.sum()
    I0, Iz = {}, {}
    for m in MOV:
        o = torch.zeros(1, dtype=torch.float64)
        a = render(Pm[m], C[m], o, o, o + Z0)[0].numpy(); I0[m] = a / a.sum()
        s = render(Pm[m], C[m], torch.zeros(21, dtype=torch.float64), torch.zeros(21, dtype=torch.float64), zz)
        s = (s / s.sum((1, 2), keepdim=True) * wz[:, None, None]).sum(0).numpy(); Iz[m] = s / s.sum()
    yy, xx = np.mgrid[:R, :R] - R // 2
    shp = {}
    for m in MOV:
        a = I0[m]; g = gfit(a)
        g.update(peak=float(a.max()), e3=float(a[6:9, 6:9].sum() / a[2:13, 2:13].sum()), cx_nm=float((a * xx).sum() * PX), cy_nm=float((a * yy).sum() * PX))
        shp[m] = g
    res['shape_z055'] = shp
    dif = {}
    for a, b in [(MOV[0], MOV[1]), (MOV[0], MOV[2]), (MOV[1], MOV[2])]:
        for nm, I in (('z055', I0), ('tirf', Iz)):
            d = I[a] - I[b]; pk = max(I[a].max(), I[b].max())
            dif[f'{a}|{b}|{nm}'] = dict(rms_pct_peak=float(100 * np.sqrt((d ** 2).mean()) / pk), max_pct_peak=float(100 * np.abs(d).max() / pk),
                                        L1=float(np.abs(d).sum()), rms_abs=float(np.sqrt((d ** 2).mean())))
    res['image_diff'] = dif
    np.savez(f'{OUT}/psf_images.npz', **{f'z055_{m}': I0[m] for m in MOV}, **{f'tirf_{m}': Iz[m] for m in MOV})
    # ---------------------------------------------------------------- C / D
    g = torch.Generator().manual_seed(20)
    nl = len(PHOT); nt = n * nl
    x = torch.rand(nt, generator=g, dtype=torch.float64) - 0.5; y = torch.rand(nt, generator=g, dtype=torch.float64) - 0.5
    z = torch.full((nt,), Z0, dtype=torch.float64)
    N = torch.tensor(PHOT, dtype=torch.float64).repeat_interleave(n); b = torch.full((nt,), BG, dtype=torch.float64)
    lev = torch.arange(nl).repeat_interleave(n)
    th = torch.stack([x, y, N, b], 1)
    pt = C[MOV[0]]['pattern']
    pat = dict(K=torch.tensor(pt['k_rad_per_px'], dtype=torch.float64), dep=torch.tensor(pt['depth'], dtype=torch.float64),
               psi=2 * math.pi * torch.rand(nt, 2, generator=g, dtype=torch.float64))
    rn2 = C[MOV[0]]['camera']['readnoise_e'] ** 2
    cr = {}
    for m in MOV:
        for mode in ('smlm', 'sf'):
            cr[m, mode] = crlb(Pm[m], C[m], th, z, mode, pat, rn2)[:, :2] * PX
    rms = lambda v: float(torch.sqrt((v ** 2).mean()))
    res['crlb_nm'] = {f'{m}|{mode}': [dict(x=rms(cr[m, mode][lev == i, 0]), y=rms(cr[m, mode][lev == i, 1]),
                                           xy=rms(cr[m, mode][lev == i])) for i in range(nl)] for m in MOV for mode in ('smlm', 'sf')}
    print(f'CRLB done {time.time() - t0:.0f}s', flush=True)
    # data from the 20 nM PSF
    T = MOV[0]
    mu, _ = model(Pm[T], C[T], th, z, 'sf', pat)
    d = torch.poisson(mu, generator=g) + math.sqrt(rn2) * torch.randn(mu.shape, generator=g, dtype=torch.float64)
    th0 = th.clone(); th0[:, :2] += 0.15 * torch.randn(nt, 2, generator=g, dtype=torch.float64)
    th0[:, 2] *= torch.exp(0.2 * torch.randn(nt, generator=g, dtype=torch.float64)); th0[:, 3] *= 1.2
    mm = {}
    bias_s = {}
    for m in MOV:
        e = (fit(Pm[m], C[m], d.sum(1, keepdim=True), th0, z, 'smlm', pat, rn2) - th)
        bias_s[m] = e[lev >= nl - 2, :2].mean(0)                    # SMLM bias of this PSF model (px), the two brightest levels
        est = {'smlm': e, 'sf': fit(Pm[m], C[m], d, th0, z, 'sf', pat, rn2) - th}
        pg = dict(pat); pg['psi'] = pat['psi'] + (pat['K'] @ bias_s[m])[None]
        est['sf_gauge'] = fit(Pm[m], C[m], d, th0, z, 'sf', pg, rn2) - th
        print(f'fits {m} done {time.time() - t0:.0f}s; SMLM bias nm {(bias_s[m] * PX).numpy().round(2)}', flush=True)
        for k, e in est.items():
            c0 = cr[T, 'smlm' if k == 'smlm' else 'sf']
            rows = []
            for i in range(nl):
                s = lev == i; ee = e[s, :2] * PX; bi = ee.mean(0); cc = c0[s]
                ok = (ee - bi).abs().max(1).values < 10 * cc.mean()           # failed fits (outliers) counted separately
                rows.append(dict(bias_x=float(bi[0]), bias_y=float(bi[1]), sd=rms(ee[ok] - ee[ok].mean(0)), rms=rms(ee[ok]), crlb=rms(cc),
                                 sd_over_crlb=rms(ee[ok] - ee[ok].mean(0)) / rms(cc), rms_over_crlb=rms(ee[ok]) / rms(cc),
                                 rms_gauge=rms(ee[ok] - (bias_s[m] * PX)[None]), outl_pct=float(100 * (~ok).double().mean()),
                                 N_bias_pct=float(100 * (e[s, 2] / th[s, 2]).mean())))
            mm[f'{m}|{k}'] = rows
    res['mismatch_true20nM'] = mm
    res['smlm_bias_nm'] = {m: (bias_s[m] * PX).tolist() for m in MOV}
    json.dump(res, open(f'{OUT}/psf_compare.json', 'w'), indent=1)
    # ---------------------------------------------------------------- text
    L.append(f'# m20check PSF comparison; z = {Z0} um; photons per 6-frame window {PHOT}; bg {BG} ph/px/frame; read noise {math.sqrt(rn2):.2f} e; n = {n} per level')
    L.append('\n## parameters'); L.append('param | ' + ' | '.join(LAB[m] for m in MOV))
    for j in par[MOV[0]]['zernike']:
        L.append(f'Z{j} [waves] | ' + ' | '.join(f'{par[m]["zernike"][j]:+.4f}' for m in MOV))
    for k in ('zernike_rms_waves', 'focus_um', 'blur_major_sd_px', 'blur_minor_sd_px', 'blur_major_angle_deg', 'gamma', 'zeta', 'resid_var'):
        L.append(f'{k} | ' + ' | '.join(f'{par[m][k]:.4f}' for m in MOV))
    for k in ('blur_px2', 'dipole_M_diag', 'dipole_M_angle_diag'):
        L.append(f'{k} | ' + ' | '.join(str(np.round(par[m][k], 4).tolist()) for m in MOV))
    L.append(f'zernike difference rms [waves]: {res["zernike_diff_rms_waves"]}')
    L.append('\n## shape at z = 0.055 (unit-sum 15 x 15; Gaussian fit sd in px)')
    for m in MOV: L.append(f'{LAB[m]:16s} ' + '  '.join(f'{k} {v:.4f}' for k, v in shp[m].items()))
    L.append('\n## image differences'); [L.append(f'{k:28s} ' + '  '.join(f'{a} {v:.4f}' for a, v in d_.items())) for k, d_ in dif.items()]
    L.append('\n## CRLB nm (x / y / rms of x and y)')
    for k, rows in res['crlb_nm'].items():
        L.append(f'{k:16s} ' + '  '.join(f'{int(p)}: {r["x"]:.2f}/{r["y"]:.2f}/{r["xy"]:.2f}' for p, r in zip(PHOT, rows)))
    L.append('\n## mismatch: data = 20 nM PSF; fit model | estimator; per photon level: bias x, y nm; sd/CRLB; rms/CRLB; rms after SMLM-bias removal nm; CRLB nm; outliers %; N bias %')
    for k, rows in mm.items():
        L.append(k)
        for p, r in zip(PHOT, rows):
            L.append(f'   {int(p):5d}: bias {r["bias_x"]:+6.2f} {r["bias_y"]:+6.2f}  sd/CRLB {r["sd_over_crlb"]:.3f}  rms/CRLB {r["rms_over_crlb"]:.3f}  '
                     f'rms-gauge {r["rms_gauge"]:.2f}  CRLB {r["crlb"]:.2f}  outl {r["outl_pct"]:.2f}  N {r["N_bias_pct"]:.1f}')
    open(f'{OUT}/psf_compare.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
