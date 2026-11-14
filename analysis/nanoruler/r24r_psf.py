"""R24 reuse check, parts A and B (2026-11-13): is the in-situ PSF of the nr40 / nr20 movie the same as that of nr80 (the PSF the
nr80 network set was trained on)?  CPU only.  Adapted copy of analysis/psfmatch/m20check_psf.py (parts A-D); CHANGES:
  - calibs out/nanoruler/calib_<name>.json (averaged-over-axes in-situ PSF), columns <name> and nr80; pixel 65 nm
  - emitter z = the calib's emitter_z_median_um (0.01) instead of 0.055; the TIRF-weighted z average is kept as a second image
  - per-axis parameters (dipole tensor per pattern orientation) from the in-situ fit json (<fit>_psf.json 'dipole_M'), and the
    per-orientation PSF images / widths rendered with them (the networks see the average)
  - shape: ellipticity (sx / sy - 1 and major / minor - 1) added; image rms difference relative to the peak AND to the rms of the
    PSF image itself (mean-subtracted)
  - CRLB / fits: bg 1.65 ph / px / frame, photons 500 / 1000 / 2000 / 4000 per 6-frame window (full-PSF units: N multiplies the
    model PSF, whose integral over the infinite plane is 1; the 15 x 15 ROI holds the fraction printed as 'roi_frac')
  - pattern = THIS movie's calib pattern block for both PSFs: k, depth, relint and the measured phase of every step (not ideal
    120 degree steps), plus a random phase per orientation and emitter; frame f of the reordered window has orientation f // 3
  - mismatch (part D): data generated with THIS movie's PSF, fitted with the own model (control) and with the nr80 PSF model;
    binomial-type standard error of rms / CRLB printed (rel. se = 1 / sqrt(2 n_coordinates))
usage: python analysis/nanoruler/r24r_psf.py <nr40|nr20> [n_per_level]  -> out/nanoruler/<name>/reuse/psf_compare.json / .txt, psf_images.npz"""
import json, math, os, sys, time
import numpy as np, torch
from scipy.optimize import least_squares
sys.path.insert(0, '/home/jc/simcode2')
from simcode.util import proctitle
from simcode import psf as spsf

proctitle.set_title('nr-r24r')
torch.set_num_threads(6)
NAME = sys.argv[1]
ROOT = '/home/jc/simcode2/out/nanoruler'
OUT = f'{ROOT}/{NAME}/reuse'; os.makedirs(OUT, exist_ok=True)
MOV = [NAME, 'nr80']
FIT = {'nr80': f'{ROOT}/psf/blps3_psf.json', 'nr40': f'{ROOT}/insitu_nr40/blps3_psf.json', 'nr20': f'{ROOT}/insitu_nr20/blps3_psf.json'}
PX = 65.0; R = 15
PHOT = [500., 1000., 2000., 4000.]
BG = 1.65
dev = 'cpu'


def load(m, dipM=None):
    c = json.load(open(f'{ROOT}/calib_{m}.json'))
    p = spsf.load_psf(f'{ROOT}/calib_{m}.json', roisize=R, oversample=3, device=dev)
    if dipM is not None: p.dipM = torch.tensor(dipM, dtype=torch.float32)
    return c, p


@torch.no_grad()
def render(p, c, x, y, z):
    out = []
    pos = torch.stack([x, y, z], 1).float()
    for q in pos.split(500):
        out.append(p(q, torch.full((len(q),), c['focus_um'])).double())
    return torch.cat(out)


def psf_d(p, c, x, y, z, h=1e-3):
    f = render(p, c, x, y, z)
    fx = (render(p, c, x + h, y, z) - render(p, c, x - h, y, z)) / (2 * h)
    fy = (render(p, c, x, y + h, z) - render(p, c, x, y - h, z)) / (2 * h)
    return f, fx, fy


def weights(pat, x, y):
    """pattern weights w [n, 6] and d/dx, d/dy; frame f: orientation f // 3, w = rel_f (1 + dep_f cos(k.r - phi_f - psi_a)) / 6"""
    K, dep, rel, phi, psi = pat['K'], pat['dep'], pat['rel'], pat['phi'], pat['psi']
    w, wx, wy = [], [], []
    for f in range(6):
        a = f // 3
        arg = K[a, 0] * x + K[a, 1] * y - phi[f] - psi[:, a]
        w.append(rel[f] * (1 + dep[f] * torch.cos(arg)) / 6)
        wx.append(-rel[f] * dep[f] * torch.sin(arg) * K[a, 0] / 6); wy.append(-rel[f] * dep[f] * torch.sin(arg) * K[a, 1] / 6)
    return torch.stack(w, 1), torch.stack(wx, 1), torch.stack(wy, 1)


def model(p, c, th, z, mode, pat):
    x, y, N, b = th.unbind(1)
    f, fx, fy = psf_d(p, c, x, y, z)
    n = len(x); f, fx, fy = f.reshape(n, 1, -1), fx.reshape(n, 1, -1), fy.reshape(n, 1, -1)
    if mode == 'smlm':
        mu = N[:, None, None] * f + 6 * b[:, None, None]
        J = torch.stack([N[:, None, None] * fx, N[:, None, None] * fy, f, torch.full_like(f, 6.0)], -1)
    else:
        w, wx, wy = weights(pat, x, y)
        w, wx, wy = w[:, :, None], wx[:, :, None], wy[:, :, None]
        NN = N[:, None, None]
        mu = NN * w * f + b[:, None, None]
        J = torch.stack([NN * (wx * f + w * fx), NN * (wy * f + w * fy), w * f, torch.ones_like(mu)], -1)
    return mu, J


def crlb(p, c, th, z, mode, pat, rn2):
    mu, J = model(p, c, th, z, mode, pat)
    v = mu + (6 * rn2 if mode == 'smlm' else rn2)
    FI = torch.einsum('nfpi,nfpj->nij', J / v[..., None], J)
    return torch.sqrt(torch.diagonal(torch.linalg.inv(FI), dim1=1, dim2=2))


def fit(p, c, d, th0, z, mode, pat, rn2, iters=25):
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
    yy, xx = np.mgrid[:R, :R] - R // 2

    def f(q):
        A, x0, y0, a, b, cxy, o = q
        return (A * np.exp(-0.5 * (a * (xx - x0) ** 2 + b * (yy - y0) ** 2 + 2 * cxy * (xx - x0) * (yy - y0))) + o - img).ravel()
    q = least_squares(f, [img.max(), 0, 0, 0.3, 0.3, 0, 0]).x
    C = np.linalg.inv(np.array([[q[3], q[5]], [q[5], q[4]]]))
    ev, evec = np.linalg.eigh(C)
    sx, sy = float(np.sqrt(C[0, 0])), float(np.sqrt(C[1, 1]))
    return dict(sx=sx, sy=sy, ell_xy_pct=100 * (sx / sy - 1), smaj=float(np.sqrt(ev[1])), smin=float(np.sqrt(ev[0])),
                ell_majmin_pct=100 * float(np.sqrt(ev[1] / ev[0]) - 1), ang_deg=float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))),
                gauss_frac=float(2 * np.pi * q[0] * np.sqrt(np.linalg.det(C)) / img.sum()))


def main():
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 800
    t0 = time.time()
    C, Pm = {}, {}
    for m in MOV: C[m], Pm[m] = load(m)
    Z0 = float(C[NAME]['emitter_z_median_um'])
    res = dict(photons=PHOT, bg_per_px_frame=BG, n_per_level=n, z_um=Z0, movies=MOV)
    L = []
    # ---------------------------------------------------------------- A parameters
    par = {}
    for m in MOV:
        c = C[m]; b = c['psf_blur_px2']; Bm = np.array([[b[0], b[2]], [b[2], b[1]]]); ev, evec = np.linalg.eigh(Bm)
        fj = json.load(open(FIT[m]))
        par[m] = dict(zernike={int(j): float(v) for j, v in zip(c['zernike_noll'], c['zernike_waves'])},
                      zernike_rms_waves=float(np.sqrt(np.sum(np.square(c['zernike_waves'])))),
                      focus_um=c['focus_um'], blur_px2=b, blur_major_sd_px=float(np.sqrt(max(ev[1], 0))), blur_minor_sd_px=float(np.sqrt(max(ev[0], 0))),
                      dipole_M_diag=[c['psf_dipole_M'][i][i] for i in range(3)], gamma=c['psf3d']['gamma'], zeta=c['psf3d']['zeta'],
                      dipole_M_angle_diag=[[Ma[i][i] for i in range(3)] for Ma in fj['dipole_M']], dipole_M_angle=fj['dipole_M'],
                      resid_var=c['psf3d']['resid_var'], nll_per_px=fj.get('nll_per_px'), NA=c['NA'], wavelength_um=c['wavelength_um'], n_med=c['n_med'],
                      readnoise_e=c['camera']['readnoise_e'], tirf_depth_um=c['tirf_depth_um'], emitter_z_median_um=c['emitter_z_median_um'],
                      pitch_nm=c['pattern']['pitch_nm'], depth=c['pattern']['depth'])
    res['params'] = par
    dz = np.array([par[MOV[0]]['zernike'][j] - par[MOV[1]]['zernike'][j] for j in par[MOV[0]]['zernike']])
    res['zernike_diff_rms_waves'] = float(np.sqrt((dz ** 2).sum())); res['zernike_diff_max'] = dict(noll=int(list(par[MOV[0]]['zernike'])[int(np.abs(dz).argmax())]), waves=float(dz[np.abs(dz).argmax()]))
    # ---------------------------------------------------------------- B images
    zz = torch.linspace(0, 0.2, 21, dtype=torch.float64); wz = torch.exp(-zz / 0.082); wz /= wz.sum()
    o = torch.zeros(1, dtype=torch.float64)
    I0, Iz, Ia, frac = {}, {}, {}, {}
    for m in MOV:
        a = render(Pm[m], C[m], o, o, o + Z0)[0].numpy(); frac[m] = float(a.sum()); I0[m] = a / a.sum()
        s = render(Pm[m], C[m], torch.zeros(21, dtype=torch.float64), torch.zeros(21, dtype=torch.float64), zz)
        s = (s / s.sum((1, 2), keepdim=True) * wz[:, None, None]).sum(0).numpy(); Iz[m] = s / s.sum()
        for k in (0, 1):
            _, pk = load(m, dipM=par[m]['dipole_M_angle'][k])
            a = render(pk, C[m], o, o, o + Z0)[0].numpy(); Ia[m, k] = a / a.sum()
    res['roi_frac'] = frac
    yy, xx = np.mgrid[:R, :R] - R // 2
    shp = {}
    for m in MOV:
        for tag, a in (('avg', I0[m]), ('tirf', Iz[m]), ('orient0', Ia[m, 0]), ('orient1', Ia[m, 1])):
            g = gfit(a)
            g.update(peak=float(a.max()), e3=float(a[6:9, 6:9].sum() / a[2:13, 2:13].sum()), cx_nm=float((a * xx).sum() * PX), cy_nm=float((a * yy).sum() * PX),
                     mxx_myy_px2=float((a * xx ** 2).sum() - (a * yy ** 2).sum()))
            shp[f'{m}|{tag}'] = g
    res['shape'] = shp
    dif = {}
    for nm, A_, B_ in (('avg_z', I0[MOV[0]], I0[MOV[1]]), ('tirf', Iz[MOV[0]], Iz[MOV[1]]), ('orient0', Ia[MOV[0], 0], Ia[MOV[1], 0]),
                       ('orient1', Ia[MOV[0], 1], Ia[MOV[1], 1]), (f'{MOV[0]}: orient1 - orient0', Ia[MOV[0], 1], Ia[MOV[0], 0]),
                       (f'{MOV[1]}: orient1 - orient0', Ia[MOV[1], 1], Ia[MOV[1], 0])):
        d = A_ - B_; pk = max(A_.max(), B_.max()); own = np.sqrt(((B_ - B_.mean()) ** 2).mean())
        dif[nm] = dict(rms_pct_peak=float(100 * np.sqrt((d ** 2).mean()) / pk), max_pct_peak=float(100 * np.abs(d).max() / pk),
                       rms_pct_psf_rms=float(100 * np.sqrt((d ** 2).mean()) / own), L1=float(np.abs(d).sum()))
    res['image_diff'] = dif
    np.savez(f'{OUT}/psf_images.npz', **{f'avg_{m}': I0[m] for m in MOV}, **{f'tirf_{m}': Iz[m] for m in MOV},
             **{f'orient{k}_{m}': Ia[m, k] for m in MOV for k in (0, 1)})
    # ---------------------------------------------------------------- C / D
    g = torch.Generator().manual_seed(20)
    nl = len(PHOT); nt = n * nl
    x = torch.rand(nt, generator=g, dtype=torch.float64) - 0.5; y = torch.rand(nt, generator=g, dtype=torch.float64) - 0.5
    z = torch.full((nt,), Z0, dtype=torch.float64)
    N = torch.tensor(PHOT, dtype=torch.float64).repeat_interleave(n); b = torch.full((nt,), BG, dtype=torch.float64)
    lev = torch.arange(nl).repeat_interleave(n)
    th = torch.stack([x, y, N, b], 1)
    pt = C[NAME]['pattern']
    t64 = lambda v: torch.tensor(v, dtype=torch.float64)
    pat = dict(K=t64(pt['k_rad_per_px']), dep=t64(pt['depth']), rel=t64(pt['relint']), phi=t64(pt['phase']),
               psi=2 * math.pi * torch.rand(nt, 2, generator=g, dtype=torch.float64))
    rn2 = C[NAME]['camera']['readnoise_e'] ** 2
    cr = {}
    for m in MOV:
        for mode in ('smlm', 'sf'):
            cr[m, mode] = crlb(Pm[m], C[m], th, z, mode, pat, rn2)[:, :2] * PX
    rms = lambda v: float(torch.sqrt((v ** 2).mean()))
    res['crlb_nm'] = {f'{m}|{mode}': [dict(x=rms(cr[m, mode][lev == i, 0]), y=rms(cr[m, mode][lev == i, 1]),
                                           xy=rms(cr[m, mode][lev == i])) for i in range(nl)] for m in MOV for mode in ('smlm', 'sf')}
    res['crlb_ratio_nr80_over_own'] = {mode: [dict(x=res['crlb_nm'][f'nr80|{mode}'][i]['x'] / res['crlb_nm'][f'{NAME}|{mode}'][i]['x'],
                                                   y=res['crlb_nm'][f'nr80|{mode}'][i]['y'] / res['crlb_nm'][f'{NAME}|{mode}'][i]['y'],
                                                   xy=res['crlb_nm'][f'nr80|{mode}'][i]['xy'] / res['crlb_nm'][f'{NAME}|{mode}'][i]['xy']) for i in range(nl)]
                                       for mode in ('smlm', 'sf')}
    print(f'CRLB done {time.time() - t0:.0f}s', flush=True)
    T = NAME
    mu, _ = model(Pm[T], C[T], th, z, 'sf', pat)
    d = torch.poisson(mu, generator=g) + math.sqrt(rn2) * torch.randn(mu.shape, generator=g, dtype=torch.float64)
    th0 = th.clone(); th0[:, :2] += 0.15 * torch.randn(nt, 2, generator=g, dtype=torch.float64)
    th0[:, 2] *= torch.exp(0.2 * torch.randn(nt, generator=g, dtype=torch.float64)); th0[:, 3] *= 1.2
    mm = {}; bias_s = {}
    for m in MOV:
        e = (fit(Pm[m], C[m], d.sum(1, keepdim=True), th0, z, 'smlm', pat, rn2) - th)
        bias_s[m] = e[lev >= nl - 2, :2].mean(0)
        est = {'smlm': e, 'sf': fit(Pm[m], C[m], d, th0, z, 'sf', pat, rn2) - th}
        pg = dict(pat); pg['psi'] = pat['psi'] + (pat['K'] @ bias_s[m])[None]
        est['sf_gauge'] = fit(Pm[m], C[m], d, th0, z, 'sf', pg, rn2) - th
        print(f'fits {m} done {time.time() - t0:.0f}s; SMLM bias nm {(bias_s[m] * PX).numpy().round(2)}', flush=True)
        for k, e in est.items():
            c0 = cr[T, 'smlm' if k == 'smlm' else 'sf']
            rows = []
            for i in range(nl):
                s = lev == i; ee = e[s, :2] * PX; bi = ee.mean(0); cc = c0[s]
                ok = (ee - bi).abs().max(1).values < 10 * cc.mean()
                nc = 2 * int(ok.sum())
                rows.append(dict(bias_x=float(bi[0]), bias_y=float(bi[1]), bias_se=float(ee[ok].std() / math.sqrt(int(ok.sum()))),
                                 sd=rms(ee[ok] - ee[ok].mean(0)), rms=rms(ee[ok]), crlb=rms(cc),
                                 sd_over_crlb=rms(ee[ok] - ee[ok].mean(0)) / rms(cc), rms_over_crlb=rms(ee[ok]) / rms(cc),
                                 rms_over_crlb_se=rms(ee[ok]) / rms(cc) / math.sqrt(2 * nc),
                                 rms_gauge=rms(ee[ok] - (bias_s[m] * PX)[None]), outl_pct=float(100 * (~ok).double().mean()),
                                 N_bias_pct=float(100 * (e[s, 2] / th[s, 2]).mean()), bg_bias=float((e[s, 3]).mean())))
            mm[f'{m}|{k}'] = rows
    res['mismatch'] = mm
    res['smlm_bias_nm'] = {m: (bias_s[m] * PX).tolist() for m in MOV}
    json.dump(res, open(f'{OUT}/psf_compare.json', 'w'), indent=1)
    # ---------------------------------------------------------------- text
    L.append(f'# {NAME} vs nr80 PSF comparison (r24r_psf.py); z = {Z0} um; photons per 6-frame window {PHOT} (full-PSF units; fraction inside the 15 x 15 ROI: '
             f'{ {m: round(v, 4) for m, v in frac.items()} }); bg {BG} ph/px/frame; read noise {math.sqrt(rn2):.2f} e; n = {n} per level; pixel {PX} nm')
    L.append('\n## A parameters'); L.append('param | ' + ' | '.join(MOV) + ' | difference')
    for j in par[MOV[0]]['zernike']:
        L.append(f'Z{j} [waves] | ' + ' | '.join(f'{par[m]["zernike"][j]:+.4f}' for m in MOV) + f' | {par[MOV[0]]["zernike"][j] - par[MOV[1]]["zernike"][j]:+.4f}')
    for k in ('zernike_rms_waves', 'focus_um', 'blur_major_sd_px', 'blur_minor_sd_px', 'gamma', 'zeta', 'resid_var', 'nll_per_px', 'emitter_z_median_um'):
        L.append(f'{k} | ' + ' | '.join(f'{par[m][k]:.4f}' for m in MOV) + f' | {par[MOV[0]][k] - par[MOV[1]][k]:+.4f}')
    for k in ('blur_px2', 'dipole_M_diag', 'dipole_M_angle_diag', 'pitch_nm', 'depth'):
        L.append(f'{k} | ' + ' | '.join(str(np.round(par[m][k], 4).tolist()) for m in MOV))
    L.append(f'zernike difference: rms over the 17 modes (root sum of squares) {res["zernike_diff_rms_waves"]:.4f} waves; largest Z{res["zernike_diff_max"]["noll"]} {res["zernike_diff_max"]["waves"]:+.4f}')
    L.append(f'\n## B shape at z = {Z0} (unit-sum 15 x 15; elliptical-Gaussian fit sd in px; ell = sx / sy - 1 in %; avg = the calib PSF (averaged over orientations), '
             'tirf = TIRF-weighted z average, orient0 / orient1 = the PSF of the frames of pattern orientation 0 (k along y) / 1 (k along x))')
    for k, v in shp.items(): L.append(f'{k:14s} ' + '  '.join(f'{a} {b_:.4f}' for a, b_ in v.items()))
    for tag in ('avg', 'tirf', 'orient0', 'orient1'):
        a_, b_ = shp[f'{MOV[0]}|{tag}'], shp[f'{MOV[1]}|{tag}']
        L.append(f'{tag}: {MOV[0]} / nr80 width ratio sx {a_["sx"] / b_["sx"]:.4f} sy {a_["sy"] / b_["sy"]:.4f}; peak ratio {a_["peak"] / b_["peak"]:.4f}')
    L.append(f'\n## B image differences ({MOV[0]} minus nr80 unless stated; rms over the 225 px, in % of the peak and in % of the rms of the PSF image about its mean)')
    for k, d_ in dif.items(): L.append(f'{k:28s} ' + '  '.join(f'{a} {v:.4f}' for a, v in d_.items()))
    L.append('\n## CRLB nm (x / y / rms of x and y) and ratio nr80 PSF / this PSF')
    for k, rows in res['crlb_nm'].items():
        L.append(f'{k:12s} ' + '  '.join(f'{int(p)}: {r["x"]:.3f}/{r["y"]:.3f}/{r["xy"]:.3f}' for p, r in zip(PHOT, rows)))
    for k, rows in res['crlb_ratio_nr80_over_own'].items():
        L.append(f'ratio {k:6s} ' + '  '.join(f'{int(p)}: {r["x"]:.4f}/{r["y"]:.4f}/{r["xy"]:.4f}' for p, r in zip(PHOT, rows)))
    L.append(f'\n## mismatch: data = {NAME} PSF; fit model | estimator (sf = true pattern phase, sf_gauge = pattern phase shifted by k . SMLM bias of that model); per photon level: '
             'bias x, y nm (se); sd/CRLB; rms/CRLB (se); rms after SMLM-bias removal nm; CRLB nm (true model); outliers %; N bias %; bg bias ph/px/frame')
    for k, rows in mm.items():
        L.append(k)
        for p, r in zip(PHOT, rows):
            L.append(f'   {int(p):5d}: bias {r["bias_x"]:+6.3f} {r["bias_y"]:+6.3f} ({r["bias_se"]:.3f})  sd/CRLB {r["sd_over_crlb"]:.3f}  rms/CRLB {r["rms_over_crlb"]:.3f} ({r["rms_over_crlb_se"]:.3f})  '
                     f'rms-gauge {r["rms_gauge"]:.2f}  CRLB {r["crlb"]:.2f}  outl {r["outl_pct"]:.2f}  N {r["N_bias_pct"]:+.2f}  bg {r["bg_bias"]:+.4f}')
    open(f'{OUT}/psf_compare.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
