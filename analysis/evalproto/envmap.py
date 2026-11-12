"""critic 37 #2: smooth illumination (photon) envelope across the FOV on real 0.5 / 2.5 / 20 nM data. CPU only.
Ours pass 1 rows (p >= 0.63, starts % 3 == 0 (0.5 / 2.5) or % 6 == 0 (20 nM)), RAW sensor coordinates (the envelope is
fixed on the sensor), 5-115 crop. Per 10-px tile: median and p75 network window photons, rows; raw-movie background
(temporal median of 600 frames, offset-subtracted, x gain -> photons / px / frame), 10-px tile medians.
Fit: log(median photons per tile) = elliptical Gaussian log-quadratic (6 params, rows-weighted) -> envelope keys for realsim
(env_x0 / env_y0 centre px, env_wx / env_wy 1/e^2... no: Gaussian sd px along the principal axes, env_th angle rad).
The bg map is fitted the same way (reported; applied only if it follows the photon envelope).  same algorithm. Input = a datasets.py json spec (its
set "ours p1" (kind ours) and grid; drift not needed) plus
  "movie": "<raw movie .tif>", "camera_maps": "<camera_maps_*.npz with offset_adu>", "envmap_key": "<output key, e.g. 0.5>"
(dev: 0.5 nM camera_maps_0.5nM.npz; 2.5 AND 20 nM camera_maps_2p5nM_m2.npz -- reproduce the dev 20 nM numbers with that map;
simcode ships models/insitu/camera_maps_{0.5nM,2p5nM_m2,20nM_m2}.npz).
usage: python analysis/evalproto/envmap.py <dataset.json> [...] [--out dir]  -> envmap_<keys>.json / .txt (spatial.py reads it)"""
import os, sys, json, numpy as np, tifffile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets as DS
from simcode import config
G = 0.56
ARGS = sys.argv[1:]
if '--out' in ARGS:
    i = ARGS.index('--out'); OUTD = ARGS[i + 1]; del ARGS[i:i + 2]
else:
    OUTD = config.out_dir('evalproto')
NT = 11; T0 = 5; TW = 10


def tiles(x, y, v, fn=np.median):
    ix = np.clip(((x - T0) / TW).astype(int), 0, NT - 1); iy = np.clip(((y - T0) / TW).astype(int), 0, NT - 1)
    M = np.full((NT, NT), np.nan); N = np.zeros((NT, NT))
    k = iy * NT + ix
    o = np.argsort(k, kind='stable'); ks = k[o]; vs = v[o]
    cut = np.r_[0, np.flatnonzero(np.diff(ks)) + 1, len(ks)]
    for a, b in zip(cut[:-1], cut[1:]):
        q = ks[a]; M.flat[q] = fn(vs[a:b]); N.flat[q] = b - a
    return M, N


def fit_gauss(M, W):
    """log M = c + quadratic form -> centre, principal sd, angle; weights W"""
    c = T0 + TW * (np.arange(NT) + 0.5)
    X, Y = np.meshgrid(c, c)        # M[iy, ix]
    ok = np.isfinite(M) & (W > 0) & (M > 0)
    xs, ys = (X[ok] - 60) / 50, (Y[ok] - 60) / 50
    Aa = np.c_[np.ones(ok.sum()), xs, ys, xs ** 2, ys ** 2, xs * ys]
    w = np.sqrt(W[ok])
    p, *_ = np.linalg.lstsq(Aa * w[:, None], np.log(M[ok]) * w, rcond=None)
    Q = -np.array([[p[3], p[5] / 2], [p[5] / 2, p[4]]])   # log E = c - 0.5 d^T (2Q) d  (in scaled units)
    ctr = np.linalg.solve(2 * Q, p[1:3])
    ev, V = np.linalg.eigh(2 * Q)
    sd = 50 / np.sqrt(np.maximum(ev, 1e-9))
    th = float(np.arctan2(V[1, 0], V[0, 0]))
    fit = (Aa @ p)
    res = np.log(M[ok]) - fit
    full = np.full((NT, NT), np.nan); full[ok] = np.exp(fit)
    return dict(x0=float(60 + 50 * ctr[0]), y0=float(60 + 50 * ctr[1]), s1=float(sd[0]), s2=float(sd[1]), th=th,
                rms_log_res=float(np.sqrt(np.average(res ** 2, weights=W[ok]))), coef=p.tolist()), full


def env_eval(P, x, y):
    """realsim envelope (unnormalized): Gaussian with principal sd s1 (axis th) and s2"""
    dx, dy = x - P['x0'], y - P['y0']
    c, s = np.cos(P['th']), np.sin(P['th'])
    u = c * dx + s * dy; v = -s * dx + c * dy
    return np.exp(-0.5 * (u ** 2 / P['s1'] ** 2 + v ** 2 / P['s2'] ** 2))


OUT = {}; LINES = []
def log(s=''):
    print(s, flush=True); LINES.append(s)


np.set_printoptions(precision=2, suppress=True, linewidth=200)
KEYS = []
for spec in ARGS:
    SP_ = json.load(open(spec)); c = SP_.get('envmap_key', SP_.get('name', '')); KEYS.append(c); grid = int(SP_.get('grid', 3))
    O = DS.load_set(dict(SP_['sets']['ours p1'], drift=None), None, grid, crop=False)
    x, y, ph = O['xr'], O['yr'], O['phot']
    k = (x >= 5) & (x < 115) & (y >= 5) & (y < 115); x, y, ph = x[k], y[k], ph[k]
    Mm, N = tiles(x, y, ph); M75, _ = tiles(x, y, ph, lambda v: np.percentile(v, 75)); M90, _ = tiles(x, y, ph, lambda v: np.percentile(v, 90))
    Pm, Fm = fit_gauss(Mm, N); P75, F75 = fit_gauss(M75, N)
    # raw bg: temporal median of 600 frames spread over the movie
    t = tifffile.TiffFile(SP_['movie']); nf = len(t.pages)
    idx = np.linspace(1000, nf - 1000, 600).astype(int)
    adu = np.stack([t.pages[int(i)].asarray() for i in idx]).astype(np.float32)
    off = np.load(SP_['camera_maps'])['offset_adu'].astype(np.float32)
    bgim = (np.median(adu, 0) - off) * G
    yy, xx = np.mgrid[0:bgim.shape[0], 0:bgim.shape[1]]
    kk = (xx >= 5) & (xx < 115) & (yy >= 5) & (yy < 115)
    Mb, Nb = tiles(xx[kk].astype(float), yy[kk].astype(float), bgim[kk].ravel() if False else bgim[kk])
    Pb, Fb = fit_gauss(Mb, Nb)
    # envelope span / log-variance over the rows
    E = env_eval(Pm, x, y); E75 = env_eval(P75, x, y)
    span = lambda M: float(np.nanmax(M) / np.nanmin(M))
    cc = np.corrcoef(np.log(Mm[np.isfinite(Mm) & np.isfinite(Mb)]), np.log(Mb[np.isfinite(Mm) & np.isfinite(Mb)]))[0, 1]
    log(f'## real {c} nM: {len(x)} rows (ours p1, raw coords)')
    log(f'median photons per 10-px tile (rows = y):\n{Mm}')
    log(f'rows per tile (k):\n{N / 1000}')
    log(f'raw bg (median frame, ph/px/frame) per tile:\n{Mb}')
    log(f'span max/min tiles: median {span(Mm):.2f}, p75 {span(M75):.2f}, p90 {span(M90):.2f}, bg {span(Mb):.2f}; corr(log med ph, log bg) {cc:.2f}')
    log(f'Gaussian fit median: centre ({Pm["x0"]:.1f}, {Pm["y0"]:.1f}) px, sd {Pm["s1"]:.1f} / {Pm["s2"]:.1f} px at {np.degrees(Pm["th"]):.0f} deg, log residual rms {Pm["rms_log_res"]:.3f}')
    log(f'Gaussian fit p75:    centre ({P75["x0"]:.1f}, {P75["y0"]:.1f}) px, sd {P75["s1"]:.1f} / {P75["s2"]:.1f} px at {np.degrees(P75["th"]):.0f} deg, log residual rms {P75["rms_log_res"]:.3f}')
    log(f'Gaussian fit bg:     centre ({Pb["x0"]:.1f}, {Pb["y0"]:.1f}) px, sd {Pb["s1"]:.1f} / {Pb["s2"]:.1f} px at {np.degrees(Pb["th"]):.0f} deg, log residual rms {Pb["rms_log_res"]:.3f}')
    log(f'envelope over the rows (median fit): sd(log E) {np.std(np.log(E)):.3f}, p5/p95 {np.percentile(E,5)/np.percentile(E,95):.2f}; (p75 fit) sd(log E) {np.std(np.log(E75)):.3f}')
    log(f'log photon sd: all rows {np.std(np.log(ph)):.3f}; after dividing by the fitted envelope {np.std(np.log(ph / E)):.3f}\n')
    OUT[c] = dict(med=Pm, p75=P75, bg=Pb, tiles_med=Mm.tolist(), tiles_n=N.tolist(), tiles_bg=Mb.tolist(), corr_bg=float(cc),
                  sdlogE=float(np.std(np.log(E))), sdlogph=float(np.std(np.log(ph))), sdlogph_flat=float(np.std(np.log(ph / E))))
tag = '_'.join(KEYS)
os.makedirs(OUTD, exist_ok=True)
json.dump(OUT, open(os.path.join(OUTD, f'envmap_{tag}.json'), 'w'), indent=1)
open(os.path.join(OUTD, f'envmap_{tag}.txt'), 'w').write('\n'.join(LINES) + '\n')
