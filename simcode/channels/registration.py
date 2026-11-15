"""Registration of the detection channels to the reference channel 0.

  PolyMap   polynomial map per channel c (channel crop coordinates, x = column, y = row, pixel centres at integers):
            fwd(c, x, y) -> channel-0 coordinates; inv(c, x0, y0) -> channel-c coordinates (a separately fitted inverse).
            Features u^i v^j (i + j <= order), u = (x - cx) / sx, v = (y - cy) / sy; the normalisation is stored in
            the json ('feats_norm'); files without it use the crop centre (cx, cy, sx, sy) = (W/2, H/2, W/2, H/2).
            json: {order, fwd: {c: [nf, 2]}, inv: {c: [nf, 2]}, feats_norm: {cx, cy, sx, sy}, ...meta}
"""
import json
import numpy as np


class PolyMap:
    def __init__(self, coef_fwd, coef_inv, order, norm):
        """coef_fwd / coef_inv: {c: [nf, 2]} for the channels c >= 1; norm = (cx, cy, sx, sy)"""
        self.cf, self.ci, self.order, self.norm = coef_fwd, coef_inv, order, tuple(float(v) for v in norm)

    @staticmethod
    def default_norm(W, H):
        return (W / 2, H / 2, W / 2, H / 2)

    def feats(self, x, y, order=None):
        cx, cy, sx, sy = self.norm
        o = self.order if order is None else order
        u, v = (np.asarray(x) - cx) / sx, (np.asarray(y) - cy) / sy
        return np.stack([u ** i * v ** j for i in range(o + 1) for j in range(o + 1 - i)], -1)

    def fwd(self, c, x, y):
        if c == 0: return np.asarray(x, float), np.asarray(y, float)
        F_ = self.feats(x, y) @ self.cf[c]; return F_[..., 0], F_[..., 1]

    def inv(self, c, x, y):
        if c == 0: return np.asarray(x, float), np.asarray(y, float)
        F_ = self.feats(x, y) @ self.ci[c]; return F_[..., 0], F_[..., 1]

    def to_dict(self, **meta):
        cx, cy, sx, sy = self.norm
        return dict(order=self.order, fwd={c: np.asarray(v).tolist() for c, v in self.cf.items()},
                    inv={c: np.asarray(v).tolist() for c, v in self.ci.items()}, feats_norm=dict(cx=cx, cy=cy, sx=sx, sy=sy), **meta)

    def save(self, fn, **meta):
        from simcode.io.work import atomic_json
        atomic_json(fn, self.to_dict(**meta), indent=1, default=str)

    @staticmethod
    def from_dict(d, size=None):
        """size = (W, H) of the channel crops: the normalisation of a map without 'feats_norm'"""
        if d.get('feats_norm'):
            n = d['feats_norm']; norm = (n['cx'], n['cy'], n['sx'], n['sy'])
        elif size is not None:
            norm = PolyMap.default_norm(*size)
        else:
            raise ValueError("registration map without 'feats_norm': give the channel crop size (W, H)")
        return PolyMap({int(c): np.array(v) for c, v in d['fwd'].items()}, {int(c): np.array(v) for c, v in d['inv'].items()}, d['order'], norm)

    @staticmethod
    def load(fn, size=None):
        return PolyMap.from_dict(json.load(open(fn)), size)


# =====================================================================================================================
# In-situ registration from the movie itself (no beads): register_insitu(reader, camera, settings) -> (PolyMap, report)
#
#   (a) per-channel independent localization of K-frame sums (K = settings['sum']; 6 = one SIMFLUX cycle, 1 = plain
#       SMLM): DoG (1.5, 5) detection, threshold x the per-frame MAD of the filtered image, nms, isolation; roi x roi
#       ROIs; astigmatic Gaussian Poisson MLE with CRLB (torch_psf). The astigmatic curve (sigma0_x, sigma0_y, gamma) is
#       calibrated per channel in situ by maximum likelihood over calib_n bright, unblended spots of the first
#       calib_frames frames (d_um fixed: it only sets the nominal z scale). astig 'auto': a symmetric Gaussian
#       (sx = sy, gamma 0, z = defocus in z_range_sym) is used for all channels unless every channel is significantly
#       astigmatic (> 0.01 nats per spot and |gamma| > 0.05 um). Per localization: Poisson deviance per px.
#   quality: deviance under the median-relative rule of torch_psf's InSituEstimator (median + k_dev max(1.4826 MAD,
#       median sqrt(2 / roi^2))), not at the offset limit, CRLB x, y < max_crlb px, z inside zlim (zlim_sym).
#   (b) coarse alignment: same-frame displacement histogram (+-40 px, 0.5 px bins) of channel 0 vs channel c, searched
#       over the flips; then mutual nearest-neighbour matching per frame, no second candidate within 2 r:
#       r1 (translation) -> affine, 1.0 px -> order, 0.7 px -> order, 0.7 px.
#   (c) robust weighted polynomial c -> 0 (k_sigma x robust sd iterative rejection) and the inverse 0 -> c fitted
#       separately; kfold held-out residual vs the CRLB of both channels; same-channel self-consistency (the same
#       emitter in consecutive windows: delta pos vs CRLB) gives the empirical precision factor per channel.
# Coordinates: channel crop px (x = column, y = row, pixel centres at integers); map feats normalised by the crop centre
# (W/2, H/2, W/2, H/2) (= the 4Pi PolyMap feats).
# =====================================================================================================================
import time

# settings of the in-situ chain beyond those of the config (simcode/pipeline/acquisition_config.py, 'registration')
INSITU_DEFAULTS = dict(order=2, frames=0, frame0=0, sum=6, chunk=600, calib_frames=1200, calib_n=3000,
                       calib_p0=[1.2, 1.2, 0.18], astig='auto', roi=13, threshold=8.0, sigma=[1.5, 5.0], isolation=6, nms=7,
                       d_um=0.27, z_range=[-0.6, 0.6], z_range_sym=[0.03, 0.8], k_dev=3.0, max_crlb=0.15,
                       zlim=[-0.5, 0.5], zlim_sym=[0.0, 0.75], flips=['none', 'lr', 'ud', 'both'], r1=2.0, k_sigma=4.0,
                       kfold=5, readnoise_e=1.2)
CHECK_LIMITS = dict(pull_max=1.5, min_pairs=2000, roundtrip_max=0.01, flip_ratio_min=5.0, self_pull_max=1.5)


def _settings(s):
    out = dict(INSITU_DEFAULTS)
    out.update({k: v for k, v in (s or {}).items() if v is not None or k not in INSITU_DEFAULTS})
    return out


# ------------------------------------------------------------------------------------------------ (a) localization
def _tp():
    """the torch_psf pieces of the in-situ chain (the pinned torch_psf of simcode/psf)"""
    from simcode.psf import filter_frames, extract_rois, AstigmaticGaussianPSF, spot_deviance
    return filter_frames, extract_rois, AstigmaticGaussianPSF, spot_deviance


def detect_spots(frames, threshold=6.0, sigma=(2.0, 6.0), nms=9, border=0, isolation=None):
    """torch_psf.detect.detect_spots with noise='mad_frame' (the threshold in units of 1.4826 x the median absolute
    deviation of each filtered frame; the pinned torch_psf has only the per-chunk 'mad'): local maxima of the DoG
    filtered frames [T, H, W] -> dict(frame, y, x, score)"""
    import torch
    import torch.nn.functional as Fn
    filter_frames = _tp()[0]
    with torch.no_grad():
        T, H, W = frames.shape
        f = filter_frames(frames, sigma, False)
        ff = f.flatten(1)
        sig = 1.4826 * (ff - ff.median(1, keepdim=True).values).abs().median(1).values[:, None, None]
        mx = Fn.max_pool2d(f[:, None], nms, 1, nms // 2)[:, 0]
        pk = (f == mx) & (f > threshold * sig)
        t, y, x = torch.nonzero(pk, as_tuple=True)
        ok = (x >= border) & (x < W - border) & (y >= border) & (y < H - border)
        if isolation is not None:
            r = int(isolation)
            cnt = Fn.conv2d(pk[:, None].float(), torch.ones(1, 1, 2 * r + 1, 2 * r + 1, device=f.device), padding=r)[:, 0]
            ok = ok & (cnt[t, y, x] < 1.5)
        t, y, x = t[ok], y[ok], x[ok]
        s = sig.expand(T, 1, 1)[t, 0, 0]
        return dict(frame=t, y=y, x=x, score=f[t, y, x] / s)


def make_psf(R, p, d_um, z_range, pixel_um, dev):
    AstigmaticGaussianPSF = _tp()[2]
    return AstigmaticGaussianPSF(R, sigma_px=(float(p[0]), float(p[1])), gamma_um=float(p[2]), d_um=d_um,
                                 z_range=tuple(z_range), device=dev, pixel_um=pixel_um)


def detect(S, threshold, sigma, nms, isolation, border, xlim=None, ylim=None):
    import torch
    p = detect_spots(S, threshold=threshold, sigma=sigma, nms=nms, border=border, isolation=isolation)
    t, y, x = p['frame'], p['y'], p['x']
    ok = torch.ones_like(t, dtype=torch.bool)
    if xlim is not None: ok &= (x >= xlim[0]) & (x < xlim[1])
    if ylim is not None: ok &= (y >= ylim[0]) & (y < ylim[1])
    return t[ok], y[ok], x[ok], p['score'][ok]


def fit_rois(psf, rois, var, iters=40, zstarts=5):
    return psf.fit(rois, var=var, iters=iters, zstarts=zstarts)


def calibrate_astig(rois, var, R, d_um, z_range, pixel_um, dev, p0=(1.2, 1.2, 0.18), n=3000):
    """per-channel in-situ astigmatic curve: (sigma0_x, sigma0_y, gamma_um) maximizing the summed Poisson likelihood of
    the MLE fits of n bright spots (d_um fixed: it only sets the z scale, which the likelihood cannot)"""
    import torch
    from scipy.optimize import minimize
    rois = rois[:n]
    v = var if (var is None or not torch.is_tensor(var) or var.dim() == 0) else var[:n]

    def f(p):
        if p[0] < 0.6 or p[1] < 0.6 or p[0] > 4 or p[1] > 4: return 1e12
        ds = fit_rois(make_psf(R, p, d_um, z_range, pixel_um, dev), rois, v, iters=30, zstarts=5)
        return -float(ds.loglik.double().sum())
    r = minimize(f, np.array(p0, float), method='Nelder-Mead',
                 options=dict(xatol=2e-3, fatol=1.0, maxiter=120, initial_simplex=None))
    return r.x, r


def localize_frames(S, psf, var, R, det_kw, frame0=0):
    """S [T, H, W] photons (GPU) -> dict of numpy arrays (one row per detection)"""
    import torch
    _, extract_rois, _, spot_deviance = _tp()
    t, y, x, sc = detect(S, **det_kw)
    if len(t) == 0: return None
    rois = extract_rois(S, t, y, x, R)
    ds = fit_rois(psf, rois, var)
    th = torch.stack([ds.pos[:, 0], ds.pos[:, 1], ds.pos[:, 2] * psf.pixel_um, ds.photons, ds.bg], 1).float().to(S.device)
    dv, _ = spot_deviance(psf, rois, var, theta=th)
    pos = ds.pos.numpy()
    return dict(frame=(t.cpu().numpy() + frame0), x=x.cpu().numpy() + pos[:, 0], y=y.cpu().numpy() + pos[:, 1],
                z=pos[:, 2] * psf.pixel_um, N=ds.photons.numpy(), bg=ds.bg.numpy(),
                cx=ds.crlb[:, 0].numpy(), cy=ds.crlb[:, 1].numpy(), cz=ds.crlb[:, 2].numpy() * psf.pixel_um,
                dev=dv, chi2=ds.chi2.numpy(), lim=ds.at_limit.numpy(), score=sc.cpu().numpy())


def cat_locs(L):
    L = [l for l in L if l is not None]
    return {k: np.concatenate([l[k] for l in L]) for k in L[0]}


def subset(L, m):
    return {k: v[m] for k, v in L.items()}


def deviance_threshold(dv, R, k=3.0):
    """median-relative rule of torch_psf.InSituEstimator: median + k max(1.4826 MAD, median sqrt(2 / R^2))"""
    med = float(np.median(dv)); mad = 1.4826 * float(np.median(np.abs(dv - med)))
    return med + k * max(mad, med * np.sqrt(2.0 / R ** 2)), med


def _cam_tensor(v, C, dev, name):
    """camera offset / gain: scalar (float, kept a python float) or per channel [C] or per pixel [C, H, W] -> broadcastable
    against [n, C, H, W]"""
    import torch
    a = np.asarray(v, np.float64) if not torch.is_tensor(v) else v
    if np.ndim(a) == 0: return float(a)
    t = torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)
    if t.dim() == 1:
        if len(t) != C: raise ValueError(f'camera {name}: {len(t)} values for {C} channels')
        return t[:, None, None]
    if t.dim() == 3:
        if t.shape[0] != C: raise ValueError(f'camera {name}: map of {t.shape[0]} channels for {C}')
        return t
    raise ValueError(f'camera {name}: a scalar, [C] or a map [C, H, W], not shape {tuple(t.shape)}')


def localize_channels(reader, camera, settings, xlim=None, ylim=None, pixel_um=None, dev='cuda', log=print):
    """(a) for every channel of a ChannelReader. camera: dict(offset (ADU; scalar, [C] or [C, H, W]), gain (e / ADU;
    same), readnoise_e (e)). -> (L: list of per-channel localization dicts, info: calibration, astig, frames, camera)"""
    import torch
    from scipy.optimize import minimize_scalar
    s = _settings(settings)
    if pixel_um is None: raise ValueError('register_insitu: pixel_um is required')
    C, H, W = reader.shape
    R, KS, CH = int(s['roi']), int(s['sum']), int(s['chunk'])
    if CH % KS: raise ValueError(f"registration.chunk ({CH}) must be a multiple of registration.sum ({KS})")
    F0 = int(s['frame0'])
    NF = int(s['frames']) if s['frames'] else reader.n - F0
    NF = min(NF, reader.n - F0); NF -= NF % CH
    NC = min(int(s['calib_frames']), NF); NC -= NC % KS
    if NF <= 0 or NC <= 0: raise ValueError(f'register_insitu: too few frames ({reader.n} from {F0}, chunk {CH})')
    rn = camera.get('readnoise_e')
    rn_note = None
    if rn is None: rn, rn_note = float(s['readnoise_e']), f"camera read noise not given: {s['readnoise_e']} e assumed"
    rn = float(np.median(np.asarray(rn)))
    gain = camera.get('gain')
    if gain is None: raise ValueError('register_insitu: camera gain (e / ADU) is required')
    off = _cam_tensor(camera['offset'], C, dev, 'offset'); gn = _cam_tensor(gain, C, dev, 'gain')
    VAR = torch.tensor(KS * rn ** 2, device=dev)               # read-noise variance of a K-frame sum (e^2)
    DET = dict(threshold=float(s['threshold']), sigma=tuple(s['sigma']), nms=int(s['nms']), isolation=int(s['isolation']),
               border=(R - 1) // 2 + 2, xlim=None if xlim is None else tuple(xlim), ylim=None if ylim is None else tuple(ylim))
    from simcode.io.movie import Prefetch
    plan = [(F0, F0 + NC)] + [(f0, f0 + CH) for f0 in range(F0, F0 + NF, CH)]
    rd = Prefetch(reader, plan)

    def sums(f0, n):
        a = torch.as_tensor(rd.read(f0, f0 + n), device=dev)                       # [n, C, H, W] ADU
        E = (a - off) * gn
        return E.reshape(n // KS, KS, C, H, W).sum(1)

    t0 = time.time()
    _, extract_rois, _, spot_deviance = _tp()
    S = sums(F0, NC)
    p0 = tuple(float(v) for v in s['calib_p0'])
    zr, zrs = tuple(s['z_range']), tuple(s['z_range_sym'])
    calib = {}
    for c in range(C):
        t, y, x, sc = detect(S[:, c], **DET)
        rois = extract_rois(S[:, c], t, y, x, R)
        psf0 = make_psf(R, p0, s['d_um'], zr, pixel_um, dev)
        ds = psf0.fit(rois, var=VAR)
        dv = spot_deviance(psf0, rois, VAR)[0]
        thr, _ = deviance_threshold(dv, R)
        good = np.flatnonzero((dv < thr) & (ds.photons.numpy() > np.quantile(ds.photons.numpy(), 0.5)) & ~ds.at_limit.numpy())
        good = good[np.random.default_rng(c).permutation(len(good))][:int(s['calib_n'])]
        rg = rois[torch.as_tensor(good, device=dev)]
        p, res = calibrate_astig(rg, VAR, R, s['d_um'], zr, pixel_um, dev, p0=p0, n=int(s['calib_n']))
        calib[c] = dict(sigma0_x=float(p[0]), sigma0_y=float(p[1]), gamma_um=float(p[2]), d_um=float(s['d_um']), n=int(len(good)),
                        n_detected=int(len(t)), nll_start=float(-ds.loglik[good].double().sum()), nll_astig=float(res.fun))
        if s['astig'] == 'auto':
            # symmetric alternative (no astigmatism): sx = sy, gamma = 0, z >= 0 = defocus
            def fsym(q):
                if not 0.6 < q[0] < 4: return 1e12
                return -float(fit_rois(make_psf(R, (q[0], q[0], 0.0), s['d_um'], zrs, pixel_um, dev), rg, VAR).loglik.double().sum())
            rs = minimize_scalar(lambda v: fsym([v]), bounds=(0.7, 3.0), method='bounded', options=dict(xatol=2e-3))
            calib[c].update(nll_sym=float(rs.fun), sigma_sym=float(rs.x),
                            astig=bool((rs.fun - res.fun) > 2 * len(good) * 0.01 and abs(p[2]) > 0.05))   # > 0.01 nats per spot
        else:
            calib[c]['astig'] = s['astig'] in (True, 'on', 'yes')
        log(f"registration: ch {c} calibration " + ', '.join(f'{k} {v:.4g}' if isinstance(v, float) else f'{k} {v}' for k, v in calib[c].items())
            + f' ({time.time() - t0:.0f} s)')
    if s['astig'] == 'auto':
        astig = all(calib[c]['astig'] for c in range(C))
    else:
        astig = s['astig'] in (True, 'on', 'yes')
    if astig:
        psfs = [make_psf(R, (calib[c]['sigma0_x'], calib[c]['sigma0_y'], calib[c]['gamma_um']), s['d_um'], zr, pixel_um, dev) for c in range(C)]
    else:
        if any('sigma_sym' not in calib[c] for c in range(C)):
            raise ValueError("registration.astig 'off' needs the symmetric calibration: use astig 'auto'")
        psfs = [make_psf(R, (calib[c]['sigma_sym'],) * 2 + (0.0,), s['d_um'], zrs, pixel_um, dev) for c in range(C)]
    del S

    L = [[] for _ in range(C)]
    for f0 in range(F0, F0 + NF, CH):
        S = sums(f0, CH)
        for c in range(C):
            L[c].append(localize_frames(S[:, c], psfs[c], VAR, R, DET, frame0=f0 // KS))
        if (f0 - F0) % (10 * CH) == 0:
            log(f'registration: frames {f0 + CH - F0} / {NF}, localizations {[sum(len(l["x"]) for l in L[c] if l is not None) for c in range(C)]} '
                f'({time.time() - t0:.0f} s)')
        del S
    rd.close()
    L = [cat_locs(L[c]) for c in range(C)]
    info = dict(calib={str(c): calib[c] for c in range(C)}, astig=bool(astig), frames=[F0, F0 + NF], frames_per_sum=KS,
                n_windows=NF // KS, n_locs=[int(len(L[c]['x'])) for c in range(C)], readnoise_e=rn,
                camera=dict(offset='map' if np.ndim(camera['offset']) else float(camera['offset']),
                            gain='map' if np.ndim(gain) else float(gain)),
                time_localize_s=round(time.time() - t0, 1))
    if rn_note: info['note'] = rn_note
    return L, info


# ------------------------------------------------------------------------------------------------ (b) matching
def flip_xy(x, y, flip, W, H):
    if flip in ('lr', 'both'): x = (W - 1) - x
    if flip in ('ud', 'both'): y = (H - 1) - y
    return x, y


def displacement_hist(A, B, max_shift=40.0, binw=0.5, max_frames=400):
    """histogram of (xA - xB, yA - yB) over all same-frame pairs; A, B dicts with frame, x, y"""
    fr = np.intersect1d(np.unique(A['frame']), np.unique(B['frame']))
    if len(fr) > max_frames: fr = fr[np.linspace(0, len(fr) - 1, max_frames).astype(int)]
    sa, sb = np.argsort(A['frame'], kind='stable'), np.argsort(B['frame'], kind='stable')
    fa, fb = A['frame'][sa], B['frame'][sb]
    dx, dy = [], []
    for f in fr:
        a = sa[np.searchsorted(fa, f):np.searchsorted(fa, f, 'right')]
        b = sb[np.searchsorted(fb, f):np.searchsorted(fb, f, 'right')]
        ddx = (A['x'][a][:, None] - B['x'][b][None]).ravel(); ddy = (A['y'][a][:, None] - B['y'][b][None]).ravel()
        k = (np.abs(ddx) < max_shift) & (np.abs(ddy) < max_shift)
        dx.append(ddx[k]); dy.append(ddy[k])
    dx = np.concatenate(dx) if dx else np.zeros(0); dy = np.concatenate(dy) if dy else np.zeros(0)
    e = np.arange(-max_shift, max_shift + binw / 2, binw)
    Hh, _, _ = np.histogram2d(dy, dx, [e, e])
    return Hh, e


def coarse_align(A, B, W, H, flips=('none', 'lr', 'ud', 'both'), max_shift=40.0, binw=0.5):
    """-> best flip, shift (dx, dy) with B -> A: x_A = flip(x_B) + dx, and the peak significance per flip
    (peak height above the histogram median in units of its Poisson sd)"""
    from scipy.ndimage import uniform_filter
    res = {}
    for fl in flips:
        bx, by = flip_xy(B['x'], B['y'], fl, W, H)
        Hh, e = displacement_hist(A, dict(frame=B['frame'], x=bx, y=by), max_shift, binw)
        Hs = uniform_filter(Hh, 3) * 9
        iy, ix = np.unravel_index(np.argmax(Hs), Hs.shape)
        bgl = np.median(Hs)
        c = 0.5 * (e[1:] + e[:-1])
        # centroid around the peak (5 x 5 bins)
        sl = (slice(max(iy - 2, 0), iy + 3), slice(max(ix - 2, 0), ix + 3))
        w = np.clip(Hh[sl] - Hh.mean(), 0, None) + 1e-9
        yy, xx = np.meshgrid(c[sl[0]], c[sl[1]], indexing='ij')
        res[fl] = dict(shift=(float((w * xx).sum() / w.sum()), float((w * yy).sum() / w.sum())),
                       signif=float((Hs[iy, ix] - bgl) / np.sqrt(max(bgl, 1.0))), peak=float(Hs[iy, ix]), bg=float(bgl))
    best = max(res, key=lambda k: res[k]['signif'])
    return best, res[best]['shift'], res


def match_frames(A, Bxy, Bframe, r, amb=None):
    """mutual nearest neighbours per frame between A (dict frame/x/y) and B (positions Bxy [n,2] already mapped into A
    coordinates) within r px; pairs with a second A candidate within amb px of the B point are dropped.
    -> index arrays ia, ib"""
    from scipy.spatial import cKDTree
    amb = 2 * r if amb is None else amb
    sa, sb = np.argsort(A['frame'], kind='stable'), np.argsort(Bframe, kind='stable')
    fa, fb = A['frame'][sa], Bframe[sb]
    IA, IB = [np.zeros(0, int)], [np.zeros(0, int)]
    for f in np.unique(fb):
        a = sa[np.searchsorted(fa, f):np.searchsorted(fa, f, 'right')]
        if len(a) == 0: continue
        b = sb[np.searchsorted(fb, f):np.searchsorted(fb, f, 'right')]
        PA = np.c_[A['x'][a], A['y'][a]]; PB = Bxy[b]
        ta, tb = cKDTree(PA), cKDTree(PB)
        k = min(2, len(a))
        d, j = ta.query(PB, k=k)
        if k == 2: d1, j1, d2 = d[:, 0], j[:, 0], d[:, 1]
        else: d1, j1, d2 = d, j, np.full(len(b), np.inf)
        dback, jback = tb.query(PA[j1], k=1)
        ok = (d1 < r) & (jback == np.arange(len(b))) & (d2 > amb)
        IA.append(a[j1[ok]]); IB.append(b[ok])
    return np.concatenate(IA), np.concatenate(IB)


# ------------------------------------------------------------------------------------------------ (c) maps
class Norm:
    """the PolyMap feats as a free function of the normalisation (cx, cy, sx, sy)"""
    def __init__(self, cx, cy, sx, sy): self.c = (cx, cy, sx, sy)

    def feats(self, x, y, order):
        cx, cy, sx, sy = self.c
        u, v = (np.asarray(x, float) - cx) / sx, (np.asarray(y, float) - cy) / sy
        return np.stack([u ** i * v ** j for i in range(order + 1) for j in range(order + 1 - i)], -1)


def wfit(A, dst, w):
    return np.linalg.lstsq(A * w[:, None], dst * w[:, None], rcond=None)[0]


def robust_poly(src, dst, w, norm, order=2, k=4.0, nit=6):
    """weighted LSQ polynomial src -> dst with iterative k x robust-sd outlier rejection on the normalized residual
    (residual x w). Returns coef [nf, 2], keep mask"""
    A = norm.feats(src[:, 0], src[:, 1], order)
    keep = np.ones(len(src), bool)
    for _ in range(nit):
        cf = wfit(A[keep], dst[keep], w[keep])
        r = (dst - A @ cf) * w[:, None]
        s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep], 0)), 0)
        nk = (np.abs(r - np.median(r[keep], 0)) < k * s).all(1)
        if (nk == keep).all(): break
        keep = nk
    return cf, keep


def kfold_resid(src, dst, w, norm, order=2, K=5, seed=0, k=4.0):
    """held-out residual dst - map(src) for every pair, the map fitted (robustly) on the other K-1 folds"""
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, K, len(src))
    r = np.zeros_like(dst)
    for f in range(K):
        tr = fold != f
        cf, _ = robust_poly(src[tr], dst[tr], w[tr], norm, order, k)
        te = ~tr
        r[te] = dst[te] - norm.feats(src[te, 0], src[te, 1], order) @ cf
    return r


def rsd(r, axis=0):
    r = np.asarray(r)
    return 1.4826 * np.median(np.abs(r - np.median(r, axis)), axis)


def link_consecutive(L, r=0.5, iso=6.0):
    """same-channel self-consistency: localizations in frame f and f+1 within r px, with no other localization within
    iso px in either frame. -> (i, j) index pairs; d = pos[j] - pos[i] has variance crlb_i^2 + crlb_j^2 if the CRLB is
    the true precision (and the emitter does not move)"""
    from scipy.spatial import cKDTree
    order = np.argsort(L['frame'], kind='stable'); fr = L['frame'][order]
    I, J = [np.zeros(0, int)], [np.zeros(0, int)]
    for f in np.unique(fr):
        a = order[np.searchsorted(fr, f):np.searchsorted(fr, f, 'right')]
        b = order[np.searchsorted(fr, f + 1):np.searchsorted(fr, f + 1, 'right')]
        if len(a) < 1 or len(b) < 1: continue
        PA, PB = np.c_[L['x'][a], L['y'][a]], np.c_[L['x'][b], L['y'][b]]
        tb = cKDTree(PB)
        d, j = tb.query(PA, k=min(2, len(b)))
        if d.ndim == 1: d, j = d[:, None], j[:, None]
        ok = (d[:, 0] < r) & ((d[:, 1] > iso) if d.shape[1] > 1 else True)
        ta = cKDTree(PA); da, _ = ta.query(PA, k=min(2, len(a)))
        if da.ndim == 2 and da.shape[1] > 1: ok &= da[:, 1] > iso
        I.append(a[ok]); J.append(b[j[ok, 0]])
    return np.concatenate(I), np.concatenate(J)


def quality(L, R, k_dev=3.0, max_crlb=0.15, zlim=None):
    """per-channel quality mask: deviance under the median-relative threshold (overlaps / blends), not at the offset
    limit, CRLB x and y < max_crlb px, z inside zlim (not at the z-range edge)"""
    thr, med = deviance_threshold(L['dev'], R, k_dev)
    m = (L['dev'] < thr) & ~L['lim'].astype(bool) & (L['cx'] < max_crlb) & (L['cy'] < max_crlb) & np.isfinite(L['x'])
    if zlim is not None: m &= (L['z'] > zlim[0]) & (L['z'] < zlim[1])
    return m, dict(dev_threshold=float(thr), dev_median=float(med), kept=float(m.mean()), n=int(len(m)))


def register_pair(L0, Lc, W, H, norm, order=2, flips=('none', 'lr', 'ud', 'both'), r1=2.0, k=4.0, K=5):
    """(b) + (c) for one channel pair, all automatic. L0, Lc: quality-filtered localization dicts.
    -> dict with flip, coarse shift, pair indices (i0, ic), forward / inverse coefficients, held-out residual, keep"""
    fl, sh, cres = coarse_align(L0, Lc, W, H, flips)
    bx, by = flip_xy(Lc['x'], Lc['y'], fl, W, H)
    src_all = np.c_[bx, by]
    # match with the translation, fit affine, re-match tighter, fit order 2, re-match at the final radius
    xy = src_all + np.array(sh)
    i0, ic = match_frames(L0, xy, Lc['frame'], r1)
    w = lambda i0, ic: 1 / np.sqrt(0.5 * (L0['cx'][i0] ** 2 + L0['cy'][i0] ** 2 + Lc['cx'][ic] ** 2 + Lc['cy'][ic] ** 2) + 0.01 ** 2)
    for o, r in ((1, 1.0), (order, 0.7), (order, 0.7)):
        dst = np.c_[L0['x'][i0], L0['y'][i0]]
        cf, keep = robust_poly(src_all[ic], dst, w(i0, ic), norm, o, k)
        xy = norm.feats(src_all[:, 0], src_all[:, 1], o) @ cf
        i0, ic = match_frames(L0, xy, Lc['frame'], r)
    src, dst, ww = src_all[ic], np.c_[L0['x'][i0], L0['y'][i0]], w(i0, ic)
    cf, keep = robust_poly(src, dst, ww, norm, order, k)
    ci, _ = robust_poly(dst, src, ww, norm, order, k)
    rho = kfold_resid(src, dst, ww, norm, order, K, k=k)                # held-out
    rin = dst - norm.feats(src[:, 0], src[:, 1], order) @ cf            # in-sample
    # inverse consistency: inv(fwd(p)) - p over the pairs
    fw = norm.feats(src[:, 0], src[:, 1], order) @ cf
    rt = norm.feats(fw[:, 0], fw[:, 1], order) @ ci - src
    return dict(flip=fl, shift=sh, coarse=cres, i0=i0, ic=ic, src=src, dst=dst, fwd=cf, inv=ci, resid_ho=rho,
                resid_in=rin, keep=keep, roundtrip_rms=float(np.sqrt((rt[keep] ** 2).mean())))


def summarize(reg, L0, Lc, keep=None, f0=None, fc=None):
    """residual vs CRLB numbers of one channel pair (held-out residual, robust; keep: the robust-fit inliers).
    f0, fc: per-axis factors (self-consistency pull rsd of each channel): 'emp' = residual / sqrt((f0 c0)^2 + (fc cc)^2),
    i.e. against the empirical localization precision instead of the CRLB"""
    i0, ic, r = reg['i0'], reg['ic'], reg['resid_ho']
    keep = reg['keep'] if keep is None else keep
    e = np.sqrt(np.c_[L0['cx'][i0] ** 2 + Lc['cx'][ic] ** 2, L0['cy'][i0] ** 2 + Lc['cy'][ic] ** 2])
    pull = r / e
    s_all = rsd(r)
    exp_ = np.sqrt(np.median(e ** 2, 0))
    out = dict(n_pairs=int(len(r)), inlier_frac=float(keep.mean()),
               median_resid=np.median(r, 0).round(4).tolist(),
               resid_rsd=s_all.round(4).tolist(), resid_rms_inliers=np.sqrt((r[keep] ** 2).mean(0)).round(4).tolist(),
               crlb_expected=exp_.round(4).tolist(),
               pull_rsd=rsd(pull).round(3).tolist(), pull_rms_inliers=np.sqrt((pull[keep] ** 2).mean(0)).round(3).tolist(),
               excess_rsd=np.sqrt(np.clip(s_all ** 2 - exp_ ** 2, 0, None)).round(4).tolist(),
               in_sample_rsd=rsd(reg['resid_in']).round(4).tolist())
    if f0 is not None:
        ee = np.c_[(f0[0] * L0['cx'][i0]) ** 2 + (fc[0] * Lc['cx'][ic]) ** 2, (f0[1] * L0['cy'][i0]) ** 2 + (fc[1] * Lc['cy'][ic]) ** 2]
        out.update(pull_emp_rsd=rsd(r / np.sqrt(ee)).round(3).tolist(), emp_expected=np.sqrt(np.median(ee, 0)).round(4).tolist())
    return out


def self_consistency(L):
    """same channel, consecutive windows: n links, rsd of the position difference, the CRLB-expected sd, pull rsd"""
    i, j = link_consecutive(L)
    d = np.c_[L['x'][j] - L['x'][i], L['y'][j] - L['y'][i]]
    e = np.sqrt(np.c_[L['cx'][i] ** 2 + L['cx'][j] ** 2, L['cy'][i] ** 2 + L['cy'][j] ** 2])
    return dict(n=int(len(i)), diff_rsd=rsd(d).round(4).tolist(), crlb_expected=np.sqrt(np.median(e ** 2, 0)).round(4).tolist(),
                pull_rsd=rsd(d / e).round(3).tolist())


def affine_of(cf1, norm):
    """order-1 coefficients (feats [1, v, u]) -> dst = A [x, y] + t, rotation (deg), scale"""
    cx, cy, sx, sy = norm.c
    A = np.array([[cf1[2, 0] / sx, cf1[1, 0] / sy], [cf1[2, 1] / sx, cf1[1, 1] / sy]])
    t = cf1[0] - A @ np.array([cx, cy])
    return dict(A=A.round(6).tolist(), t=t.round(4).tolist(), rotation_deg=float(np.degrees(np.arctan2(A[1, 0] - A[0, 1], A[0, 0] + A[1, 1]))),
                scale=float(np.sqrt(abs(np.linalg.det(A)))))


def unflip_coefs(cf, ci, flip, W, H, norm, order):
    """register_pair fits in the flipped channel-c frame (src = flip(x_c)); the stored PolyMap works on the channel crop
    coordinates: fwd(x_c) = P_fwd(flip(x_c)), inv(x_0) = flip(P_inv(x_0)). A flip is affine, so both are again
    polynomials of the same order: exact refit on a grid. flip 'none': the coefficients unchanged."""
    if flip == 'none': return cf, ci
    gx, gy = np.meshgrid(np.linspace(0, W - 1, 4 * order + 4), np.linspace(0, H - 1, 4 * order + 4))
    gx, gy = gx.ravel(), gy.ravel()
    A = norm.feats(gx, gy, order)
    fx, fy = flip_xy(gx, gy, flip, W, H)
    cf2 = np.linalg.lstsq(A, norm.feats(fx, fy, order) @ cf, rcond=None)[0]
    p = A @ ci
    ux, uy = flip_xy(p[:, 0], p[:, 1], flip, W, H)
    ci2 = np.linalg.lstsq(A, np.c_[ux, uy], rcond=None)[0]
    return cf2, ci2


def register_locs(L, W, H, settings=None, astig=True, norm=None, log=print, keep_pairs=False):
    """(b) + (c) on per-channel localizations L (list of dicts, channel 0 = reference) of W x H channel crops.
    -> (PolyMap with fwd / inv for channels 1..C-1, report). astig: the localizations come from the astigmatic model
    (zlim) or the symmetric one (zlim_sym). keep_pairs: also return the pair arrays per channel (report['_pairs'],
    not json-serialisable)."""
    s = _settings(settings)
    t0 = time.time()
    R, order, C = int(s['roi']), int(s['order']), len(L)
    norm = Norm(*(PolyMap.default_norm(W, H) if norm is None else norm))
    zlim = s['zlim'] if astig else s['zlim_sym']
    Q, LQ, SC = {}, [], {}
    for c in range(C):
        m, q = quality(L[c], R, k_dev=float(s['k_dev']), max_crlb=float(s['max_crlb']), zlim=None if zlim is None else tuple(zlim))
        Q[str(c)] = q; LQ.append(subset(L[c], m))
        SC[str(c)] = self_consistency(LQ[c])
    rep = dict(order=order, norm=list(norm.c), astig=bool(astig), quality=Q, self=SC, channels={})
    fwd, inv, pairs = {}, {}, {}
    for c in range(1, C):
        reg = register_pair(LQ[0], LQ[c], W, H, norm, order, flips=tuple(s['flips']), r1=float(s['r1']), k=float(s['k_sigma']),
                            K=int(s['kfold']))
        sm = summarize(reg, LQ[0], LQ[c], f0=SC['0']['pull_rsd'], fc=SC[str(c)]['pull_rsd'])
        sig = {k: round(v['signif'], 1) for k, v in reg['coarse'].items()}
        others = [v for k, v in sig.items() if k != reg['flip']]
        cf1, _ = robust_poly(reg['src'], reg['dst'], 1 / np.sqrt(0.5 * (LQ[0]['cx'][reg['i0']] ** 2 + LQ[c]['cx'][reg['ic']] ** 2) + 1e-4), norm, 1)
        sm.update(flip=reg['flip'], coarse_shift=np.round(reg['shift'], 3).tolist(), coarse_signif=sig,
                  flip_signif_ratio=float(sig[reg['flip']] / max(max(others), 1.0)) if others else None,
                  roundtrip_rms=reg['roundtrip_rms'], affine=affine_of(cf1, norm))
        rep['channels'][str(c)] = sm
        fwd[c], inv[c] = unflip_coefs(reg['fwd'], reg['inv'], reg['flip'], W, H, norm, order)
        if keep_pairs: pairs[c] = {k: reg[k] for k in ('i0', 'ic', 'src', 'dst', 'resid_ho', 'keep')}
        log(f"registration: ch {c} flip {reg['flip']} (signif {sig}), shift {sm['coarse_shift']}, {sm['n_pairs']} pairs, "
            f"held-out rsd {sm['resid_rsd']} px, CRLB {sm['crlb_expected']}, pull {sm['pull_rsd']} (emp {sm['pull_emp_rsd']}), "
            f"round trip {sm['roundtrip_rms']:.2e} px")
    rep['time_register_s'] = round(time.time() - t0, 1)
    if keep_pairs: rep['_pairs'] = pairs
    return PolyMap(fwd, inv, order, norm.c), rep


def register_insitu(reader, camera, settings, xlim=None, pixel_um=None, dev='cuda', log=print, keep_pairs=False):
    """the in-situ registration of one movie (file): reader = ChannelReader (read(f0, f1) -> [n, C, H, W] ADU, shape
    (C, H, W)); camera = dict(offset, gain, readnoise_e) (offset / gain scalars, [C] or maps [C, H, W] of the channel
    crops); settings = the 'registration' config section (missing keys: INSITU_DEFAULTS); xlim = (x0, x1): detect only
    in these columns of the crops (the illuminated field); keep_pairs: also the pair arrays per channel (report['_pairs'],
    not json-serialisable: score another map on the same pairs). -> (PolyMap c -> 0 for c = 1..C-1 with a separately fitted
    inverse, feats normalised by the crop centre; report: json-serialisable dict)"""
    C, H, W = reader.shape
    if C < 2: raise ValueError('register_insitu: at least two channels')
    L, info = localize_channels(reader, camera, settings, xlim=xlim, pixel_um=pixel_um, dev=str(dev), log=log)
    pm, rep = register_locs(L, W, H, settings, astig=info['astig'], log=log, keep_pairs=keep_pairs)
    rep = dict(info, **rep, channel_shape=[C, H, W], xlim=None if xlim is None else list(xlim),
               settings={k: v for k, v in _settings(settings).items() if k not in ('mode', 'map')})
    return pm, rep


def checks(per_file_reports, limits=None):
    """pipeline checks of the in-situ registration reports {file: report} -> [dict(name, value, ok, warn_only)]:
    held-out pull (rsd, x and y; warn: the excess is mostly emitter-specific, not map error), pairs per channel, inverse
    round trip, flip significance ratio (chosen flip vs the best other one), self-consistency pull per channel (warn)."""
    lim = dict(CHECK_LIMITS, **(limits or {}))
    out = []
    for fi, rep in per_file_reports.items():
        if not rep or 'channels' not in rep:
            out.append(dict(name=f'file {fi}: registration report', value=None, ok=False, warn_only=False)); continue
        for c, s in rep['channels'].items():
            p = f'file {fi} ch {c}'
            out.append(dict(name=f'{p}: held-out pull rsd < {lim["pull_max"]}', value=max(s['pull_rsd']), ok=max(s['pull_rsd']) < lim['pull_max'], warn_only=True))
            out.append(dict(name=f'{p}: pairs >= {lim["min_pairs"]}', value=s['n_pairs'], ok=s['n_pairs'] >= lim['min_pairs'], warn_only=False))
            out.append(dict(name=f'{p}: inverse round trip < {lim["roundtrip_max"]} px', value=s['roundtrip_rms'], ok=s['roundtrip_rms'] < lim['roundtrip_max'], warn_only=False))
            fr = s.get('flip_signif_ratio')
            if fr is not None:
                out.append(dict(name=f'{p}: flip {s["flip"]} significance ratio > {lim["flip_ratio_min"]}', value=fr, ok=fr > lim['flip_ratio_min'], warn_only=False))
        for c, q in rep.get('self', {}).items():
            v = max(q['pull_rsd']) if q['n'] else None
            out.append(dict(name=f'file {fi} ch {c}: self-consistency pull rsd < {lim["self_pull_max"]}', value=v,
                            ok=v is not None and v < lim['self_pull_max'], warn_only=True))
    return out
