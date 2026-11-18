"""In-situ PSF fit of the automatic pipeline (stage psf; docs/QUICKSTART.md, docs/PSF_POLARIZATION.md).

fit(tracks, out, ...) = archive-2026-11-15:simcode/calibration/fit_psf_tracks.py as a function (same algorithm and defaults),
but with the optics and the camera constants as arguments (fit_psf_tracks hard-codes wavelength 0.70 um, pixel 0.10833 um, offset 99.3 ADU,
read noise 1.9 e), and with a stable sample draw (2026-11-18, select(): by a hash of each sample's own key instead of a
permutation of the list, so a few added / removed samples change only those few). The fit itself is torch_psf fit_tracks (variable projection: per-frame I and per-(sample, pattern
angle) bg solved in closed form, per-sample x, y, z and the PSF parameters by Adam; it also fits the line blur); here: the
sample selection (select(); torch_psf split_checkerboard for the checkerboard split), the camera conversion, the pattern-angle
groups (frame % 6 // 3) and the photoselection hook (dip_tensor: gamma, zeta per pattern angle). Supported terms: Zernike
5-21 + focus, blur 'none' | 'line' | 'cov' (general covariance), photoselection 0 | 3, --init-fit / --freeze-base. Writes <out>_psf.pt / _psf.json in
the fit_psf_tracks format.

heldout(...): the PSFs frozen, x, y, z refit per sample (torch_psf heldout_tracks: spline-table refit, multi-start in z,
NLL / chi2 per sample with the vectorial model); the same samples for every PSF (paired comparison: paired()).
pattern_from_intensities(...): k per orientation (FFT + DFT refinement of the 3-step phasor over the pitch range), per-step
phase / modulation / relative intensity.
"""
import json, math, time
import numpy as np, torch
from simcode import psf as spsf
from simcode.psf import Tracks, fit_tracks, heldout_tracks, paired_nll, split_checkerboard


class Fit:
    """the PSF model of one fit: the VectorialPSF (Zernikes, NA, apodization), the focus / read noise / offset (G), a fixed
    line blur (BL0: the fitted blur of an init fit, or the result of a blur='line' fit) and the photoselection terms
    (GAM, ZETA, per pattern angle)"""
    def __init__(self, R, optics, dev='cuda', blur='none', photosel=0, kdir=None, init_fit=None, freeze_base=False,
                 init_zcoef=None, readnoise_e=1.9, offset_adu=99.3, gamma_init=0.05, zmax=0.2):
        self.dev, self.R, self.c, self.zmax = dev, R, (R - 1) / 2, zmax
        self.model = spsf.VectorialPSF.from_optics(spsf.Optics.from_dict(optics), R, oversample=2, device=dev)
        m = self.model
        if init_zcoef is not None:
            with torch.no_grad(): m.zcoef.copy_(torch.as_tensor(init_zcoef).to(dev))
        m.zcoef.requires_grad_(True); m.apod.requires_grad_(False); m.NA.requires_grad_(False)
        self.G = G = dict(focus=torch.tensor(0.0, device=dev), logrn=torch.tensor(math.log(readnoise_e), device=dev),
                          offset=torch.tensor(float(offset_adu), device=dev))
        G['focus'].requires_grad_(True)
        self.BL0 = None
        if init_fit:
            S0 = torch.load(init_fit, weights_only=False) if isinstance(init_fit, str) else init_fit
            with torch.no_grad():
                m.zcoef.copy_(S0['zcoef'].to(dev)); G['focus'].fill_(float(S0['focus']))
            if 'blur_px2' in S0.get('extras', {}): self.BL0 = torch.tensor(S0['extras']['blur_px2'][0], device=dev)
            if freeze_base:
                m.zcoef.requires_grad_(False); G['focus'].requires_grad_(False)
        self.GAM = self.ZETA = None
        if photosel:
            assert photosel == 3 and kdir is not None
            kc = torch.tensor(kdir, device=dev, dtype=torch.float32)
            KD = kc / kc.norm(dim=1, keepdim=True)
            self.EDIR = torch.stack([-KD[:, 1], KD[:, 0]], 1)
            self.GAM = torch.tensor(gamma_init, device=dev, requires_grad=True)
            self.ZETA = torch.tensor(1 / 3, device=dev, requires_grad=True)
        self.blur_mode, self.photosel = blur, photosel

    # ---- per-angle PSF terms (= fit_psf_tracks.dip_tensor / psf_of)
    def dip_tensor(self, k):
        e = torch.zeros(3, device=self.dev); e[:2] = self.EDIR[k]
        Z = self.ZETA
        base = torch.diag(torch.stack([(1 - Z) / 2, (1 - Z) / 2, Z]))
        return (1 - self.GAM) * base + self.GAM * e[:, None] * e[None, :]

    def n_angles(self):
        return 2 if self.GAM is not None else 1

    def angle_models(self):
        """one VectorialPSF per pattern angle at the fitted focus, with the fixed blur and the angle's dipole tensor set"""
        out = []
        for k in range(self.n_angles()):
            v = self.model.resized(self.R); v.focus_um = float(self.G['focus'])
            v.blur = self.BL0
            v.dipM = self.dip_tensor(k).detach() if self.GAM is not None else None
            out.append(v)
        return out

    def extras_json(self):
        r = {}
        if self.BL0 is not None: r['blur_px2'] = [self.BL0.cpu().tolist()]
        if self.GAM is not None:
            r['gamma'] = self.GAM.item(); r['dipole_M'] = [self.dip_tensor(k).detach().cpu().tolist() for k in range(2)]
            r['e_dir'] = self.EDIR.cpu().tolist(); r['zeta'] = self.ZETA.item()
        return r


def load_maps(path):
    """per-pixel camera maps of an npz (simcode.calibration.camera_maps / the calibration camera maps): dict(offset_adu
    [H, W], readnoise_e [H, W], path); None for None"""
    if not path: return None
    M = np.load(path)
    return dict(offset_adu=np.asarray(M['offset_adu'], np.float32), readnoise_e=np.asarray(M['readnoise_e'], np.float32), path=str(path))


def crop_maps(D, idx, maps):
    """(offset [B, R, R] ADU, read noise [B, R, R] e) of the samples idx at their crops (rois = movie[cy:cy+R, cx:cx+R])"""
    R = D['rois'].shape[-1]; c = np.asarray(D['corner'])[idx]
    ii = c[:, 1][:, None, None] + np.arange(R)[None, :, None]; jj = c[:, 0][:, None, None] + np.arange(R)[None, None, :]
    return maps['offset_adu'][ii, jj], maps['readnoise_e'][ii, jj]


def tracks_of(D, idx, gain, offset_adu, dev, maps=None):
    """(Tracks in photons, pattern-angle groups [B, T], read noise) of the samples idx of a tracks.npz. maps (load_maps):
    the per-pixel offset map is subtracted instead of offset_adu and the read noise is the per-pixel map at each crop
    ([B, R, R] tensor, e rms; fit_tracks / heldout_tracks model var = mu + rn^2 per pixel); without maps: None (the
    caller's scalar read noise)"""
    mask = torch.tensor(D['mask'][idx], device=dev); frames = torch.tensor(D['frames'][idx], device=dev)
    rn = None
    if maps is not None:
        off, rnm = crop_maps(D, idx, maps)
        rois = (torch.tensor(D['rois'][idx].astype(np.float32), device=dev) - torch.tensor(off, device=dev)[:, None]) * gain
        rn = torch.tensor(rnm, device=dev)
    else:
        rois = (torch.tensor(D['rois'][idx].astype(np.float32), device=dev) - float(offset_adu)) * gain
    tr = Tracks(rois, mask, frames, torch.tensor(D['corner'][idx], device=dev, dtype=torch.float32),
                torch.tensor(D['xy_init'][idx], device=dev, dtype=torch.float32))
    return tr, torch.where(mask, (frames % 6) // 3, torch.zeros_like(frames)), rn


def sample_keys(D):
    """[B, 3] int64 key of each sample that depends only on the sample itself: its first frame and its crop corner (x, y).
    (The samples of one track share the corner and differ in the first frame; two isolated tracks cannot share both.)"""
    fr, co = np.asarray(D['frames']), np.asarray(D['corner'])
    return np.stack([fr[:, 0], co[:, 0], co[:, 1]], 1).astype(np.int64)


_GOLD = np.uint64(0x9E3779B97F4A7C15)


def _mix64(h):
    """splitmix64 finaliser of a uint64 array (wrap-around arithmetic)"""
    h = (h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return h ^ (h >> np.uint64(31))


def sample_hash(D, salt=0):
    """uint64 pseudo-random number per sample from its key (sample_keys) and the salt; uses no random-number state"""
    K = np.ascontiguousarray(sample_keys(D)).view(np.uint64)
    with np.errstate(over='ignore'):
        h = _mix64(np.full(len(K), _GOLD * np.uint64(int(salt) + 1), dtype=np.uint64))
        for j in range(K.shape[1]):
            h = _mix64(h ^ (K[:, j] + _GOLD))
    return h


def candidates(D, parity=None, tile=16):
    """indices of the samples of one checkerboard parity (0 / 1; torch_psf split_checkerboard of the track positions), or all"""
    if parity is None: return np.arange(len(D['rois']))
    pos_ = D['corner'] + D['xy_init']
    return np.flatnonzero(split_checkerboard(pos_[:, 0], pos_[:, 1], tile) == parity)


def select(D, nsamples, parity=None, tile=16, salt=0):
    """nsamples pseudo-random samples (sorted indices) of one checkerboard parity or of all (candidates()): the nsamples
    candidates with the smallest sample_hash(D, salt). A sample's rank depends only on its own key (first frame, crop
    corner), not on its position in the list, so adding or removing a few samples changes only those few in the
    selection (a removed selected sample is replaced by the next one). Another salt gives another, independent draw."""
    cand = candidates(D, parity, tile)
    h, K = sample_hash(D, salt)[cand], sample_keys(D)[cand]
    o = np.lexsort((K[:, 2], K[:, 1], K[:, 0], h))       # by hash; equal hashes (never in practice) by the key
    return np.sort(cand[o[:nsamples]])


def fit(tracks, out, optics, gain, nsamples=8000, iters=600, zmax=0.2, blur='none', photosel=0, kdir=None, init_fit=None,
        freeze_base=False, init_zcoef=None, readnoise_e=1.9, offset_adu=99.3, split='', tile=16, dev='cuda', log=print,
        return_state=False, salt=0, maps=None):
    """split: '' (all samples) or 'cb0' / 'cb1' (checkerboard parity 0 / 1); salt: the sample draw (select()); maps:
    per-pixel camera maps (load_maps; then offset_adu / readnoise_e are not used for the data)"""
    torch.manual_seed(0); np.random.seed(0)
    D = np.load(tracks) if isinstance(tracks, str) else tracks
    R = D['rois'].shape[-1]
    F = Fit(R, optics, dev, blur, photosel, kdir, init_fit, freeze_base, init_zcoef, readnoise_e, offset_adu, zmax=zmax)
    sel = select(D, nsamples, int(split[2]) if split else None, tile, salt)
    tr, groups, rnp = tracks_of(D, sel, gain, F.G['offset'], dev, maps)
    m, G = F.model, F.G
    m.blur = F.BL0                                                          # fixed blur of an init fit (None = none)
    hook = (lambda k: dict(dipM=F.dip_tensor(k))) if F.GAM is not None else None
    logf = lambda it, tot, t, zc: (it % 100 == 0 or it == iters - 1) and log(
        f'{it:5d} nll/px {tot:.4f}  |Z| {zc.norm().item():.3f}')
    torch.cuda.synchronize(); t0 = time.time()
    est = fit_tracks(tr, m, readnoise_e=float(torch.exp(G['logrn'])) if rnp is None else rnp, zmax_um=zmax, iters=iters, warmup=100,
                     focus_um=float(G['focus'].detach()), blur=blur, groups=groups, ngroups=2, group_psf_kwargs=hook,
                     extra_params=[v for v in (F.GAM, F.ZETA) if v is not None], fix_zernike=not m.zcoef.requires_grad,
                     fix_focus=not G['focus'].requires_grad, log=logf)
    torch.cuda.synchronize(); t_fit = time.time() - t0
    m.blur = None
    with torch.no_grad():
        m.zcoef.copy_(est['psf'].zcoef); G['focus'].fill_(est['report']['focus_um'])
    if blur in ('line', 'cov'): F.BL0 = est['psf'].blur                     # fitted blur -> fixed covariance
    rep = est['report']
    res = dict(gain=gain, readnoise_e=torch.exp(G['logrn']).item(), offset=G['offset'].item(), focus=G['focus'].item(),
               NA=m.NA.item(), apod=m.apod.tolist(), zernike=m.zcoef.tolist(), resid_var=rep['resid_var'], zmax=zmax,
               blur_mode=blur, photosel=photosel, split=split, tile=tile, salt=salt, nll_per_px=rep['nll_per_px'],
               nll_per_px_last_iter=rep['nll_per_px_before_last_step'], n_train=int(len(sel)), fit_s=t_fit,
               z_median=float(est['z'].median()), optics=optics, maps=maps['path'] if maps else None, **F.extras_json())
    log(f'normalized residual var {res["resid_var"]:.4f}, nll/px (vectorial) {res["nll_per_px"]:.5f}, fit {t_fit:.0f} s')
    if out:
        json.dump(res, open(out + '_psf.json', 'w'), indent=1)
        ex = F.extras_json()
        torch.save(dict(zcoef=m.zcoef.detach().cpu(), NA=m.NA.detach().cpu(), apod=m.apod.detach().cpu(),
                        **{k: v.detach().cpu() for k, v in G.items()}, gain=gain, zmax=zmax, optics=optics, maps=maps['path'] if maps else None,
                        **({'extras': ex} if ex else {})), out + '_psf.pt')
    if return_state:
        st = dict(I=est['I'].cpu(), r=(est['xy'] + F.c + tr.corner).cpu(), z=est['z'].cpu(), frames=tr.frames.cpu(),
                  mask=tr.mask.cpu(), sel=sel)
        return res, st
    return res


def model_from_pt(pt, R, dev='cuda'):
    """frozen Fit object from a _psf.pt (this module's or fit_psf_tracks's; optics default = the DMD setup)"""
    S = torch.load(pt, weights_only=False) if isinstance(pt, str) else pt
    o = S.get('optics') or dict(pixel_um=0.10833, NA=float(S['NA']), wavelength_um=0.70, n_imm=1.518, n_med=1.33)
    o = dict(o); o['NA'] = float(S['NA'])
    F = Fit(R, o, dev, zmax=float(S.get('zmax', 0.2)), readnoise_e=float(torch.exp(torch.as_tensor(S['logrn']))), offset_adu=float(S['offset']))
    with torch.no_grad():
        F.model.zcoef.copy_(S['zcoef'].to(dev)); F.G['focus'].fill_(float(S['focus']))
    ex = S.get('extras', {})
    if 'blur_px2' in ex: F.BL0 = torch.tensor(ex['blur_px2'][0], device=dev)
    if 'dipole_M' in ex:
        M = torch.tensor(ex['dipole_M'], device=dev)
        F.GAM = torch.tensor(0.0, device=dev)       # marks per-angle; dip_tensor overridden below
        F.dip_tensor = lambda k, M=M: M[k]
    for p in [F.model.zcoef, F.G['focus']]: p.requires_grad_(False)
    return F, float(S['gain'])


def heldout(tracks, pts, test='cb1', nsamples=6000, iters=300, zstarts=(0.02, 0.08, 0.15), tile=16, salt=1, dev='cuda', log=print,
            maps=None, keep_z=False):
    """per-sample NLL / chi2 of several frozen PSFs on the same held-out samples (torch_psf heldout_tracks; the refit on
    the PSF's spline table, 8 x oversampled, 10 nm in z); the test samples = select(D, nsamples, test parity, salt).
    maps: per-pixel camera maps (load_maps) of the data model (the same for every PSF; None: each PSF's scalar offset /
    read noise). Returns {name: dict(nll [n], chi2 [n], nll_per_px, chi2_per_px, n[, z, I])}"""
    D = np.load(tracks) if isinstance(tracks, str) else tracks
    R = D['rois'].shape[-1]
    sel = select(D, nsamples, int(test[2]) if test else None, tile, salt)
    out = {}
    for name, pt in pts.items():
        F, gain = model_from_pt(pt, R, dev)
        tr, groups, rnp = tracks_of(D, sel, gain, F.G['offset'], dev, maps)
        vm = F.angle_models()
        with torch.no_grad():
            spl = [v.to_spline(R, (0.0, F.zmax), oversample=8, dz_um=0.01, zpad=3, xy_limit=2.0) for v in vm]
        h = heldout_tracks(tr, vm, readnoise_e=float(torch.exp(F.G['logrn'])) if rnp is None else rnp, zmax_um=F.zmax, groups=groups, ngroups=2,
                           refit_psfs=spl, z_starts_um=zstarts, iters=iters)
        out[name] = {k: h[k] for k in ('nll', 'chi2', 'nll_per_px', 'chi2_per_px', 'n') + (('z', 'I') if keep_z else ())}
        log(f'heldout {name}: nll/px {h["nll_per_px"]:.5f}, chi2/px {h["chi2_per_px"]:.4f} ({len(sel)} samples, {test})')
    return out


def paired(H, a, b):
    """paired NLL difference b - a per sample: mean, se, z (torch_psf paired_nll)"""
    p = paired_nll(H[a]['nll'], H[b]['nll'])
    return dict(mean=p['mean'], se=p['se'], z=p['z'])


def replicate_values(fits, H=None, names=('base', 'bl', 'ps3')):
    """the numbers of one replicate of the fit chain (fits: {name: fit() result}; H: heldout() of its PSFs or None) whose
    sampling spread spread() reports: focus / |Z| per fit, the Zernikes and line blur of bl (the final PSF's base),
    gamma / zeta of ps3, held-out NLL / px per fit and the paired differences (mean NLL per sample) of consecutive fits"""
    v = {}
    for n in names:
        if n not in fits or n == 'ps3': continue
        v[f'{n}.focus_um'] = fits[n]['focus']; v[f'{n}.zernike_norm'] = float(np.linalg.norm(fits[n]['zernike']))
    b = fits.get('bl', fits.get('base'))
    v['bl.zernike'] = list(b['zernike'])
    if 'blur_px2' in b: v['bl.blur_sxx_syy_sxy'] = list(b['blur_px2'][0])
    if 'ps3' in fits:
        v['ps3.gamma'] = fits['ps3'].get('gamma'); v['ps3.zeta'] = fits['ps3'].get('zeta')
    v['fit_s'] = float(sum(fits[n].get('fit_s', 0.0) for n in names if n in fits))
    if H is not None:
        hn = [n for n in names if n in H]
        for n in hn: v[f'heldout.{n}.nll_per_px'] = H[n]['nll_per_px']
        for a, c in zip(hn[:-1], hn[1:]): v[f'heldout.{c}_minus_{a}'] = paired(H, a, c)['mean']
    return v


def spread(reps, n, N, nsamples=None):
    """sampling spread of the replicate_values of k draws of n samples each from the same N candidates (select() with
    salts 0 .. k-1; the draws overlap when n is not << N). Per quantity: mean, sd_draws (sd over the draws) and sd =
    sd_draws / sqrt(1 - n / N): the sd of the fit for independent sets of n samples (the variance between draws of n of
    N is (1 / n - 1 / N) / (1 / n) of it, sampling without replacement; None when n >= N, the draws are all the same).
    List quantities (Zernikes, blur) per element, plus for the Zernikes the rms wavefront change sqrt(sum sd^2) (waves).
    nsamples (default n): the size of the fit the sd is for; != n: sd x sqrt(n / nsamples) (1 / sqrt(n) scaling of the
    sampling sd; sd_draw_size = the sd for independent sets of n)."""
    k = len(reps); fpc = 1 / math.sqrt(1 - n / N) if n < N else None
    nsamples = nsamples or n; sc = math.sqrt(n / nsamples)
    out = dict(k=k, n=int(n), n_candidates=int(N), nsamples=int(nsamples), fpc=fpc, scale=sc, mean={}, sd_draws={}, sd={})
    if sc != 1: out['sd_draw_size'] = {}
    for q in reps[0]:
        if q in ('fit_s', 'wall_s') or reps[0][q] is None: continue
        X = np.array([r[q] for r in reps], dtype=float)
        m, sdr = X.mean(0), X.std(0, ddof=1) if k > 1 else np.zeros_like(X[0])
        sdn = sdr * fpc if fpc is not None else None
        tl = (lambda a: a.tolist() if np.ndim(a) else float(a))
        rms = lambda a: float(np.sqrt(np.sum(a ** 2)))
        out['mean'][q] = tl(m); out['sd_draws'][q] = tl(sdr); out['sd'][q] = tl(sdn * sc) if sdn is not None else None
        if sc != 1: out['sd_draw_size'][q] = tl(sdn) if sdn is not None else None
        if q == 'bl.zernike':
            out['sd_draws']['bl.zernike_rms'] = rms(sdr)
            out['sd']['bl.zernike_rms'] = rms(sdn * sc) if sdn is not None else None
            if sc != 1: out['sd_draw_size']['bl.zernike_rms'] = rms(sdn) if sdn is not None else None
    out['fit_s_per_draw'] = float(np.mean([r['fit_s'] for r in reps]))
    w = [r['wall_s'] for r in reps if 'wall_s' in r]
    if w: out['wall_s_per_draw'] = float(np.mean(w))          # fits + held-out test of one extra draw
    return out


def pattern_from_intensities(st, pitch_px=(1.6, 2.8), dev='cuda', log=print):
    """k [2,2] rad/px, per-step phase [6], modulation [6], relint [6] from per-frame track intensities (blocked order:
    slot = frame % 6, orientation = slot // 3). Model I_s = A relint_s (1 + m_s cos(k_a . r - phi_s)); positions = pixel-centre
    coordinates of the movie."""
    I, fr, mk, r = st['I'].numpy(), st['frames'].numpy(), st['mask'].numpy(), st['r'].numpy()
    med = np.array([np.median(I[b][mk[b]]) if mk[b].any() else 0 for b in range(len(I))])
    on = mk & (I > 0.25 * med[:, None])
    res = dict(k=[], phase=[0.0] * 6, mod=[0.0] * 6, relint=[0.0] * 6, n=[], pitch_px=[])
    for a in range(2):
        rows, vs = [], []
        for b in range(len(I)):
            f = fr[b]
            for t in range(I.shape[1] - 2):
                if not mk[b, t] or f[t] % 6 != 3 * a: continue
                if not (on[b, t:t + 3].all() and f[t + 2] == f[t] + 2): continue
                v = I[b, t:t + 3]; rows.append(b); vs.append(v / v.mean())
        rows = np.array(rows); vs = np.array(vs)
        if len(rows) < 200: raise RuntimeError(f'pattern: only {len(rows)} complete fully-on triples for orientation {a}')
        z = (vs * np.exp(-2j * np.pi * np.arange(3) / 3)).sum(1)
        p = r[rows]
        # FFT of the phasor binned at 0.25 px, peak inside the pitch annulus (both signs of k), then DFT refinement
        h = 0.25; L = int(np.ceil((p.max(0) - p.min(0)).max() / h)) + 1; Nf = 1 << int(np.ceil(np.log2(2 * L)))
        ij = np.round((p - p.min(0)) / h).astype(int)
        g = np.zeros((Nf, Nf), complex); np.add.at(g, (ij[:, 1], ij[:, 0]), z)
        Fg = np.abs(np.fft.fft2(g))
        kf = 2 * np.pi * np.fft.fftfreq(Nf, d=h)
        KX, KY = np.meshgrid(kf, kf)
        kk = np.hypot(KX, KY)
        ok = (kk > 2 * np.pi / pitch_px[1]) & (kk < 2 * np.pi / pitch_px[0])
        j = np.argmax(np.where(ok, Fg, 0))
        k0 = np.array([KX.flat[j], KY.flat[j]])        # fft2 kernel exp(-i k.r): peak at k where z ~ exp(+i k.r)
        P = torch.tensor(p, device=dev, dtype=torch.float64); Z = torch.tensor(z, device=dev)
        kt = torch.tensor(k0, device=dev, dtype=torch.float64, requires_grad=True)
        o = torch.optim.LBFGS([kt], lr=0.5, max_iter=100, line_search_fn='strong_wolfe')
        def clo():
            o.zero_grad(); v = -(Z * torch.exp(-1j * (P @ kt))).sum().abs() / len(Z); v.backward(); return v
        o.step(clo)
        k = kt.detach().cpu().numpy()
        ph = p @ k
        for s in range(3):
            X = np.stack([np.ones_like(ph), np.cos(ph), np.sin(ph)], 1)
            c = np.linalg.lstsq(X, vs[:, s], rcond=None)[0]
            res['phase'][3 * a + s] = float(np.arctan2(c[2], c[1])); res['mod'][3 * a + s] = float(np.hypot(c[1], c[2]) / c[0])
            res['relint'][3 * a + s] = float(c[0])
        res['k'].append(k.tolist()); res['n'].append(int(len(rows))); res['pitch_px'].append(float(2 * np.pi / np.hypot(*k)))
        log(f'pattern orientation {a}: k ({k[0]:.5f}, {k[1]:.5f}) rad/px, pitch {res["pitch_px"][-1]:.4f} px, {len(rows)} triples, '
            f'phases {np.round(res["phase"][3*a:3*a+3], 3).tolist()}, mod {np.round(res["mod"][3*a:3*a+3], 3).tolist()}')
    # relint: per-orientation mean of the triple-normalised values is 1 by construction; the angle ratio from the frame means
    sec = [0 if abs(k[0]) > abs(k[1]) else 1 for k in res['k']]
    res['sectors'] = sec; res['sector_ok'] = sec[0] != sec[1]
    return res


def centre_vs_z(st, D, px_nm, zbins=(0.0, 0.03, 0.06, 0.1, 0.15, 0.21)):
    """z-dependent lateral offset of the crop centre (docs/AUTO_PIPELINE.md section 5 note): d = vectorial-fit position minus
    the track's Gaussian centre (corner + xy_init), per sample, against the fitted z. Returns the slope (nm per 100 nm z) and
    the z-binned medians (nm) of dx, dy."""
    sel = st['sel']
    g = D['corner'][sel] + D['xy_init'][sel]
    d = (st['r'].numpy() - g) * px_nm; z = st['z'].numpy()
    out = dict(n=int(len(z)))
    for i, ax in enumerate('xy'):
        p = np.polyfit(z, d[:, i], 1); out[f'slope_{ax}_nm_per_100nm'] = float(p[0] * 0.1)
        out[f'median_{ax}_nm_by_z'] = [float(np.median(d[(z >= lo) & (z < hi), i])) if ((z >= lo) & (z < hi)).sum() > 50 else None
                                       for lo, hi in zip(zbins[:-1], zbins[1:])]
    out['z_bins_um'] = list(zbins)
    return out
