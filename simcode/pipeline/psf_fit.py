"""In-situ PSF fit of the automatic pipeline (stage psf; docs/QUICKSTART.md, docs/PSF_POLARIZATION.md).

fit(tracks, out, ...) = archive-2026-11-15:simcode/calibration/fit_psf_tracks.py as a function (same algorithm, defaults, random draws), but with the
optics and the camera constants as arguments (fit_psf_tracks hard-codes wavelength 0.70 um, pixel 0.10833 um, offset 99.3 ADU,
read noise 1.9 e). The fit itself is torch_psf fit_tracks (variable projection: per-frame I and per-(sample, pattern
angle) bg solved in closed form, per-sample x, y, z and the PSF parameters by Adam; it also fits the line blur); here: the
sample selection (torch_psf split_checkerboard for the checkerboard split), the camera conversion, the pattern-angle
groups (frame % 6 // 3) and the photoselection hook (dip_tensor: gamma, zeta per pattern angle). Supported terms: Zernike
5-21 + focus, blur 'none' | 'line', photoselection 0 | 3, --init-fit / --freeze-base. Writes <out>_psf.pt / _psf.json in
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


def tracks_of(D, idx, gain, offset_adu, dev):
    """(Tracks in photons, pattern-angle groups [B, T]) of the samples idx of a tracks.npz"""
    mask = torch.tensor(D['mask'][idx], device=dev); frames = torch.tensor(D['frames'][idx], device=dev)
    rois = (torch.tensor(D['rois'][idx].astype(np.float32), device=dev) - float(offset_adu)) * gain
    tr = Tracks(rois, mask, frames, torch.tensor(D['corner'][idx], device=dev, dtype=torch.float32),
                torch.tensor(D['xy_init'][idx], device=dev, dtype=torch.float32))
    return tr, torch.where(mask, (frames % 6) // 3, torch.zeros_like(frames))


def select(D, nsamples, parity=None, tile=16, rng=np.random):
    """nsamples random samples (sorted), of one checkerboard parity (0 / 1; torch_psf split_checkerboard of the track
    positions) or of all; rng: np.random (fit: the np.random.seed(0) state of fit_psf_tracks) or a RandomState"""
    if parity is None:
        return np.sort(rng.permutation(len(D['rois']))[:nsamples])
    pos_ = D['corner'] + D['xy_init']
    cand = np.flatnonzero(split_checkerboard(pos_[:, 0], pos_[:, 1], tile) == parity)
    return np.sort(cand[rng.permutation(len(cand))[:nsamples]])


def fit(tracks, out, optics, gain, nsamples=8000, iters=600, zmax=0.2, blur='none', photosel=0, kdir=None, init_fit=None,
        freeze_base=False, init_zcoef=None, readnoise_e=1.9, offset_adu=99.3, split='', tile=16, dev='cuda', log=print,
        return_state=False):
    """split: '' (all samples) or 'cb0' / 'cb1' (checkerboard parity 0 / 1)"""
    torch.manual_seed(0); np.random.seed(0)
    D = np.load(tracks) if isinstance(tracks, str) else tracks
    R = D['rois'].shape[-1]
    F = Fit(R, optics, dev, blur, photosel, kdir, init_fit, freeze_base, init_zcoef, readnoise_e, offset_adu, zmax=zmax)
    sel = select(D, nsamples, int(split[2]) if split else None, tile)
    tr, groups = tracks_of(D, sel, gain, F.G['offset'], dev)
    m, G = F.model, F.G
    m.blur = F.BL0                                                          # fixed blur of an init fit (None = none)
    hook = (lambda k: dict(dipM=F.dip_tensor(k))) if F.GAM is not None else None
    logf = lambda it, tot, t, zc: (it % 100 == 0 or it == iters - 1) and log(
        f'{it:5d} nll/px {tot:.4f}  |Z| {zc.norm().item():.3f}')
    torch.cuda.synchronize(); t0 = time.time()
    est = fit_tracks(tr, m, readnoise_e=float(torch.exp(G['logrn'])), zmax_um=zmax, iters=iters, warmup=100,
                     focus_um=float(G['focus'].detach()), blur=blur, groups=groups, ngroups=2, group_psf_kwargs=hook,
                     extra_params=[v for v in (F.GAM, F.ZETA) if v is not None], fix_zernike=not m.zcoef.requires_grad,
                     fix_focus=not G['focus'].requires_grad, log=logf)
    torch.cuda.synchronize(); t_fit = time.time() - t0
    m.blur = None
    with torch.no_grad():
        m.zcoef.copy_(est['psf'].zcoef); G['focus'].fill_(est['report']['focus_um'])
    if blur == 'line': F.BL0 = est['psf'].blur                              # fitted line blur -> fixed covariance
    rep = est['report']
    res = dict(gain=gain, readnoise_e=torch.exp(G['logrn']).item(), offset=G['offset'].item(), focus=G['focus'].item(),
               NA=m.NA.item(), apod=m.apod.tolist(), zernike=m.zcoef.tolist(), resid_var=rep['resid_var'], zmax=zmax,
               blur_mode=blur, photosel=photosel, split=split, tile=tile, nll_per_px=rep['nll_per_px'],
               nll_per_px_last_iter=rep['nll_per_px_before_last_step'], n_train=int(len(sel)), fit_s=t_fit,
               z_median=float(est['z'].median()), optics=optics, **F.extras_json())
    log(f'normalized residual var {res["resid_var"]:.4f}, nll/px (vectorial) {res["nll_per_px"]:.5f}, fit {t_fit:.0f} s')
    if out:
        json.dump(res, open(out + '_psf.json', 'w'), indent=1)
        ex = F.extras_json()
        torch.save(dict(zcoef=m.zcoef.detach().cpu(), NA=m.NA.detach().cpu(), apod=m.apod.detach().cpu(),
                        **{k: v.detach().cpu() for k, v in G.items()}, gain=gain, zmax=zmax, optics=optics,
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


def heldout(tracks, pts, test='cb1', nsamples=6000, iters=300, zstarts=(0.02, 0.08, 0.15), tile=16, seed=1, dev='cuda', log=print):
    """per-sample NLL / chi2 of several frozen PSFs on the same held-out samples (torch_psf heldout_tracks; the refit on
    the PSF's spline table, 8 x oversampled, 10 nm in z). Returns {name: dict(nll [n], chi2 [n], nll_per_px, chi2_per_px, n)}"""
    D = np.load(tracks) if isinstance(tracks, str) else tracks
    R = D['rois'].shape[-1]
    sel = select(D, nsamples, int(test[2]) if test else None, tile, np.random.RandomState(seed))
    out = {}
    for name, pt in pts.items():
        F, gain = model_from_pt(pt, R, dev)
        tr, groups = tracks_of(D, sel, gain, F.G['offset'], dev)
        vm = F.angle_models()
        with torch.no_grad():
            spl = [v.to_spline(R, (0.0, F.zmax), oversample=8, dz_um=0.01, zpad=3, xy_limit=2.0) for v in vm]
        h = heldout_tracks(tr, vm, readnoise_e=float(torch.exp(F.G['logrn'])), zmax_um=F.zmax, groups=groups, ngroups=2,
                           refit_psfs=spl, z_starts_um=zstarts, iters=iters)
        out[name] = {k: h[k] for k in ('nll', 'chi2', 'nll_per_px', 'chi2_per_px', 'n')}
        log(f'heldout {name}: nll/px {h["nll_per_px"]:.5f}, chi2/px {h["chi2_per_px"]:.4f} ({len(sel)} samples, {test})')
    return out


def paired(H, a, b):
    """paired NLL difference b - a per sample: mean, se, z (torch_psf paired_nll)"""
    p = paired_nll(H[a]['nll'], H[b]['nll'])
    return dict(mean=p['mean'], se=p['se'], z=p['z'])


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
