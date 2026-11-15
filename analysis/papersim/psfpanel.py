"""PSF panel + CRLB small print for render figures (papersim overviews; reusable for the real-data figures).

  info = psf_info(spec, photons, bg_px_frame, z_um=None, pattern=dict(pitch_nm=224, depth=0.9, angles_deg=(0, 90)), px_nm=108.33)
      spec = a PSF spec of analysis/baselines/zpsf.py ('gauss:<sigma px>', an in-situ calib .json, a z-stack .npz): the PSF the
      data were simulated / processed with. Returns the xy image at the emitter height z_um (default: the PSF's fit plane =
      the calib's emitter_z_median_um), an xz section through the emitter over the PSF's z range, the rms aberration of the
      calib's Zernike terms Noll >= 5 (mλ; None for a Gaussian PSF) and the CRLBs at `photons` per 6-frame window and
      `bg_px_frame` background photons / px / frame (Poisson, 16 x 16 px ROI, mean over 256 sub-pixel positions and pattern
      phases; numerical derivatives):
        SMLM     the 6-frame sum, parameters (x, y, [z,] I, bg)
        SIMFLUX  the 6 frames with the known pattern (2 angles x 3 steps, I / 6 (1 + depth cos(k . r - phi)) per frame, one
                 bg), parameters (x, y, [z,] I, bg)
      z is a free parameter for a 3D PSF (calib / z-stack): x, y, z CRLBs with z free (and x, y with z known, 'xy_zfixed');
      2D for a Gaussian PSF.
  draw(ax_xy, ax_xz, info, cmap='inferno')   log-scale images (3 decades), 500 nm scale bars; xz over z 0 - 0.8 um with
                                             the emitter height (cyan dotted) and the focal plane (white dashed) marked
  crlb_text(info)                            the small print (aberration, photons / bg used, CRLBs in nm)
  add_panel(fig, rect, info)                 both images + the text in a figure rectangle [x0, y0, w, h] (figure fraction)
  sim_psf_info(movie_dir, dexp)              psf_info for a papersim movie (its gen.json; _v2psf: detected photons + read noise)
  panel_label(info) / caption_text(info)     the paper style (analysis/paper/fig_exp_new.py): a two-line label on the figure,
                                             conditions / CRLBs / xz line legend in the caption text"""
import os, sys, json, math
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, 'analysis', 'baselines')); sys.path.insert(0, REPO)
import zpsf

P_ROI = 16
N_SAMPLES = 256
ZMAX_IMG = 0.8           # um, z range of the xz section


def _dev(device):
    return device or ('cuda' if torch.cuda.is_available() else 'cpu')


def rms_aberration_mwaves(spec):
    """rms of the Zernike coefficients Noll >= 5 (Noll-normalised, waves) of a calib json -> mλ; None if not a calib"""
    if not spec.endswith(('.json', '.zip')): return None
    from simcode.config import resolve
    from simcode.psf import read_calib
    c = read_calib(resolve(spec))
    n = np.asarray(c.get('zernike_noll', []), int); w = np.asarray(c.get('zernike_waves', []), float)
    if not len(w): return 0.0
    return float(np.sqrt(np.sum(w[n >= 5] ** 2)) * 1e3)


def _g(psf, xc, yc, P, z):
    return psf.eval(xc, yc, P, z)[0]


def _crlb(model, th, steps, var_add=0.0):
    """mean over samples of sqrt(diag(F^-1)); F from central differences of model(th) [n, M] (Poisson; var_add > 0: + Gaussian
    read noise of that variance per value, as the sCMOS shifted-Poisson approximation: F = J^T J / (mu + var_add))"""
    mu = model(th).clamp_min(1e-9) + var_add
    J = []
    for i, h in enumerate(steps):
        d = torch.zeros_like(th); d[:, i] = h
        J.append((model(th + d) - model(th - d)) / (2 * h))
    J = torch.stack(J, 1)                                                     # [n, p, M]
    F = torch.einsum('npm,nqm->npq', J / mu[:, None], J)
    C = torch.linalg.inv(F.double()).diagonal(dim1=1, dim2=2).clamp_min(0).sqrt()
    return C.mean(0).float().cpu().numpy()


def psf_info(spec, photons, bg_px_frame, z_um=None, pattern=None, px_nm=108.33, device=None, seed=0, rn2_e=0.0, detected=False, zmax=0.2):
    """(2026-11-15 opt-in) rn2_e: read-noise variance per pixel and frame (e^2) in the CRLB; detected: `photons` are detected
    photons (the intensity parameter = photons / collected fraction of the 41 x 41 px PSF image at z); zmax: PSF z range (um)"""
    dev = _dev(device)
    psf = zpsf.load(spec, device=dev, zmax=zmax)
    is3d = not spec.startswith('gauss:')
    z0 = float(psf.z_fit if z_um is None else z_um)
    pattern = dict(dict(pitch_nm=224.0, depth=0.9, angles_deg=(0.0, 90.0)), **(pattern or {}))
    # images
    Pi = 15; zero = torch.zeros(1, device=dev)
    xy = _g(psf, zero, zero, Pi, torch.full((1,), z0, device=dev))[0].cpu().numpy()
    # xz section over 0 - ZMAX_IMG um (a separate, taller knot stack of the same calib; a Gaussian PSF has no z dependence)
    pz = zpsf.render_calib(spec, dev, P=Pi + 4, zmax=ZMAX_IMG, hz=0.02) if spec.endswith('.json') else psf
    zs = np.linspace(0.0, ZMAX_IMG, 61)
    zt = torch.as_tensor(zs, device=dev, dtype=torch.float32)
    xz = _g(pz, torch.zeros_like(zt), torch.zeros_like(zt), Pi, zt)[:, Pi // 2, :].cpu().numpy()     # [nz, x]
    # CRLB
    rng = np.random.default_rng(seed); n = N_SAMPLES; P = P_ROI
    off = torch.as_tensor(rng.uniform(-0.5, 0.5, (n, 2)), device=dev, dtype=torch.float32)
    ph0 = torch.as_tensor(rng.uniform(0, 2 * np.pi, (n, len(pattern['angles_deg']))), device=dev, dtype=torch.float32)
    kmag = 2 * np.pi / (pattern['pitch_nm'] / px_nm)
    K = torch.tensor([[kmag * math.cos(math.radians(a)), kmag * math.sin(math.radians(a))] for a in pattern['angles_deg']], device=dev)
    I = float(photons); B = float(bg_px_frame); coll = None
    if detected and is3d:
        coll = float(_g(psf, zero, zero, 41, torch.full((1,), z0, device=dev))[0].sum()); I = I / coll
    def unpack(th):
        x, y = th[:, 0], th[:, 1]
        if is3d: z, I_, b = th[:, 2], th[:, 3], th[:, 4]
        else: z, I_, b = torch.full_like(x, z0), th[:, 2], th[:, 3]
        return x, y, z, I_, b
    def smlm(th):
        x, y, z, I_, b = unpack(th)
        return (I_[:, None, None] * _g(psf, x, y, P, z) + 6 * b[:, None, None]).flatten(1)
    def sf(th):
        x, y, z, I_, b = unpack(th)
        g = _g(psf, x, y, P, z)
        out = []
        for a in range(len(pattern['angles_deg'])):
            for s in range(3):
                ph = (K[a, 0] * x + K[a, 1] * y) - ph0[:, a] - 2 * np.pi * s / 3
                e = (1 + pattern['depth'] * torch.cos(ph)) / (3 * len(pattern['angles_deg']))
                out.append(I_[:, None, None] * e[:, None, None] * g + b[:, None, None])
        return torch.stack(out, 1).flatten(1)
    zc = torch.full((n,), z0, device=dev)
    cols = [off[:, 0], off[:, 1]] + ([zc] if is3d else []) + [torch.full((n,), I, device=dev), torch.full((n,), B, device=dev)]
    th = torch.stack(cols, 1)
    steps = [1e-3, 1e-3] + ([1e-3] if is3d else []) + [1e-2 * I, 1e-3 * max(B, 1)]
    with torch.no_grad():
        cs, cf = _crlb(smlm, th, steps, 6 * rn2_e), _crlb(sf, th, steps, rn2_e)
        cr = dict(smlm=dict(x=cs[0] * px_nm, y=cs[1] * px_nm), sf=dict(x=cf[0] * px_nm, y=cf[1] * px_nm))
        if is3d:
            cr['smlm']['z'] = cs[2] * 1e3; cr['sf']['z'] = cf[2] * 1e3
            # x, y with z known (the 2D fits at the fixed emitter plane)
            th2 = th[:, [0, 1, 3, 4]]; st2 = [steps[0], steps[1], steps[3], steps[4]]
            ins = lambda m: (lambda t: m(torch.cat([t[:, :2], zc[:, None], t[:, 2:]], 1)))
            c2s, c2f = _crlb(ins(smlm), th2, st2, 6 * rn2_e), _crlb(ins(sf), th2, st2, rn2_e)
            cr['smlm']['xy_zfixed'] = [c2s[0] * px_nm, c2s[1] * px_nm]; cr['sf']['xy_zfixed'] = [c2f[0] * px_nm, c2f[1] * px_nm]
    return dict(spec=spec, desc=psf.desc, is3d=is3d, z_um=z0, xy=xy, xz=xz, z_grid_um=zs, pixel_nm=px_nm, roi_px=Pi,
                rms_mwaves=rms_aberration_mwaves(spec), photons=float(photons), intensity_param=I, collected=coll, rn2_e=rn2_e, bg_px_frame=B, pattern=pattern, crlb=cr,
                focus_um=getattr(psf, 'meta', {}).get('focus_um'))


def draw(ax_xy, ax_xz, info, cmap='inferno', decades=3, fontsize=6):
    from matplotlib.colors import LogNorm
    px = info['pixel_nm']; Pi = info['roi_px']; h = Pi * px / 2
    z0, z1 = info['z_grid_um'][0] * 1e3, info['z_grid_um'][-1] * 1e3
    for ax, img, ext, ttl in ((ax_xy, info['xy'], [-h, h, -h, h], f"PSF xy (z = {info['z_um'] * 1e3:.0f} nm), log" if info['is3d'] else 'PSF xy (2D), log'),
                              (ax_xz, info['xz'], [-h, h, z0, z1], 'PSF xz, log')):
        if ax is None: continue
        m = img.max()
        ax.imshow(np.clip(img, m * 10 ** -decades, None), cmap=cmap, norm=LogNorm(m * 10 ** -decades, m), extent=ext, origin='lower',
                  aspect='equal' if ax is ax_xy else 'auto', interpolation='bilinear')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl, fontsize=fontsize + 1, pad=2)
        x0 = -0.85 * h
        if ax is ax_xy:        # scale bar bottom-left; in the xz section at the top (clear of the height / focus lines)
            y0 = ext[2] + 0.1 * (ext[3] - ext[2]); ax.plot([x0, x0 + 500], [y0, y0], 'w-', lw=2)
            ax.text(x0, y0 + 0.04 * (ext[3] - ext[2]), '500 nm', color='w', fontsize=fontsize, va='bottom')
        else:
            y0 = ext[2] + 0.9 * (ext[3] - ext[2]); ax.plot([x0, x0 + 500], [y0, y0], 'w-', lw=2)
            ax.text(x0, y0 - 0.04 * (ext[3] - ext[2]), '500 nm', color='w', fontsize=fontsize, va='top')
    if ax_xz is not None:
        ax_xz.axhline(info['z_um'] * 1e3, color='c', lw=0.6, ls=':')
        if info.get('focus_um') is not None: ax_xz.axhline(info['focus_um'] * 1e3, color='w', lw=0.6, ls='--')
        ax_xz.set_ylabel('z (nm)', fontsize=fontsize); ax_xz.set_yticks([0, 400, 800]); ax_xz.tick_params(labelsize=fontsize)


def crlb_text(info):
    c = info['crlb']; L = []
    if info['spec'].startswith('gauss:'):
        L.append(f"Gaussian PSF σ {info['spec'][6:]} px; no aberrations")
    else:
        L.append(f"vectorial PSF\n{os.path.splitext(os.path.basename(info['spec']))[0]}\nrms aberration (Noll ≥ 5) {info['rms_mwaves']:.0f} mλ")
    L.append(f"CRLB at {info['photons']:.0f} {'detected ' if info.get('collected') else ''}ph / 6-frame window,\nbg {info['bg_px_frame']:.2f} ph/px/frame"
             + (f",\nread noise {info['rn2_e']:.2f} e²/px/frame" if info.get('rn2_e') else '') + (f" (z = {info['z_um'] * 1e3:.0f} nm)" if info['is3d'] else '') + ':')
    for k, lab in (('smlm', 'SMLM'), ('sf', 'SIMFLUX')):
        r = c[k]
        if info['is3d']:
            L.append(f"{lab} x/y/z {r['x']:.1f}/{r['y']:.1f}/{r['z']:.0f} nm\n  (x/y with z known {r['xy_zfixed'][0]:.1f}/{r['xy_zfixed'][1]:.1f})")
        else:
            L.append(f"{lab} x/y {r['x']:.1f}/{r['y']:.1f} nm")
    p = info['pattern']; L.append(f"(pattern {p['pitch_nm']:.0f} nm, depth {p['depth']:g}, 2 x 3 steps)")
    return '\n'.join(L)


def add_panel(fig, rect, info, fontsize=6):
    """xy image (top), xz (middle), text (bottom) inside rect = [x0, y0, w, h] in figure fractions"""
    x0, y0, w, h = rect
    a1 = fig.add_axes([x0, y0 + 0.62 * h, w, 0.36 * h]); a2 = fig.add_axes([x0, y0 + 0.36 * h, w, 0.22 * h])
    draw(a1, a2, info, fontsize=fontsize)
    fig.text(x0, y0 + 0.33 * h, crlb_text(info), fontsize=fontsize, va='top', ha='left', family='sans-serif')
    return a1, a2


def sim_psf_info(md, dexp):
    """psf_info for the papersim movie directory md (gen.json): Dexp = the movie's median brightness / mean bg, else the nominal
    photons / bg; a v2 camera movie (camera_sim.npz): detected photons, mean read-noise variance of the crop, PSF to 0.40 um"""
    g0 = json.load(open(os.path.join(md, 'gen.json')))
    spec = 'gauss:' + g0['psf'][5:] if g0['psf'].startswith('gauss') else g0['psf']
    if dexp: ph, bg = g0['brightness_fit']['fit_median'], g0['bg_per_frame']['mean']
    else: ph, bg = g0['intensity_per_6frames'], g0['bg']
    pat = g0.get('pattern') or {}
    kw = {}
    if '_v2psf' in os.path.basename(md.rstrip('/')):     # every v2-PSF set (_v2psf, _v2psf_wsm, _v2psf_wsmaxis, ...)
        rn = np.load(os.path.join(md, 'camera_sim.npz'))['readnoise_e']
        kw = dict(detected=True, rn2_e=float((rn.astype(float) ** 2).mean()), zmax=0.4)
    return psf_info(spec, ph, bg, pattern={k: pat[k] for k in ('angles_deg', 'pitch_nm', 'depth') if k in pat}, **kw)


def panel_label(info):
    return f"Gaussian PSF, σ {info['spec'][6:]} px" if not info['is3d'] else f"In-situ vectorial PSF fit\nrms aberration {info['rms_mwaves']:.0f} mλ"


def caption_text(info, what='PSF panel'):
    """caption sentences: the PSF panel (incl. the xz line legend) and the single-emitter SMLM / SIMFLUX CRLBs with their definition"""
    c = info['crlb']; p = info['pattern']
    if info['is3d']:
        psfcap = (f"{what}: the in-situ vectorial PSF fit the movies were simulated with (Zernike Noll 5-21), xy section at the emitters' median "
                  f"height ({info['z_um'] * 1e3:.0f} nm) and xz section, log scale (3 decades); rms aberration over Noll >= 5 {info['rms_mwaves']:.0f} mλ. "
                  "xz section: white dashed line = focal plane, cyan dotted line = median emitter height.")
        xyz = lambda r: f"x/y/z {r['x']:.1f}/{r['y']:.1f}/{r['z']:.0f} nm"
    else:
        psfcap = (f"{what}: the Gaussian PSF (σ {info['spec'][6:]} px, no aberrations, no z dependence) the movies were simulated with (the preprint "
                  "simulation), xy and xz sections, log scale (3 decades); xz section: cyan dotted line = emitter plane.")
        xyz = lambda r: f"x/y {r['x']:.1f}/{r['y']:.1f} nm"
    rncap = f" and camera read noise {info['rn2_e'] ** 0.5:.1f} e- rms" if info.get('rn2_e') else ''
    crlbcap = (f"Cramér-Rao lower bounds for a single isolated emitter (mean over sub-pixel positions and pattern phases, 16 x 16 px ROI"
               + (", z free" if info['is3d'] else '') + f") at {info['photons']:.0f} {'detected ' if info.get('collected') else ''}photons per 6-frame window, "
               f"background {info['bg_px_frame']:.2f} photons/px/frame{rncap}: SMLM (the 6-frame sum) {xyz(c['smlm'])}, SIMFLUX (the 6 frames with the "
               f"known pattern: pitch {p['pitch_nm']:.0f} nm, depth {p['depth']:g}, 2 x 3 phase steps) {xyz(c['sf'])}.")
    return psfcap + '\n' + crlbcap
