"""[2026-11-10 simreal2] copy of field/sim_field_movie.py (= sc6/sim_movie.py + SIM_PFIELD pattern phase / depth field) + realism knobs:
  SIMP_<key>=<value>  RealSim preset override (float; 'true'/'false' for bools), e.g. SIMP_dens, SIMP_tau, SIMP_bg, SIMP_bg_sd,
                      SIMP_fluct, SIMP_fil_density, SIMP_fil_on, SIMP_fil_w, SIMP_fil_resample, SIMP_psf_wr / psf_fr / psf_zs, SIMP_z_sd
  SIM_DIM_DENS        ON density (per px per frame) of a second, dim emitter population on the SAME filaments (dim binders /
                      local fluctuating light); SIM_DIM_MED / SIM_DIM_SD window-photon median (label units) / log-sd; SIM_DIM_TAU
                      on-time (frames). GT rows of this population carry dim = 1.
  SIM_CALIB=<json>    (2026-11-11, opt-in) PSF from this calibration json instead of the preset's calib_0.5nM.json
  SIM_PSF_BLUR=sx,sy  (2026-11-11, opt-in) anisotropic Gaussian blur (px) of every rendered PSF (after the halo)
  SIM_PSFDIV=<json>   (2026-11-11, opt-in) PSF-diversity LUT bank replacing rs.luts (realsim.psf_div_luts); with SIM_SITE_PSF=0 one LUT per event
  SIM_EV_SHIFT=<nm>   (2026-11-11, opt-in) per-event emission offset sd per axis (PSF rendered at r + d; GT / illumination at r; stored in shift_dx / dy)
  (2026-11-11) a SIM_CALIB with a per-pattern-angle PSF (psf_dipole_M_angle / psf_blur_angle_px2; RealSim LUT tuples) renders
                      each frame with its angle's PSF (frame % 6 // 3); not combined with SIM_LAZY / SIM_PSF_DYN / SIM_HALO2 / SIM_PSF_BLUR
  SIM_PSF_DYN=1       per-frame PSF shape fluctuation (random jittered LUT per emitter per frame), SIMP_npsf-free: uses rs.luts
[2026-11-10 field copy of sc6/sim_movie.py] + SIM_PFIELD=<field .pt> (simcode2/pipeline/pattern_field.py format): a static
per-position pattern phase field phi_a(r) and depth factor f_a(r) are injected into the illumination
(il = rel (1 + m f_a(r) cos(k.r - phase - phi_a(r)))); no extra random draws (same seed = same events / noise as the
field-free movie). The true field is saved as <out>/true_field.pt.

Simulated 0.5 nM movie in the real-movie format (iteration 31, pipeline test of run_uncalibrated --frontend sc).
rs05pm-like: RealSim '0.5nM' preset (density, on-times tau 65, z, filaments), paper photons (data.paper_bright, no per-window
jitter), full mismatch: emitter variation (angle / step / frame factors), PSF jitter (4 LUTs) + halo, fluctuating bg,
camera error (static residual offset 0.3 ADU, gain error 1 %), one continuous pattern over the movie (frame % 6 = step,
angle = step // 3, eps(t) random walk). 120 x 120 px (the real sensor crop; camera maps of the 0.5 nM calibration),
uint16 ADU multi-page TIFF. Rendering in 60-frame blocks.
Outputs: <out>/sim_movie.tif, <out>/sim_gt.npz (events, pattern per frame), <out>/gt_pattern.pt (common pattern format,
for --ref_pattern).
usage: CUDA_VISIBLE_DEVICES=1 python sim_movie.py <out_dir> [n_frames] [seed]
Optional env overrides (2026-11-09, SIMCODE-baseline worker; defaults unchanged): SIM_PRESET ('0.5nM' | '2.5nM', RealSim
preset + paper_bright preset), SIM_MED / SIM_SD (paper_bright med0 / log-sd, e.g. label-unit brightness 1280 / 0.45).
(2026-11-09, sim-realism worker): SIM_FRAME_SD / SIM_STEP_SD / SIM_ANG_SD (emitter-variation factors), SIM_HALO2 / SIM_HALO2_SIG
(broad PSF halo, realsim.add_halo2: a fraction HALO2 of every emitter's light in a broad exponential-disc component of scale
HALO2_SIG px), SIM_FILSTAT / SIM_FILSTAT_SIG (filament-correlated dim static light, realsim.filament_static_map: FILSTAT ph/px/frame x the
unit-mean filament line map blurred SIG px; FOV-mean bg unchanged), SIM_STRUCT (struct_frac override: 1 = filament geometry).
(2026-11-10/11, evaluation protocol; all opt-in, absent = unchanged):
  SIMP_site_ev / site_wsd / site_L / site_jit   discrete docking sites with repeated binding (realsim.RealSim._sites); z per site
  SIM_SITE_PSF=1      (default 1 when sites are on) PSF-jitter LUT group per SITE instead of per event (static per-site PSF term)
  SIM_DIM_SITES=1     dim binders also bound to (their own) sites (default: free events as before)
  SIM_SHIFT_LIN / SIM_SHIFT_RMS / SIM_SHIFT_SIG   imaging shift field (static, all methods see it): the PSF is rendered at
                      r + d(r), d = LIN nm per 100 px x (r - centre) (scale-like) + smooth random field (rms RMS nm, sigma SIG px);
                      the illumination and the GT stay at r. GT: shift_dx / shift_dy (px) per event.
  SIMP_env_x0 / env_y0 / env_s1 / env_s2 / env_th / env_bg   (critic 37 #2) illumination envelope (realsim.RealSim.envelope)
  SIM_LAZY=1          per-60-frame-block PSF rendering (memory ~ active events) and sparse GT window photons
                      (ph6_ev / ph6_blk / ph6_val, with frame_sd + pattern field included) instead of the dense ph6 [N, NF/6]."""
import json, math, os, sys
from simcode2 import config as _cfg
import numpy as np, torch, tifffile
from simcode2.hd.realsim import RealSim
from simcode2.hd.data import paper_bright

dev = 'cuda'


def psf_blur(P_, sx, sy):
    """(2026-11-11 PSF-shape mismatch, opt-in SIM_PSF_BLUR=sx,sy px) anisotropic Gaussian image blur of PSF ROIs [N,R,R] (FFT, zero-padded)"""
    R_ = P_.shape[-1]
    fy_, fx_ = torch.meshgrid(torch.fft.fftfreq(2 * R_, device=P_.device), torch.fft.fftfreq(2 * R_, device=P_.device), indexing='ij')
    K_ = torch.exp(-2 * math.pi ** 2 * (sx ** 2 * fx_ ** 2 + sy ** 2 * fy_ ** 2))
    pad_ = torch.nn.functional.pad(P_, (0, R_, 0, R_))
    out_ = torch.fft.ifft2(torch.fft.fft2(pad_) * K_).real
    h_ = R_ // 2   # kernel centred at index 0 (circular) -> no shift; crop the original window
    return out_[..., :R_, :R_].clamp(min=0)


def main():
    out = sys.argv[1]; NF = int(sys.argv[2]) if len(sys.argv) > 2 else 4800; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed)
    H = W = 120
    PRE = os.environ.get('SIM_PRESET', '0.5nM')
    MED = float(os.environ['SIM_MED']) if os.environ.get('SIM_MED') else None
    SD = float(os.environ['SIM_SD']) if os.environ.get('SIM_SD') else None
    over = {}
    for k_, v_ in os.environ.items():
        if k_.startswith('SIMP_'):
            over[k_[5:]] = (v_.lower() == 'true') if v_.lower() in ('true', 'false') else float(v_)
    if os.environ.get('SIM_CALIB'):   # (2026-11-11 PSF-shape mismatch, opt-in) render with this calib json's PSF (zernike_waves / focus_um);
        over['calib'] = os.environ['SIM_CALIB']   # the jittered LUTs are centred on it; camera maps / pattern presets unchanged
    rs = RealSim(PRE, H, W, NF, seed=seed, **over)
    if os.environ.get('SIM_PSFDIV'):   # (2026-11-11, opt-in) PSF-diversity LUT bank (realsim.psf_div_luts, json); own generator, no global draws
        from simcode2.hd.realsim import psf_div_luts
        rs.luts = psf_div_luts(rs, json.loads(os.environ['SIM_PSFDIV']), seed); print(f'PSF bank: {len(rs.luts)} LUTs', flush=True)
    print('preset overrides', over, flush=True)
    p = rs.p
    # (2026-11-09, sim-realism worker) emitter-variation overrides: SIM_FRAME_SD / SIM_STEP_SD / SIM_ANG_SD (defaults = preset);
    # the random draws are unchanged (same seed -> same events, bg, noise; only the factors scale)
    for ev, key in (('SIM_FRAME_SD', 'frame_sd'), ('SIM_STEP_SD', 'step_sd'), ('SIM_ANG_SD', 'ang_sd')):
        if os.environ.get(ev): p[key] = float(os.environ[ev])
    for _ in range(20):
        P = rs.pattern(1)
        if int(P['steps'][0, 0]) == 0: break
    patt = P['patt'][0]                                                     # [NF,5] true pattern per frame
    em = rs.emitters(1, struct_frac=float(os.environ['SIM_STRUCT']) if os.environ.get('SIM_STRUCT') else None)
    N = len(em['x'])
    em['bright'] = paper_bright(PRE, 1, em['fov'], em['z'], rs, jitter=None, med0=MED, sd=SD)
    NSITE = int(float(os.environ.get('SIM_SITES', 0) or 0))
    if NSITE > 0 and N > NSITE:   # (2026-11-10 retrain_r, opt-in) punctate repeated docking sites: every event binds at one of NSITE fixed
        # sites (= the positions / z of the first NSITE events; random order), +- SIM_SITE_JIT nm linker jitter (default 3); site brightness
        # factor lognormal SIM_SITE_BSD (default 0.25) shared by all its events (a per-site static term); extra random draws
        si = torch.randint(0, NSITE, (N,), device=dev); jit = float(os.environ.get('SIM_SITE_JIT', 3.0)) / 108.33
        sb = torch.exp(float(os.environ.get('SIM_SITE_BSD', 0.25)) * torch.randn(NSITE, device=dev))
        b0 = em['bright'] * torch.exp(-(em['z'] - rs.p['z_med']) / rs.sim.tirf) ** -1      # TIRF factor removed, re-applied at the site z
        em['x'] = em['x'][:NSITE][si] + jit * torch.randn(N, device=dev); em['y'] = em['y'][:NSITE][si] + jit * torch.randn(N, device=dev)
        em['z'] = em['z'][:NSITE][si]
        em['bright'] = b0 * sb[si] * torch.exp(-(em['z'] - rs.p['z_med']) / rs.sim.tirf)
        print(f'sites: {NSITE} for {N} events', flush=True)
    dim = torch.zeros(N, device=dev)
    site = em.get('site')
    SINFO = {k_: em[k_] for k_ in ('site_x', 'site_y', 'site_z', 'site_a', 'site_b', 'site_w') if k_ in em}
    if float(os.environ.get('SIM_DIM_DENS', 0) or 0) > 0:
        e2 = rs.emitters(1, dens=float(os.environ['SIM_DIM_DENS']), tau=float(os.environ.get('SIM_DIM_TAU', p['tau'])), struct_frac=1.0,
                         fil_pts_in=em['fil_pts'], use_sites=bool(int(os.environ.get('SIM_DIM_SITES', '0'))))
        e2['bright'] = paper_bright(PRE, 1, e2['fov'], e2['z'], rs, jitter=None, med0=float(os.environ['SIM_DIM_MED']),
                                    sd=float(os.environ.get('SIM_DIM_SD', 0.5)))
        for k_ in ('fov', 'x', 'y', 'z', 'bright', 't0', 't1'): em[k_] = torch.cat([em[k_], e2[k_]])
        if site is not None:
            s2 = e2['site'] + len(SINFO['site_x']) if 'site' in e2 else torch.full((len(e2['x']),), -1, dtype=torch.long, device=dev)
            if 'site' in e2:
                for k_ in SINFO: SINFO[k_] = torch.cat([SINFO[k_], e2[k_]])
            site = torch.cat([site, s2])
        dim = torch.cat([dim, torch.ones(len(e2['x']), device=dev)]); N = len(em['x'])
        print(f'dim population: {len(e2["x"])} events', flush=True)
    x, y, z, br, t0, t1 = (em[k] for k in ('x', 'y', 'z', 'bright', 't0', 't1'))
    ENV = rs.envelope(x, y)   # (critic 37 #2, opt-in SIMP_env_*) illumination envelope on emitter brightness (GT photons include it)
    if ENV is not None: br = br * ENV; print(f'envelope: emitter factor p5/p50/p95 {[round(float(v), 2) for v in torch.quantile(ENV, torch.tensor([0.05, 0.5, 0.95], device=dev))]}', flush=True)
    # per-emitter variation (fixed per event) + PSF group + halo
    ea = torch.exp(p['ang_sd'] * torch.randn(N, 2, device=dev)); ea = ea / ea.mean(1, keepdim=True)
    es = 1 + p['step_sd'] * torch.randn(N, 6, device=dev)
    grp = torch.randint(0, len(rs.luts), (N,), device=dev)
    sim = rs.sim; R = sim.R
    LAZY = bool(int(os.environ.get('SIM_LAZY', '0')))
    BLUR = [float(v) for v in os.environ['SIM_PSF_BLUR'].split(',')] if os.environ.get('SIM_PSF_BLUR') else None
    if site is not None and bool(int(os.environ.get('SIM_SITE_PSF', '1'))):   # (eval protocol) PSF LUT group per site
        sgrp = torch.randint(0, len(rs.luts), (len(SINFO['site_x']),), device=dev)
        grp = torch.where(site >= 0, sgrp[site.clamp(min=0)], grp)
    # (eval protocol) imaging shift field: render the PSF at r + d(r); illumination and GT stay at r
    xr_, yr_ = x, y
    SDX = SDY = None
    if os.environ.get('SIM_SHIFT_LIN') or os.environ.get('SIM_SHIFT_RMS'):
        from simcode2.hd.realsim import smooth_field, PX_NM
        lin = float(os.environ.get('SIM_SHIFT_LIN', 0) or 0) / PX_NM / 100.0
        rms = float(os.environ.get('SIM_SHIFT_RMS', 0) or 0) / PX_NM
        sg = float(os.environ.get('SIM_SHIFT_SIG', '20'))
        m6 = 8; Fs = smooth_field(2, H + 2 * m6, W + 2 * m6, sg) * rms                     # [2, H', W'] unit sd x rms
        gx = (x + m6).clamp(0, W + 2 * m6 - 1.001).long(); gy = (y + m6).clamp(0, H + 2 * m6 - 1.001).long()
        SDX = lin * (x - W / 2) + Fs[0, gy, gx]; SDY = lin * (y - H / 2) + Fs[1, gy, gx]
        xr_, yr_ = x + SDX, y + SDY
    if float(os.environ.get('SIM_EV_SHIFT', 0) or 0) > 0:   # (2026-11-11, opt-in) per-EVENT emission offset (nm sd per axis): PSF at r + d_event
        es_ = float(os.environ['SIM_EV_SHIFT']) / 108.33
        EVX = es_ * torch.randn(N, device=dev); EVY = es_ * torch.randn(N, device=dev)
        xr_, yr_ = xr_ + EVX, yr_ + EVY
        SDX = EVX if SDX is None else SDX + EVX; SDY = EVY if SDY is None else SDY + EVY
    def mkpsf(ix_):
        xs, ys = xr_[ix_], yr_[ix_]
        P_ = torch.zeros(len(ix_), R, R, device=dev); base_ = sim.lut
        for gi, lut in enumerate(rs.luts):
            m_ = grp[ix_] == gi
            if m_.any():
                sim.lut = lut; P_[m_] = sim.render_psf(xs[m_], ys[m_], z[ix_][m_])
        sim.lut = base_
        o_ = torch.arange(R, device=dev, dtype=torch.float32) - R // 2
        fx_, fy_ = xs - torch.round(xs), ys - torch.round(ys)
        gb_ = torch.exp(-0.5 * (((o_[None, None, :] - fx_[:, None, None]) ** 2 + (o_[None, :, None] - fy_[:, None, None]) ** 2) / 16.0))
        gb_ = gb_ / gb_.sum((1, 2), keepdim=True) * P_.sum((1, 2), keepdim=True)
        P_ = (1 - hh[ix_]) * P_ + hh[ix_] * gb_
        return psf_blur(P_, *BLUR) if BLUR else P_
    ANG = isinstance(rs.luts[0], tuple); psf1 = None
    if ANG: assert not (LAZY or os.environ.get('SIM_PSF_DYN') or os.environ.get('SIM_HALO2') or BLUR), 'per-angle PSF: unsupported combination'
    if LAZY:
        hh = p['halo'] * torch.rand(N, 1, 1, device=dev)
        psf = None
    else:
        psf = torch.zeros(N, R, R, device=dev); base = sim.lut
        if ANG: psf1 = torch.zeros(N, R, R, device=dev)
        for gi, lut in enumerate(rs.luts):
            m_ = grp == gi
            if m_.any():
                sim.lut = lut[0] if ANG else lut; psf[m_] = sim.render_psf(xr_[m_], yr_[m_], z[m_])
                if ANG: sim.lut = lut[1]; psf1[m_] = sim.render_psf(xr_[m_], yr_[m_], z[m_])
        sim.lut = base
        o_ = torch.arange(R, device=dev, dtype=torch.float32) - R // 2
        fx_, fy_ = xr_ - torch.round(xr_), yr_ - torch.round(yr_)
        gb = torch.exp(-0.5 * (((o_[None, None, :] - fx_[:, None, None]) ** 2 + (o_[None, :, None] - fy_[:, None, None]) ** 2) / 16.0))
        gb = gb / gb.sum((1, 2), keepdim=True) * psf.sum((1, 2), keepdim=True)
        hh = p['halo'] * torch.rand(N, 1, 1, device=dev)
        psf = (1 - hh) * psf + hh * gb
        if ANG: psf1 = (1 - hh) * psf1 + hh * gb
        if BLUR: psf = torch.cat([psf_blur(psf[i_:i_ + 4096], *BLUR) for i_ in range(0, len(psf), 4096)])
        del gb
    PSFD = None
    if os.environ.get('SIM_PSF_DYN'):
        assert not LAZY   # (simreal2) per-frame PSF shape fluctuation: every frame each emitter is rendered with a random one
        # of the jittered LUTs (rs.luts; widths via SIMP_psf_wr / psf_fr / psf_zs), instead of one LUT per event (no extra draws above)
        PSFD = torch.zeros(len(rs.luts), N, R, R, device=dev)
        for gi, lut in enumerate(rs.luts):
            sim.lut = lut; PSFD[gi] = sim.render_psf(x, y, z)
        sim.lut = base
        PSFD = (1 - hh)[None] * PSFD + hh[None] * gb[None]
    if os.environ.get('SIM_HALO2'):   # broad emitter-correlated halo (realsim.add_halo2); no extra random draws
        assert not LAZY
        from simcode2.hd.realsim import add_halo2
        psf = add_halo2(psf, x, y, float(os.environ['SIM_HALO2']), float(os.environ.get('SIM_HALO2_SIG', '4.0')))
    # background over the whole movie (fluct mode), steps = frame % 6
    steps = (torch.arange(NF, device=dev) % 6)[None]
    bgt, _ = rs.background(1, steps, mode='fluct')
    bgt = bgt[0]
    FILSTAT = float(os.environ.get('SIM_FILSTAT', 0) or 0)
    if FILSTAT > 0:   # filament-correlated dim static light (realsim.filament_static_map), FOV-mean bg unchanged
        from simcode2.hd.realsim import filament_static_map
        fp = em['fil_pts'][0] if em['fil_pts'] is not None and bool(em['fil_struct'][0]) else None
        Dm = filament_static_map(fp, H, W, float(os.environ.get('SIM_FILSTAT_SIG', '1.5'))).to(dev)
        bgt = (bgt + FILSTAT * (Dm[None] - 1)).clamp(min=0.3) if float(Dm.mean()) > 0 else bgt
        np.save(f'{out}/filstat_map.npy', Dm.cpu().numpy())
    cam = np.load(p['cam'])
    from simcode2.pipeline.common import load_calib  # noqa (format reference)
    off_map = torch.tensor(np.load(_cfg.model('insitu/camera_maps_0.5nM.npz'))['offset_adu'], device=dev).float()
    rn = rs.rn_map[:H, :W]
    gain_true = sim.gain * (1 + 0.01 * torch.randn(1, device=dev).item())
    off_true = off_map + float(os.environ.get("SIM_OFFERR", 0.3)) * torch.randn(H, W, device=dev)               # residual offset error vs the calibration map
    hr = R // 2; margin = 6
    cx = torch.round(xr_).long() + margin + 1 + hr; cy = torch.round(yr_).long() + margin + 1 + hr
    oy, ox = torch.meshgrid(torch.arange(R, device=dev), torch.arange(R, device=dev), indexing='ij')
    ADU = np.empty((NF, H, W), np.uint16)
    PH = torch.zeros(N, NF // 6, device=dev) if not LAZY else None         # GT photons per 6-frame block (diagnostic)
    PHS = []                                                                # (lazy) sparse (event, block, photons)
    BL = 60
    FPH = FDF = None
    if os.environ.get('SIM_PFIELD'):
        from simcode2.pipeline.pattern_field import field_at
        Fi = torch.load(os.environ['SIM_PFIELD'], weights_only=False)
        dp_, df_ = field_at(Fi, x.cpu().numpy().astype(np.float64), y.cpu().numpy().astype(np.float64))
        FPH = torch.tensor(dp_, device=dev, dtype=torch.float32); FDF = torch.tensor(df_, device=dev, dtype=torch.float32)
        torch.save(Fi, f'{out}/true_field.pt')
    for a in range(0, NF, BL):
        fr = torch.arange(a, min(a + BL, NF), device=dev, dtype=torch.float32); Tb = len(fr)
        act = torch.nonzero((t1 > a) & (t0 < a + Tb))[:, 0]
        img = torch.zeros(Tb, H + 2 * (margin + hr) + 2, W + 2 * (margin + hr) + 2, device=dev)
        if len(act):
            onf = (torch.minimum(t1[act, None], fr[None] + 1) - torch.maximum(t0[act, None], fr[None])).clamp(0, 1)
            pt = patt[a:a + Tb]
            if FPH is not None:
                ang_ = (fr.long() % 6) // 3
                dph = FPH[act][:, ang_]; dmf = FDF[act][:, ang_]
                il = pt[None, :, 4] * (1 + (pt[None, :, 3] * dmf).clamp(max=1.0) * torch.cos(pt[None, :, 0] * x[act, None] + pt[None, :, 1] * y[act, None] - pt[None, :, 2] - dph))
            else:
                il = pt[None, :, 4] * (1 + pt[None, :, 3] * torch.cos(pt[None, :, 0] * x[act, None] + pt[None, :, 1] * y[act, None] - pt[None, :, 2]))
            st = (fr.long() % 6)
            mult = (ea[act][:, st // 3] * es[act][:, st] * (1 + p['frame_sd'] * torch.randn(len(act), Tb, device=dev))).clamp(min=0.2)
            ph = br[act, None] * il * onf * mult
            # (critic 31 #8) accumulate: several frames of the block map to the same 6-frame window (duplicate indices)
            if PH is not None:
                PH.index_put_((act[:, None].expand(-1, Tb), (fr.long() // 6)[None].expand(len(act), -1)), ph, accumulate=True)
            else:
                nbk = (Tb + 5) // 6; blk0 = a // 6
                acc = torch.zeros(len(act), nbk, device=dev)
                acc.index_add_(1, (fr.long() // 6) - blk0, ph)
                ii, bb = torch.nonzero(acc > 0, as_tuple=True)
                PHS.append(torch.stack([act[ii].float(), (bb + blk0).float(), acc[ii, bb]], 1).cpu())
            iy = (cy[act, None, None] - hr + oy).reshape(len(act), -1); ix = (cx[act, None, None] - hr + ox).reshape(len(act), -1)
            Hp, Wp = img.shape[-2], img.shape[-1]
            lin = torch.arange(Tb, device=dev)[None, :, None] * Hp * Wp + (iy * Wp + ix)[:, None, :]
            if psf1 is not None:   # per-angle PSF: frame angle = (frame % 6) // 3
                a1 = (st // 3 == 1)[None, :, None]
                img.view(-1).index_add_(0, lin.reshape(-1), (ph[:, :, None] * torch.where(a1, psf1[act].reshape(len(act), 1, -1),
                                                                                           psf[act].reshape(len(act), 1, -1))).reshape(-1))
            elif PSFD is None:
                pa_ = psf[act] if psf is not None else mkpsf(act)
                img.view(-1).index_add_(0, lin.reshape(-1), (ph[:, :, None] * pa_.reshape(len(act), 1, -1)).reshape(-1))
            else:
                gs = torch.randint(0, PSFD.shape[0], (len(act), Tb), device=dev)
                for gi in range(PSFD.shape[0]):
                    img.view(-1).index_add_(0, lin.reshape(-1), ((ph * (gs == gi))[:, :, None] * PSFD[gi, act].reshape(len(act), 1, -1)).reshape(-1))
        o0 = margin + hr + 1
        mu = img[:, o0:o0 + H, o0:o0 + W] + bgt[a:a + Tb]
        e = torch.poisson(mu) + torch.randn_like(mu) * rn * float(os.environ.get("SIM_RNMULT", 1.0))
        ADU[a:a + Tb] = torch.round(e / gain_true + off_true).clamp(0, 65535).cpu().numpy().astype(np.uint16)
    tifffile.imwrite(f'{out}/sim_movie.tif', ADU)
    EXTRA = {}
    if PH is None:
        Q = torch.cat(PHS).numpy() if PHS else np.zeros((0, 3), np.float32)
        EXTRA.update(ph6_ev=Q[:, 0].astype(np.int64), ph6_blk=Q[:, 1].astype(np.int64), ph6_val=Q[:, 2].astype(np.float32))
    if site is not None:
        EXTRA['site'] = site.cpu().numpy()
        for k_, v_ in SINFO.items(): EXTRA[k_] = v_.cpu().numpy()
        EXTRA['grp'] = grp.cpu().numpy()
    if SDX is not None:
        EXTRA.update(shift_dx=SDX.cpu().numpy(), shift_dy=SDY.cpu().numpy())
    np.savez_compressed(f'{out}/sim_gt.npz', **EXTRA, dim=dim.cpu().numpy(), knobs=json.dumps(dict(over=over, env={k_: v_ for k_, v_ in os.environ.items() if k_.startswith('SIM')})), x=x.cpu().numpy(), y=y.cpu().numpy(), z=z.cpu().numpy(), bright=br.cpu().numpy(),
                        t0=t0.cpu().numpy(), t1=t1.cpu().numpy(), ea=ea.cpu().numpy(), es=es.cpu().numpy(), patt=patt.cpu().numpy(),
                        **({'ph6': PH.cpu().numpy()} if PH is not None else {}), gain_true=gain_true, halo2=float(os.environ.get('SIM_HALO2', 0)), halo2_sig=float(os.environ.get('SIM_HALO2_SIG', 0)), frame_sd=p['frame_sd'], step_sd=p['step_sd'], filstat=FILSTAT, struct=float(em['fil_struct'][0]) if em['fil_struct'] is not None else 0.0, nf=NF, preset=PRE, med0=MED if MED else np.nan, sd=SD if SD else np.nan)
    # GT pattern in the common file format (pattern_frames): phase(f) = phi[s] + dphi_a(f), knots every frame
    k = P['k'][0].cpu().numpy()                                             # [2,2] rows = angle
    ph_ = patt[:, 2].cpu().numpy(); s_ = np.arange(NF) % 6
    phi = np.array([ph_[s] for s in range(6)])                              # frame s = first occurrence (eps(s))
    eps = ph_ - phi[s_]
    m = np.array([float(patt[s, 3]) for s in range(6)]); rel = np.array([float(patt[s, 4]) for s in range(6)])
    G = dict(k=torch.tensor(k), phi=torch.tensor(phi), m=torch.tensor(m), logrel=torch.tensor(np.log(rel)),
             dphi=torch.tensor(np.stack([eps, eps])), knot_frames=1.0, angle_of_step=[0, 0, 0, 1, 1, 1])
    torch.save(G, f'{out}/gt_pattern.pt')
    medph = float(PH[PH > 0].median()) if PH is not None else float(np.median(EXTRA['ph6_val'])) if len(EXTRA['ph6_val']) else 0.0
    print(f'movie {NF} frames, {N} events' + (f', {len(SINFO["site_x"])} sites' if site is not None else '') +
          f', mean window photons (on) {medph:.0f}, gain_true {gain_true:.4f}', flush=True)


if __name__ == '__main__':
    main()
