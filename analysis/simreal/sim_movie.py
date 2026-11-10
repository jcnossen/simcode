"""(development sc6/sim_movie.py; the m05q / m05qh / m25qh movies, analysis/simreal/recipes.sh)
Simulated 0.5 nM movie in the real-movie format (iteration 31, pipeline test of run_uncalibrated --frontend sc).
rs05pm-like: RealSim '0.5nM' preset (density, on-times tau 65, z, filaments), paper photons (data.paper_bright, no per-window
jitter), full mismatch: emitter variation (angle / step / frame factors), PSF jitter (4 LUTs) + halo, fluctuating bg,
camera error (static residual offset 0.3 ADU, gain error 1 %), one continuous pattern over the movie (frame % 6 = step,
angle = step // 3, eps(t) random walk). 120 x 120 px (the real sensor crop; camera maps of the 0.5 nM calibration),
uint16 ADU multi-page TIFF. Rendering in 60-frame blocks.
Outputs: <out>/sim_movie.tif, <out>/sim_gt.npz (events, pattern per frame), <out>/gt_pattern.pt (common pattern format,
for --ref_pattern).
usage: CUDA_VISIBLE_DEVICES=1 python analysis/simreal/sim_movie.py <out_dir> [n_frames] [seed]
Optional env overrides (2026-11-09, SIMCODE-baseline worker; defaults unchanged): SIM_PRESET ('0.5nM' | '2.5nM', RealSim
preset + paper_bright preset), SIM_MED / SIM_SD (paper_bright med0 / log-sd, e.g. label-unit brightness 1280 / 0.45).
(2026-11-09, sim-realism worker): SIM_FRAME_SD / SIM_STEP_SD / SIM_ANG_SD (emitter-variation factors), SIM_HALO2 / SIM_HALO2_SIG
(broad PSF halo, realsim.add_halo2: a fraction HALO2 of every emitter's light in a broad exponential-disc component of scale
HALO2_SIG px), SIM_FILSTAT / SIM_FILSTAT_SIG (filament-correlated dim static light, realsim.filament_static_map: FILSTAT ph/px/frame x the
unit-mean filament line map blurred SIG px; FOV-mean bg unchanged), SIM_STRUCT (struct_frac override: 1 = filament geometry)."""
import math, os, sys
from simcode2 import config as _cfg
import numpy as np, torch, tifffile
from simcode2.hd.realsim import RealSim
from simcode2.hd.data import paper_bright

dev = 'cuda'


def main():
    out = sys.argv[1]; NF = int(sys.argv[2]) if len(sys.argv) > 2 else 4800; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed)
    H = W = 120
    PRE = os.environ.get('SIM_PRESET', '0.5nM')
    MED = float(os.environ['SIM_MED']) if os.environ.get('SIM_MED') else None
    SD = float(os.environ['SIM_SD']) if os.environ.get('SIM_SD') else None
    rs = RealSim(PRE, H, W, NF, seed=seed)
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
    x, y, z, br, t0, t1 = (em[k] for k in ('x', 'y', 'z', 'bright', 't0', 't1'))
    # per-emitter variation (fixed per event) + PSF group + halo
    ea = torch.exp(p['ang_sd'] * torch.randn(N, 2, device=dev)); ea = ea / ea.mean(1, keepdim=True)
    es = 1 + p['step_sd'] * torch.randn(N, 6, device=dev)
    grp = torch.randint(0, len(rs.luts), (N,), device=dev)
    sim = rs.sim; R = sim.R
    psf = torch.zeros(N, R, R, device=dev); base = sim.lut
    for gi, lut in enumerate(rs.luts):
        m_ = grp == gi
        if m_.any():
            sim.lut = lut; psf[m_] = sim.render_psf(x[m_], y[m_], z[m_])
    sim.lut = base
    o_ = torch.arange(R, device=dev, dtype=torch.float32) - R // 2
    fx_, fy_ = x - torch.round(x), y - torch.round(y)
    gb = torch.exp(-0.5 * (((o_[None, None, :] - fx_[:, None, None]) ** 2 + (o_[None, :, None] - fy_[:, None, None]) ** 2) / 16.0))
    gb = gb / gb.sum((1, 2), keepdim=True) * psf.sum((1, 2), keepdim=True)
    hh = p['halo'] * torch.rand(N, 1, 1, device=dev)
    psf = (1 - hh) * psf + hh * gb
    if os.environ.get('SIM_HALO2'):   # broad emitter-correlated halo (realsim.add_halo2); no extra random draws
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
    off_true = off_map + 0.3 * torch.randn(H, W, device=dev)               # residual offset error vs the calibration map
    hr = R // 2; margin = 6
    cx = torch.round(x).long() + margin + 1 + hr; cy = torch.round(y).long() + margin + 1 + hr
    oy, ox = torch.meshgrid(torch.arange(R, device=dev), torch.arange(R, device=dev), indexing='ij')
    ADU = np.empty((NF, H, W), np.uint16)
    PH = torch.zeros(N, NF // 6, device=dev)                               # GT photons per 6-frame block (diagnostic)
    BL = 60
    for a in range(0, NF, BL):
        fr = torch.arange(a, min(a + BL, NF), device=dev, dtype=torch.float32); Tb = len(fr)
        act = torch.nonzero((t1 > a) & (t0 < a + Tb))[:, 0]
        img = torch.zeros(Tb, H + 2 * (margin + hr) + 2, W + 2 * (margin + hr) + 2, device=dev)
        if len(act):
            onf = (torch.minimum(t1[act, None], fr[None] + 1) - torch.maximum(t0[act, None], fr[None])).clamp(0, 1)
            pt = patt[a:a + Tb]
            il = pt[None, :, 4] * (1 + pt[None, :, 3] * torch.cos(pt[None, :, 0] * x[act, None] + pt[None, :, 1] * y[act, None] - pt[None, :, 2]))
            st = (fr.long() % 6)
            mult = (ea[act][:, st // 3] * es[act][:, st] * (1 + p['frame_sd'] * torch.randn(len(act), Tb, device=dev))).clamp(min=0.2)
            ph = br[act, None] * il * onf * mult
            # (critic 31 #8) accumulate: several frames of the block map to the same 6-frame window (duplicate indices)
            PH.index_put_((act[:, None].expand(-1, Tb), (fr.long() // 6)[None].expand(len(act), -1)), ph, accumulate=True)
            iy = (cy[act, None, None] - hr + oy).reshape(len(act), -1); ix = (cx[act, None, None] - hr + ox).reshape(len(act), -1)
            Hp, Wp = img.shape[-2], img.shape[-1]
            lin = torch.arange(Tb, device=dev)[None, :, None] * Hp * Wp + (iy * Wp + ix)[:, None, :]
            img.view(-1).index_add_(0, lin.reshape(-1), (ph[:, :, None] * psf[act].reshape(len(act), 1, -1)).reshape(-1))
        o0 = margin + hr + 1
        mu = img[:, o0:o0 + H, o0:o0 + W] + bgt[a:a + Tb]
        e = torch.poisson(mu) + torch.randn_like(mu) * rn
        ADU[a:a + Tb] = torch.round(e / gain_true + off_true).clamp(0, 65535).cpu().numpy().astype(np.uint16)
    tifffile.imwrite(f'{out}/sim_movie.tif', ADU)
    np.savez_compressed(f'{out}/sim_gt.npz', x=x.cpu().numpy(), y=y.cpu().numpy(), z=z.cpu().numpy(), bright=br.cpu().numpy(),
                        t0=t0.cpu().numpy(), t1=t1.cpu().numpy(), ea=ea.cpu().numpy(), es=es.cpu().numpy(), patt=patt.cpu().numpy(),
                        ph6=PH.cpu().numpy(), gain_true=gain_true, halo2=float(os.environ.get('SIM_HALO2', 0)), halo2_sig=float(os.environ.get('SIM_HALO2_SIG', 0)), frame_sd=p['frame_sd'], step_sd=p['step_sd'], filstat=FILSTAT, struct=float(em['fil_struct'][0]) if em['fil_struct'] is not None else 0.0, nf=NF, preset=PRE, med0=MED if MED else np.nan, sd=SD if SD else np.nan)
    # GT pattern in the common file format (pattern_frames): phase(f) = phi[s] + dphi_a(f), knots every frame
    k = P['k'][0].cpu().numpy()                                             # [2,2] rows = angle
    ph_ = patt[:, 2].cpu().numpy(); s_ = np.arange(NF) % 6
    phi = np.array([ph_[s] for s in range(6)])                              # frame s = first occurrence (eps(s))
    eps = ph_ - phi[s_]
    m = np.array([float(patt[s, 3]) for s in range(6)]); rel = np.array([float(patt[s, 4]) for s in range(6)])
    G = dict(k=torch.tensor(k), phi=torch.tensor(phi), m=torch.tensor(m), logrel=torch.tensor(np.log(rel)),
             dphi=torch.tensor(np.stack([eps, eps])), knot_frames=1.0, angle_of_step=[0, 0, 0, 1, 1, 1])
    torch.save(G, f'{out}/gt_pattern.pt')
    print(f'movie {NF} frames, {N} events, mean window photons (on) {float(PH[PH > 0].median()):.0f}, gain_true {gain_true:.4f}', flush=True)


if __name__ == '__main__':
    main()
