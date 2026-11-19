"""STEP 0 of the v3 retrain: does the training simulator (simcode.simulate: Simulator / AcquisitionSim / TrainingBatches) and
the DECODE training LUT (analysis/baselines/decode/decode_paper.psf_luts) render the in-situ PSF v3 exactly as fitted?

Per dataset (1809_2, 2p5nM_m2), references:
  FIT    the fitted model itself (simcode.pipeline.psf_fit.model_from_pt of the fit's _psf.pt: Zernikes, focus, blur, the
         per-angle dipole tensors), as fitted (VectorialPSF oversample 2, npupil 64), on a 41 x 41 ROI
  CAL_k  torch_psf PSF.load(calib, angle=k) of the calib files (avg json, axis json, avg psf.zip) at the same sampling
checks (peak-normalised max |a - b| / max b, and the max relative difference over pixels > 1e-3 of the peak):
  A  calib file == fit                (axis json angle k vs FIT angle k; avg json vs the mean of the FIT angles)   float level
  B  simulator LUT nodes == torch_psf (AcquisitionSim LUT of TrainingBatches(calib=[avg]) at the LUT nodes vs PSF.load at the
                                       simulator's sampling oversample 3; axis json: lut_angle[k])                  float level
  C  sampling: oversample 3 (the default before the fix) vs 2 (fit), same parameters                                   report
  D  LUT interpolation off the nodes (random sub-pixel / z) vs direct rendering                                      report
  E  DECODE LUT (psf_luts of the avg json; spline knot stack) vs direct rendering at its nodes                       report
  F  scene rendering: one emitter rendered by AcquisitionSim.scene (no variation, no jitter, exact pattern) vs
     photons x direct PSF; per-angle PSF of the axis json per frame                                                  float level
usage: python step0_psf.py <out.json>"""
import os, sys, json, math
import numpy as np, torch
from simcode import psf as spsf
from simcode.pipeline import psf_fit

dev = 'cuda'
M = os.path.expanduser('~/simcode2/out/train_v3/calib')   # frozen copies of the v3 calibrations (with the fit sampling)
DEC = os.path.expanduser(os.environ.get('DEC_REPO', '~/simcode2-decv4'))
SETS = {
    '1809_2': dict(pt='/mnt/data/dmdsf/bn_dnapaint/1809/cover1/2/simcode_results/psf_v3/psf/fit_ps3b_f0.3_psf.pt',
                   avg=f'{M}/calib_1809_2_psf_v3.json', axis=f'{M}/calib_1809_2_psf_v3_axis.json',
                   zipf=f'{M}/calib_1809_2_psf_v3.psf.zip'),
    '2p5nM_m2': dict(pt='/mnt/data/dmdsf/bn_dnapaint/dnapaint/2.5nM/2/simcode_results/psf_v3/psf/fit_ps3_calib_2p5nM_m2_psf_v2_psf.pt',
                     avg=f'{M}/calib_2p5nM_m2_psf_v3.json', axis=f'{M}/calib_2p5nM_m2_psf_v3_axis.json',
                     zipf=f'{M}/calib_2p5nM_m2_psf_v3.psf.zip'),
}
R = 41
ZS = (0.0, 0.05, 0.1, 0.15, 0.2)


def cmp(a, b):
    a = a.double(); b = b.double()
    pk = b.abs().amax(dim=(-2, -1), keepdim=True)
    d = (a - b).abs()
    m = b > 1e-3 * pk
    return dict(max_abs_over_peak=float((d / pk).max()), max_rel_px=float((d[m] / b[m]).max()), sum_a=float(a.sum((-2, -1)).mean()),
                sum_b=float(b.sum((-2, -1)).mean()))


def nodes(zs, sub):
    Z, SY, SX = torch.meshgrid(torch.as_tensor(zs, device=dev, dtype=torch.float32), sub, sub, indexing='ij')
    return torch.stack([SX.reshape(-1), SY.reshape(-1), Z.reshape(-1)], 1)


def render(psf, pos):
    with torch.no_grad():
        return torch.cat([spsf.render(psf, p) for p in pos.split(1024)])


def main(out):
    res = {}
    for name, S in SETS.items():
        r = res[name] = {}
        F, _ = psf_fit.model_from_pt(S['pt'], R, dev)
        vm = F.angle_models()                            # as fitted: oversample 2
        rng = torch.Generator(device='cpu').manual_seed(1)
        pos = torch.cat([torch.rand(64, 2, generator=rng) - 0.5, torch.rand(64, 1, generator=rng) * 0.2], 1).to(dev)
        pos = torch.cat([pos, nodes(ZS, torch.tensor([-0.5, 0.0, 0.25], device=dev))])
        fit = [render(v, pos) for v in vm]
        # A: calib files vs fit (same sampling: oversample 2)
        for k in range(2):
            ca = spsf.PSF.load(S['axis'], roisize=R, device=dev, angle=k)
            r[f'A_axis_angle{k}_vs_fit'] = cmp(render(ca, pos), fit[k])
        cavg = spsf.PSF.load(S['avg'], roisize=R, device=dev)
        r['calib_sampling'] = dict(npupil=cavg.npupil, oversample=cavg.os)
        r['A_avg_vs_fit_mean'] = cmp(render(cavg, pos), 0.5 * (fit[0] + fit[1]))
        czip = spsf.PSF.load(S['zipf'], roisize=R, device=dev)
        r['A_zip_vs_json'] = cmp(render(czip, pos), render(cavg, pos))
        r['angle_vs_avg'] = cmp(fit[0], 0.5 * (fit[0] + fit[1]))   # size of the per-angle photoselection term
        # C: sampling 3 vs 2
        c3 = spsf.PSF.load(S['avg'], roisize=R, oversample=3, device=dev)
        r['C_os3_vs_os2'] = cmp(render(c3, pos), render(cavg, pos))
        c4 = spsf.PSF.load(S['avg'], roisize=R, oversample=4, device=dev)
        r['C_os3_vs_os4'] = cmp(render(c3, pos), render(c4, pos))
        # B: simulator LUT of the training generator (TrainingBatches -> AcquisitionSim -> Simulator)
        from simcode.simulate.batches import TrainingBatches
        for kind in ('avg', 'axis'):
            tb = TrainingBatches(64, 64, 6, seed=0, calib=[S[kind]])
            rs = tb.rsm[0]['0.5nM']; sim = rs.sim
            r[f'heights_{kind}'] = tb.heights()
            r[f'camera_{kind}'] = dict(gain=sim.gain, offset=sim.offset, rn_scalar=sim.rn, rn_map_median=float(rs.rn_map.median()),
                                       rn_map_rms=float(rs.rn_map.pow(2).mean().sqrt()), rn_map_shape=list(rs.rn_map.shape))
            sub = (torch.arange(sim.ls + 1, device=dev) / sim.ls) - 0.5
            iz = [0, 5, 10, 15, 20, 30]
            P = nodes([float(sim.zs[i]) for i in iz], sub[::4])
            luts = sim.lut_angle if sim.lut_angle is not None else [sim.lut]
            for k, lut in enumerate(luts):
                ref = render(spsf.PSF.load(S[kind], roisize=sim.R, device=dev, angle=(k if kind == 'axis' else None)), P)
                L = lut[iz][:, ::4, ::4].reshape(-1, sim.R, sim.R)
                r[f'B_lut_{kind}_angle{k}_vs_torchpsf'] = cmp(L, ref)
                if kind == 'axis':   # also against the FIT at the simulator's sampling
                    v3 = vm[k].resized(sim.R)
                    r[f'B_lut_axis_angle{k}_vs_fit'] = cmp(L, render(v3, P))
            if kind == 'avg':
                ref0 = render(vm[0].resized(sim.R), P); ref1 = render(vm[1].resized(sim.R), P)
                r['B_lut_avg_vs_fit_mean'] = cmp(sim.lut[iz][:, ::4, ::4].reshape(-1, sim.R, sim.R), 0.5 * (ref0 + ref1))
                r['n_luts_bank'] = len(rs.luts); r['lut_shape'] = list(sim.lut.shape); r['lut_z_um'] = [float(sim.zs[0]), float(sim.zs[-1])]
                # D: interpolation off the nodes
                g = torch.Generator(device='cpu').manual_seed(2)
                n = 2000
                x = (torch.rand(n, generator=g) - 0.5).to(dev); y = (torch.rand(n, generator=g) - 0.5).to(dev)
                z = (torch.rand(n, generator=g) * 0.2).to(dev)
                Li = sim.render_psf(x, y, z)
                Di = render(spsf.PSF.load(S[kind], roisize=sim.R, device=dev), torch.stack([x, y, z], 1))
                r['D_lut_interp_vs_direct'] = cmp(Li, Di)
                # F: scene rendering of single emitters (one emitter per FOV, no variation / jitter / halo, flat bg 0, exact pattern)
                F_ = scene_check(rs, S[kind], sim)
                r['F_scene_avg'] = F_
            else:
                r['F_scene_axis'] = scene_check(rs, S[kind], sim)
            del tb, rs, sim; torch.cuda.empty_cache()
        # E: DECODE LUT
        r['E_decode'] = decode_check(S['avg'], vm)
        print(name, json.dumps(r, indent=1), flush=True)
    json.dump(res, open(out, 'w'), indent=1)


def scene_check(rs, calib, sim):
    """frames of AcquisitionSim.scene with one emitter per FOV (camera noise removed: mean image = img + bg), compared with
    phot[t] x the direct torch_psf render of that frame's angle"""
    nb = 8; H = rs.H
    g = torch.Generator(device='cpu').manual_seed(3)
    x = (20 + 20 * torch.rand(nb, generator=g)).to(dev); y = (20 + 20 * torch.rand(nb, generator=g)).to(dev)
    z = (0.2 * torch.rand(nb, generator=g)).to(dev)
    em = dict(fov=torch.arange(nb, device=dev), x=x, y=y, z=z, bright=torch.full((nb,), 1000.0, device=dev),
              t0=torch.full((nb,), -10.0, device=dev), t1=torch.full((nb,), 100.0, device=dev))
    # mean image: patch poisson / randn / rounding out
    pois, rnd, rnr = torch.poisson, torch.randn_like, torch.round
    torch.poisson = lambda m: m; torch.randn_like = lambda m: torch.zeros_like(m)
    torch.round = lambda m: m if m.dim() == 4 else rnr(m)               # no ADU rounding; emitter pixel rounding unchanged
    try:
        o = rs.scene(nb, em=em, bgmode='flat', bg=1e-9, emitter_var=False, psf_var=False, exact_est=True, cam_err=False)
    finally:
        torch.poisson, torch.randn_like, torch.round = pois, rnd, rnr
    img = o['frames'] - o['bg']                                         # [nb, T, H, W] photons of the emitter alone
    phot = o['gt']['phot']                                              # [nb, T]
    ang = o['steps'] // 3
    psfs = {}
    has = sim.lut_angle is not None
    for k in (range(2) if has else [None]):
        p_ = spsf.PSF.load(calib, roisize=sim.R, device=dev, angle=k)
        pos = torch.stack([x - torch.round(x), y - torch.round(y), z], 1)
        psfs[k] = render(p_, pos)                                       # [nb, R, R]
    hr = sim.R // 2; errs = []; errs_lut = []
    for b in range(nb):
        cx, cy = int(torch.round(x[b])), int(torch.round(y[b]))
        for t in range(rs.T):
            k = int(ang[b, t]) if has else None
            ref = phot[b, t] * psfs[k][b]
            got = img[b, t, cy - hr:cy + hr + 1, cx - hr:cx + hr + 1]
            errs.append(cmp(got[None], ref[None]))
    return dict(max_abs_over_peak=max(e['max_abs_over_peak'] for e in errs), max_rel_px=max(e['max_rel_px'] for e in errs),
                note='direct render at the emitter positions (includes the LUT interpolation of D); per-angle PSF per frame' if has else
                'direct render at the emitter positions (includes the LUT interpolation of D)')


def decode_check(calib, vm):
    sys.path[:0] = [os.path.join(DEC, 'analysis/baselines/decode')]
    import decode_paper as DP
    lz = 21; zmax = 0.2
    lut, desc = DP.psf_luts(calib, lz=lz, zmax=zmax)                   # [lz, 33, 33, 41, 41]
    ls = lut.shape[1] - 1
    sub = torch.arange(ls + 1, device=dev, dtype=torch.float32) / ls - 0.5
    iz = [0, 5, 10, 15, 20]
    P = nodes([i / (lz - 1) * zmax for i in iz], sub[::4])
    ref = render(spsf.PSF.load(calib, roisize=41, device=dev), P)
    L = lut[iz][:, ::4, ::4].reshape(-1, 41, 41)
    fit = 0.5 * (render(vm[0].resized(41), P) + render(vm[1].resized(41), P))
    out = dict(desc=desc, lut_shape=list(lut.shape), module=DP.__file__, vs_direct=cmp(L, ref), vs_fit_mean=cmp(L, fit))
    return out


if __name__ == '__main__':
    main(sys.argv[1])
