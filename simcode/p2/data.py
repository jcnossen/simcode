"""
Realistic training batches for the pass-2 networks (copy of simcode/hd/data.Gen.batch with the realistic settings and a
pattern-estimate input; the original generator is unchanged).

  preset per batch '0.5nM' / '2.5nM' (realsim: in-situ PSF + jitter, camera maps, structured / fluctuating bg, filaments
         (struct_frac 0.5 / 0.85) + random sites)
  photons  mix 'paper' (default for the head): per-emitter log-normal of the fully-on window photons in LABEL units,
           median 1260 (0.5 nM) / 1890 (2.5 nM) x per-window jitter 0.7-1.4, log-sd 0.45, + 10 % bright tail log-uniform
           2k-20k (as sc6n / sc5nb); mix 'det' (half of the detector batches): the det_n1 mix (per-emitter log-uniform
           per-frame 40-2500 / per-window median) so the detector keeps its dim-emitter sensitivity
  intensity noise  frame_sd 0.09, step_sd U(0.07, 0.25) per batch (measured c_total ~ 0.22 = frame 0.09 + static 0.20,
           realism/c_calib.txt; the lower end = the old preset 0.07), angle factor sd preset
  kinetics on-time log-uniform 15-80 frames; density log-uniform 0.0005-0.04 /px; pair_frac 0.25, pair separation
           uniform pair_sep (detector 0.3-3.0 px) or the sc5 curriculum (head: pair_curr)
  pattern  generic draw (hd/realsim.RealSim._generic: pitch 1.8-4 px, orientations, independent per-frame phases, frame order,
           per-angle depth, power) in a share 1 - p_std of the FOVs, the SIMCODE (DMD) pattern in the others (p_std default
           P_STD = 0.5; the recipe's first stages 0.25, models/README.md); o['patt_p2'] = illum.perturb(true pattern) (realistic
           estimation errors), o['illum_p2'] = illum_maps of it
"""
import torch
from simcode.hd.realsim import lu
from simcode.hd.data import Gen, paper_bright
from simcode.p2 import illum as IL

dev = 'cuda'
QMED = {'0.5nM': 1260.0, '2.5nM': 1890.0}
P_STD = 0.5     # share of FOVs with the SIMCODE pattern (the rest: generic draw); 2026-11-13 out/p2generic g2ss


class Gen2(Gen):
    def __init__(self, H=64, W=64, T=6, seed=0, calib=None, p_std=P_STD):
        super().__init__(H, W, T, seed, calib=calib)       # calib (2026-11-11, opt-in): PSF calib mixture, see hd/data.Gen
        if not 0.0 <= p_std < 1.0: raise ValueError(f'p_std {p_std}: the share of SIMCODE-pattern FOVs must be in [0, 1)')
        for rs in self.all_rs():
            rs.p['frame_sd'] = 0.09
            rs.p_std = float(p_std)      # generic pattern draw (hd/realsim.RealSim._generic) in a share 1 - p_std of the FOVs

    def batch2(self, nb, mix='paper', dens=(0.0005, 0.04), pair_frac=0.25, pair_sep=(0.5, 3.0), pair_curr=False,
               tau_r=(15.0, 80.0), bright_tail=(0.1, 2000.0, 20000.0), pert_scale=1.0, bgr=(1.0, 30.0),
               photons=(40.0, 6000.0), det_em=(40.0, 2500.0), step_sd=(0.07, 0.25), pert_model='v1'):
        pre = '0.5nM' if torch.rand(1).item() < 0.5 else '2.5nM'
        rs = self.pick(pre)
        g = lambda *s: torch.rand(*s, device=dev)
        d = lu(*dens, nb); tau = lu(*tau_r, nb)
        med = lu(*det_em, nb)
        em = rs.emitters(nb, dens=d, tau=tau, bright=med, pair_frac=pair_frac, pair_sep=pair_sep, pair_curr=pair_curr)
        N = len(em['fov'])
        if mix == 'paper':
            em['bright'] = paper_bright(pre, nb, em['fov'], em['z'], rs, med0=QMED[pre], sd=0.45)
            if bright_tail is not None and bright_tail[0] > 0:
                sel = torch.rand(N, device=dev) < bright_tail[0]
                bt = lu(bright_tail[1], bright_tail[2], N) / 6.0 * torch.exp(-(em['z'] - rs.p['z_med']) / rs.sim.tirf)
                em['bright'] = torch.where(sel, bt, em['bright'])
        else:      # det_n1 mix (simcode/det/up2.batch)
            per_em = (g(nb) < 0.5)[em['fov']]
            b_em = lu(*det_em, N) * torch.exp(-(em['z'] - rs.p['z_med']) / rs.sim.tirf)
            em['bright'] = torch.where(per_em, b_em, em['bright']).clamp(photons[0] * 0.3, photons[1] * 3)
        u = torch.rand(1).item()
        mode = 'flat' if u < 0.2 else ('struct' if u < 0.5 else 'fluct')
        b0 = lu(*bgr, nb)
        sdf = 0.4 * g(nb)
        dep = 0.5 + 0.5 * torch.rand(1).item()
        save = dict(rs.p)
        rs.p['depth'] = (dep, min(1.0, dep + 0.03))
        rs.p['step_sd'] = step_sd[0] + (step_sd[1] - step_sd[0]) * torch.rand(1).item()
        try:
            o = rs.scene(nb, em=em, bgmode=mode, bg=b0, bg_sd=sdf * b0, bg_hp=0.05 * g(nb) * b0,
                         fluct=1.5 * torch.rand(1).item() * (b0 / 11.7).sqrt())
        finally:
            rs.p.clear(); rs.p.update(save)
        o['preset'] = pre; o['bgmode'] = mode
        if self.calibs: o['calib_i'] = self.ci
        o['patt_p2'] = IL.perturb(o['patt'], self.H, self.W, pert_scale, pert_model)
        o['illum_p2'] = IL.illum_maps(o['patt_p2'], self.H, self.W)
        return o
