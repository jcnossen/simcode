"""
Realistic training batches (2026-11-10, "retrain on realistic sims"; the realism report =,
A2 / A3 / A6). One generator for all four networks (pass-1 detector / head, pass-2 detector / head); the old generators
(hd/data.Gen.batch, det/up2.batch, p2/data.Gen2.batch2) are unchanged.

Recipe 'r1' (per batch unless stated):
  preset      '0.5nM' / '2.5nM' (50 / 50; realsim PSF + jitter, camera maps, bg modes, emitter variation)
  geometry    struct_frac 0.9 (10 % uniform FOVs for generality); filaments fil_on 0.97 + fil_resample (no uniform fallback:
              outside points are redrawn), fil_density log-uniform 0.5-1.2 filament px per px^2 (real-like counts: the
              realistic movies use 0.6 / 0.8)
  density     main ON density log-uniform 0.0005-0.1 per px per frame (20 nM-like up to 0.1), pairs pair_frac 0.2
  dim binders with prob. 0.6 per batch: a second population on the SAME filaments (structured FOVs only), density
              U(0, 1) x the FOV's main density, window photons median log-uniform 150-500 (log-sd 0.4), on-time
              log-uniform 10-30 frames (2.5 nM scorecard: 0.02 / px / frame, 300 ph, tau 20)
  photons     mix 'paper': label-unit fully-on window photons, median 700 (0.5 nM) / 750 (2.5 nM) (realistic scorecard;
              m05r / m25r) x per-FOV jitter log-uniform 0.6-1.6, log-sd 0.5, clipped 80-5000, + 10 % bright tail
              log-uniform 2k-20k (as sc5nb); x TIRF factor. mix 'det': the det_n1 mix (dim-emitter sensitivity)
  intensity   frame_sd U(0.06, 0.12), step_sd U(0.05, 0.16) (real autoc c 0.12-0.14; static misfit now explicit below)
  pattern fld static per-FOV phase field: rms U(0, 0.18) rad per angle (0-6.4 nm; measured own10 2.7-5.7 nm), smooth sigma
              U(5, 12) px (10 px cells); depth factor 1 + U(0, 0.08) x smooth field (sigma 25 px) (measured sd 0.05-0.09);
              the networks / solve get the global pattern only (patt_est with the realsim estimate errors; pass 2: perturbed)
  PSF         unchanged; bg log-uniform 1-40 ph/px/frame (modes as before); depth U(0.6, 1.0)
API: RGen(H, W, T, seed).rbatch(nb, mix, tau_r, pair_frac, pair_sep, pair_curr, p2, pert_model) -> o (as Gen.batch; with
p2: + patt_p2 / illum_p2 as p2/data.Gen2.batch2); .batch2(...) = rbatch(p2=True) (drop-in for Gen2 in p2/det2 + p2/head2).
GT rows of the dim population carry o['gt']['dim'] = 1 (they are ordinary targets for the losses).
"""
import torch
from simcode2.hd.realsim import lu
from simcode2.hd.data import Gen, PAPER_CLIP
from simcode2.p2 import illum as IL

dev = 'cuda'
R1 = dict(struct_frac=0.9, fil_on=0.97, fil_density=(0.5, 1.2), dens=(0.0005, 0.1), pair_frac=0.2,
          dim_p=0.6, dim_ratio=(0.0, 1.0), dim_med=(150.0, 500.0), dim_sd=0.4, dim_tau=(10.0, 30.0),
          med={'0.5nM': 700.0, '2.5nM': 750.0}, med_jit=(0.6, 1.6), med_sd=0.5, tail=(0.1, 2000.0, 20000.0),
          det_em=(40.0, 2500.0), photons=(40.0, 6000.0),
          frame_sd=(0.06, 0.12), step_sd=(0.05, 0.16), pf_rad=0.18, pf_sig=(5.0, 12.0), pf_dep=0.08, pf_dsig=25.0,
          bgr=(1.0, 40.0), depth=(0.6, 1.0))


def _win_bright(med_fov, fov, z, rs, sd):
    W = (med_fov[fov] * torch.exp(sd * torch.randn(len(fov), device=dev))).clamp(*PAPER_CLIP)
    return W / 6.0 * torch.exp(-(z - rs.p['z_med']) / rs.sim.tirf)


class RGen(Gen):
    def __init__(self, H=64, W=64, T=6, seed=0, recipe=None):
        super().__init__(H, W, T, seed)
        self.R = dict(R1, **(recipe or {}))

    def rbatch(self, nb, mix='paper', tau_r=(15.0, 80.0), pair_frac=None, pair_sep=(0.5, 3.0), pair_curr=False,
               p2=False, pert_model='v1', pert_scale=1.0, dens=None):
        R = self.R
        pre = '0.5nM' if torch.rand(1).item() < 0.5 else '2.5nM'
        rs = self.rs[pre]
        g = lambda *s: torch.rand(*s, device=dev)
        save = dict(rs.p)
        try:
            fd = lu(*R['fil_density'], 1).item()
            rs.p.update(fil_on=R['fil_on'], fil_resample=True, fil_density=fd)
            d = lu(*(dens or R['dens']), nb); tau = lu(*tau_r, nb)
            med = lu(*R['det_em'], nb)
            pf = R['pair_frac'] if pair_frac is None else pair_frac
            em = rs.emitters(nb, dens=d, tau=tau, struct_frac=R['struct_frac'], bright=med, pair_frac=pf, pair_sep=pair_sep,
                             pair_curr=pair_curr)
            N = len(em['fov'])
            tirf = lambda z_: torch.exp(-(z_ - rs.p['z_med']) / rs.sim.tirf)
            if mix == 'paper':
                mf = R['med'][pre] * lu(*R['med_jit'], nb)
                em['bright'] = _win_bright(mf, em['fov'], em['z'], rs, R['med_sd'])
                if R['tail'] and R['tail'][0] > 0:
                    sel = g(N) < R['tail'][0]
                    em['bright'] = torch.where(sel, lu(R['tail'][1], R['tail'][2], N) / 6.0 * tirf(em['z']), em['bright'])
            else:      # det_n1 mix
                per_em = (g(nb) < 0.5)[em['fov']]
                b_em = lu(*R['det_em'], N) * tirf(em['z'])
                em['bright'] = torch.where(per_em, b_em, em['bright']).clamp(R['photons'][0] * 0.3, R['photons'][1] * 3)
            dim = torch.zeros(N, device=dev)
            if em['fil_pts'] is not None and bool(em['fil_struct'].any()) and torch.rand(1).item() < R['dim_p']:
                r = R['dim_ratio'][0] + (R['dim_ratio'][1] - R['dim_ratio'][0]) * g(nb)
                e2 = rs.emitters(nb, dens=(r * d).clamp(min=1e-6), tau=lu(*R['dim_tau'], nb), struct_frac=1.0, fil_pts_in=em['fil_pts'])
                k2 = em['fil_struct'][e2['fov']]
                if bool(k2.any()):
                    e2 = {k_: e2[k_][k2] for k_ in ('fov', 'x', 'y', 'z', 't0', 't1')}
                    e2['bright'] = _win_bright(lu(*R['dim_med'], nb), e2['fov'], e2['z'], rs, R['dim_sd'])
                    for k_ in ('fov', 'x', 'y', 'z', 'bright', 't0', 't1'): em[k_] = torch.cat([em[k_], e2[k_]])
                    dim = torch.cat([dim, torch.ones(len(e2['x']), device=dev)])
            u = torch.rand(1).item()
            mode = 'flat' if u < 0.2 else ('struct' if u < 0.5 else 'fluct')
            b0 = lu(*R['bgr'], nb)
            sdf = 0.4 * g(nb)
            dep = R['depth'][0] + (R['depth'][1] - R['depth'][0]) * torch.rand(1).item()
            rs.p['depth'] = (dep, min(1.0, dep + 0.03))
            rs.p['frame_sd'] = R['frame_sd'][0] + (R['frame_sd'][1] - R['frame_sd'][0]) * torch.rand(1).item()
            rs.p['step_sd'] = R['step_sd'][0] + (R['step_sd'][1] - R['step_sd'][0]) * torch.rand(1).item()
            rs.p.update(pf_rad=R['pf_rad'], pf_sig=R['pf_sig'], pf_dep=R['pf_dep'], pf_dsig=R['pf_dsig'])
            o = rs.scene(nb, em=em, bgmode=mode, bg=b0, bg_sd=sdf * b0, bg_hp=0.05 * g(nb) * b0,
                         fluct=1.5 * torch.rand(1).item() * (b0 / 11.7).sqrt())
        finally:
            rs.p.clear(); rs.p.update(save)
        o['gt']['dim'] = dim
        o['preset'] = pre; o['bgmode'] = mode
        if p2:
            o['patt_p2'] = IL.perturb(o['patt'], self.H, self.W, pert_scale, pert_model)
            o['illum_p2'] = IL.illum_maps(o['patt_p2'], self.H, self.W)
        return o

    def batch2(self, nb, mix='paper', pair_sep=(0.5, 3.0), pair_curr=False, tau_r=(15.0, 80.0), pert_model='v1', pert_scale=1.0, **kw):
        """drop-in for p2/data.Gen2.batch2 (other Gen2 knobs ignored: the recipe sets them)"""
        return self.rbatch(nb, mix=mix, tau_r=tau_r, pair_sep=pair_sep, pair_curr=pair_curr, p2=True, pert_model=pert_model,
                           pert_scale=pert_scale)
