"""
Training batches for HDNet from the experiment-matched simulator (simcode/hd/realsim.py) with widened ranges:
  preset per batch ('0.5nM' or '2.5nM' pattern / camera / emitter-variation model, PSF jitter on)
  ON density log-uniform dens (default 0.0005-0.06 per frame per px), on-time log-uniform 2-80 frames
  per-frame photons (full on, mean illumination) log-uniform 40-6000: half the FOVs per emitter over the full range,
  half with a per-FOV median x lognormal(0.5)
  close pairs: pair_frac of the events get a partner at 0.5-3 px
  bg level log-uniform 1-30 ph/px/frame; mode per FOV group: flat / struct (sd up to 40 %) / fluct (scale 0-1.5)
  depth randomised in [0.5, 1.0]
GT set for the loss: emitters inside the image with >= 0.5 on-frames and >= 50 window photons.
"""
import math
import torch
from simcode.hd.realsim import RealSim, lu

dev = 'cuda'


class Gen:
    def __init__(self, H=64, W=64, T=6, seed=0, calib=None):
        # calib (2026-11-11 psf3d retrain; None = the presets' PSF): list of in-situ calib jsons (e.g. the averaged
        # *_psf3d_avg.json of a movie) replacing the presets' PSF; one is drawn uniformly per batch (o['calib_i']), the
        # preset (pattern / camera / emitter model) is drawn as before. Both presets of one calib share its Simulator / LUT bank.
        self.calibs = None
        if calib:
            self.calibs = list(calib); self.rsm = []
            for c in self.calibs:
                d_ = {p: RealSim(p, H, W, T, seed=seed, calib=c) for p in ('0.5nM', '2.5nM')}
                d_['2.5nM'].sim = d_['0.5nM'].sim; d_['2.5nM'].luts = d_['0.5nM'].luts
                self.rsm.append(d_)
            torch.cuda.empty_cache()
            self.rs = self.rsm[0]; self.ci = 0
        else:
            self.rs = {p: RealSim(p, H, W, T, seed=seed) for p in ('0.5nM', '2.5nM')}
        self.H, self.W, self.T = H, W, T

    def all_rs(self):
        return [r for d_ in self.rsm for r in d_.values()] if self.calibs else list(self.rs.values())

    def pick(self, pre):
        """RealSim of preset `pre`; with a calib mixture, first draws the calib of this batch (self.ci, one extra draw)"""
        if self.calibs:
            self.ci = int(torch.randint(0, len(self.calibs), (1,)).item()); self.rs = self.rsm[self.ci]
        return self.rs[pre]

    def batch(self, nb, dens=(0.0005, 0.06), photons=(40.0, 6000.0), pair_frac=0.25, bgr=(1.0, 30.0), pair_curr=False,
              med_photons=None, em_photons=None, paper=False, tau_r=(2.0, 80.0), bright_tail=None):
        # bright_tail (iteration 31, critic 30 #5): (frac, lo, hi) -> that fraction of the emitters gets log-uniform lo-hi
        # fully-on window photons (x the TIRF factor) after the paper draw
        # med_photons: per-FOV median range (default = photons); em_photons: per-emitter draw range (default = photons)
        # (iteration 30 sc4 mix; defaults reproduce the sc2 / sc2L batches exactly)
        pre = '0.5nM' if torch.rand(1).item() < 0.5 else '2.5nM'
        rs = self.pick(pre)
        g = lambda *s: torch.rand(*s, device=dev)
        d = lu(*dens, nb); tau = lu(*tau_r, nb)
        med = lu(*(med_photons or photons), nb)
        em = rs.emitters(nb, dens=d, tau=tau, bright=med, pair_frac=pair_frac, pair_curr=pair_curr)
        N = len(em['fov'])
        per_em = (g(nb) < 0.5)[em['fov']]
        b_em = lu(*(em_photons or photons), N) * torch.exp(-(em['z'] - rs.p['z_med']) / rs.sim.tirf)
        em['bright'] = torch.where(per_em, b_em, em['bright']).clamp(photons[0] * 0.3, photons[1] * 3)
        if paper:      # iteration 30b (user): paper-matched photons, replaces both draws above
            em['bright'] = paper_bright(pre, nb, em['fov'], em['z'], rs)
        if bright_tail is not None and bright_tail[0] > 0:
            sel = torch.rand(N, device=dev) < bright_tail[0]
            bt = lu(bright_tail[1], bright_tail[2], N) / 6.0 * torch.exp(-(em['z'] - rs.p['z_med']) / rs.sim.tirf)
            em['bright'] = torch.where(sel, bt, em['bright'])
        u = torch.rand(1).item()
        mode = 'flat' if u < 0.2 else ('struct' if u < 0.5 else 'fluct')
        b0 = lu(*bgr, nb)
        sdf = 0.4 * g(nb)
        dep = 0.5 + 0.5 * torch.rand(1).item()
        save = dict(rs.p)
        rs.p['depth'] = (dep, min(1.0, dep + 0.03))
        try:
            o = rs.scene(nb, em=em, bgmode=mode, bg=b0, bg_sd=sdf * b0, bg_hp=0.05 * g(nb) * b0,
                         fluct=1.5 * torch.rand(1).item() * (b0 / 11.7).sqrt())
        finally:
            rs.p.clear(); rs.p.update(save)
        o['preset'] = pre; o['bgmode'] = mode
        if self.calibs: o['calib_i'] = self.ci
        return o


PAPER_MED = {'0.5nM': 450.0, '2.5nM': 800.0}     # window photons (6 frames, fully on), SIMCODE paper data p50 ~ 640 / 940
PAPER_SD, PAPER_CLIP = 0.55, (80.0, 5000.0)


def paper_bright(pre, nb, fov, z, rs, jitter=(0.7, 1.4), med0=None, sd=None):
    """per-frame brightness from a per-emitter log-normal of the 6-frame (fully-on) window photons: median PAPER_MED[pre]
    x a per-window jitter (log-uniform 0.7-1.4), log-sd 0.55, clipped to [80, 5000], then x the TIRF factor
    exp(-(z - z_med) / tirf)"""
    m0 = PAPER_MED[pre] if med0 is None else med0
    sd = PAPER_SD if sd is None else sd
    med = m0 * lu(*jitter, nb) if jitter else torch.full((nb,), m0, device=dev)
    N = len(fov)
    W = (med[fov] * torch.exp(sd * torch.randn(N, device=dev))).clamp(*PAPER_CLIP)
    return W / 6.0 * torch.exp(-(z - rs.p['z_med']) / rs.sim.tirf)


def gt_set(gt, H, W, min_on=0.5, min_phot=50.0):
    tot = gt['phot'].sum(1); on = gt['onf'].sum(1)
    ins = (gt['x'] > -0.5) & (gt['x'] < W - 0.5) & (gt['y'] > -0.5) & (gt['y'] < H - 0.5)
    return ins & (on >= min_on) & (tot >= min_phot)
