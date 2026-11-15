"""
Training batches of the detector and the emitter network from the acquisition simulator (simcode/simulate/acquisition.py),
one generator class TrainingBatches with three batch kinds:
  emitter_batch       emitter-network batches of pass 1, widened ranges:
    preset per batch ('0.5nM' or '2.5nM' pattern / camera / emitter-variation model, PSF jitter on)
    ON density log-uniform dens (default 0.0005-0.06 per frame per px), on-time log-uniform 2-80 frames
    per-frame photons (full on, mean illumination) log-uniform 40-6000: half the FOVs per emitter over the full range,
    half with a per-FOV median x lognormal(0.5); paper=True: the paper-matched photons (paper_bright) instead
    close pairs: pair_frac of the events get a partner at 0.5-3 px
    bg level log-uniform 1-30 ph/px/frame; mode per FOV group: flat / struct (sd up to 40 %) / fluct (scale 0-1.5)
    depth randomised in [0.5, 1.0]
  detector_batch      detector batches of pass 1: as emitter_batch with density log-uniform 0.0005-0.04 /px, per-window median
                      per-frame brightness log-uniform 40-2500 (half the windows) / per-emitter log-uniform em_photons (other
                      half), pair_frac 0.25 with pair separations uniform pair_sep (default 0.3-3.0 px, random direction)
  illumination_batch  batches of the pass-2 networks (needs p_std, the pass-2 setup below): the realistic settings and a
                      pattern-estimate input
    photons  mix 'paper' (default for the emitter network): per-emitter log-normal of the fully-on window photons in LABEL
             units, median 1260 (0.5 nM) / 1890 (2.5 nM) x per-window jitter 0.7-1.4, log-sd 0.45, + 10 % bright tail
             log-uniform 2k-20k; mix 'det' (half of the detector batches): the detector_batch photon mix (per-emitter
             log-uniform per-frame 40-2500 / per-window median) so the detector keeps its dim-emitter sensitivity
    intensity noise  frame_sd 0.09, step_sd U(0.07, 0.25) per batch (measured c_total ~ 0.22 = frame 0.09 + static 0.20,
             realism/c_calib.txt; the lower end = the old preset 0.07), angle factor sd preset
    kinetics on-time log-uniform 15-80 frames; density log-uniform 0.0005-0.04 /px; pair_frac 0.25, pair separation
             uniform pair_sep (detector 0.3-3.0 px) or the pair curriculum (emitter network: pair_curr)
    pattern  generic draw (simulate/acquisition.AcquisitionSim._generic: pitch 1.8-4 px, orientations, independent per-frame
             phases, frame order, per-angle depth, power) in a share 1 - p_std of the FOVs, the SIMCODE (DMD) pattern in the
             others; o['patt_input'] = illumination.perturb(true pattern) (realistic estimation errors), o['illum_input'] =
             illum_maps of it
TrainingBatches(..., p_std=None) = the pass-1 setup (the presets' intensity noise, the SIMCODE pattern only); p_std = a share
in [0, 1) = the pass-2 setup (frame_sd 0.09 and the generic pattern draw in a share 1 - p_std of the FOVs; P_STD = 0.5, the
recipe's first stages 0.25, models/README.md).
Emitter height z (3D emitter network, 2026-11-14): Z_UNIF of the emitters uniform over Z_RANGE_UM, the rest the presets'
lognormal (median 50-55 nm), both with the TIRF brightness factor; z of the GT in px relative to the calibration's focal
plane: z_target_px.
GT set for the loss: emitters inside the image with >= 0.5 on-frames and >= 50 window photons (gt_set).
"""
import math
import torch
from simcode.simulate.acquisition import AcquisitionSim, lu
from simcode.pattern import illumination as IL

dev = 'cuda'
# emitter heights of the training data (um above the coverslip). From the calibrations of the paper data (TIRF depth 82 nm,
# emitter median 55 nm): at 0.3 um the excitation is exp(-(0.3 - 0.055) / 0.082) = 5 % of the median emitter's and the
# collected PSF fraction 0.58 of the one at the coverslip (analysis/z3d/psf_zscan.py), i.e. < 40 photons per window for the
# paper's photon distribution: no detectable emitter lies above, so the network's z range is [0, 0.3] um.
Z_RANGE_UM = (0.0, 0.3)
Z_UNIF = 0.5


QMED = {'0.5nM': 1260.0, '2.5nM': 1890.0}     # illumination_batch 'paper' mix: median window photons in label units
P_STD = 0.5             # share of FOVs with the SIMCODE pattern (the rest: generic draw); 2026-11-13 out/p2generic g2ss
EM_PHOTONS = (40.0, 2500.0)                  # per-emitter photons of the per-emitter half of detector_batch (every model's value)
BRIGHT_TAIL = (0.1, 2000.0, 20000.0)         # (frac, lo, hi) window photons of the bright-tail draw (emitter_batch bright_tail)


def z_target_px(z_um, sim):
    """emitter height (um above the coverslip) -> the network's z: px of the lateral pixel size relative to the focal plane of
    the PSF calibration (torch_psf DATASET_FORMAT; z_px = (z - focus_um) / pixelsize_um)"""
    return (z_um - sim.focus) / sim.c['pixelsize_um']


class TrainingBatches:
    def __init__(self, H=64, W=64, T=6, seed=0, calib=None, p_std=None):
        # calib (2026-11-11 psf3d retrain; None = the presets' PSF): list of in-situ calib jsons (e.g. the averaged
        # *_psf3d_avg.json of a movie) replacing the presets' PSF; one is drawn uniformly per batch (o['calib_i']), the
        # preset (pattern / camera / emitter model) is drawn as before. Both presets of one calib share its Simulator / LUT bank.
        self.calibs = None
        if calib:
            self.calibs = list(calib); self.rsm = []
            for c in self.calibs:
                d_ = {p: AcquisitionSim(p, H, W, T, seed=seed, calib=c, zmax=Z_RANGE_UM[1], z_unif=Z_UNIF) for p in ('0.5nM', '2.5nM')}
                d_['2.5nM'].sim = d_['0.5nM'].sim; d_['2.5nM'].luts = d_['0.5nM'].luts
                self.rsm.append(d_)
            torch.cuda.empty_cache()
            self.rs = self.rsm[0]; self.ci = 0
        else:
            self.rs = {p: AcquisitionSim(p, H, W, T, seed=seed, zmax=Z_RANGE_UM[1], z_unif=Z_UNIF) for p in ('0.5nM', '2.5nM')}
        self.H, self.W, self.T = H, W, T
        self.p_std = p_std
        if p_std is not None:      # pass-2 setup (illumination_batch)
            if not 0.0 <= p_std < 1.0: raise ValueError(f'p_std {p_std}: the share of SIMCODE-pattern FOVs must be in [0, 1)')
            for rs in self.all_rs():
                rs.p['frame_sd'] = 0.09
                rs.p_std = float(p_std)      # generic pattern draw (simulate/acquisition.AcquisitionSim._generic) in a share 1 - p_std of the FOVs

    def all_rs(self):
        return [r for d_ in self.rsm for r in d_.values()] if self.calibs else list(self.rs.values())

    def pick(self, pre):
        """AcquisitionSim of preset `pre`; with a calib mixture, first draws the calib of this batch (self.ci, one extra draw)"""
        if self.calibs:
            self.ci = int(torch.randint(0, len(self.calibs), (1,)).item()); self.rs = self.rsm[self.ci]
        return self.rs[pre]

    def emitter_batch(self, nb, dens=(0.0005, 0.06), photons=(40.0, 6000.0), pair_frac=0.25, bgr=(1.0, 30.0), pair_curr=False,
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



    def detector_batch(self, nb, dens=(0.0005, 0.04), photons=(40.0, 6000.0), med_photons=(40.0, 2500.0), em_photons=None,
                       pair_frac=0.25, pair_sep=(0.3, 3.0), bgr=(1.0, 30.0)):
        """detector batches of pass 1 (module doc)"""
        pre = '0.5nM' if torch.rand(1).item() < 0.5 else '2.5nM'
        rs = self.pick(pre)
        g = lambda *s: torch.rand(*s, device=dev)
        d = lu(*dens, nb); tau = lu(2.0, 80.0, nb)
        med = lu(*med_photons, nb)
        em = rs.emitters(nb, dens=d, tau=tau, bright=med, pair_frac=pair_frac, pair_sep=pair_sep)
        N = len(em['fov'])
        per_em = (g(nb) < 0.5)[em['fov']]
        b_em = lu(*(em_photons or photons), N) * torch.exp(-(em['z'] - rs.p['z_med']) / rs.sim.tirf)
        em['bright'] = torch.where(per_em, b_em, em['bright']).clamp(photons[0] * 0.3, photons[1] * 3)
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
        o['preset'] = pre
        return o

    def illumination_batch(self, nb, mix='paper', dens=(0.0005, 0.04), pair_frac=0.25, pair_sep=(0.5, 3.0), pair_curr=False,
                           tau_r=(15.0, 80.0), bright_tail=(0.1, 2000.0, 20000.0), pert_scale=1.0, bgr=(1.0, 30.0),
                           photons=(40.0, 6000.0), det_em=(40.0, 2500.0), step_sd=(0.07, 0.25), pert_model='v1'):
        """batches of the pass-2 networks with the pattern estimate (module doc); needs the pass-2 setup (p_std)"""
        if self.p_std is None: raise ValueError('illumination_batch needs TrainingBatches(..., p_std=<share>) (the pass-2 setup)')
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
        else:      # the detector_batch photon mix
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
        o['patt_input'] = IL.perturb(o['patt'], self.H, self.W, pert_scale, pert_model)
        o['illum_input'] = IL.illum_maps(o['patt_input'], self.H, self.W)
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
