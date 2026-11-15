"""DECODE baseline for the high-density sim comparison (critic 35 #6). Shared pieces for training and inference.

DECODE = the fork the SIMCODE paper used (~/decode2, editable install in simcode-py311; imported read-only), SigmaMUNet,
3-frame input (channels_in 3), driven through decode.instance.DecodeInstance exactly as ~/smlmtorch/scripts/decode_localize.py.

Paper usage (checked 2026-11-10):
  * 20 nM movie 2 (fig_dmdsim_bn_dnapaint_20nM.ipynb): model 'decode-1f' (bead-calibrated cspline zstack1_3dcal.mat,
    trained at 30 emitters / 40x40 px, 3000 +- 3000 ph, lifetime 30), run per frame -> decode.hdf5 and on non-overlapping
    6-frame sums (sumframes=6, frame *= 6) -> decode-summed.hdf5; the figure / FRC use decode-summed as 'DECODE'.
  * 2.5 nM movie 2 (fig_dmdsim_bn_dnapaint-2.5nM.ipynb, run_pipeline2.py): 'decode-3f-g1.3' (Gaussian sigma 1.3 px,
    40 / 40x40, 400 +- 800 ph, lifetime 2) on 3-frame sums -> decode-summed3.hdf5.
  The paper models are copied to the scratchpad (DecodeInstance writes param_out.yaml / logs into the model dir; /mnt/data
  must not be modified).

Matched models (this work): DECODE trained on DECODE's own simulator with the sim movie's PSF, density, photons, bg, camera:
  PSF      = the movie's vectorial PSF LUTs (AcquisitionSim '2.5nM' calibration, 41x41 px ROI, base + 3 jittered LUTs drawn with a
             different seed than the movie's, + the 0-5 % sigma-4 px halo), z 10-150 nm (sim z p1-p99 22-140 nm)
  density  = emitters per 40x40 frame drawn per epoch uniformly in DENS_RANGE (covers m25qh 0.024 and m20qh 0.044 /px^2)
  photons  = lognormal (median, log-sd) of the sim's brightness (per frame: 313, 0.45 x the sine pattern with depth
             0.74-0.82, pitch 2.07 px, relint 0.88-1.12, x per-frame jitter (1 + 0.2 N)(1 + 0.09 N) as the movie's step/frame
             factors; 6-frame sum: 1870, 0.45, no pattern); on-time exponential, mean 65 frames (10.8 sums)
  bg       = per-block level U(BG) x (1 + smooth field sd 0.27 at sigma 8 px) + fine field (sd 0.45 ph at 2.5 px) x per-frame
             factor U(0.91, 1.09) (movie: bg 10.8-12.5, bg_sd 3-3.6 at 8 px, bg_hp 0.45, bg_ang 1.09 / 0.91)
  camera   = Poisson + 0.9 e read noise, 0.56 e/ADU floor, baseline 100 ADU, output in photons (as the movie's (ADU-off)*gain)
"""
import os, sys, math, glob
from simcode import config as _cfg
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np, torch
import decode
from decode.utils import param_io
from decode.utils.types import RecursiveNamespace
from decode.instance import DecodeInstance
from decode.simulation.psf_kernel import PSF
from decode.simulation.background import Background

SP = _cfg.root('out')
SPD = f'{SP}/decode_hd'
LUTPSF = f'{SPD}/lutpsf.txt'          # dummy 'psf file': models live in SPD/lutpsf/<name>/
PAPER_PSF = f'{SPD}/paper/zstack1_3dcal.mat'
CAM = _cfg.model('insitu/camera_maps_0.5nM.npz')

VARIANTS = {
    # per-frame DECODE (pattern-modulated frames)
    'frame': dict(med=313.0, lsd=0.45, th=30.0, life=65.0, bg=(7.0, 18.0), read=0.9, pattern=True, jitter=True,
                  label_min=50.0, phot_max=3500.0, dens=(30.0, 85.0)),
    # DECODE on non-overlapping 6-frame sums (the paper's 'decode-summed')
    'sum6': dict(med=1870.0, lsd=0.45, th=120.0, life=65.0 / 6, bg=(42.0, 108.0), read=0.9 * math.sqrt(6), pattern=False,
                 jitter=False, label_min=150.0, phot_max=16000.0, dens=(32.0, 90.0)),
}


# ------------------------------------------------------------------------------------------------- PSF from the sim LUTs
class LutPSF(PSF):
    def __init__(self, xextent, yextent, img_shape, luts, zmax_um, halo=0.05, device='cuda'):
        super().__init__(xextent, yextent, zextent=None, img_shape=img_shape)
        self.luts = luts
        self.lz, self.ls = luts[0].shape[0], luts[0].shape[1] - 1
        self.R = luts[0].shape[-1]
        self.zmax = zmax_um; self.halo = halo; self.device = device

    def _render(self, lut, x, y, z):          # = simcode.simulate.simulator.Simulator.render_psf
        fx, fy = x - torch.round(x), y - torch.round(y)
        tx = (fx + 0.5) * self.ls; ty = (fy + 0.5) * self.ls
        ix = tx.floor().long().clamp(0, self.ls - 1); iy = ty.floor().long().clamp(0, self.ls - 1)
        wx = (tx - ix)[:, None, None]; wy = (ty - iy)[:, None, None]
        tz = (z / self.zmax).clamp(0, 1) * (self.lz - 1)
        iz = tz.floor().long().clamp(0, self.lz - 2); wz = (tz - iz)[:, None, None]
        out = 0
        for dz, w_z in ((0, 1 - wz), (1, wz)):
            a = lambda dy, dx: lut[iz + dz, iy + dy, ix + dx]
            out = out + w_z * ((1 - wy) * ((1 - wx) * a(0, 0) + wx * a(0, 1)) + wy * ((1 - wx) * a(1, 0) + wx * a(1, 1)))
        return out

    def rois(self, x, y, z_um):
        N = len(x); R = self.R
        # per-emitter (position-hashed, so constant over an emitter's frames) LUT group and halo fraction
        h = torch.frac(torch.sin(x * 12.9898 + y * 78.233) * 43758.5453).abs()
        grp = (h * len(self.luts)).long().clamp(max=len(self.luts) - 1)
        hh = self.halo * torch.frac(h * 7.0)
        psf = torch.zeros(N, R, R, device=self.device)
        for gi, lut in enumerate(self.luts):
            m = grp == gi
            if m.any(): psf[m] = self._render(lut, x[m], y[m], z_um[m])
        o = torch.arange(R, device=self.device, dtype=torch.float32) - R // 2
        fx, fy = x - torch.round(x), y - torch.round(y)
        gb = torch.exp(-0.5 * (((o[None, None, :] - fx[:, None, None]) ** 2 + (o[None, :, None] - fy[:, None, None]) ** 2) / 16.0))
        gb = gb / gb.sum((1, 2), keepdim=True) * psf.sum((1, 2), keepdim=True)
        return (1 - hh[:, None, None]) * psf + hh[:, None, None] * gb

    def forward(self, xyz, weight, frame_ix=None, n_frames=None):
        xyz, weight, frame_ix, n_frames = super().forward(xyz, weight, frame_ix, n_frames)
        H, W = self.img_shape; R = self.R; P = R // 2
        canvas = torch.zeros(n_frames, H + 2 * P, W + 2 * P, device=self.device)
        if len(xyz):
            xyz = xyz.to(self.device).float(); weight = weight.to(self.device).float(); frame_ix = frame_ix.to(self.device).long()
            o = torch.arange(R, device=self.device) - P
            for s in range(0, len(xyz), 20000):
                x, y, z = xyz[s:s + 20000, 0], xyz[s:s + 20000, 1], xyz[s:s + 20000, 2] * 1e-3
                ps = self.rois(x, y, z) * weight[s:s + 20000, None, None]
                cx = torch.round(x).long() + P; cy = torch.round(y).long() + P
                xx = (cx[:, None, None] + o[None, None, :]).expand(-1, R, R); yy = (cy[:, None, None] + o[None, :, None]).expand(-1, R, R)
                ok = (xx >= 0) & (xx < W + 2 * P) & (yy >= 0) & (yy < H + 2 * P)
                f = frame_ix[s:s + 20000, None, None].expand(-1, R, R)
                lin = (f * (H + 2 * P) + yy) * (W + 2 * P) + xx
                canvas.view(-1).index_add_(0, lin[ok], ps[ok])
        return canvas[:, P:P + H, P:P + W].cpu()


def smooth_field(n, H, W, sig, device):
    k = int(3 * sig); o = torch.arange(-k, k + 1, device=device, dtype=torch.float32)
    g = torch.exp(-0.5 * (o / sig) ** 2); g = g / g.sum()
    x = torch.randn(n, 1, H + 2 * k, W + 2 * k, device=device)
    x = torch.nn.functional.conv2d(x, g.view(1, 1, 1, -1)); x = torch.nn.functional.conv2d(x, g.view(1, 1, -1, 1))[:, 0]
    return (x - x.mean((1, 2), keepdim=True)) / x.std((1, 2), keepdim=True).clamp(min=1e-6)


class StructBg(Background):
    """bg = level_block x (1 + 0.27 field8) + 0.45 x sqrt(level/11) field2.5, x per-frame factor; blocks of 48 frames."""
    def __init__(self, bg_range, block=48):
        super().__init__(forward_return='tuple'); self.lo, self.hi = bg_range; self.block = block

    def sample(self, size, device=torch.device('cpu')):
        T, H, W = size[-3], size[-2], size[-1]
        nb = (T + self.block - 1) // self.block
        lev = self.lo + (self.hi - self.lo) * torch.rand(nb, 1, 1, device=device)
        st = lev * (1 + 0.27 * smooth_field(nb, H, W, 8.0, device)) + 0.45 * torch.sqrt(lev / 11.0) * smooth_field(nb, H, W, 2.5, device)
        bg = st.repeat_interleave(self.block, 0)[:T] * (0.91 + 0.18 * torch.rand(T, 1, 1, device=device))
        return bg.clamp(min=0.3).reshape(size)


# ------------------------------------------------------------------------------------------------- params / instance
def make_param(variant, epochs=400, pseudo=8192, batch=32, device='cuda:0'):
    V = VARIANTS[variant]
    p = RecursiveNamespace(**param_io.load_reference())
    p.Camera.baseline = 100.0; p.Camera.convert2photons = True; p.Camera.e_per_adu = 0.56; p.Camera.em_gain = None
    p.Camera.px_size = [108.33, 108.33]; p.Camera.qe = 1.0; p.Camera.read_sigma = V['read']; p.Camera.spur_noise = 0.0
    p.CameraPreset = None
    S = p.Simulation
    S.img_size = [40, 40]; S.emitter_extent = [[-0.5, 39.5], [-0.5, 39.5], [10.0, 150.0]]; S.psf_extent = S.emitter_extent
    S.psf_type = 'LutPSF'; S.roi_size = None; S.xy_unit = 'px'; S.mode = 'acquisition'
    S.emitter_av = float(np.mean(V['dens']))
    mean = V['med'] * math.exp(V['lsd'] ** 2 / 2); sd = mean * math.sqrt(math.exp(V['lsd'] ** 2) - 1)
    S.intensity_mu_sig = [mean, sd]; S.intensity_th = V['th']; S.lifetime_avg = V['life']; S.bg_uniform = list(V['bg'])
    S.frame_block_size = 1
    S.excitation_pattern.type = 'sine' if V['pattern'] else 'uniform'
    S.excitation_pattern.angles_deg = [1.27, 91.38]; S.excitation_pattern.pitch_px = [2.07, 2.07]
    S.excitation_pattern.depth_range = [0.74, 0.82]; S.excitation_pattern.relint_range = [0.88, 1.12]
    S.excitation_pattern.equidistant_steps = False
    H = p.HyperParameter
    a = H.arch_param
    a.depth_union = 2; a.init_custom = True; a.intensity_scale = 'sigmoid'     # as the paper's decode-3f-g1.3
    H.architecture = 'SigmaMUNet'; H.channels_in = 3; H.batch_size = batch; H.pseudo_ds_size = pseudo; H.epochs = epochs
    H.emitter_label_photon_min = V['label_min']; H.max_number_targets = 250; H.enable_pattern_prior = False
    H.opt_param.lr = 2e-4; H.opt_param.weight_decay = 0.1
    H.learning_rate_scheduler = 'ReduceLROnPlateau'; H.learning_rate_scheduler_param.factor = 0.5; H.learning_rate_scheduler_param.patience = 10
    p.Hardware.device = device; p.Hardware.device_ix = 0; p.Hardware.device_simulation = 'cpu'   # PSF renders on the GPU, returns CPU frames
    p.Hardware.num_worker_train = 0; p.Hardware.torch_threads = 4
    p.TestSet.test_size = 512; p.TestSet.bg_uniform = list(V['bg']); p.TestSet.img_size = [40, 40]
    p.Scaling.phot_max = V['phot_max']
    p.InOut.calibration_file = None
    return p


_LUTS = None


def sim_luts(device='cuda'):
    """base + 3 jittered LUTs of the movie's PSF model (AcquisitionSim '2.5nM' calibration) with seed 3 (the movie used seed 7)."""
    global _LUTS
    if _LUTS is None:
        from simcode.simulate.acquisition import AcquisitionSim
        # (2026-11-11, opt-in) DEC_CALIB=<calib json>: the movie's in-situ 3D PSF (e.g. models/insitu/calib_*_psf3d.json) instead of
        # the preset's; a per-pattern-angle PSF enters as its angle average (DECODE is pattern-free; exact for the 6-frame sums)
        over = {'calib': os.environ['DEC_CALIB']} if os.environ.get('DEC_CALIB') else {}
        rs = AcquisitionSim('2.5nM', 40, 40, 6, seed=3, **over)
        luts = [(l[0] + l[1]) / 2 if isinstance(l, tuple) else l for l in rs.luts]
        _LUTS = (luts, rs.zmax, rs.p['halo'])
    return _LUTS


class LutDecodeInstance(DecodeInstance):
    """DecodeInstance with the sim-LUT PSF, structured bg and (per-frame variant) per-frame intensity jitter / lognormal
    brightness / per-epoch density draw."""
    variant = 'frame'

    def __init__(self, param, model_name, variant):
        LutDecodeInstance.variant = variant
        os.makedirs(SPD, exist_ok=True)
        if not os.path.exists(LUTPSF): open(LUTPSF, 'w').write('dummy psf file; models in lutpsf/<name>/\n')
        super().__init__(param, psf_path=LUTPSF, model_name=model_name)
        self._patch_sim(self.sim_train, randomize=True); self._patch_sim(self.sim_test, randomize=False)

    def create_psf(self):
        luts, zmax, halo = sim_luts()
        return LutPSF(self.param.Simulation.psf_extent[0], self.param.Simulation.psf_extent[1], self.param.Simulation.img_size,
                      luts, zmax, halo=halo, device=self.param.Hardware.device)

    def _init_optics_models(self):
        super()._init_optics_models()
        self.bg_model = StructBg(self.param.Simulation.bg_uniform)
        self.bg_model_test = StructBg(self.param.TestSet.bg_uniform)

    def _patch_sim(self, sim, randomize):
        V = VARIANTS[self.variant]; es = sim.em_sampler
        es.intensity_dist = torch.distributions.LogNormal(math.log(V['med']), V['lsd'])
        if randomize:
            orig = es.sample
            def sample():
                es._em_avg = float(np.random.uniform(*V['dens']))
                es._emitter_av_total = es._em_avg * es._num_frames_plus / (es.lifetime_avg + 1)
                return orig()
            es.sample = sample
        if V['jitter']:
            fwd = sim.forward
            def forward(em, ix_low=None, ix_high=None, excitation_patterns=None):
                em = em.clone()
                j = (1 + 0.2 * torch.randn(len(em))) * (1 + 0.09 * torch.randn(len(em)))
                em.phot = em.phot * j.clamp(min=0.05).to(em.phot.device)
                return fwd(em, ix_low, ix_high, excitation_patterns)
            sim.forward = forward


def load_trained(model_name, variant, device='cuda:0', model_file=None):
    p = param_io.load_params(f'{SPD}/lutpsf/{model_name}/param_train.yaml')
    p.HyperParameter.pseudo_ds_size = 64; p.TestSet.test_size = 16
    p.Hardware.device = device; p.Hardware.device_simulation = 'cpu'
    inst = LutDecodeInstance(p, model_name, variant)
    inst.load_model(model_file)
    return inst


def _patch_fastpsf():
    """fastpsf's SMAP .mat reader fails with the current numpy (dz is a (1,1) array); same logic, np.asarray(...).item()."""
    import scipy.io
    from fastpsf import CSplineCalibration
    def from_file_nmeth(cls, filename):
        mat = scipy.io.loadmat(filename)
        try:
            spline = mat["SXY"]['cspline'].item(); coefs = spline['coeff'].item(); dz = float(np.asarray(spline['dz'].item()).item())
        except KeyError:
            spline = mat['cspline'].item(); coefs = spline[0]; dz = float(np.asarray(spline[1]).item())
        if len(coefs.shape) == 2: coefs = coefs[0, 0]
        coefs = np.ascontiguousarray(coefs, dtype=np.float32)
        nx, ny, nz = coefs.shape[:-1]
        coefs = np.transpose(coefs.reshape((nx, ny, nz, 4, 4, 4)), (2, 0, 1, 3, 5, 4)).reshape((nz, ny, nx, 64))
        nz = float(coefs.shape[0])
        return cls(-nz * dz * 1e-3 / 2, nz * dz * 1e-3 / 2, coefs)
    CSplineCalibration.from_file_nmeth = classmethod(from_file_nmeth)


def load_paper(model_name, model_file='model_2.pt', device='cuda:0'):
    _patch_fastpsf()
    """the paper's DECODE model (scratchpad copy), loaded as decode_localize.py does."""
    p = param_io.load_params(f'{SPD}/paper/zstack1_3dcal/{model_name}/param_out.yaml')
    p.HyperParameter.pseudo_ds_size = 64; p.HyperParameter.batch_size = 16
    p.Hardware.num_worker_train = 0; p.Hardware.torch_threads = 1; p.Hardware.device_ix = 0
    p.Hardware.device = device; p.Hardware.device_simulation = 'cpu'
    p.Simulation.frame_block_size = 1; p.HyperParameter.enable_pattern_prior = False
    p.TestSet.test_size = 16; p.TestSet.bg_uniform = 10
    inst = DecodeInstance(p, psf_path=PAPER_PSF, model_name=model_name)
    inst.load_model(f'{SPD}/paper/zstack1_3dcal/{model_name}/{model_file}')
    return inst


# ------------------------------------------------------------------------------------------------- movie inference
def infer_movie(inst, tif, sumframes=1, start=0, maxframes=None, batch_size=64):
    """DECODE on the movie (photons = (ADU - offset map) * gain, clamped at 0, as decode_localize.py). sumframes > 1:
    non-overlapping sums of frames start + sumframes * k ... ; output frame = start + sumframes * k (= window start).
    -> dict of numpy arrays: frame, x, y (px, pixel-centre = integer, x = column), sx, sy, phot, sphot, bg, prob"""
    import tifffile
    cam = np.load(CAM)
    gain = float(cam['gain']); off = cam['offset_adu'].astype(np.float32)
    with tifffile.TiffFile(tif) as t:
        n = len(t.pages)
        end = n if maxframes is None else min(n, start + maxframes)
        nsum = (end - start) // sumframes
        def gen():
            for k in range(nsum):
                fr = [t.pages[start + k * sumframes + i].asarray() for i in range(sumframes)]
                yield np.sum(fr, 0) if sumframes > 1 else fr[0]
        # sums of ADU: subtract sumframes x offset
        em, shp = inst.process_iterable(gen(), gain=gain, offset=off * sumframes, sumframes=1, batch_size=batch_size, infer_batch_size=16)
    em = em[em.frame_ix < nsum] if len(em) else em
    D = dict(frame=(start + sumframes * em.frame_ix.numpy()).astype(np.int64), x=em.xyz[:, 0].numpy(), y=em.xyz[:, 1].numpy(),
             sx=em.xyz_sig[:, 0].numpy(), sy=em.xyz_sig[:, 1].numpy(), phot=em.phot.numpy(),
             sphot=em.phot_sig.numpy() if em.phot_sig is not None else np.zeros(len(em)),
             bg=em.bg.numpy() if em.bg is not None else np.zeros(len(em)), prob=em.prob.numpy())
    return D
