"""SMLM and SIMFLUX baselines of the paper rerun: the SIMCODE paper's classical pipeline (~/smlmtorch/scripts/run_pipeline.py
process(), SMLM + SIMFLUX steps), with one PSF input (analysis/baselines/zpsf.py) instead of the fixed Gaussian sigma 1.3 px.

Paper settings, kept (config.pickle of the published 1809/2 and 2.5 nM runs): gain 0.56 e/ADU, pattern_frames [[0,1,2],[3,4,5]],
108.33 nm px, ROI 8 px, detection threshold 3.5 on the 6-frame sum, moving window (every window start), spots_per_frame cap 100
(smlmtorch SpotDetector default), fit_smlm(max_crlb_xy=None, max_chisq=4) = chi^2 <= 4 x ROI^2, fastpsf's fit-parameter limits
(x, y in (1.5, 5.5) for the 8-px ROI; I > 10; bg > 1), SMLM pattern estimate estimate_mp_smlm() (copied
verbatim from run_pipeline.py; angles from the DFT, 20000 spots per phase bin, accept_percentile 50, 8 iterations, depth fixed
0.9), mp_sf = its constant-phase-offset version, modulation-error filter < 0.1, SIMFLUX fit (50 iterations) of the SMLM ROIs,
|SIMFLUX - SMLM| < 0.5 px and the same parameter limits.

Changes (documented):
  * PSF: --psf <spec> (zpsf.py). For a spline PSF the 2D fits are at the fixed emitter plane psf.z_fit (the analogue of the
    fixed-sigma Gaussian; default the calib's emitter_z_median_um). Detection is unchanged (the paper's sigma 1.3 px Gaussian
    template, threshold 3.5), so every PSF fits the same candidate ROIs (--det_template psf: the PSF's own image; with the
    in-situ PSF's lower peak the same 3.5 threshold keeps only ~7 % of the candidates on 1809/2).
  * LM-MLE: fastpsf has no spline-PSF SIMFLUX estimator and no fixed-z spline SMLM model, so both fits use one torch
    Levenberg-Marquardt Poisson MLE (Fisher-scoring normal equations, adaptive damping, parameters clamped to the limits,
    --iters_smlm / --iters_sf, default 50 = the paper's / fastpsf's) on the same likelihood and model as fastpsf
    (SMLM mu = I g + bg on the 6-frame sum; SIMFLUX mu_k = I e_k g + bg_k, e_k = relint (1 + depth sin(k.r - phase)), bg per
    frame). The per-frame intensity / bg fits (ibg, input of the pattern estimate and the modulation error) use fastpsf's
    native EstimateIBg with our PSF images. --fitter fastpsf (gauss: only) runs the original fastpsf code path, as a check.
  * Camera: --camera maps (default) = the movie's in-situ camera calibration (per-pixel offset map, models/insitu/
    camera_maps_<movie>.npz, median 99.3 ADU) + gain 0.56; --camera paper = the paper's scalar offset 106.34 ADU (over-subtracts
    ~7 ADU = 3.9 e/px/frame, see the calib notes); --camera calib = the camera of the --psf calib json (its 'camera' block:
    gain_e_per_adu + the offset map of camera.maps next to the json), i.e. camera v2 (gain 0.4256, v2 offset map) for the
    PSF-v2 calibrations. Only gain + offset enter (Poisson MLE as fastpsf; no read-noise term), as for the other two options.
  * Fit plane (2026-11-15): --psf_z <um> or --psf_z_run <SIMCODE v2 run dir> = the median pass-1 emitter height of that run
    (rows p >= 0.63 inside the crop [5, 115) px, z -> focus_um + z * pixelsize_um of the calib, as analysis/paper/psfdata.py);
    --psf_zmax: top of the rendered knot stack (um, default 0.2; must lie above the fit plane).
  * SIMFLUX initial values (--sf_init fixed, default): smlmtorch's fit_simflux indexes the per-ROI summed fits with the
    filtered dataset's index instead of the ROI id, so the published SIMFLUX started each ROI from ANOTHER ROI's SMLM fit and
    applied |SIMFLUX - SMLM| < 0.5 px against that wrong reference (most fits rejected). Here each ROI starts from its own
    SMLM fit; --sf_init paper reproduces the original indexing.
  * Output locations only (nothing is written next to the raw movie): <out>/movie/ symlinks, <out>/work/<run>/ (smlmtorch ROI
    cache, its results dir), <out>/smlm/, <out>/sf/.

usage:
  CUDA_VISIBLE_DEVICES=1 python analysis/baselines/paper_pipeline.py <dataset 1809_2|2.5nM_2|20nM_2> --psf <spec>
        [--out out/paper2/<dataset>] [--maxframes N] [--camera maps|paper|calib] [--fitter torch|fastpsf] [--steps smlm,sf]
        [--psf_z um | --psf_z_run <run dir>] [--psf_zmax um]
  PSF v2 + camera v2 (Figs 3/4 baselines, 2026-11-15): --psf <models/.../calib_<movie>_psf_v2.json> --camera calib
        --psf_z_run <the movie's SIMCODE v2 run> --psf_zmax 0.4 --out out/paper2/<dataset>/baselines_psfv2
outputs: <out>/smlm/smlm.npy (+ .hdf5; SFDataset, as the paper's smlm.npy), mp_smlm.pickle, info.json;
         <out>/sf/sf.hdf5, mp_sf.pickle, info.json; <out>/run.json (settings, PSF, timing)."""
import os, sys, json, time, argparse, pickle
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
import numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import smlmtorch.util.progbar
smlmtorch.util.progbar.USE_AUTO_TQDM = False
from smlmtorch import config_dict, progbar
from smlmtorch.simflux.localizer import SFLocalizer
from smlmtorch.simflux.dataset import SFDataset
from smlmtorch.simflux.localizer_report import LocalizationReporter
import smlmtorch.simflux.pattern_estimator as pe
import smlmtorch.util.caching as cache
from fastpsf import Context, GaussianPSFMethods

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import zpsf

DATA = '/mnt/data/dmdsf/bn_dnapaint'
DATASETS = {
    '1809_2': dict(tif=f'{DATA}/1809/cover1/2/simflux_5.5ms_500mW.tif', cam='camera_maps_1809_2.npz', nframes=120000),
    '2.5nM_2': dict(tif=f'{DATA}/dnapaint/2.5nM/2/cover1_600mW_5.5ms_cf660.tif', cam='camera_maps_2p5nM_m2.npz', nframes=180000),
    '20nM_2': dict(tif=f'{DATA}/dnapaint/20nM/2/cover1_500mW_5.5ms_cf660.tif', cam='camera_maps_20nM_m2.npz', nframes=120000),
}
PAPER = dict(gain=0.56, offset=106.34, pattern_frames=np.array([[0, 1, 2], [3, 4, 5]]), pixelsize=108.33, roisize=8,
             detection_threshold=3.5, zrange=[-0.3, 0.3], smlm_max_chisq=4, mod_error_threshold=0.1, fix_depth=0.9,
             smlm_spots_per_bin=20000, sf_iterations=50, distFromSMLM=0.5)


def link_movie(tif, movie_dir):
    """symlink every part of a multipart TIFF (name.tif, name_X1.tif, ...) into movie_dir"""
    from smlmtorch.util.multipart_tiff import tiff_get_filenames
    os.makedirs(movie_dir, exist_ok=True)
    for f in tiff_get_filenames(tif):
        d = os.path.join(movie_dir, os.path.basename(f))
        if not os.path.lexists(d): os.symlink(f, d)
    return os.path.join(movie_dir, os.path.basename(tif))


# ------------------------------------------------------------------------------------------------ torch LM-MLE
def lm_mle(model, d, theta, lo, hi, iters=50, lam0=1e-2):
    """Poisson MLE by Levenberg-Marquardt. model(theta [N,K]) -> mu [N,M], J [N,K,M]; d [N,M]; lo / hi [K] limits (clamped).
    Returns theta, info dict (per-row loglik, final undamped Fisher step, accepted-iteration count)."""
    N, K = theta.shape
    mu, J = model(theta); mu = mu.clamp_min(1e-6)
    L = (d * torch.log(mu) - mu).sum(1)
    lam = torch.full((N,), lam0, device=d.device)
    eye = torch.eye(K, device=d.device)
    nacc = torch.zeros(N, device=d.device)
    for it in range(iters):
        g = (J * (d / mu - 1)[:, None]).sum(2)
        H = torch.einsum('nkm,nlm->nkl', J / mu[:, None], J)
        Hd = H + lam[:, None, None] * (torch.diag_embed(H.diagonal(dim1=1, dim2=2)) + 1e-9 * eye)
        step = torch.linalg.solve(Hd, g[..., None])[..., 0]
        step = torch.nan_to_num(step)
        tn = torch.maximum(torch.minimum(theta + step, hi), lo)
        mun, Jn = model(tn); mun = mun.clamp_min(1e-6)
        Ln = (d * torch.log(mun) - mun).sum(1)
        acc = Ln >= L
        theta = torch.where(acc[:, None], tn, theta)
        mu = torch.where(acc[:, None], mun, mu); J = torch.where(acc[:, None, None], Jn, J); L = torch.where(acc, Ln, L)
        lam = torch.where(acc, lam * 0.1, lam * 10).clamp(1e-6, 1e6)
        nacc += acc.float()
    # convergence diagnostic: the undamped Fisher-scoring step at the result (clamped rows at a limit are flagged separately)
    g = (J * (d / mu - 1)[:, None]).sum(2)
    H = torch.einsum('nkm,nlm->nkl', J / mu[:, None], J)
    full = torch.nan_to_num(torch.linalg.solve(H + 1e-9 * eye, g[..., None])[..., 0])
    atlim = ((theta <= lo + 1e-6) | (theta >= hi - 1e-6)).any(1)
    crlb = torch.sqrt(torch.linalg.inv(H + 1e-12 * eye).diagonal(dim1=1, dim2=2).clamp_min(0))
    chisq = ((d - mu) ** 2 / mu).sum(1)
    return theta, dict(loglik=L, full_step=full, atlim=atlim, crlb=crlb, chisq=chisq, nacc=nacc, mu=mu)


def smlm_model(psf, P):
    """theta = (x, y, I, bg) in fastpsf ROI coordinates (pixel j has its centre at j; ROI centre (P-1)/2); mu = I g + bg on the summed ROI"""
    def f(th):
        g, gx, gy = psf.eval(th[:, 0] - (P - 1) / 2, th[:, 1] - (P - 1) / 2, P)
        I = th[:, 2, None, None]
        mu = I * g + th[:, 3, None, None]
        J = torch.stack([I * gx, I * gy, g, torch.ones_like(g)], 1)
        return mu.flatten(1), J.flatten(2)
    return f


def sf_model(psf, P, mod):
    """SIMFLUX (fastpsf SIMFLUX_Gauss2D model, any PSF): mod [N,K,5] = (kx, ky, phase, depth, relint) with the phase already in
    ROI coordinates; mu_k = I relint_k (1 + depth_k sin(kx x + ky y - phase_k)) g + bg (bg per frame)."""
    kx, ky, ph, dep, rel = [mod[..., i] for i in range(5)]
    def f(th):
        g, gx, gy = psf.eval(th[:, 0] - (P - 1) / 2, th[:, 1] - (P - 1) / 2, P)
        a = kx * th[:, 0, None] + ky * th[:, 1, None] - ph
        e = rel * (1 + dep * torch.sin(a)); de = rel * dep * torch.cos(a)            # [N,K]
        I = th[:, 2, None, None, None]
        G = g[:, None]; e4 = e[..., None, None]
        mu = I * e4 * G + th[:, 3, None, None, None]
        Jx = I * ((de * kx)[..., None, None] * G + e4 * gx[:, None])
        Jy = I * ((de * ky)[..., None, None] * G + e4 * gy[:, None])
        J = torch.stack([Jx, Jy, e4 * G, torch.ones_like(mu)], 1)
        return mu.flatten(1), J.flatten(2)
    return f


def limits(P):   # fastpsf Gauss2D / SIMFLUX ParamLimits (x, y in (1.5, P - 2.5); I > 10; bg > 1)
    return np.array([1.5, 1.5, 10.0, 1.0], np.float32), np.array([P - 2.5, P - 2.5, 1e9, 1e9], np.float32)


def init_smlm(d, P):
    """initial (x, y, I, bg): bg = median of the ROI border, I = sum above bg, x / y = centroid of the positive excess"""
    b = torch.cat([d[:, 0], d[:, -1], d[:, 1:-1, 0], d[:, 1:-1, -1]], 1)
    bg = b.median(1).values.clamp_min(1.0)
    ex = (d - bg[:, None, None]).clamp_min(0)
    s = ex.sum((1, 2)).clamp_min(1e-3)
    o = torch.arange(P, device=d.device, dtype=torch.float32)
    x = (ex.sum(1) * o).sum(1) / s; y = (ex.sum(2) * o).sum(1) / s
    c = (P - 1) / 2
    return torch.stack([x.clamp(c - 1.5, c + 1.5), y.clamp(c - 1.5, c + 1.5), s.clamp_min(20.0), bg], 1)


class TorchSMLMFitter:
    """drop-in for smlmtorch SMLMFitter2D in SFLocalizer (process / estimates / cache_cfg / detection_template / save_stats)"""
    param_names = ['x', 'y', 'I', 'bg']

    def __init__(self, psf, roisize, ctx, iters=50, batch=40000, dev='cuda'):
        self.psf, self.P, self.iters, self.batch, self.dev = psf, roisize, iters, batch, torch.device(dev)
        P = roisize
        with torch.no_grad():
            t = psf.eval(torch.zeros(1, device=self.dev), torch.zeros(1, device=self.dev), P)[0]
        self.detection_template = t.cpu().numpy().astype(np.float32)          # [1,P,P], emitter at the ROI centre
        self.cache_cfg = [psf.desc, iters, 'torchlm1']
        self._ibg_est = GaussianPSFMethods(ctx).CreatePSF_XYIBg(P, 1.3, cuda=True)   # only for its native EstimateIBg
        self.stats = {}

    def ibg(self, th, pixels):
        """per-frame (I, bg) at the fitted position: fastpsf native EstimateIBg (as SMLMFitter.process) with our PSF images"""
        n, K, P = pixels.shape[0], pixels.shape[1], self.P
        with torch.no_grad():
            g = self.psf.eval(th[:, 0] - (P - 1) / 2, th[:, 1] - (P - 1) / 2, P)[0]
        psfs = np.ascontiguousarray(np.repeat(g.cpu().numpy(), K, axis=0), np.float32)
        imgs = np.ascontiguousarray(pixels.reshape(n * K, P, P), np.float32)
        ibg = np.zeros((n * K, 2), np.float32); ibg_crlb = np.zeros((n * K, 2), np.float32)
        self._ibg_est._EstimateIBg(imgs, psfs, ibg, ibg_crlb, n * K, P * P, 400, True)
        return ibg.reshape(n, K, 2), ibg_crlb.reshape(n, K, 2)

    def fit_sums(self, summed):
        P = self.P; lo, hi = [torch.tensor(v, device=self.dev) for v in limits(P)]
        d = torch.as_tensor(summed, device=self.dev).float()
        with torch.no_grad():
            th, info = lm_mle(smlm_model(self.psf, P), d.flatten(1), init_smlm(d, P), lo, hi, self.iters)
        return th, info

    def process(self, roi_iterator, imgshape):
        est, crlb, chisq, roipos, framenum, ibg, ibgc = [], [], [], [], [], [], []
        diag = dict(n=0, step_xy_lt_1e3=0, step_xy_lt_1e2=0, atlim=0, nacc=[])
        numrois, rois = roi_iterator()
        with progbar(total=numrois) as pb:
            pb.set_description('torch LM-MLE (PSF: ' + self.psf.desc[:40] + ')')
            for rois_info, pixels, readnoise in rois:
                for s in range(0, len(pixels), self.batch):
                    px = pixels[s:s + self.batch]; ri = rois_info[s:s + self.batch]
                    th, info = self.fit_sums(px.sum(1))
                    est.append(th.cpu().numpy()); crlb.append(info['crlb'].cpu().numpy()); chisq.append(info['chisq'].cpu().numpy())
                    roipos.append(np.stack((ri['y'], ri['x']), 1)); framenum.append(ri['id'])
                    a, b = self.ibg(th, px); ibg.append(a); ibgc.append(b)
                    fs = info['full_step'][:, :2].abs().max(1).values; ok = ~info['atlim']
                    diag['n'] += len(px); diag['atlim'] += int(info['atlim'].sum())
                    diag['step_xy_lt_1e3'] += int(((fs < 1e-3) & ok).sum()); diag['step_xy_lt_1e2'] += int(((fs < 1e-2) & ok).sum())
                    diag['nacc'].append(info['nacc'].cpu().numpy())
                    pb.update(len(px))
        self.estimates = np.concatenate(est); self.crlb = np.concatenate(crlb); self.chisq = np.concatenate(chisq)
        self.roipos = np.concatenate(roipos); self.framenum = np.concatenate(framenum)
        nacc = np.concatenate(diag.pop('nacc'))
        diag['nacc_median'] = float(np.median(nacc)); diag['n_not_atlim'] = diag['n'] - diag['atlim']
        u, c = np.unique(self.framenum, return_counts=True)
        diag['windows'] = int(len(u)); diag['windows_at_cap100'] = int((c >= 100).sum()); diag['rois_per_window_max'] = int(c.max())
        self.stats = diag
        print('LM diagnostics:', diag, flush=True)
        ds = SFDataset.fromEstimates(self.estimates, self.param_names, roipos=self.roipos, crlb=self.crlb, framenum=self.framenum,
                                     imgshape=imgshape, numPatterns=ibg[0].shape[1], chisq=self.chisq)
        ds.ibg[:] = np.concatenate(ibg); ds.ibg_crlb[:] = np.concatenate(ibgc)
        ds.filterNaN()
        fits = self.estimates[ds.roi_id]; lo, hi = limits(self.P)
        ds = ds[(fits > lo[None]).all(1) & (fits < hi[None]).all(1)]
        return ds

    def save_stats(self, result_dir):
        json.dump(self.stats, open(os.path.join(result_dir, 'lm_stats.json'), 'w'), indent=1)


class PSFLocalizer(SFLocalizer):
    """SFLocalizer with the torch fitter (any zpsf PSF) instead of SMLMFitter2D / 3D"""
    def __init__(self, tif_path, psf, iters=50, det_template='paper', **kw):
        self._psf, self._iters, self._det_template = psf, iters, det_template
        super().__init__(tif_path, **kw)

    def create_psf_model(self):
        torch.cuda.set_device(self.device)
        self.fastpsf_ctx = Context(debugMode=self.fastpsf_debug_mode)
        self.smlm_fitter = TorchSMLMFitter(self._psf, self.roisize, self.fastpsf_ctx, self._iters)
        if self._det_template == 'paper':   # the paper's detection template: the sigma 1.3 px Gaussian (fastpsf ExpectedValue)
            g = GaussianPSFMethods(self.fastpsf_ctx).CreatePSF_XYIBg(self.roisize, [1.3, 1.3], cuda=True)
            self.smlm_fitter.detection_template = g.ExpectedValue([[self.roisize / 2, self.roisize / 2, 1, 0]])
        self.detection_template = self.smlm_fitter.detection_template


def fit_simflux_torch(sfloc, psf, mp, ds, iterations=50, distFromSMLM=0.5, batch=20000, init='fixed'):
    """= SFLocalizer.fit_simflux (same ROIs, modulation constants, output dataset and filters) with the torch SIMFLUX LM on psf.
    init: 'fixed' = each ROI starts from ITS OWN summed (SMLM) fit, summed_fits[ds.roi_id[ix]]; 'paper' = as smlmtorch's
    fit_simflux, summed_fits[ix] with ix = the ds index (summed_fits is indexed by ROI id, so after the limits / chi^2 /
    modulation-error filters this takes another ROI's fit as the initial value AND as the reference of the |SIMFLUX - SMLM|
    < 0.5 px filter). Returns (sf_ds, diag)."""
    P = sfloc.roisize; npat = sfloc.pattern_frames.size; dev = sfloc.device
    assert mp.pattern_frames.size == ds.ibg.shape[1]
    mod_per_frame = mp.mod_at_frame(np.arange(ds.numFrames), npat)
    id_map = np.ones(sfloc.numrois, dtype=np.int32) * -1
    id_map[ds.roi_id] = np.arange(len(ds))
    lo, hi = [torch.tensor(v, device=dev) for v in limits(P)]
    fits, crlb, org, roi_ids = [], [], [], []
    diag = dict(n=0, atlim=0, step_xy_lt_1e3=0, step_xy_lt_1e2=0)
    numrois, it = sfloc.roi_iterator()
    start_ix = 0
    with progbar(total=len(ds)) as pb:
        for rois_info, pixels, _ in it:
            nrois = len(pixels)
            ix = id_map[start_ix:start_ix + nrois]
            rois_info = rois_info[ix >= 0]; pixels = pixels[ix >= 0]; ix = ix[ix >= 0]
            start_ix += nrois
            for s in range(0, len(ix), batch):
                ixb = ix[s:s + batch]; ri = rois_info[s:s + batch]; pxb = pixels[s:s + batch]
                initial = sfloc.summed_fits[ds.roi_id[ixb] if init == 'fixed' else ixb, :4].copy(); initial[:, -1] /= len(mp.mod)
                mb = mod_per_frame[ds.frame[ixb]]
                mb['phase'] -= ri['y'][:, None] * mb['k'][:, :, 1]
                mb['phase'] -= ri['x'][:, None] * mb['k'][:, :, 0]
                mod = np.stack([mb['k'][..., 0], mb['k'][..., 1], mb['phase'], mb['depth'], mb['relint']], -1).astype(np.float32)
                d = torch.as_tensor(pxb, device=dev).float().flatten(1)
                th0 = torch.maximum(torch.minimum(torch.as_tensor(initial, device=dev).float(), hi), lo)
                with torch.no_grad():
                    th, info = lm_mle(sf_model(psf, P, torch.as_tensor(mod, device=dev)), d, th0, lo, hi, iterations)
                fits.append(th.cpu().numpy()); crlb.append(info['crlb'].cpu().numpy()); org.append(initial); roi_ids.append(ixb)
                fs = info['full_step'][:, :2].abs().max(1).values; ok = ~info['atlim']
                diag['n'] += len(ixb); diag['atlim'] += int(info['atlim'].sum())
                diag['step_xy_lt_1e3'] += int(((fs < 1e-3) & ok).sum()); diag['step_xy_lt_1e2'] += int(((fs < 1e-2) & ok).sum())
                pb.update(len(ixb))
    fits = np.concatenate(fits); crlb = np.concatenate(crlb); org = np.concatenate(org)
    sf_ds = SFDataset.fromEstimates(fits, ['x', 'y', 'I', 'bg'], roipos=ds.data.roipos, crlb=crlb, framenum=ds.frame,
                                    imgshape=sfloc.imgshape, numPatterns=npat, pixelsize=sfloc.pixelsize)
    sf_ds.ibg[:] = ds.ibg
    sf_ds.roi_id[:] = np.concatenate(roi_ids)
    lp = np.concatenate([sf_ds.local_pos.T, [sf_ds.photons], [sf_ds.background]], 0).T
    l0, l1 = limits(P)
    dist2D = np.sqrt(np.sum((org[:, :2] - fits[:, :2]) ** 2, 1))
    sel = (lp > l0[None]).all(1) & (lp < l1[None]).all(1) & (dist2D < distFromSMLM)
    diag.update(n_out=int(sel.sum()), n_dist_rejected=int((dist2D >= distFromSMLM).sum()))
    print('SIMFLUX LM diagnostics:', diag, flush=True)
    return sf_ds[sel], diag


# ------------------------------------------------------------------------------------------------ verbatim from run_pipeline.py
def estimate_mp_smlm(smlm_ds, cfg, smlm_spots_per_bin=20000, mod_error_threshold=0.1, fix_depth=None):

    # filtered dataset for pattern estimation
    if smlm_ds.dims==3:
        smlm_pe_ds = smlm_ds[ np.abs(smlm_ds.pos[:,2]) < 0.1]
    else:
        smlm_pe_ds = smlm_ds[:]

    print(smlm_pe_ds)

    #mp_ds = ds[ds.chisq<2*cfg.r    oisize**2] # strongly filtered dataset for pattern estimation
    mp_smlm = pe.estimate_angles(pitch_minmax_nm=[150,500], ds=smlm_pe_ds,
                            pattern_frames = cfg['pattern_frames'],
                            result_dir = cfg['result_dir'],
                            device = cfg['device'], moving_window=True)

    if len(smlm_ds) // smlm_spots_per_bin < 3:
        smlm_spots_per_bin = len(smlm_ds) // 5
        print(f'SMLM pattern estimation warning: smlm_spots_per_bin too high, reducing to {smlm_spots_per_bin}')

    mp_smlm = pe.estimate_phases(smlm_pe_ds, mp_smlm, spots_per_bin=smlm_spots_per_bin,
                            accept_percentile=50, iterations=8, verbose=False, device=cfg['device'])
    mp_smlm.print_info(cfg['pixelsize'])

    mp_smlm.save(cfg['result_dir']+"mp_smlm.pickle")

    if fix_depth is not None:
        mp_smlm.depths = fix_depth

    col = ['r', 'g', 'k']
    fig,ax=plt.subplots(2, 1,figsize=(10,8),sharex=True)
    #mp_smlm_co.plot_phase_drift(ax=ax, colors=col, label='CO {0}', linestyle='-', lw=1)
    mp_smlm.plot_phase_drift(ax=ax, colors=col, label='Pattern {0}', linestyle='-', lw=2)
    plt.savefig(cfg['result_dir']+'smlm-phase-drift.png')

    lr = LocalizationReporter(smlm_ds, cfg['result_dir'], mp_smlm)
    lr.draw_pattern(0, 2, me_threshold=mod_error_threshold)
    return mp_smlm
# ------------------------------------------------------------------------------------------------


def run_height(run_dir, calib, crop=(5.0, 5.0, 110.0), p_min=0.63):
    """median pass-1 emitter height (um) of a SIMCODE v2 run: pass1_*.npz rows with p >= p_min inside the crop; z (px relative
    to the calib's focal plane) -> focus_um + z * pixelsize_um (analysis/paper/psfdata.py convention)"""
    import glob
    c = json.load(open(calib)); x0, y0, s = crop; z = []
    files = sorted(glob.glob(os.path.join(run_dir, 'pass1_*.npz')))
    assert files, f'no pass1_*.npz in {run_dir}'
    for f in files:
        with np.load(f) as d:
            m = (d['p'] >= p_min) & (d['x'] >= x0) & (d['x'] < x0 + s) & (d['y'] >= y0) & (d['y'] < y0 + s)
            z.append(c['focus_um'] + d['z'][m].astype(np.float64) * c['pixelsize_um'])
    z = np.concatenate(z); q = np.nanpercentile(z, [25, 50, 75])
    return float(q[1]), dict(run=os.path.abspath(run_dir), n_chunks=len(files), n_rows=int(len(z)), p_min=p_min, crop=list(crop),
                             z_median_um=float(q[1]), z_q25_q75_um=[float(q[0]), float(q[2])], z_nan_frac=float(np.mean(~np.isfinite(z))))


def crlb_summary(ds, px=108.33):
    c = ds.crlb.pos[:, :2]
    return dict(n=int(len(ds)), median_crlb_nm=(np.median(c, 0) * px).round(2).tolist(),
                frac_crlb_lt_015px=float(np.mean((c < 0.15).all(1))), median_photons=float(np.median(ds.photons)),
                median_bg=float(np.median(ds.background)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', choices=list(DATASETS))
    ap.add_argument('--psf', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--maxframes', type=int, default=None)
    ap.add_argument('--camera', default='maps', choices=['maps', 'paper', 'calib'],
                    help="'calib' = the camera block of the --psf calib json (gain + offset map; camera v2 for the PSF-v2 calibs)")
    ap.add_argument('--fitter', default='torch', choices=['torch', 'fastpsf'])
    ap.add_argument('--steps', default='smlm,sf')
    ap.add_argument('--iters_smlm', type=int, default=50)
    ap.add_argument('--iters_sf', type=int, default=PAPER['sf_iterations'])
    ap.add_argument('--det_template', default='paper', choices=['paper', 'psf'],
                    help="detection template: 'paper' = the Gaussian sigma 1.3 px of the paper (identical ROI set for every PSF; "
                         "default), 'psf' = the fit PSF's image (its lower peak cuts the candidates at the same threshold)")
    ap.add_argument('--sf_init', default='fixed', choices=['fixed', 'paper'],
                    help="SIMFLUX initial value / reference: 'fixed' = the ROI's own SMLM fit; 'paper' = smlmtorch's indexing")
    ap.add_argument('--psf_z', type=float, default=None, help='fit plane (um) for a spline PSF; default from the PSF')
    ap.add_argument('--psf_z_run', default=None, help='SIMCODE v2 run dir: fit plane = its median pass-1 emitter height (run_height)')
    ap.add_argument('--psf_zmax', type=float, default=0.2, help='top of the knot stack of a calib json PSF (um)')
    a = ap.parse_args()
    assert not (a.psf_z is not None and a.psf_z_run), '--psf_z and --psf_z_run are exclusive'
    D = DATASETS[a.dataset]
    out = os.path.abspath(a.out or os.path.join(REPO, 'out', 'paper2', a.dataset))
    run = f'{a.fitter}_{a.camera}' + (f'_{a.maxframes}' if a.maxframes else '')
    os.makedirs(out, exist_ok=True)
    tif = link_movie(D['tif'], os.path.join(REPO, 'out', 'paper2', a.dataset, 'movie'))
    T = {}
    t0 = time.time()
    cam_src = None
    if a.camera == 'maps':
        cam = np.load(os.path.join(REPO, 'models', 'insitu', D['cam']))
        gain, offset = float(cam['gain']), cam['offset_adu'].astype(np.float32)
    elif a.camera == 'calib':
        assert a.psf.endswith(('.json', '.zip')), '--camera calib needs a calibration (calib json or psf.zip) --psf'
        from simcode.psf import read_calib, camera_maps, camera_maps_path
        C = read_calib(a.psf); cc = C['camera']; cam_src = camera_maps_path(a.psf, C)
        cam = camera_maps(a.psf, C)
        gain, offset = float(cc['gain_e_per_adu']), cam['offset_adu'].astype(np.float32)
        if 'gain' in cam: assert abs(float(cam['gain']) - gain) < 1e-6, (cam_src, float(cam['gain']), gain)
    else:
        gain, offset = PAPER['gain'], PAPER['offset']
    height = None
    if a.psf_z_run:
        a.psf_z, height = run_height(a.psf_z_run, a.psf)
        print('fit plane from the run:', json.dumps(height), flush=True)
    psf = zpsf.load(a.psf, z_fit=a.psf_z, zmax=a.psf_zmax)
    if getattr(psf, 'z_fit', 0) > a.psf_zmax - 0.02 and not a.psf.startswith('gauss:'):
        raise SystemExit(f'fit plane {psf.z_fit:.3f} um too close to the knot-stack top --psf_zmax {a.psf_zmax}')
    rel = lambda p: os.path.relpath(p, os.path.dirname(tif))
    kw = dict(pattern_frames=PAPER['pattern_frames'], pixelsize=PAPER['pixelsize'], detection_threshold=PAPER['detection_threshold'],
              roisize=PAPER['roisize'], device='cuda:0', zrange=PAPER['zrange'], gain=gain, offset=offset, maxframes=a.maxframes,
              psf_sigma_binsize=None, cache_name=rel(f'{out}/work/{run}/cache'), result_dir=rel(f'{out}/work/{run}/results'))
    if a.fitter == 'fastpsf':
        assert a.psf.startswith('gauss:'), '--fitter fastpsf supports only a Gaussian PSF'
        sfloc = SFLocalizer(tif, psf_calib=[psf.sigma, psf.sigma], **kw)
    else:
        sfloc = PSFLocalizer(tif, psf, iters=a.iters_smlm, det_template=a.det_template,
                             psf_calib=psf.desc if a.det_template == 'psf' else [1.3, 1.3], **kw)
    cfg = config_dict(dict(psf_calib=psf.desc, roisize=PAPER['roisize'], detection_threshold=PAPER['detection_threshold'],
                           pattern_frames=PAPER['pattern_frames'], gain=gain, offset=(a.camera if a.camera == 'maps' else PAPER['offset']),
                           pixelsize=PAPER['pixelsize'], zrange=PAPER['zrange'], result_dir=sfloc.result_dir, device='cuda:0'))
    sfloc.detect_spots(ignore_cache=False, moving_window=True)
    torch.cuda.synchronize(); T['detect_s'] = time.time() - t0
    smlm_ds = sfloc.fit_smlm(max_crlb_xy=None, ignore_cache=False, max_chisq=PAPER['smlm_max_chisq'])
    torch.cuda.synchronize(); T['smlm_fit_s'] = time.time() - t0 - T['detect_s']
    lm_stats = getattr(sfloc.smlm_fitter, 'stats', {})
    if not lm_stats and os.path.exists(os.path.join(sfloc.result_dir, 'lm_stats.json')):
        lm_stats = json.load(open(os.path.join(sfloc.result_dir, 'lm_stats.json')))
    os.makedirs(f'{out}/smlm', exist_ok=True)
    smlm_ds.save(f'{out}/smlm/smlm.npy'); smlm_ds.save(f'{out}/smlm/smlm.hdf5')
    n_fit = len(sfloc.summed_fits) if hasattr(sfloc, 'summed_fits') else None
    info = dict(dataset=a.dataset, psf=psf.desc, psf_z_fit_um=getattr(psf, 'z_fit', None), psf_z_from_run=height,
                camera=a.camera, camera_maps=cam_src, gain=gain, fitter=a.fitter,
                maxframes=a.maxframes, numrois=int(sfloc.numrois), n_after_chisq=int(len(smlm_ds)), lm=lm_stats,
                chisq_per_px_median=float(np.median(smlm_ds.chisq) / PAPER['roisize'] ** 2), **crlb_summary(smlm_ds))
    print('SMLM:', json.dumps(info), flush=True)
    t1 = time.time()
    mp_smlm = estimate_mp_smlm(smlm_ds, cfg, PAPER['smlm_spots_per_bin'], PAPER['mod_error_threshold'], PAPER['fix_depth'])
    mp_smlm.save(f'{out}/smlm/mp_smlm.pickle')
    mp_sf = mp_smlm.const_phase_offsets()
    mp_sf.save(f'{out}/smlm/mp_sf.pickle')
    T['pattern_s'] = time.time() - t1
    json.dump(info, open(f'{out}/smlm/info.json', 'w'), indent=1)
    if 'sf' in a.steps:
        t2 = time.time()
        me = mp_sf.mod_error(smlm_ds); me_sel = me < PAPER['mod_error_threshold']
        if a.fitter == 'fastpsf':
            sf_ds = sfloc.fit_simflux(mp_sf, smlm_ds[me_sel], iterations=PAPER['sf_iterations'], lambda_=500, normalizeWeights=True,
                                      ignore_cache=True)
            sdiag = {}
        else:
            sf_ds, sdiag = fit_simflux_torch(sfloc, psf, mp_sf, smlm_ds[me_sel], PAPER['sf_iterations'], PAPER['distFromSMLM'],
                                             init=a.sf_init)
        torch.cuda.synchronize(); T['sf_fit_s'] = time.time() - t2
        os.makedirs(f'{out}/sf', exist_ok=True)
        sf_ds['pixelsize'] = PAPER['pixelsize']
        sf_ds.save(f'{out}/sf/sf.hdf5'); mp_sf.save(f'{out}/sf/mp_sf.pickle')
        sinfo = dict(n_modsel=int(me_sel.sum()), n_out=int(len(sf_ds)), lm=sdiag, pattern=dict(
            pitch_nm=(2 * np.pi / np.linalg.norm(mp_sf.mod['k'][:, :2], axis=1) * PAPER['pixelsize']).round(1).tolist(),
            depth=mp_sf.mod['depth'].tolist()), **crlb_summary(sf_ds))
        if len(sf_ds):
            l = lambda d: d.crlb_filter(0.5, inplace=False).crlb.pos[:, :2].mean()
            sinfo['improvement_factor_xy'] = float(l(smlm_ds) / l(sf_ds))
        print('SIMFLUX:', json.dumps(sinfo), flush=True)
        json.dump(sinfo, open(f'{out}/sf/info.json', 'w'), indent=1)
    T['total_s'] = time.time() - t0
    json.dump(dict(args=vars(a), timing=T, run=run, gain=gain, offset=cam_src or (a.camera if a.camera == 'maps' else PAPER['offset']),
                   psf_z_fit_um=getattr(psf, 'z_fit', None), psf_z_from_run=height),
              open(f'{out}/run.json', 'w'), indent=1)
    print('timing', T, flush=True)


if __name__ == '__main__':
    main()
