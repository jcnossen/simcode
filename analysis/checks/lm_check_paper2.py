"""Check of the torch LM-MLE of analysis/baselines/paper_pipeline.py on the SAME ROIs (the smlmtorch ROI cache of a paper_pipeline
run): (1) Gaussian sigma 1.3 px, torch LM vs fastpsf's native LM (the paper's fitter): parameter differences, log-likelihood, CRLB,
chi^2; (2) convergence: 50 vs 200 iterations; (3) Gaussian vs spline PSF on these ROIs (photons, bg, CRLB, chi^2 per px).
usage: CUDA_VISIBLE_DEVICES=1 python analysis/checks/lm_check_paper2.py <rois.npy of a run> <psf spec> [max_rois]"""
import os, sys, json
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'baselines'))
import paper_pipeline as PP, zpsf
import smlmtorch.smlm.detection as detection
from fastpsf import Context, GaussianPSFMethods

rois_fn, spec = sys.argv[1], sys.argv[2]
nmax = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
info, px = detection.load_rois(rois_fn, maxrois=nmax)
P = px.shape[-1]; summed = px.sum(1).astype(np.float32)
dev = 'cuda'
lo, hi = [torch.tensor(v, device=dev) for v in PP.limits(P)]
d = torch.as_tensor(summed, device=dev)
R = {}

# fastpsf Gaussian (as SMLMFitter2D: CreatePSF_XYIBg(roisize, [1.3, 1.3]), default LM settings)
ctx = Context()
g = GaussianPSFMethods(ctx).CreatePSF_XYIBg(P, [1.3, 1.3], cuda=True)
ef = g.Estimate(summed)[0]; cf = g.CRLB(ef); xf = g.ChiSquare(ef, summed)
gp = zpsf.GaussPSF(1.3)
def run(psf, iters, th0=None):
    with torch.no_grad():
        return PP.lm_mle(PP.smlm_model(psf, P), d.flatten(1), PP.init_smlm(d, P) if th0 is None else th0, lo, hi, iters)
th50, i50 = run(gp, 50); th200, i200 = run(gp, 200)
ok = ((ef > PP.limits(P)[0]).all(1) & (ef < PP.limits(P)[1]).all(1))
t = th50.cpu().numpy(); t2 = th200.cpu().numpy()
okt = ok & ~i50['atlim'].cpu().numpy() & ~i200['atlim'].cpu().numpy()
# log-likelihood of the fastpsf estimate under the same model
with torch.no_grad():
    mu, _ = PP.smlm_model(gp, P)(torch.as_tensor(ef, device=dev)); mu = mu.clamp_min(1e-6)
    Lf = (d.flatten(1) * torch.log(mu) - mu).sum(1).cpu().numpy()
    chis_f_ours = ((d.flatten(1) - mu) ** 2 / mu).sum(1).cpu().numpy()
L50 = i50['loglik'].cpu().numpy(); L200 = i200['loglik'].cpu().numpy()
q = lambda v: np.percentile(v, [50, 90, 99]).round(5).tolist()
R['n_rois'] = int(len(summed)); R['n_inside_limits_both'] = int(okt.sum())
R['gauss_torch50_vs_fastpsf'] = dict(
    dxy_px_p50_90_99=q(np.abs(t[okt, :2] - ef[okt, :2]).max(1)), dI_rel_p50_90_99=q(np.abs(t[okt, 2] / ef[okt, 2] - 1)),
    dloglik_torch_minus_fastpsf_p1_50_99=np.percentile((L50 - Lf)[okt], [1, 50, 99]).round(4).tolist(),
    frac_torch_loglik_ge_fastpsf_minus_1e3=float(np.mean((L50 - Lf)[okt] > -1e-3)),
    crlb_x_ratio_median=float(np.median(i50['crlb'][:, 0].cpu().numpy()[okt] / cf[okt, 0])),
    chisq_fastpsf_vs_ours_at_fastpsf_est_median_ratio=float(np.median(xf[okt] / chis_f_ours[okt])))
R['gauss_torch_50_vs_200_iter'] = dict(dxy_px_p50_90_99=q(np.abs(t[okt, :2] - t2[okt, :2]).max(1)),
                                       dloglik_200_minus_50_p50_99=np.percentile((L200 - L50)[okt], [50, 99]).round(5).tolist(),
                                       full_step_xy_50_p50_90_99=q(i50['full_step'][:, :2].abs().max(1).values.cpu().numpy()[okt]))
# chi^2-filter pass fraction (paper: chi2 <= 4 * ROI^2) and median chi^2 per px, Gaussian vs the spline PSF on these ROIs
sp = zpsf.load(spec)
ths, isp = run(sp, 50)
for nm, inf, th in (('gauss', i50, th50), ('spline', isp, ths)):
    th = th.cpu().numpy(); okk = ((th > PP.limits(P)[0]).all(1) & (th < PP.limits(P)[1]).all(1))
    c = inf['chisq'].cpu().numpy(); cr = inf['crlb'].cpu().numpy()
    pas = okk & (c <= 4 * P * P)
    bright = pas & (th[:, 2] > 1000)
    R[f'{nm}_on_same_rois'] = dict(n_limits=int(okk.sum()), n_chisq_pass=int(pas.sum()), chisq_per_px_median=float(np.median(c[okk]) / P / P),
                                   chisq_per_px_median_I_gt_1000=float(np.median(c[bright]) / P / P),
                                   crlb_nm_median=(np.median(cr[pas, :2], 0) * 108.33).round(2).tolist(),
                                   photons_median=float(np.median(th[pas, 2])), bg_median=float(np.median(th[pas, 3])),
                                   n_crlb_lt_015=int((pas & (cr[:, :2] < 0.15).all(1)).sum()),
                                   full_step_xy_lt_1e3=float(np.mean(inf['full_step'][:, :2].abs().max(1).values.cpu().numpy()[okk] < 1e-3)))
# paired: spline vs Gaussian on rows that pass both
tg = th50.cpu().numpy(); ts = ths.cpu().numpy()
both = okt & ((ts > PP.limits(P)[0]).all(1) & (ts < PP.limits(P)[1]).all(1)) & (ts[:, 2] > 1000)
R['spline_vs_gauss_bright_I_gt_1000'] = dict(n=int(both.sum()), photons_ratio_median=float(np.median(ts[both, 2] / tg[both, 2])),
                                             dx_px_median=float(np.median(ts[both, 0] - tg[both, 0])), dy_px_median=float(np.median(ts[both, 1] - tg[both, 1])),
                                             dloglik_spline_minus_gauss_median=float(np.median((isp['loglik'] - i50['loglik']).cpu().numpy()[both])))
print(json.dumps(R, indent=1))

# (4) SIMFLUX: torch SIMFLUX LM (Gaussian) vs fastpsf's SIMFLUX_Gauss2D estimator (the paper's), identical inputs: the first
# max_rois ROIs of the cache that pass the run's chi^2 + modulation-error selection, with the run's mp_sf (env SF_RUN=<out dir
# of a --fitter fastpsf run whose ROI cache is rois_fn>)
if os.environ.get('SF_RUN'):
    from smlmtorch.simflux.dataset import SFDataset
    from smlmtorch.simflux.pattern_estimator import ModulationPattern
    from fastpsf.simflux import SIMFLUX
    run = os.environ['SF_RUN']
    ds = SFDataset.load(f'{run}/smlm/smlm.npy'); mp = ModulationPattern.load(f'{run}/sf/mp_sf.pickle')
    ds = ds[(mp.mod_error(ds) < 0.1) & (ds.roi_id < len(px))]
    rid = ds.roi_id
    E = np.zeros((len(px), 4), np.float32)
    est = g.Estimate(summed)[0]                       # the summed fits (fit_simflux initial values)
    init = est[rid].copy(); init[:, -1] /= 6
    mod_pf = mp.mod_at_frame(np.arange(ds.numFrames), 6)
    mb = mod_pf[ds.frame]
    mb['phase'] -= info['y'][rid][:, None] * mb['k'][:, :, 1]; mb['phase'] -= info['x'][rid][:, None] * mb['k'][:, :, 0]
    rmf = np.reshape(mb.view(np.float32), (len(mb), 6 * 6))
    with SIMFLUX(ctx).CreateEstimator_Gauss2D([1.3, 1.3], 6, P, 6) as sfm:
        sfm.SetLevMarParams(500, iterations=50)
        ff = sfm.Estimate(px[rid], initial=init, constants=rmf)[0]
        cff = sfm.CRLB(ff, constants=rmf)
    mod = np.stack([mb['k'][..., 0], mb['k'][..., 1], mb['phase'], mb['depth'], mb['relint']], -1).astype(np.float32)
    dd = torch.as_tensor(px[rid], device=dev).float().flatten(1)
    th0 = torch.maximum(torch.minimum(torch.as_tensor(init, device=dev), hi), lo)
    with torch.no_grad():
        model = PP.sf_model(gp, P, torch.as_tensor(mod, device=dev))
        ts, isf = PP.lm_mle(model, dd, th0, lo, hi, 50)
        muf, _ = model(torch.as_tensor(ff, device=dev)); muf = muf.clamp_min(1e-6)
        Lff = (dd * torch.log(muf) - muf).sum(1).cpu().numpy()
    ts = ts.cpu().numpy(); Lt = isf['loglik'].cpu().numpy()
    okk = ((ff > PP.limits(P)[0]).all(1) & (ff < PP.limits(P)[1]).all(1)) & ~isf['atlim'].cpu().numpy()
    near = okk & (np.hypot(*(ff[:, :2] - init[:, :2]).T) < 0.5)
    R2 = dict(n=int(len(rid)), n_inside=int(okk.sum()), n_near_smlm=int(near.sum()),
              dxy_px_p50_90_99_near=q(np.abs(ts[near, :2] - ff[near, :2]).max(1)),
              dloglik_torch_minus_fastpsf_p1_50_99_near=np.percentile((Lt - Lff)[near], [1, 50, 99]).round(4).tolist(),
              crlb_x_ratio_median=float(np.median(isf['crlb'][:, 0].cpu().numpy()[near] / cff[near, 0])),
              dist_from_smlm_lt_05_torch=int((okk & (np.hypot(*(ts[:, :2] - init[:, :2]).T) < 0.5)).sum()))
    print('SIMFLUX torch vs fastpsf:', json.dumps(R2))
