"""S4 = SI S10: precision vs CRLB on isolated emitters, with the averaged 1809/2 PSF for every method.

Protocol (preprint: smlmtorch 5f7fdcb scripts/estimation_vs_crlb_varying_bg.py / estimation_vs_crlb_fastpsf.py; values kept):
  x axis    total signal photons per 6-frame window, np.logspace(2, 3.5, 10) (100-3162), N = 1000 emitters per bin (1008 here)
  ROI       6 x 16 x 16 px per emitter, emitter uniform in [5, 11) px of its ROI (edge 5), 108.33 nm px, pitch 224 nm, depth 0.9,
            2 angles (0 / 90 deg) x 3 steps (phases 0, 2 pi / 3, 4 pi / 3 + a random offset per angle and FOV), rel 1/6 per frame
  bg        A: constant 1 photon / px / frame ('const1');  B, C: per-frame bg ~ U(1, 6) ('unif16')
  trim      TRIM_KEEP 0.98 (the 2 % largest |error| removed per method and bin, as the script's rmsd(); the SI caption says 5 %,
            the 5 % numbers are stored too)
Layout (new, for the networks): the ROIs are the cells of a 4 x 4 grid inside an 80 x 80 px FOV (8 px empty margin, cells 16 px),
so the 21 px head patch of SIMCODE v2 sees real background instead of zero padding (the v2 networks need context beyond a
16 px ROI); all 16 emitters of a FOV share its per-frame bg (bg_t ~ U(1, 6) per FOV and frame). Pure Poisson, photons in the
simulator's PSF normalisation (common.py), emitter z = 0.055 um (the PSF's fit plane).
Methods:
  SMLM      torch LM-MLE (analysis/baselines/paper_pipeline.py lm_mle + smlm_model, the paper's 50 iterations) on the 6-frame sum
            of the 16 x 16 ROI, parameters (x, y, I, bg), start = GT + U(-0.5, 0.5) px (as the old script), the zpsf spline PSF
  SIMFLUX   the same LM with paper_pipeline.sf_model (one bg for the 6 frames, as the paper's SIMFLUX), known GT pattern, same start
  SIMFLUX bg per frame  (not in the preprint) the same with one bg per frame (x, y, I, bg_1..6; start bg_t = ROI-border median):
            the MLE of the U(1, 6) data-generating model
  NN        SIMCODE v2 network output before the pattern solve: the head's pattern-free position prior (pass 1 head; and the
            pass-2 head's prior); predicted precision = the head's sigma
  SIMCODE v2  explicit pattern solve on the network intensities with the GT pattern (pass 1 = pattern-free networks + solve;
            pass 2 = pattern-aware networks fed the GT pattern + solve, the production final output); predicted = calibrated sigma
            Detection is the network's own (detector p >= 0.5, matched to the GT within 1 px); SMLM / SIMFLUX fit every emitter.
CRLB from the ground truth (Poisson, zpsf PSF at z_fit, 16 x 16 ROI): SMLM (x, y, I, bg on the sum), SIMFLUX with one bg
(x, y, I, bg) and SIMFLUX with a bg per frame (x, y, I, bg_1..6; the bound that applies under the U(1, 6) condition).
Per bin the stored numbers are: RMSD per axis (trim 2 %, trim 5 %, none), mean and RMS of the predicted sigma, mean CRLBs
(the old script's .mean(0)), recall (networks), n.
PSF v2 (2026-11-15, --psf v2; the default for a model trained on the in-situ PSF v2, e.g. the current m1809): scenes rendered with
the in-situ PSF v2 of 1809/2 (common.RendererV2 LUT: 41 x 41 px, 10 nm planes 0-0.40 um) at ONE fixed height for every emitter =
the calibration's emitter_z_median_um (0.116 um; the old scenes: 0.055 um = the old PSF's fit plane); the x axis is DETECTED
photons (as the v2-PSF movies: intensity parameter = photons / collected(z), collected(0.116 um) = the rendered 41 x 41 px image
sum); still pure Poisson in photons (no camera / read noise, as the preprint protocol); SMLM / SIMFLUX fit and the CRLBs use
zpsf.load(PSF v2) at the same height (2D fits, z known); the networks' PSF template column = the scene PSF (as simcode.run,
which builds it from the movie's calibration).
usage: CUDA_VISIBLE_DEVICES=0 python analysis/papersim/s10.py [--model m1809] [--bins all|i,j] [--cond const1,unif16]
          [--n 1000] [--out out/papersim/s10] [--psf vectorial|v2|gauss1.3] [--tag NAME]
outputs: <out>/s10_<tag>.json (+ per-emitter npz), s10_<tag>.png; tag default <model> (+ _gauss1.3; + v2 for --psf v2)"""
import os, sys, json, time, math, argparse, importlib.util
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, REPO)
import common as C
_s = importlib.util.spec_from_file_location('paper_pipeline', os.path.join(REPO, 'analysis', 'baselines', 'paper_pipeline.py'))
PP = importlib.util.module_from_spec(_s); _s.loader.exec_module(PP)
dev = 'cuda'
CELL, NC, MARGIN = 16, 4, 8
FOV = NC * CELL + 2 * MARGIN       # 80
EDGE = 5
TRIM_KEEP = 0.98
PHOTONS = np.logspace(2, 3.5, 10)


def scene(nfov, ph, cond, rng):
    """-> dict frames [nfov,6,FOV,FOV] photons, gt x / y (FOV px) [nfov*16], fov, cell corner, bg [nfov,6], patt [nfov,6,5]"""
    n = nfov * NC * NC
    fov = np.repeat(np.arange(nfov), NC * NC)
    ci = np.tile(np.arange(NC * NC), nfov)
    cx0 = MARGIN + (ci % NC) * CELL; cy0 = MARGIN + (ci // NC) * CELL
    x = cx0 + EDGE + rng.random(n) * (CELL - 2 * EDGE); y = cy0 + EDGE + rng.random(n) * (CELL - 2 * EDGE)
    bg = np.full((nfov, 6), 1.0) if cond == 'const1' else rng.uniform(1, 6, (nfov, 6))
    P0 = C.gt_pattern([0, 90])
    off = rng.random((nfov, 2)) * 2 * np.pi
    phi = P0['phi'][None] + np.repeat(off, 3, 1)                                  # [nfov,6]
    s = np.arange(6); a = s // 3
    patt = np.zeros((nfov, 6, 5))
    patt[..., 0] = P0['k'][a, 0][None]; patt[..., 1] = P0['k'][a, 1][None]; patt[..., 2] = phi; patt[..., 3] = C.DEPTH; patt[..., 4] = 1.0
    il = 1 + C.DEPTH * np.cos(patt[fov, :, 0] * x[:, None] + patt[fov, :, 1] * y[:, None] - patt[fov, :, 2])   # [n,6]
    phf = ph / 6 * il
    fr = (fov[:, None] * 6 + s[None]).ravel()
    return dict(fr=fr, x=x, y=y, phf=phf, fov=fov, cx0=cx0, cy0=cy0, bg=bg, patt=patt, n=n, nfov=nfov)


def render(S, R, gen):
    if isinstance(R, C.RendererV2):     # fixed height R.z for every emitter; phf is the intensity parameter
        mu = R.expected(0, S['nfov'] * 6, S['fr'], np.repeat(S['x'], 6), np.repeat(S['y'], 6), np.full(S['n'] * 6, R.z), S['phf'].ravel())
    else:
        mu = R.expected(0, S['nfov'] * 6, S['fr'], np.repeat(S['x'], 6), np.repeat(S['y'], 6), S['phf'].ravel())
    mu = mu.reshape(S['nfov'], 6, FOV, FOV) + torch.as_tensor(S['bg'], device=dev, dtype=torch.float32)[..., None, None]
    return torch.poisson(mu, generator=gen)


def rois(frames, S):
    """[n,6,16,16] cell ROIs"""
    f = frames.cpu().numpy()
    return np.stack([f[b, :, y0:y0 + CELL, x0:x0 + CELL] for b, x0, y0 in zip(S['fov'], S['cx0'], S['cy0'])])


def crlb(mu, J):
    Fm = torch.einsum('nkm,nlm->nkl', J / mu.clamp_min(1e-9)[:, None], J)
    return torch.sqrt(torch.linalg.inv(Fm + 1e-12 * torch.eye(Fm.shape[1], device=dev)).diagonal(dim1=1, dim2=2).clamp_min(0))


def sf_mod(S):
    """paper_pipeline.sf_model modulation [n,6,5] (kx, ky, phase in ROI coordinates, sin convention, depth, relint 1/6)"""
    p = S['patt'][S['fov']]                                                       # [n,6,5]
    ph = p[..., 2] - p[..., 0] * S['cx0'][:, None] - p[..., 1] * S['cy0'][:, None] - np.pi / 2
    return torch.as_tensor(np.stack([p[..., 0], p[..., 1], ph, p[..., 3], np.full(p.shape[:2], 1 / 6)], -1), device=dev, dtype=torch.float32)


def sf_pf_model(psf, P, mod):
    """SIMFLUX with one bg per frame: theta (x, y, I, bg_1..6)"""
    base = PP.sf_model(psf, P, mod)
    def f(th):
        mu, J = base(th[:, :4])
        K = mod.shape[1]; n = th.shape[0]
        mu = mu.reshape(n, K, -1) - th[:, 3, None, None] + th[:, 3:, None]
        Jb = torch.zeros(n, K, K, P * P, device=th.device); Jb[:, torch.arange(K), torch.arange(K)] = 1.0
        return mu.reshape(n, -1), torch.cat([J[:, :3], Jb.reshape(n, K, -1)], 1)
    return f


def classical(S, R6, psf, rng, start='gt'):
    P = CELL; n = S['n']
    lo, hi = [torch.tensor(v, device=dev) for v in PP.limits(P)]
    xl = torch.as_tensor(S['x'] - S['cx0'], device=dev, dtype=torch.float32); yl = torch.as_tensor(S['y'] - S['cy0'], device=dev, dtype=torch.float32)
    ph = torch.as_tensor(S['phf'].sum(1), device=dev, dtype=torch.float32)
    bgv = torch.as_tensor(S['bg'][S['fov']], device=dev, dtype=torch.float32)                       # [n,6]
    d6 = torch.as_tensor(R6, device=dev, dtype=torch.float32)
    jit = torch.as_tensor(rng.random((n, 2)) - 0.5, device=dev, dtype=torch.float32)
    mod = sf_mod(S)
    out = {}
    with torch.no_grad():
        th_gt = torch.stack([xl, yl, ph, bgv.sum(1)], 1)
        mu, J = PP.smlm_model(psf, P)(th_gt); out['crlb_smlm'] = crlb(mu, J)[:, :2]
        if start == 'smlmfit':      # paper_pipeline: SMLM start = centroid / border-median init of the 6-frame sum
            th0 = torch.maximum(torch.minimum(PP.init_smlm(d6.sum(1), P), hi), lo)
        else:
            th0 = torch.stack([xl + jit[:, 0], yl + jit[:, 1], ph, bgv.sum(1)], 1)
        th, info = PP.lm_mle(PP.smlm_model(psf, P), d6.sum(1).flatten(1), th0, lo, hi, 50)
        th_smlm = th.clone()
        out['smlm'] = th[:, :2] - torch.stack([xl, yl], 1); out['smlm_sig'] = info['crlb'][:, :2]
        th_gt1 = torch.stack([xl, yl, ph, bgv.mean(1)], 1)
        mu, J = PP.sf_model(psf, P, mod)(th_gt1); out['crlb_sf'] = crlb(mu, J)[:, :2]
        mu, J = sf_pf_model(psf, P, mod)(torch.cat([th_gt1[:, :3], bgv], 1)); out['crlb_sf_pfbg'] = crlb(mu, J)[:, :2]
        if start == 'smlmfit':      # paper_pipeline (init 'fixed'): SIMFLUX start = the SMLM fit of the 6-frame sum, bg / 6
            th0 = torch.cat([th_smlm[:, :3], th_smlm[:, 3:4] / 6], 1)
            th0 = torch.maximum(torch.minimum(th0, hi), lo)
        else:
            th0 = torch.stack([xl + jit[:, 0], yl + jit[:, 1], ph, bgv.mean(1)], 1)
        th, info = PP.lm_mle(PP.sf_model(psf, P, mod), d6.flatten(1), th0, lo, hi, 50)
        out['sf'] = th[:, :2] - torch.stack([xl, yl], 1); out['sf_sig'] = info['crlb'][:, :2]
        # (not in the preprint) SIMFLUX MLE with one bg per frame, start bg_t = the frame's ROI-border median
        brd = torch.cat([d6[:, :, 0], d6[:, :, -1], d6[:, :, 1:-1, 0], d6[:, :, 1:-1, -1]], -1).median(-1).values.clamp_min(0.1)
        lo9 = torch.cat([lo[:3], torch.full((6,), 0.01, device=dev)]); hi9 = torch.cat([hi[:3], torch.full((6,), 1e9, device=dev)])
        th0 = torch.cat([th0[:, :3], brd], 1)
        th, info = PP.lm_mle(sf_pf_model(psf, P, mod), d6.flatten(1), th0, lo9, hi9, 50)
        out['sf_pfbg'] = th[:, :2] - torch.stack([xl, yl], 1); out['sf_pfbg_sig'] = info['crlb'][:, :2]
    return {k: v.cpu().numpy() for k, v in out.items()}


class Nets:
    def __init__(self, ns, psf_calib=None):
        from simcode.emitter import model as SCM
        from simcode.detector import model as DM
        from simcode.solve import solve as SV
        from simcode.processing import gate_dump as DU
        from simcode.simulate.simulator import Simulator
        from simcode import config
        self.SCM, self.SV, self.DU = SCM, SV, DU
        sim = Simulator(psf_calib or ns.get('psf_calib', C.PSF_CALIB), H=FOV, W=FOV, T=6)
        self.col = SCM.PSFCol(sim)
        det, _ = DM.load(ns['detector'])
        self.p1 = (det, SCM.load(config.resolve(ns['emitter'])), self.col,
                   json.load(open(config.resolve(ns['cal_sim']))))
        d2, _ = DM.load(ns['detector_pass2'])
        self.p2 = (d2, SCM.load(config.resolve(ns['emitter_pass2'])), True)
        self.p2cal = json.load(open(config.resolve(ns['cal_sim_pass2'])))

    @torch.no_grad()
    def run(self, frames, patt):
        from simcode.benchmark import gate_scenes as GA
        from simcode.pattern import illumination as IL
        o = dict(frames=frames, patt_est=patt)
        r1 = GA.out_sc(self.p1, o, 0.2)
        il = IL.illum_maps(patt, FOV, FOV)
        r2 = self.DU.front_out(self.p2, o, il, self.col)
        so = self.SV.solve_emitters(r2, patt, r2['b'], self.p2cal)
        r2.update(x_sc=so['r'][:, 0], y_sc=so['r'][:, 1], scal_sc=so['s_cal'], s_prior=r2['lsxy'].exp())
        return r1, r2


def match(S, b, x, y, score, rmax=1.0):
    """per GT emitter: index of the nearest detection (score >= 0.5) of its FOV within rmax px, else -1"""
    idx = np.full(S['n'], -1)
    ok = score >= 0.5
    for f in range(S['nfov']):
        g = np.nonzero(S['fov'] == f)[0]; dd = np.nonzero(ok & (b == f))[0]
        if not len(dd): continue
        d = np.hypot(S['x'][g, None] - x[None, dd], S['y'][g, None] - y[None, dd])
        j = d.argmin(1); m = d[np.arange(len(g)), j] < rmax
        idx[g[m]] = dd[j[m]]
    return idx


def rmsd(e, keep):
    e = e[np.isfinite(e).all(1)]
    if len(e) < 10: return [np.nan, np.nan]
    if keep < 1:
        nrm = np.hypot(e[:, 0], e[:, 1]); e = e[nrm <= np.quantile(nrm, keep)]
    return np.sqrt((e ** 2).mean(0)).tolist()


def summarize(err, sig):
    return dict(rmsd_trim2=rmsd(err, TRIM_KEEP), rmsd_trim5=rmsd(err, 0.95), rmsd=rmsd(err, 1.0),
                pred_mean=np.nanmean(sig, 0).tolist(), pred_rms=np.sqrt(np.nanmean(sig ** 2, 0)).tolist(), n=int(np.isfinite(err).all(1).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='m1809')
    ap.add_argument('--bins', default='all')
    ap.add_argument('--cond', default='const1,unif16')
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=10)
    ap.add_argument('--out', default=os.path.join(C.OUT, 's10'))
    ap.add_argument('--psf', default='vectorial', help="'vectorial' (averaged 1809/2), 'v2' (in-situ PSF v2 of 1809/2 at the calibration's median "
                    "height, detected photons; module doc) or 'gauss1.3' (the preprint's PSF: rendering AND the SMLM / SIMFLUX fits)")
    ap.add_argument('--tag', default='', help='output name (default: see usage)')
    ap.add_argument('--nets', type=int, default=1, help='0: classical methods only (network rows nan)')
    ap.add_argument('--start', default='gt', choices=['gt', 'smlmfit'],
                    help="SMLM / SIMFLUX start: gt = GT + U(-0.5, 0.5) px (preprint script); smlmfit = paper_pipeline: SMLM from the "
                         "centroid / border-median init of the 6-frame sum, SIMFLUX (both bg models) from that SMLM fit")
    a = ap.parse_args()
    from simcode.util.proctitle import set_title; set_title('psim-s10')
    from simcode.config import model_set
    os.makedirs(a.out, exist_ok=True)
    ns = model_set(a.model)
    gauss = a.psf.startswith('gauss'); v2 = a.psf == 'v2'
    coll = 1.0
    if v2:
        psf = PP.zpsf.load(C.PSF_CALIB_V2)                  # fit plane z_fit = the calib's emitter_z_median_um
        R = C.RendererV2(FOV, FOV); R.z = float(psf.z_fit)
        coll = float(R.collected(R.z))                      # detected photons -> intensity parameter
        ns = dict(ns, psf_calib=C.PSF_CALIB_V2)
    else:
        psf = PP.zpsf.load(f'gauss:{a.psf[5:]}' if gauss else C.PSF_CALIB)
        R = C.Renderer(FOV, FOV, psf=a.psf)
    if gauss: ns = dict(ns, psf_calib=C.GAUSS_CALIB)        # the head's PSF template = the movie's Gaussian (calib key psf_gauss_sigma_um)
    nets = Nets(ns) if a.nets else None
    tag = a.tag or (a.model + ('_' + a.psf if gauss else '') + ('v2' if v2 else ''))
    bins = range(len(PHOTONS)) if a.bins == 'all' else [int(v) for v in a.bins.split(',')]
    nfov = int(math.ceil(a.n / (NC * NC)))
    res = dict(protocol=dict(photons=PHOTONS.tolist(), n_per_bin=nfov * NC * NC, trim_keep=TRIM_KEEP, roi=CELL, edge=EDGE, fov=FOV,
                             pitch_nm=C.PITCH_NM, depth=C.DEPTH, psf=a.psf if gauss else (C.PSF_CALIB_V2 if v2 else C.PSF_CALIB), z_um=R.z, model=a.model, nets=ns, units='px',
                             photons_are=('detected (intensity parameter = photons / %.4f, the 41 x 41 px PSF image sum at z)' % coll) if v2 else 'intensity parameter',
                             collected_at_z=coll, classical_start=a.start), cond={})
    fn = os.path.join(a.out, f's10_{tag}.json')
    if os.path.exists(fn):
        old = json.load(open(fn)); res['cond'] = old.get('cond', {})
    for cond in a.cond.split(','):
        RC = res['cond'].setdefault(cond, {})
        for bi in bins:
            ph = float(PHOTONS[bi]); t0 = time.time()
            rng = np.random.default_rng([a.seed, bi, 0 if cond == 'const1' else 1])
            gen = torch.Generator(device=dev); gen.manual_seed(a.seed * 1000 + bi * 10 + (0 if cond == 'const1' else 1))
            S = scene(nfov, ph / coll, cond, rng)
            frames = render(S, R, gen)
            cl = classical(S, rois(frames, S), psf, rng, a.start)
            patt = torch.as_tensor(S['patt'], device=dev, dtype=torch.float32)
            if nets is not None:
                r1, r2 = nets.run(frames, patt)
                r1 = {k: v.float().cpu().numpy() for k, v in r1.items()}; r2 = {k: v.float().cpu().numpy() for k, v in r2.items()}
            else:
                r1 = r2 = None
            gxy = np.c_[S['x'], S['y']]
            out = dict(photons=ph, intensity_param=ph / coll, n=S['n'], crlb_smlm=np.mean(cl['crlb_smlm'], 0).tolist(), crlb_sf=np.mean(cl['crlb_sf'], 0).tolist(),
                       crlb_sf_pfbg=np.mean(cl['crlb_sf_pfbg'], 0).tolist(), bg_mean=float(S['bg'].mean()))
            out['smlm'] = summarize(cl['smlm'], cl['smlm_sig']); out['sf'] = summarize(cl['sf'], cl['sf_sig'])
            out['sf_pfbg'] = summarize(cl['sf_pfbg'], cl['sf_pfbg_sig'])
            pp_ = C.PITCH_NM / C.PX_NM          # fringe off: |error| > half a pattern period along x or y (the pattern axes 0 / 90 deg)
            for k in ('smlm', 'sf', 'sf_pfbg'):
                e = cl[k]; ok = np.isfinite(e).all(1)
                out[k]['fringe_off_frac'] = float((np.abs(e[ok]).max(1) > pp_ / 2).mean()) if ok.any() else float('nan')
            for nm, r, xk, yk, sk in (('nn_p1', r1, 'x_prior', 'y_prior', 's_prior'), ('v2_p1', r1, 'x_sc', 'y_sc', 'scal_sc'),
                                      ('nn_p2', r2, 'x', 'y', 's_prior'), ('v2_p2', r2, 'x_sc', 'y_sc', 'scal_sc')):
                if r is None:
                    out[nm] = summarize(np.full((S['n'], 2), np.nan), np.full((S['n'], 2), np.nan)); out[nm]['recall'] = float('nan'); continue
                mi = match(S, r['b'].astype(int), r['x_det'], r['y_det'], r['score'])
                m = mi >= 0
                e = np.full((S['n'], 2), np.nan); s_ = np.full((S['n'], 2), np.nan)
                e[m] = np.c_[r[xk][mi[m]], r[yk][mi[m]]] - gxy[m]; s_[m] = r[sk][mi[m]]
                out[nm] = summarize(e, s_); out[nm]['recall'] = float(m.mean())
            RC[str(bi)] = out
            np.savez_compressed(os.path.join(a.out, f's10_{tag}_{cond}_b{bi}.npz'), gx=S['x'], gy=S['y'], fov=S['fov'], bg=S['bg'],
                                **{k: v for k, v in cl.items()})
            pr = lambda k: f"{k} {out[k]['rmsd_trim2'][0] * C.PX_NM:.2f}/{out[k]['pred_mean'][0] * C.PX_NM:.2f}"
            print(f"{cond} ph {ph:7.1f}: CRLB smlm {out['crlb_smlm'][0] * C.PX_NM:.2f} sf {out['crlb_sf'][0] * C.PX_NM:.2f} "
                  f"sf_pfbg {out['crlb_sf_pfbg'][0] * C.PX_NM:.2f} nm | x RMSD(trim 2%)/pred nm: " +
                  ' '.join(pr(k) for k in ('smlm', 'sf', 'sf_pfbg', 'nn_p1', 'v2_p1', 'nn_p2', 'v2_p2')) +
                  f" | recall p2 {out['v2_p2']['recall']:.3f} | {time.time() - t0:.1f}s", flush=True)
            json.dump(res, open(fn, 'w'), indent=1)
    plot(res, os.path.join(a.out, f's10_{tag}.png'))
    print('written', fn)


def plot(res, fn):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    conds = list(res['cond'])
    fig, axs = plt.subplots(1, len(conds), figsize=(7.5 * len(conds), 5.5), squeeze=False)
    sty = dict(smlm=('k', 'o', 'SMLM MLE'), sf=('g', 's', 'SIMFLUX MLE'), sf_pfbg=('olive', 'D', 'SIMFLUX MLE, bg per frame'), nn_p1=('tab:blue', '^', 'NN output (pass 1)'),
               nn_p2=('c', '^', 'NN output (pass 2)'), v2_p1=('salmon', 'v', 'SIMCODE v2 pass 1'), v2_p2=('tab:red', 'v', 'SIMCODE v2'))
    for ax, cond in zip(axs[0], conds):
        B = sorted(res['cond'][cond], key=int); ph = [res['cond'][cond][b]['photons'] for b in B]
        g = lambda k, f: np.array([res['cond'][cond][b][k][f][0] if isinstance(res['cond'][cond][b][k], dict) else res['cond'][cond][b][k][0] for b in B]) * C.PX_NM
        for k, ls in (('crlb_smlm', '--'), ('crlb_sf', ':'), ('crlb_sf_pfbg', '-.')):
            ax.loglog(ph, np.array([res['cond'][cond][b][k][0] for b in B]) * C.PX_NM, ls, color='0.4', label=k.replace('crlb_', 'CRLB '))
        for k, (c, mk, lab) in sty.items():
            ax.loglog(ph, g(k, 'rmsd_trim2'), '-', marker=mk, color=c, label=lab + ' RMSD')
            ax.loglog(ph, g(k, 'pred_mean'), ':', marker='x', color=c, lw=1, label=lab + ' predicted')
        ax.set_xlabel('total signal photons'); ax.set_ylabel('x precision (nm)'); ax.grid(True, which='both', alpha=0.3)
        ax.set_title({'const1': 'A: bg 1 photon/px/frame', 'unif16': 'B/C: bg ~ U(1, 6) per frame'}.get(cond, cond))
        ax.legend(fontsize=6.5, ncol=2)
    plt.tight_layout(); plt.savefig(fn, dpi=90); plt.close(fig)


if __name__ == '__main__':
    main()
