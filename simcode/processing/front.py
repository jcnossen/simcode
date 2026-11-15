"""
Reference implementation of the per-chunk pass 1 and of the pattern solve of its rows (run_uncalibrated --pass2_mode solve; the
pipeline runs the fast versions in front_fast.py, these are the test oracle). No linear-intensity stage anywhere: no
pixel-level per-frame intensity fit, no wing subtraction, no pixel-level position refinement.

  pass 1 (pattern-free)  per 6-frame window (stride 3, absolute start % 3 == 0, as pass1.py):
                         pass-1 detector (sub-pixel NMS + top-k, p >= 0.2 for the neighbour map) -> pass-1 EmitterNet on the
                         p >= 0.5 candidates (21 x 21 x 6 patch + neighbour-splat / detector-probability channels) ->
                         network per-frame intensities I_t with the full T x T covariance (Cholesky L), per-frame bg,
                         pattern-free position prior (x, y) +- s.
                         Stored per detection (all detections of each window, flag `out` as pass1.py: the window whose
                         central half holds the time centroid of the network intensities, inside the crop, own frames),
                         in the pass-1 file format (wstart, x, y, sx, sy, z, p, phot, phot_t, I, sI, bg, out, tc, frame)
                         + L [n,6,6], x_det, y_det. pattern_estimation.py runs on these network intensities unchanged.
  pass 2                 the explicit intensity-domain pattern solve (simcode/solve/solve.solve_emitters: fringe starts,
                         kinetic model selection, box 1.5, calibrated sigma, fringe-start posterior mean) with
                         the estimated pattern (pattern_frames of pattern_est.pt) on the stored head outputs; output in
                         the pass-2 format (x_hy0, y_hy0, sig_hy0 (mean calibrated sigma), kin_hy, fringe_hy, pf_hy,
                         chi2_hy, B_hy, Klin_hy) + sc columns (sx_cal, sy_cal, q_dof, phot, x_prior, y_prior, dnll), rows
                         = the pass-1 `out` detections.
Coordinates: movie (camera) pixels.
"""
import json, time
import numpy as np, torch
from simcode.psf import TiffMovie
from simcode.processing.common import region, pattern_frames, CAND_PMIN

T = 6
from simcode import config as _cfg
DETECTOR = _cfg.model(_cfg.DETECTOR)
EMITTER = _cfg.model(_cfg.EMITTER)
CAL = _cfg.model(_cfg.CAL_REAL)
THR0 = 0.2



class Pass1Front:
    def __init__(self, calib_path, emitter=EMITTER, detector=DETECTOR, dev='cuda'):
        from simcode.emitter import model as EM
        from simcode.detector import model as DM
        from simcode.simulate.simulator import Simulator
        self.EM = EM
        self.detector, _ = DM.load(detector)
        self.emitter = EM.load(emitter)
        self.col = EM.PSFCol(Simulator(calib_path, H=32, W=32, T=T, device=dev))
        self.dev = dev

    @torch.no_grad()
    def run_chunk(self, movie: TiffMovie, cal, f0, f1, crop=(5, 5, 110), margin=5, pad=12, batch=16, timing=None):
        from simcode.detector.select import select_subpixel
        dev = self.dev; EM = self.EM
        timing = timing if timing is not None else {}
        sync = lambda: torch.cuda.synchronize()
        X0, Y0, S = region(movie, crop, margin)
        own0, own1 = f0, min(f1, movie.n)
        l0 = max(f0 - pad, 0) // 6 * 6; l1 = min(own1 + pad, movie.n)
        t_ = time.time()
        mov = movie.read(l0, l1, Y0, X0, S)
        off = torch.tensor(cal['offset'][Y0:Y0 + S, X0:X0 + S], device=dev)
        frames = (torch.tensor(mov, device=dev) - off) * cal['gain']
        sync(); timing['read'] = timing.get('read', 0) + time.time() - t_
        F = frames.shape[0]
        starts = list(range(0, F - T + 1, T // 2))
        k_cand = int(640 * S * S / 64 ** 2)
        x0c, y0c, szc = crop
        rows = []
        for bi in range(0, len(starts), batch):
            st = starts[bi:bi + batch]; B = len(st)
            t_ = time.time()
            x = torch.stack([frames[s0:s0 + T] for s0 in st])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                dd = self.detector(x)
            lgm = dd['logit'].float(); lgm = lgm[:, 0] if lgm.dim() == 4 else lgm
            c = select_subpixel(lgm, dd['dxy'].float(), k_cand, p_min=THR0, radius=0.7)
            keep = torch.isfinite(c['logit'])
            b, iy, ix, xs, ys, lg = (c[k][keep] for k in ('b', 'iy', 'ix', 'x', 'y', 'logit'))
            sync(); timing['detector'] = timing.get('detector', 0) + time.time() - t_
            if len(b) == 0: continue
            t_ = time.time()
            nbr = EM.nbr_map(dd['logit'].float(), b, iy, ix, xs, ys, lg)
            s5 = torch.sigmoid(lg) >= CAND_PMIN                 # stored-candidate cut (processing.common)
            b, iy, ix, xs, ys, lg = (v[s5] for v in (b, iy, ix, xs, ys, lg))
            if len(b) == 0: continue
            outs = []
            for c0 in range(0, len(b), 4096):
                sl = slice(c0, c0 + 4096)
                oh = EM.apply_to_candidates(self.emitter, self.col, x, None, b[sl], iy[sl], ix[sl], xs[sl], ys[sl], nbr=nbr, p=torch.sigmoid(lg[sl]))
                outs.append(dict({k: oh[k].float() for k in ('I', 'L', 'bg', 'x', 'y', 'lsxy')}, **EM.emitter_z(self.emitter, oh)))
            oh = {k: torch.cat([q[k] for q in outs]) for k in outs[0]}
            sync(); timing['head'] = timing.get('head', 0) + time.time() - t_
            I = oh['I']; Ip = I.clamp(min=0)
            tc = (Ip * torch.arange(T, device=dev).float()[None]).sum(1) / Ip.sum(1).clamp(min=1e-3)
            wabs = torch.tensor(st, device=dev) + l0
            ws = wabs[b]
            central = (tc >= T / 4 - 0.5) & (tc < 3 * T / 4 - 0.5)
            central |= (ws == l0) & (tc < T / 4 - 0.5)
            central |= (ws == l0 + starts[-1]) & (tc >= 3 * T / 4 - 0.5)
            gx, gy = oh['x'] + X0, oh['y'] + Y0
            inside = (gx >= x0c) & (gx < x0c + szc) & (gy >= y0c) & (gy < y0c + szc)
            fabs = ws.float() + tc
            out = central & inside & (fabs >= own0) & (fabs < own1)
            sp = oh['lsxy'].exp()
            sI = (oh['L'] ** 2).sum(-1).sqrt()
            r = dict(wstart=ws, x=gx, y=gy, sx=sp[:, 0], sy=sp[:, 1], z=oh['z'], sz=oh['lsz'].exp(), bzx=oh['bz'][:, 0], bzy=oh['bz'][:, 1],
                     p=torch.sigmoid(lg.float()),
                     phot=I.sum(1), phot_t=I, I=I, sI=sI, bg=oh['bg'], L=oh['L'], out=out, tc=tc, frame=fabs,
                     x_det=xs + X0, y_det=ys + Y0)
            rows.append({k_: v.float().cpu().numpy() if v.is_floating_point() else v.cpu().numpy() for k_, v in r.items()})
        R = {k_: np.concatenate([r_[k_] for r_ in rows]) for k_ in rows[0]}
        R['wstart'] = R['wstart'].astype(np.int64)
        return R, dict(X0=X0, Y0=Y0, S=S, own=(own0, own1), loaded=(l0, l1))


def z_inputs(R, idx, tt):
    """the head's z outputs of stored pass-1 rows (columns z, sz, bzx, bzy) -> z, lsz, bz of the solve input; rows of a run made
    before 2026-11-14 (no sz column: 2D heads): NaN"""
    if 'sz' not in R:
        nan = tt(np.full(len(idx), np.nan))
        return dict(z=nan, lsz=nan, bz=torch.stack([nan, nan], 1))
    return dict(z=tt(R['z'][idx]), lsz=torch.log(tt(R['sz'][idx])), bz=torch.stack([tt(R['bzx'][idx]), tt(R['bzy'][idx])], 1))


@torch.no_grad()
def run_chunk_pattern_solve(R, info, pattern, solve_cal, dev='cuda', chunk=8192, timing=None):
    """pattern solve on the stored pass-1 head outputs of the `out` rows (identical rows for every pattern)"""
    from simcode.solve import solve as SV
    from simcode.solve import z_conditioning as ZC
    timing = timing if timing is not None else {}
    t_ = time.time()
    cal = json.load(open(solve_cal)) if isinstance(solve_cal, str) else dict(solve_cal)
    keep = R['out'].astype(bool)
    X0, Y0 = info['X0'], info['Y0']
    ws = R['wstart'][keep]
    n = int(keep.sum())
    tt = lambda a: torch.tensor(np.asarray(a), device=dev, dtype=torch.float32)
    out = {c: np.full(n, np.nan, np.float32) for c in ('x_hy0', 'y_hy0', 'sig_hy0', 'kin_hy', 'fringe_hy', 'pf_hy', 'chi2_hy', 'B_hy',
                                                    'Klin_hy', 'sx_cal', 'sy_cal', 'q_dof', 'phot', 'x_prior', 'y_prior', 'dnll',
                                                    'x_map', 'y_map', 'z_hy0', 'sz_cal', 'cov_xz', 'cov_yz', 'z_prior')}
    uw, inv = np.unique(ws, return_inverse=True)
    PW = pattern_frames(pattern, (uw[:, None] + np.arange(T)[None]).reshape(-1), X0, Y0, dev).reshape(len(uw), T, 5)
    for c0 in range(0, n, chunk):
        sl = slice(c0, c0 + chunk)
        idx = np.nonzero(keep)[0][sl]
        o = dict(I=tt(R['I'][idx]), L=tt(R['L'][idx]), x=tt(R['x'][idx] - X0), y=tt(R['y'][idx] - Y0),
                 lsxy=torch.log(tt(np.c_[R['sx'][idx], R['sy'][idx]])), x_det=tt(R['x_det'][idx] - X0), y_det=tt(R['y_det'][idx] - Y0),
                 **z_inputs(R, idx, tt))
        b = torch.tensor(inv[sl], device=dev)
        so = SV.solve_emitters(o, PW, b, cal)
        for k_, v in ZC.z_columns(o, so).items(): out[k_][sl] = v.cpu().numpy()
        r = so['r'].cpu().numpy(); sc_ = so['s_cal'].cpu().numpy()
        out['x_hy0'][sl] = r[:, 0] + X0; out['y_hy0'][sl] = r[:, 1] + Y0
        out['sx_cal'][sl] = sc_[:, 0]; out['sy_cal'][sl] = sc_[:, 1]; out['sig_hy0'][sl] = sc_.mean(1)
        out['kin_hy'][sl] = so['kin'].float().cpu().numpy(); out['pf_hy'][sl] = so['p'].float().cpu().numpy()
        out['q_dof'][sl] = so['q'].cpu().numpy(); out['chi2_hy'][sl] = so['q'].cpu().numpy()
        out['phot'][sl] = R['phot'][idx]; out['x_prior'][sl] = R['x'][idx]; out['y_prior'][sl] = R['y'][idx]
        if 'dnll' in so: out['dnll'][sl] = so['dnll'].cpu().numpy()
        rm = so.get('r_map', so['r']).cpu().numpy(); out['x_map'][sl] = rm[:, 0] + X0; out['y_map'][sl] = rm[:, 1] + Y0
    torch.cuda.synchronize(); timing['solve'] = timing.get('solve', 0) + time.time() - t_
    out.update(wstart=ws, frame=R['frame'][keep], p=R['p'][keep], I_net=R['I'][keep])
    return out
