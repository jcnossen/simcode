"""
Reference implementation of the per-chunk pass 1 (run_uncalibrated stage pass1; the pipeline runs the fast version in inference_fast.py,
this is the test oracle) and z_inputs (the stored pass-1 z outputs as solve inputs, used by the fast pattern solve). No linear-intensity stage anywhere: no
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
  pattern solve          (stage pass1_solve: inference_fast.run_chunk_pattern_solve_fast) the explicit intensity-domain pattern solve (simcode/solve/solve.solve_emitters: fringe starts,
                         kinetic model selection, box 1.5, calibrated sigma, fringe-start posterior mean) with
                         the estimated pattern (pattern_frames of pattern_est.pt) on the stored head outputs; output in
                         the pass-2 format (x_hy0, y_hy0, sig_hy0 (mean calibrated sigma), kin_hy, fringe_hy, pf_hy,
                         chi2_hy, B_hy, Klin_hy) + sc columns (sx_cal, sy_cal, q_dof, phot, x_prior, y_prior, dnll), rows
                         = the pass-1 `out` detections.
Coordinates: movie (camera) pixels.
"""
import time
import numpy as np, torch
from simcode.psf import TiffMovie
from simcode.processing.common import region, CAND_PMIN

T = 6
from simcode import config as _cfg
DETECTOR = _cfg.model(_cfg.DETECTOR)
EMITTER = _cfg.model(_cfg.EMITTER)
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
                outs.append(dict({k: oh[k].float() for k in ('I', 'L', 'C', 'bg', 'x', 'y', 'lsxy')}, **EM.emitter_z(self.emitter, oh)))
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
                     phot=I.sum(1), phot_t=I, I=I, sI=sI, bg=oh['bg'], L=oh['L'], C=oh['C'], out=out, tc=tc, frame=fabs,
                     x_det=xs + X0, y_det=ys + Y0)
            rows.append({k_: v.float().cpu().numpy() if v.is_floating_point() else v.cpu().numpy() for k_, v in r.items()})
        R = {k_: np.concatenate([r_[k_] for r_ in rows]) for k_ in rows[0]}
        R['wstart'] = R['wstart'].astype(np.int64)
        return R, dict(X0=X0, Y0=Y0, S=S, own=(own0, own1), loaded=(l0, l1))


def z_inputs(R, idx, tt):
    """the head's z outputs of stored pass-1 rows (columns z, sz, bzx, bzy) -> z, lsz, bz of the solve input"""
    return dict(z=tt(R['z'][idx]), lsz=torch.log(tt(R['sz'][idx])), bz=torch.stack([tt(R['bzx'][idx]), tt(R['bzy'][idx])], 1))
