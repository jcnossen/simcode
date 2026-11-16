"""
Pipeline pass 2 with the pattern-aware networks (run_uncalibrated stage pass2). Pass 1 (pattern-free detector + emitter
network) and the pattern estimation are unchanged (simcode/processing/pass1_inference.py, pattern_estimation.py on the network
intensities). Then per 6-frame window (stride 3, the pass-1 grid):
  illumination maps of the ESTIMATED pattern (pattern_est.pt -> processing.common.pattern_frames, loaded-region coordinates)
  -> pass-2 Detector (frames + illumination) -> sub-pixel NMS, p >= 0.2 neighbour map, p >= 0.3 (CAND_PMIN) candidates
  -> pass-2 EmitterNet (patch + neighbour + illumination channels): I_t, full covariance, bg_t, prior (pattern-informed)
  -> explicit intensity-domain pattern solve (solve/solve.solve_emitters: fringe-start posterior, cmult, box 1.5, calibrated sigma)
     with the pass-2 head prior.
  Output rows: windows' central detections (time centroid in the central half, inside the crop, own frames) as pass 1,
  pass-2 format columns (x_hy0, y_hy0, sx_cal, sy_cal, sig_hy0, q_dof, phot, x_prior, y_prior, kin_hy, p, wstart, frame,
  I_net) + x_dir / y_dir / sx_dir / sy_dir (variant b: the pass-2 head prior itself with its sigma).
  z (z_hy0, sz_cal, cov_xz, cov_yz, z_prior) from PASS 1 (2026-11-14; the pass-2 head's z is not used): each row is linked
  to a pass-1 candidate of the same window (solve/z_conditioning.link_pass1: detector points, one-to-one, 1 px; p1_row = its index in
  pass1_<f0>.npz, -1 = none -> z NaN) and that candidate's 3D prior is conditioned on the pass-2 solved x, y
  (z_conditioning.pass1_z_columns).
Reference implementation (test oracle of simcode/processing/inference_fast.Pass2FrontFast, which the pipeline uses).
"""
import json, time
import numpy as np, torch
from simcode.psf import TiffMovie
from simcode.processing.common import region, pattern_frames, CAND_PMIN

T = 6
from simcode import config as _cfg
DETECTOR_PASS2 = _cfg.model(_cfg.DETECTOR_PASS2)
EMITTER_PASS2 = _cfg.model(_cfg.EMITTER_PASS2)



class Pass2Front:
    def __init__(self, calib_path, emitter=EMITTER_PASS2, detector=DETECTOR_PASS2, dev='cuda'):
        from simcode.emitter import model as EM
        from simcode.detector import model as DM
        from simcode.simulate.simulator import Simulator
        self.detector, _ = DM.load(detector)
        self.emitter = EM.load(emitter)      # a pattern-free (pass-1) network also works (detector-only pass 2)
        self.col = EM.PSFCol(Simulator(calib_path, H=32, W=32, T=T, device=dev))
        self.dev = dev

    @torch.no_grad()
    def run_chunk(self, movie: TiffMovie, cal, f0, f1, pattern, solve_cal, p1, p1_cal, crop=(5, 5, 110), margin=5, pad=12, batch=16, timing=None):
        """p1 = the chunk's stored pass-1 rows (pass1_<f0>.npz), p1_cal = the pass-1 solve calibration: the z columns come
        from pass 1 (z_conditioning.pass1_z_columns: the linked pass-1 candidate's z prior conditioned on the pass-2 solved x, y)"""
        from simcode.pattern import illumination as IL
        from simcode.emitter import model as EM
        from simcode.solve import solve as SV
        from simcode.solve import z_conditioning as ZC
        dev = self.dev
        timing = timing if timing is not None else {}
        sync = lambda: torch.cuda.synchronize()
        scal = json.load(open(solve_cal)) if isinstance(solve_cal, str) else dict(solve_cal)
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
        PF = pattern_frames(pattern, np.arange(l0, l1), X0, Y0, dev)             # [F,5] region coordinates
        k_cand = int(640 * S * S / 64 ** 2)
        x0c, y0c, szc = crop
        rows = []
        for bi in range(0, len(starts), batch):
            st = starts[bi:bi + batch]
            t_ = time.time()
            x = torch.stack([frames[s0:s0 + T] for s0 in st])
            PW = torch.stack([PF[s0:s0 + T] for s0 in st])
            il = IL.illum_maps(PW, S, S)
            c, nbr, lgm = EM.detect_candidates(self.detector, x, il, k=k_cand)
            sync(); timing['detector'] = timing.get('detector', 0) + time.time() - t_
            if len(c['b']) == 0: continue
            t_ = time.time()
            s5 = torch.sigmoid(c['logit']) >= CAND_PMIN
            b, iy, ix, xs, ys, lg = (c[k][s5] for k in ('b', 'iy', 'ix', 'x', 'y', 'logit'))
            if len(b) == 0: continue
            p = torch.sigmoid(lg)
            outs = []
            for c0 in range(0, len(b), 4096):
                sl = slice(c0, c0 + 4096)
                args = (x, il, b[sl], iy[sl], ix[sl], xs[sl], ys[sl], nbr, p[sl])
                oh = EM.apply_to_candidates(self.emitter, self.col, *args)
                r = {k: oh[k].float() for k in ('I', 'L', 'bg', 'x', 'y', 'lsxy', 'x_det', 'y_det')}
                outs.append(r)
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
            keep = central & inside & (fabs >= own0) & (fabs < own1)
            if not keep.any(): continue
            t_ = time.time()
            k_ = torch.nonzero(keep)[:, 0]
            o = {kk: v[k_] for kk, v in oh.items()}
            bw = b[k_]
            so = SV.solve_emitters(o, PW, bw, scal)
            sync(); timing['solve'] = timing.get('solve', 0) + time.time() - t_
            sp = oh['lsxy'][k_].exp()
            ph = I[k_].sum(1)
            r = dict(wstart=ws[k_], x_hy0=so['r'][:, 0] + X0, y_hy0=so['r'][:, 1] + Y0, sx_cal=so['s_cal'][:, 0], sy_cal=so['s_cal'][:, 1],
                     sig_hy0=so['s_cal'].mean(1), q_dof=so['q'], kin_hy=so['kin'].float(), phot=ph, p=p[k_],
                     x_prior=oh['x'][k_] + X0, y_prior=oh['y'][k_] + Y0, x_det=oh['x_det'][k_] + X0, y_det=oh['y_det'][k_] + Y0,
                     x_dir=oh['x'][k_] + X0, y_dir=oh['y'][k_] + Y0, sx_dir=sp[:, 0], sy_dir=sp[:, 1],
                     frame=fabs[k_], I_net=I[k_])
            rows.append({kk: v.float().cpu().numpy() if v.is_floating_point() else v.cpu().numpy() for kk, v in r.items()})
        R = {kk: np.concatenate([r_[kk] for r_ in rows]) for kk in rows[0]}
        R['wstart'] = R['wstart'].astype(np.int64)
        tt = lambda v: torch.as_tensor(v, device=dev)
        Z = ZC.pass1_z_columns(p1, p1_cal, R['wstart'], R['x_det'], R['y_det'], torch.stack([tt(R['x_hy0']), tt(R['y_hy0'])], 1),
                                torch.stack([tt(R['sx_cal']), tt(R['sy_cal'])], 1))
        R.update({k: v.cpu().numpy() for k, v in Z.items()})
        return R


def direct_view(O):
    """variant b as a pass-2 file: the pass-2 head prior as the final position"""
    D = dict(O)
    D['x_hy0'], D['y_hy0'], D['sx_cal'], D['sy_cal'] = O['x_dir'], O['y_dir'], O['sx_dir'], O['sy_dir']
    D['sig_hy0'] = 0.5 * (O['sx_dir'] + O['sy_dir'])
    return D
