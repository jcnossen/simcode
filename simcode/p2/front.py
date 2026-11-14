"""
Pipeline pass 2 with the pattern-aware networks (run_uncalibrated --frontend sc2p). Pass 1 (det_n1 + sc5nb, pattern-free)
and the pattern estimation are the 'sc' front-end unchanged (simcode/pipeline/sc_front.py, pattern_est.py on the network
intensities). Then per 6-frame window (stride 3, the pass-1 grid):
  illumination maps of the ESTIMATED pattern (pattern_est.pt -> pipeline.common.pattern_frames, loaded-region coordinates)
  -> pass-2 detector DetP2 (frames + illumination) -> sub-pixel NMS, p >= 0.2 neighbour map, p >= 0.5 candidates
  -> pass-2 head SCHeadP2 (patch + neighbour + illumination channels): I_t, full covariance, bg_t, prior (pattern-informed)
  -> explicit intensity-domain pattern solve (hd/sc.sc_solve: fringe-start posterior, cmult, box 1.5, calibrated sigma)
     with the pass-2 head prior.
  Output rows: windows' central detections (time centroid in the central half, inside the crop, own frames) as pass 1,
  pass-2 format columns (x_hy0, y_hy0, sx_cal, sy_cal, sig_hy0, q_dof, phot, x_prior, y_prior, kin_hy, p, wstart, frame,
  I_net) + x_dir / y_dir / sx_dir / sy_dir (variant b: the pass-2 head prior itself with its sigma).
Reference implementation (test oracle of simcode/pipeline/sc_front_fast.Pass2P2Fast, which the pipeline uses).
"""
import json, time
import numpy as np, torch
from simcode.psf import TiffMovie
from simcode.pipeline.common import region, pattern_frames, CAND_PMIN

T = 6
from simcode import config as _cfg
DET_P2 = _cfg.model(_cfg.P2_DET)
HEAD_P2 = _cfg.model(_cfg.P2_HEAD)



class Pass2P2:
    def __init__(self, calib_path, head=HEAD_P2, det=DET_P2, dev='cuda'):
        from simcode.hd import sc as SCM
        from simcode.p2 import head2 as H2
        from simcode.sim.simulator import Simulator
        self.SCM, self.H2 = SCM, H2
        self.det, self.needs_il = H2.load_det(det)
        self.head = H2.load(head) if ('p2_head' in head or H2.is_p2_head(head)) else SCM.load(head)     # detector-only pass 2: a pattern-free head
        self.col = SCM.PSFCol(Simulator(calib_path, H=32, W=32, T=T, device=dev))
        self.dev = dev

    @torch.no_grad()
    def run_chunk(self, movie: TiffMovie, cal, f0, f1, pattern, sc_cal, crop=(5, 5, 110), margin=5, pad=12, batch=16, timing=None):
        from simcode.p2 import illum as IL
        dev = self.dev; SCM, H2 = self.SCM, self.H2
        timing = timing if timing is not None else {}
        sync = lambda: torch.cuda.synchronize()
        scal = json.load(open(sc_cal)) if isinstance(sc_cal, str) else dict(sc_cal)
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
            c, nbr, lgm = H2.candidates(self.det, self.needs_il, x, il, k=k_cand)
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
                oh = H2.head_apply(self.head, self.col, *args)
                r = dict({k: oh[k].float() for k in ('I', 'L', 'bg', 'x', 'y', 'lsxy', 'x_det', 'y_det')}, **SCM.head_z(self.head, oh))
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
            so = SCM.sc_solve(o, PW, bw, scal)
            so.update(SCM.z_columns(o, so))
            sync(); timing['solve'] = timing.get('solve', 0) + time.time() - t_
            sp = oh['lsxy'][k_].exp()
            ph = I[k_].sum(1)
            r = dict(wstart=ws[k_], x_hy0=so['r'][:, 0] + X0, y_hy0=so['r'][:, 1] + Y0, sx_cal=so['s_cal'][:, 0], sy_cal=so['s_cal'][:, 1],
                     sig_hy0=so['s_cal'].mean(1), q_dof=so['q'], kin_hy=so['kin'].float(), phot=ph, p=p[k_],
                     x_prior=oh['x'][k_] + X0, y_prior=oh['y'][k_] + Y0, x_det=oh['x_det'][k_] + X0, y_det=oh['y_det'][k_] + Y0,
                     x_dir=oh['x'][k_] + X0, y_dir=oh['y'][k_] + Y0, sx_dir=sp[:, 0], sy_dir=sp[:, 1],
                     frame=fabs[k_], I_net=I[k_], **{k: so[k] for k in ('z_hy0', 'sz_cal', 'cov_xz', 'cov_yz', 'z_prior')})
            rows.append({kk: v.float().cpu().numpy() if v.is_floating_point() else v.cpu().numpy() for kk, v in r.items()})
        R = {kk: np.concatenate([r_[kk] for r_ in rows]) for kk in rows[0]}
        R['wstart'] = R['wstart'].astype(np.int64)
        return R


def direct_view(O):
    """variant b as a pass-2 file: the pass-2 head prior as the final position"""
    D = dict(O)
    D['x_hy0'], D['y_hy0'], D['sx_cal'], D['sy_cal'] = O['x_dir'], O['y_dir'], O['sx_dir'], O['sy_dir']
    D['sig_hy0'] = 0.5 * (O['sx_dir'] + O['sy_dir'])
    return D
