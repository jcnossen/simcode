"""
Fast versions of the per-chunk fronts (2026-11-10, throughput; the pipeline's code path). Same networks, same
candidate rules, same outputs and file formats as the reference implementations simcode/processing/front.py (pass 1,
pattern solve) and simcode/processing/pass2_front.py (pattern-aware pass 2), which are kept as the test oracle (tests/test_fast_paths.py).

Queued fixed-size batching (shared by pass 1 and pass 2):
  detector  on `det_batch` windows at a time (default 64) -> padded top-k -> neighbour maps from the padded list (no
            boolean compaction needed; invalid entries carry p = 0) -> the p >= CAND_PMIN candidates (one index compaction per
            detector batch) are appended to a GPU queue of (window index, pixel, sub-pixel point, logit); the chunk's frames
            and neighbour maps stay on the GPU (zero-padded once per chunk), so a queue entry needs no patch storage
  head      drains the queue in fixed batches of `emitter_batch` (default 1024; the last one padded by repeating the last
            entry) -> static shapes; patches are gathered from the padded chunk buffers (identical values to layers.blocks.extract_patches);
            pass 2's illumination patches are evaluated analytically per candidate (= illumination.illum_maps + zero padding)
  streams   detector and head on two CUDA streams (the head of batch k overlaps the detector of batch k+1; the only host
            syncs are the per-detector-batch candidate counts, on the detector stream)
  outputs   written into preallocated GPU tensors, post-processed (time centroid, central / inside / own flags) once per chunk,
            copied to the CPU once per chunk
  compile   the heads are not compiled (torch.compile of EmitterNet changed the outputs far beyond the bf16 noise on the real
            0.5 nM chunk: I up to 275 ph, p99.9 59 ph vs 1.4 ph run-to-run); `compile` ('auto' | 'on' | 'off') applies to the
            pattern solve's LM step (solve/solve_fast.py)
Pattern solve (the explicit intensity-domain solve; not "pass 2"): simcode/solve/solve_fast.solve_emitters_fast over the whole chunk
in batches of `solve_batch` rows (default 16384).
Timing keys: read, detector, head, solve (GPU busy time per stage from CUDA events; with streams the stage sums can exceed
the wall time) + wall.
"""
import json, time
import numpy as np, torch
import torch.nn.functional as F
from simcode.psf import TiffMovie
from simcode.processing.common import region, pattern_frames, CAND_PMIN

T = 6
PP = 21
THR0 = 0.2
from simcode import config as _cfg
DETECTOR = _cfg.model(_cfg.DETECTOR)
DETECTOR_PASS2 = _cfg.model(_cfg.DETECTOR_PASS2)
EMITTER_PASS2 = _cfg.model(_cfg.EMITTER_PASS2)
EMITTER = _cfg.model(_cfg.EMITTER)


# ------------------------------------------------------------------------------------------------ helpers
def nbr_map_padded(logit, b, iy, ix, x, y, lg):
    """= simcode.emitter.model.nbr_map on the kept (finite-logit) entries, computed on the padded top-k list without a boolean
    compaction (invalid entries: p = sigmoid(-inf) = 0, out-of-range stamp pixels: weight 0 at a clamped index)"""
    from simcode.emitter.model import OFFS, SIG_SPLAT
    B, H, W = logit.shape
    p = torch.sigmoid(lg.float())
    o = OFFS.to(logit.device)
    py = iy[:, None, None] + o[None, :, None]; px = ix[:, None, None] + o[None, None, :]
    v = p[:, None, None] * torch.exp(-((px - x[:, None, None]) ** 2 + (py - y[:, None, None]) ** 2) / (2 * SIG_SPLAT ** 2))
    ok = (py >= 0) & (py < H) & (px >= 0) & (px < W)
    v = torch.where(ok, v, torch.zeros_like(v))
    bb = b[:, None, None].expand_as(py)
    idx = (bb * H + py.clamp(0, H - 1)) * W + px.clamp(0, W - 1)
    sp = torch.zeros(B * H * W, device=logit.device)
    sp.index_put_((idx.reshape(-1),), v.reshape(-1), accumulate=True)
    return torch.stack([sp.reshape(B, H, W), torch.sigmoid(logit.float())], 1)


def gather_patches(bufP, rows, iy, ix, P=PP):
    """bufP [N0, C, H + P - 1, W + P - 1] zero-padded by P // 2; rows [n] (index into N0) or [n, C'] (per-channel rows,
    e.g. frames s0 + t) -> [n, C', P, P]; = layers.blocks.extract_patches on the unpadded buffer"""
    o = torch.arange(P, device=bufP.device)
    yy = (iy[:, None] + o[None])[:, None, :, None]
    xx = (ix[:, None] + o[None])[:, None, None, :]
    if rows.dim() == 1:
        C = bufP.shape[1]
        return bufP[rows[:, None, None, None], torch.arange(C, device=bufP.device)[None, :, None, None], yy, xx]
    return bufP[rows[:, :, None, None], 0, yy, xx]


def illum_patches(PWc, iy, ix, H, W, P=PP):
    """illumination patches [n, 2T, P, P] of the per-candidate window pattern PWc [n, T, 5] at the candidate pixels, as
    patches(illumination.illum_maps(PW, H, W), b, iy, ix) (zero outside the H x W region)"""
    h = P // 2
    o = torch.arange(P, device=PWc.device)
    yy = (iy[:, None] + o[None] - h).float()[:, None, :, None]                 # [n,1,P,1] region rows
    xx = (ix[:, None] + o[None] - h).float()[:, None, None, :]
    inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    q = PWc.float()[..., None, None]                                           # [n,T,5,1,1]
    a = q[:, :, 0] * xx + q[:, :, 1] * yy - q[:, :, 2]
    rel, m = q[:, :, 4], q[:, :, 3]
    il = torch.cat([rel * (1 + m * torch.cos(a)) - 1, rel * m * torch.sin(a)], 1)
    return torch.where(inside.expand_as(il[:, :1]).expand_as(il), il, torch.zeros_like(il))


class _Ev:
    """per-stage GPU time from CUDA events (no host sync until read)"""
    def __init__(self):
        self.ev = {}

    def rec(self, key, stream=None):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        s = stream or torch.cuda.current_stream()
        e0.record(s)
        return lambda: (e1.record(s), self.ev.setdefault(key, []).append((e0, e1)))

    def totals(self):
        torch.cuda.synchronize()
        return {k: sum(a.elapsed_time(b) for a, b in v) / 1e3 for k, v in self.ev.items()}


# ------------------------------------------------------------------------------------------------ queued front-end
class QueuedFront:
    """detector -> GPU queue -> fixed-size emitter-network batches (see the module doc); emitters: dict name -> EmitterNet (a
    pass-2 network, net.illum, also gets the illumination patches; a pass-2 detector, det.illum, the illumination maps)"""

    def __init__(self, detector, emitters, col, dev='cuda'):
        self.detector, self.col, self.dev = detector, col, dev
        self.emitters = dict(emitters)

    def _head_call(self, nm, pt, off, nbp, ilp):
        h = self.emitters[nm]
        psf = self.col(off) if self.col is not None else None
        if h.illum:
            return h(pt, None, off, psf, nbp, ilp)
        return h(pt, None, off, psf, nbp)

    @torch.no_grad()
    def run(self, frames, starts, k_cand, PWall=None, det_batch=64, emitter_batch=1024, streams=True, outs=('I', 'L', 'bg', 'x', 'y', 'lsxy', 'z', 'lsz', 'bz'),
            ev=None):
        """frames [F,S,S] (photons, chunk), starts: window starts (relative frame index), PWall [nW,T,5] (pass 2) ->
        dict of per-candidate GPU tensors (queue order = window order, top-k order within a window, as the originals):
        w (window index), iy, ix, x_det, y_det, logit + per head {name: {out: tensor}}"""
        from simcode.detector.select import select_subpixel
        from simcode.emitter.model import own_stamp
        dev = self.dev; ev = ev or _Ev()
        Fn, S, _ = frames.shape; h = PP // 2
        nW = len(starts)
        st_t = torch.tensor(starts, device=dev, dtype=torch.long)
        framesP = F.pad(frames, (h, h, h, h))[:, None]                          # [F,1,S+20,S+20]
        nbrP = torch.zeros(nW, 2, S + 2 * h, S + 2 * h, device=dev)
        s_det = torch.cuda.Stream() if streams else torch.cuda.current_stream()
        s_head = torch.cuda.Stream() if streams else torch.cuda.current_stream()
        main = torch.cuda.current_stream()
        s_det.wait_stream(main); s_head.wait_stream(main)
        Q = {k: [] for k in ('w', 'iy', 'ix', 'x', 'y', 'lg')}
        res = {nm: {k: [] for k in outs} for nm in self.emitters}
        pend = []                      # queued (not yet processed) candidate pieces: dicts of [m] tensors
        npend = 0; keep_alive = []

        def drain(final=False):
            nonlocal pend, npend
            while npend >= emitter_batch or (final and npend > 0):
                with torch.cuda.stream(s_head):
                    s_head.wait_stream(s_det)
                    cat = {k: torch.cat([p_[k] for p_ in pend]) for k in pend[0]}
                    take = min(emitter_batch, npend)
                    cur = {k: v[:take] for k, v in cat.items()}
                    rest = {k: v[take:] for k, v in cat.items()}
                    if take < emitter_batch:                  # pad the final batch to the static size (repeat the last entry)
                        pad = emitter_batch - take
                        cur = {k: torch.cat([v, v[-1:].expand(pad, *v.shape[1:])]) for k, v in cur.items()}
                    done = ev.rec('head', s_head)
                    w, iy, ix, xs, ys, lg = (cur[k] for k in ('w', 'iy', 'ix', 'x', 'y', 'lg'))
                    off = torch.stack([xs - ix.float(), ys - iy.float()], 1)
                    rows = st_t[w][:, None] + torch.arange(T, device=dev)[None]
                    pt = gather_patches(framesP, rows, iy, ix)
                    q = gather_patches(nbrP, w, iy, ix)
                    p = torch.sigmoid(lg)
                    stmp = own_stamp(off, p.float(), iy, ix, S, S)
                    nbp = torch.stack([(q[:, 0] - stmp).clamp(min=0), q[:, 1]], 1)
                    ilp = illum_patches(PWall[w], iy, ix, S, S) if PWall is not None else None
                    for nm in self.emitters:
                        o = self._head_call(nm, pt, off, nbp, ilp)
                        o['x'] = xs + o['dxy'][:, 0]; o['y'] = ys + o['dxy'][:, 1]
                        for k in outs:
                            res[nm][k].append(o[k][:take].float())
                    done()
                    keep_alive.append(cur)
                    pend = [rest]; npend -= take

        for bi in range(0, nW, det_batch):
            wi = list(range(bi, min(bi + det_batch, nW)))
            with torch.cuda.stream(s_det):
                done = ev.rec('detector', s_det)
                x = torch.stack([frames[starts[i]:starts[i] + T] for i in wi])
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    if self.detector.illum:
                        from simcode.pattern import illumination as IL
                        dd = self.detector(x, IL.illum_maps(PWall[wi[0]:wi[-1] + 1], S, S))
                    else:
                        dd = self.detector(x)
                lgm = dd['logit'].float(); lgm = lgm[:, 0] if lgm.dim() == 4 else lgm
                c = select_subpixel(lgm, dd['dxy'].float(), k_cand, p_min=THR0, radius=0.7)
                nb = nbr_map_padded(lgm, c['b'], c['iy'], c['ix'], c['x'], c['y'], c['logit'])
                nbrP[bi:bi + len(wi), :, h:h + S, h:h + S] = nb
                sel = torch.nonzero(torch.isfinite(c['logit']) & (torch.sigmoid(c['logit']) >= CAND_PMIN))[:, 0]   # host sync (det stream)
                piece = dict(w=c['b'][sel] + bi, iy=c['iy'][sel], ix=c['ix'][sel], x=c['x'][sel], y=c['y'][sel], lg=c['logit'][sel])
                done()
            for k in Q: Q[k].append(piece[k])
            if len(sel):
                pend.append(piece); npend += len(sel)
            drain()
        drain(final=True)
        main.wait_stream(s_head); main.wait_stream(s_det)
        out = {k: torch.cat(v) if v else torch.zeros(0, device=dev) for k, v in Q.items()}
        out['emitters'] = {nm: {k: torch.cat(v) if v else torch.zeros(0, device=dev) for k, v in r.items()} for nm, r in res.items()}
        out['_keep'] = keep_alive
        return out


def _flags(I, ws, gx, gy, l0, last_start, crop, own0, own1, dev):
    """time centroid + the `out` rule of front.Pass1Front.run_chunk (central half, first / last loaded window ends, crop, own frames)"""
    Ip = I.clamp(min=0)
    tc = (Ip * torch.arange(T, device=dev).float()[None]).sum(1) / Ip.sum(1).clamp(min=1e-3)
    central = (tc >= T / 4 - 0.5) & (tc < 3 * T / 4 - 0.5)
    central |= (ws == l0) & (tc < T / 4 - 0.5)
    central |= (ws == l0 + last_start) & (tc >= 3 * T / 4 - 0.5)
    x0c, y0c, szc = crop
    inside = (gx >= x0c) & (gx < x0c + szc) & (gy >= y0c) & (gy < y0c + szc)
    fabs = ws.float() + tc
    return tc, fabs, central & inside & (fabs >= own0) & (fabs < own1)


def _np(r):
    return {k_: v.float().cpu().numpy() if v.is_floating_point() else v.cpu().numpy() for k_, v in r.items()}


# ------------------------------------------------------------------------------------------------ pass 1
class Pass1FrontFast:
    """= front.Pass1Front (same constructor arguments) with the queued front-end"""

    def __init__(self, calib_path, emitter=EMITTER, detector=DETECTOR, dev='cuda'):
        from simcode.emitter import model as EM
        from simcode.detector import model as DM
        from simcode.simulate.simulator import Simulator
        self.EM = EM
        self.detector, _ = DM.load(detector)
        self.emitter = EM.load(emitter)
        self.col = EM.PSFCol(Simulator(calib_path, H=32, W=32, T=T, device=dev))
        self.dev = dev
        self.front = QueuedFront(self.detector, {'h': self.emitter}, self.col, dev)

    @torch.no_grad()
    def run_chunk(self, movie: TiffMovie, cal, f0, f1, crop=(5, 5, 110), margin=5, pad=12, det_batch=64, emitter_batch=1024, streams=True,
                  timing=None, keep_gpu=False):
        dev = self.dev
        timing = timing if timing is not None else {}
        tw = time.time()
        X0, Y0, S = region(movie, crop, margin)
        own0, own1 = f0, min(f1, movie.n)
        l0 = max(f0 - pad, 0) // 6 * 6; l1 = min(own1 + pad, movie.n)
        t_ = time.time()
        mov = movie.read(l0, l1, Y0, X0, S)
        off = torch.tensor(cal['offset'][Y0:Y0 + S, X0:X0 + S], device=dev)
        frames = (torch.tensor(mov, device=dev) - off) * cal['gain']
        torch.cuda.synchronize(); timing['read'] = timing.get('read', 0) + time.time() - t_
        Fn = frames.shape[0]
        starts = list(range(0, Fn - T + 1, T // 2))
        k_cand = int(640 * S * S / 64 ** 2)
        ev = _Ev()
        q = self.front.run(frames, starts, k_cand, det_batch=det_batch, emitter_batch=emitter_batch, streams=streams, ev=ev)
        oh = dict(q['emitters']['h']); oh.update(self.EM.emitter_z(self.emitter, oh))
        I = oh['I']
        ws = torch.tensor(starts, device=dev)[q['w']] + l0
        gx, gy = oh['x'] + X0, oh['y'] + Y0
        tc, fabs, out = _flags(I, ws, gx, gy, l0, starts[-1], crop, own0, own1, dev)
        sp = oh['lsxy'].exp()
        sI = (oh['L'] ** 2).sum(-1).sqrt()
        r = dict(wstart=ws, x=gx, y=gy, sx=sp[:, 0], sy=sp[:, 1], z=oh['z'], sz=oh['lsz'].exp(), bzx=oh['bz'][:, 0], bzy=oh['bz'][:, 1],
                 p=torch.sigmoid(q['lg'].float()),
                 phot=I.sum(1), phot_t=I, I=I, sI=sI, bg=oh['bg'], L=oh['L'], out=out, tc=tc, frame=fabs,
                 x_det=q['x'] + X0, y_det=q['y'] + Y0)
        R = r if keep_gpu else _np(r)
        if not keep_gpu: R['wstart'] = R['wstart'].astype(np.int64)
        for k, v in ev.totals().items(): timing[k] = timing.get(k, 0) + v
        timing['wall'] = timing.get('wall', 0) + time.time() - tw
        del q, oh, frames
        if not keep_gpu: torch.cuda.empty_cache()     # shared GPUs: return the chunk buffers to the driver
        return R, dict(X0=X0, Y0=Y0, S=S, own=(own0, own1), loaded=(l0, l1))


# ------------------------------------------------------------------------------------------------ pattern solve (pass-1 rows)
@torch.no_grad()
def run_chunk_pattern_solve_fast(R, info, pattern, solve_cal, dev='cuda', solve_batch=16384, timing=None, compile='auto'):
    """pattern solve (solve_fast.solve_emitters_fast) of the stored pass-1 head outputs of the `out` rows, results kept on the GPU until the
    end of the chunk"""
    from simcode.solve import solve_fast as MF
    timing = timing if timing is not None else {}
    t_ = time.time()
    cal = json.load(open(solve_cal)) if isinstance(solve_cal, str) else dict(solve_cal)
    keep = R['out'].astype(bool)
    X0, Y0 = info['X0'], info['Y0']
    ws = R['wstart'][keep]
    n = int(keep.sum())
    idx_all = np.nonzero(keep)[0]
    tt = lambda a: torch.tensor(np.asarray(a), device=dev, dtype=torch.float32)
    uw, inv = np.unique(ws, return_inverse=True)
    PW = pattern_frames(pattern, (uw[:, None] + np.arange(T)[None]).reshape(-1), X0, Y0, dev).reshape(len(uw), T, 5)
    G = {k: tt(R[k][idx_all]) for k in ('I', 'L', 'x', 'y', 'sx', 'sy', 'x_det', 'y_det')}
    from simcode.processing.front import z_inputs
    from simcode.solve import z_conditioning as ZC
    G.update(z_inputs(R, idx_all, tt))
    binv = torch.tensor(inv, device=dev)
    acc = {k: [] for k in ('r', 's_cal', 'kin', 'p', 'q', 'dnll', 'r_map', 'z_hy0', 'sz_cal', 'cov_xz', 'cov_yz', 'z_prior')}
    for c0 in range(0, n, solve_batch):
        sl = slice(c0, c0 + solve_batch)
        o = dict(I=G['I'][sl], L=G['L'][sl], x=G['x'][sl] - X0, y=G['y'][sl] - Y0, lsxy=torch.log(torch.stack([G['sx'][sl], G['sy'][sl]], 1)),
                 x_det=G['x_det'][sl] - X0, y_det=G['y_det'][sl] - Y0, z=G['z'][sl], lsz=G['lsz'][sl], bz=G['bz'][sl])
        so = MF.solve_emitters_fast(o, PW, binv[sl], cal, compile=compile)
        so.update(ZC.z_columns(o, so))
        for k in acc: acc[k].append(so.get(k, so['r']) if k != 'dnll' else so.get('dnll', torch.full_like(so['q'], float('nan'))))
    A = {k: torch.cat(v).float().cpu().numpy() if v else np.zeros((0, 2) if k in ('r', 's_cal', 'r_map') else 0, np.float32) for k, v in acc.items()}
    out = {}
    out['x_hy0'] = A['r'][:, 0] + X0; out['y_hy0'] = A['r'][:, 1] + Y0
    out['sig_hy0'] = A['s_cal'].mean(1) if n else np.zeros(0, np.float32)
    out['kin_hy'] = A['kin'].astype(np.float32)
    out['fringe_hy'] = np.full(n, np.nan, np.float32)
    out['pf_hy'] = A['p'].astype(np.float32)
    out['chi2_hy'] = A['q'].astype(np.float32)
    out['B_hy'] = np.full(n, np.nan, np.float32); out['Klin_hy'] = np.full(n, np.nan, np.float32)
    out['sx_cal'] = A['s_cal'][:, 0]; out['sy_cal'] = A['s_cal'][:, 1]
    out['q_dof'] = A['q'].astype(np.float32)
    out['phot'] = R['phot'][idx_all].astype(np.float32)
    out['x_prior'] = R['x'][idx_all].astype(np.float32); out['y_prior'] = R['y'][idx_all].astype(np.float32)
    out['dnll'] = A['dnll'].astype(np.float32)
    out['x_map'] = A['r_map'][:, 0] + X0; out['y_map'] = A['r_map'][:, 1] + Y0
    for k in ('z_hy0', 'sz_cal', 'cov_xz', 'cov_yz', 'z_prior'): out[k] = A[k]
    out = {k: np.asarray(v, np.float32) for k, v in out.items()}
    torch.cuda.synchronize(); timing['solve'] = timing.get('solve', 0) + time.time() - t_
    out.update(wstart=ws, frame=R['frame'][keep], p=R['p'][keep], I_net=R['I'][keep])
    return out


# ------------------------------------------------------------------------------------------------ pattern-aware pass 2
class Pass2FrontFast:
    """= pass2_front.Pass2Front (same constructor arguments) with the queued front-end and one chunk-wide pattern solve"""

    def __init__(self, calib_path, emitter=None, detector=None, dev='cuda', compile='auto'):
        from simcode.emitter import model as EM
        from simcode.detector import model as DM
        from simcode.solve import z_conditioning as ZC
        from simcode.simulate.simulator import Simulator
        emitter = emitter or EMITTER_PASS2; detector = detector or DETECTOR_PASS2
        self.ZC = ZC
        self.detector, _ = DM.load(detector)        # a pattern-free (pass-1) detector or network also works (detector-only pass 2)
        self.emitter = EM.load(emitter)
        self.col = EM.PSFCol(Simulator(calib_path, H=32, W=32, T=T, device=dev))
        self.dev = dev
        self.front = QueuedFront(self.detector, dict(h=self.emitter), self.col, dev)
        self.compile = compile          # pattern solve LM step

    @torch.no_grad()
    def run_chunk(self, movie: TiffMovie, cal, f0, f1, pattern, solve_cal, p1, p1_cal, crop=(5, 5, 110), margin=5, pad=12, det_batch=64,
                  emitter_batch=1024, streams=True, solve_batch=16384, timing=None, return_head=False):
        """p1 = the chunk's stored pass-1 rows (pass1_<f0>.npz), p1_cal = the pass-1 solve calibration: z from pass 1
        (= pass2_front.Pass2Front.run_chunk; solve/z_conditioning.pass1_z_columns)"""
        from simcode.solve import solve_fast as MF
        dev = self.dev
        timing = timing if timing is not None else {}
        tw = time.time()
        scal = json.load(open(solve_cal)) if isinstance(solve_cal, str) else dict(solve_cal)
        front = self.front
        X0, Y0, S = region(movie, crop, margin)
        own0, own1 = f0, min(f1, movie.n)
        l0 = max(f0 - pad, 0) // 6 * 6; l1 = min(own1 + pad, movie.n)
        t_ = time.time()
        mov = movie.read(l0, l1, Y0, X0, S)
        off = torch.tensor(cal['offset'][Y0:Y0 + S, X0:X0 + S], device=dev)
        frames = (torch.tensor(mov, device=dev) - off) * cal['gain']
        torch.cuda.synchronize(); timing['read'] = timing.get('read', 0) + time.time() - t_
        Fn = frames.shape[0]
        starts = list(range(0, Fn - T + 1, T // 2))
        PF = pattern_frames(pattern, np.arange(l0, l1), X0, Y0, dev)
        PWall = torch.stack([PF[s0:s0 + T] for s0 in starts])                   # [nW,T,5]
        k_cand = int(640 * S * S / 64 ** 2)
        ev = _Ev()
        outs = ('I', 'L', 'bg', 'x', 'y', 'lsxy')
        q = front.run(frames, starts, k_cand, PWall=PWall, det_batch=det_batch, emitter_batch=emitter_batch, streams=streams, outs=outs, ev=ev)
        oh = dict(q['emitters']['h']); oh['x_det'] = q['x']; oh['y_det'] = q['y']
        I = oh['I']
        ws = torch.tensor(starts, device=dev)[q['w']] + l0
        tc, fabs, keep = _flags(I, ws, oh['x'] + X0, oh['y'] + Y0, l0, starts[-1], crop, own0, own1, dev)
        k_ = torch.nonzero(keep)[:, 0]
        o = {kk: v[k_] for kk, v in oh.items()}
        bw = q['w'][k_]
        done = ev.rec('solve')
        acc = {k: [] for k in ('r', 's_cal', 'q', 'kin')}
        for c0 in range(0, len(k_), solve_batch):
            sl = slice(c0, c0 + solve_batch)
            oc = {kk: v[sl] for kk, v in o.items()}
            so = MF.solve_emitters_fast(oc, PWall, bw[sl], scal, compile=self.compile)
            for k in acc: acc[k].append(so[k])
        done()
        so = {k: torch.cat(v) if v else torch.zeros((0, 2) if k in ('r', 's_cal') else 0, device=dev) for k, v in acc.items()}
        sp = oh['lsxy'][k_].exp()
        ph = I[k_].sum(1)
        p = torch.sigmoid(q['lg'])
        r = dict(wstart=ws[k_], x_hy0=so['r'][:, 0] + X0, y_hy0=so['r'][:, 1] + Y0, sx_cal=so['s_cal'][:, 0], sy_cal=so['s_cal'][:, 1],
                 sig_hy0=so['s_cal'].mean(1), q_dof=so['q'], kin_hy=so['kin'].float(), phot=ph, p=p[k_],
                 x_prior=oh['x'][k_] + X0, y_prior=oh['y'][k_] + Y0, x_det=oh['x_det'][k_] + X0, y_det=oh['y_det'][k_] + Y0,
                 x_dir=oh['x'][k_] + X0, y_dir=oh['y'][k_] + Y0, sx_dir=sp[:, 0], sy_dir=sp[:, 1],
                 frame=fabs[k_], I_net=I[k_])
        r.update(self.ZC.pass1_z_columns(p1, p1_cal, r['wstart'], r['x_det'], r['y_det'], torch.stack([r['x_hy0'], r['y_hy0']], 1),
                                          torch.stack([r['sx_cal'], r['sy_cal']], 1)))      # z from pass 1 (the pass-2 head's z is not used)
        R = _np(r); R['wstart'] = R['wstart'].astype(np.int64)
        for k, v in ev.totals().items(): timing[k] = timing.get(k, 0) + v
        timing['wall'] = timing.get('wall', 0) + time.time() - tw
        if return_head:
            return R, dict(o={kk: v for kk, v in o.items()}, PWall=PWall, bw=bw, scal=scal)
        del q, oh, o, frames
        torch.cuda.empty_cache()
        return R
