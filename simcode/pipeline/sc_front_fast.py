"""
Fast (opt-in) versions of the sc front-end chunks (2026-11-10, throughput; run_uncalibrated --fast). Same networks, same
candidate rules, same outputs and file formats as simcode/pipeline/sc_front.py (pass 1, pattern solve) and
simcode/p2/front.py (pattern-aware pass 2); the originals are unchanged.

Queued fixed-size batching (shared by pass 1 and pass 2):
  detector  on `det_batch` windows at a time (default 64) -> padded top-k -> neighbour maps from the padded list (no
            boolean compaction needed; invalid entries carry p = 0) -> the p >= 0.5 candidates (one index compaction per
            detector batch) are appended to a GPU queue of (window index, pixel, sub-pixel point, logit); the chunk's frames
            and neighbour maps stay on the GPU (zero-padded once per chunk), so a queue entry needs no patch storage
  head      drains the queue in fixed batches of `head_batch` (default 1024; the last one padded by repeating the last
            entry) -> static shapes; patches are gathered from the padded chunk buffers (identical values to sc.patches);
            pass 2's illumination patches are evaluated analytically per candidate (= illum.illum_maps + zero padding)
  streams   detector and head on two CUDA streams (the head of batch k overlaps the detector of batch k+1; the only host
            syncs are the per-detector-batch candidate counts, on the detector stream)
  outputs   written into preallocated GPU tensors, post-processed (time centroid, central / inside / own flags) once per chunk,
            copied to the CPU once per chunk
  compile   the head is NOT compiled by default: torch.compile of SCHead4 changed the outputs far beyond the bf16 noise on the
            real 0.5 nM chunk (I up to 275 ph, p99.9 59 ph vs 1.4 ph run-to-run) -> `compile_head` ('off' default) exists only
            for diagnostics; `compile` ('auto' | 'on' | 'off') applies to the pattern solve's LM step (msolve_fast)
Pattern solve (the explicit intensity-domain solve; not "pass 2"): simcode/hd/msolve_fast.sc_solve_fast over the whole chunk
in batches of `solve_batch` rows (default 16384), exact options by default (solve_opts for the approximate ones).
Timing keys: read, detector, head, solve (GPU busy time per stage from CUDA events; with streams the stage sums can exceed
the wall time) + wall.
"""
import json, time
import numpy as np, torch
import torch.nn.functional as F
from simcode.pipeline.common import Movie, region, pattern_frames

T = 6
PP = 21
THR0 = 0.2
from simcode import config as _cfg
DET_N1 = _cfg.model(_cfg.DET)
HEAD = _cfg.model(_cfg.HEAD)


# ------------------------------------------------------------------------------------------------ helpers
def nbr_map_padded(logit, b, iy, ix, x, y, lg):
    """= simcode.hd.sc4.nbr_map on the kept (finite-logit) entries, computed on the padded top-k list without a boolean
    compaction (invalid entries: p = sigmoid(-inf) = 0, out-of-range stamp pixels: weight 0 at a clamped index)"""
    from simcode.hd.sc4 import OFFS, SIG_SPLAT
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
    e.g. frames s0 + t) -> [n, C', P, P]; = sc.patches on the unpadded buffer"""
    o = torch.arange(P, device=bufP.device)
    yy = (iy[:, None] + o[None])[:, None, :, None]
    xx = (ix[:, None] + o[None])[:, None, None, :]
    if rows.dim() == 1:
        C = bufP.shape[1]
        return bufP[rows[:, None, None, None], torch.arange(C, device=bufP.device)[None, :, None, None], yy, xx]
    return bufP[rows[:, :, None, None], 0, yy, xx]


def illum_patches(PWc, iy, ix, H, W, P=PP):
    """illumination patches [n, 2T, P, P] of the per-candidate window pattern PWc [n, T, 5] at the candidate pixels, as
    patches(illum.illum_maps(PW, H, W), b, iy, ix) (zero outside the H x W region)"""
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
    """detector -> GPU queue -> fixed-size head batches (see the module doc); heads: dict name -> (module, kind) with kind
    'sc' (sc.apply_head inputs: patch, off, psf, nbr) or 'p2' (+ illumination patches)"""

    def __init__(self, det, needs_il, heads, col, dev='cuda', compile_head='off', channels_last=False):
        from simcode import fastutil
        self.det, self.needs_il, self.col, self.dev = det, needs_il, col, dev
        self.heads = {}
        for nm, (h, kind) in heads.items():
            if channels_last: h = h.to(memory_format=torch.channels_last)
            if compile_head in ('trunk', 'trunk_on'):
                # (2026-11-11, speed branch) torch.compile of the bf16 CNN trunk + MLP only (SCHead4.trunk); the fp32 read-out,
                # covariance and intensity algebra stay eager. 'trunk' = compile when available (else eager), 'trunk_on' =
                # compile or raise. The module itself is not modified (a reference path sharing it stays eager).
                bm = h.trunk
                tf = fastutil.maybe_compile(bm, 'auto' if compile_head == 'trunk' else 'on', dynamic=False)
                self.heads[nm] = (h, kind, h, tf if tf is not bm else None)
                continue
            if compile_head == 'blk':
                # only the residual conv stack compiled (opt-in): head 30 % faster on a busy GPU, but I / x differences 4x the
                # bf16 run-to-run noise on the real 0.5 nM chunk (dI median 0.30 vs 0.08 ph, max 4.3 vs 1.2 ph) -> not default
                if fastutil.resolve('auto') and not hasattr(h, '_blk_eager'):
                    h._blk_eager = h.blk; h.blk = fastutil.maybe_compile(h.blk, 'auto', dynamic=False)
                self.heads[nm] = (h, kind, h, None)
            else:
                self.heads[nm] = (fastutil.maybe_compile(h, compile_head, dynamic=False), kind, h, None)

    def _head_call(self, nm, pt, off, nbp, ilp):
        hc, kind, h, tf = (self.heads[nm] + (None,))[:4]
        psf = self.col(off) if self.col is not None else None
        kw = {} if tf is None else dict(trunk_fn=tf)
        if kind == 'p2':
            return hc(pt, None, off, psf, nbp, ilp, **kw)
        if getattr(h, 'nbr', False):
            return hc(pt, None, off, psf, nbp, **kw)
        return hc(pt, None, off, psf)

    @torch.no_grad()
    def run(self, frames, starts, k_cand, PWall=None, det_batch=64, head_batch=1024, streams=True, outs=('I', 'L', 'bg', 'x', 'y', 'lsxy'),
            ev=None):
        """frames [F,S,S] (photons, chunk), starts: window starts (relative frame index), PWall [nW,T,5] (pass 2) ->
        dict of per-candidate GPU tensors (queue order = window order, top-k order within a window, as the originals):
        w (window index), iy, ix, x_det, y_det, logit + per head {name: {out: tensor}}"""
        from simcode.det.model import select_subpixel
        from simcode.hd.sc4 import own_stamp
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
        res = {nm: {k: [] for k in outs} for nm in self.heads}
        pend = []                      # queued (not yet processed) candidate pieces: dicts of [m] tensors
        npend = 0; keep_alive = []
        if PWall is not None:
            Km_il = None

        def drain(final=False):
            nonlocal pend, npend
            while npend >= head_batch or (final and npend > 0):
                with torch.cuda.stream(s_head):
                    s_head.wait_stream(s_det)
                    cat = {k: torch.cat([p_[k] for p_ in pend]) for k in pend[0]}
                    take = min(head_batch, npend)
                    cur = {k: v[:take] for k, v in cat.items()}
                    rest = {k: v[take:] for k, v in cat.items()}
                    if take < head_batch:                  # pad the final batch to the static size (repeat the last entry)
                        pad = head_batch - take
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
                    for nm in self.heads:
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
                    if self.needs_il:
                        from simcode.p2 import illum as IL
                        dd = self.det(x, IL.illum_maps(PWall[wi[0]:wi[-1] + 1], S, S))
                    else:
                        dd = self.det(x)
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
        out['heads'] = {nm: {k: torch.cat(v) if v else torch.zeros(0, device=dev) for k, v in r.items()} for nm, r in res.items()}
        out['_keep'] = keep_alive
        return out


def _flags(I, ws, gx, gy, l0, last_start, crop, own0, own1, dev):
    """time centroid + the `out` rule of sc_front.run_chunk (central half, first / last loaded window ends, crop, own frames)"""
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
CAND_PMIN = 0.5     # stored-candidate cut (head runs on p >= CAND_PMIN); run_uncalibrated --cand_pmin (2026-11-10, default unchanged)


class Pass1SCFast:
    """= sc_front.Pass1SC (same constructor arguments) with the queued front-end"""

    def __init__(self, calib_path, head=HEAD, det=DET_N1, dev='cuda', compile_head='off', channels_last=False, compile=None):
        from simcode.hd import sc as SCM
        from simcode.sim.simulator import Simulator
        self.SCM = SCM
        self.det, _ = SCM.load_det_front(det)
        self.det.eval()
        self.head = SCM.load(head)
        self.col = SCM.PSFCol(Simulator(calib_path, H=32, W=32, T=T, device=dev))
        self.dev = dev
        self.front = QueuedFront(self.det, False, {'h': (self.head, 'sc')}, self.col, dev, compile_head=compile_head, channels_last=channels_last)

    @torch.no_grad()
    def run_chunk(self, movie: Movie, cal, f0, f1, crop=(5, 5, 110), margin=5, pad=12, det_batch=64, head_batch=1024, streams=True,
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
        q = self.front.run(frames, starts, k_cand, det_batch=det_batch, head_batch=head_batch, streams=streams, ev=ev)
        oh = q['heads']['h']
        I = oh['I']
        ws = torch.tensor(starts, device=dev)[q['w']] + l0
        gx, gy = oh['x'] + X0, oh['y'] + Y0
        tc, fabs, out = _flags(I, ws, gx, gy, l0, starts[-1], crop, own0, own1, dev)
        sp = oh['lsxy'].exp()
        sI = (oh['L'] ** 2).sum(-1).sqrt()
        r = dict(wstart=ws, x=gx, y=gy, sx=sp[:, 0], sy=sp[:, 1], z=torch.full_like(gx, 0.05), p=torch.sigmoid(q['lg'].float()),
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
def run_chunk_pattern_solve_fast(R, info, pattern, sc_cal, dev='cuda', solve_batch=16384, timing=None, solve_opts=None, field=None):
    """= sc_front.run_chunk_pass2 (same output columns) with msolve_fast.sc_solve_fast, results kept on the GPU until the
    end of the chunk. field (2026-11-10, opt-in, default None = unchanged): a pattern-field dict (pattern_field.py; e.g. a
    depth-only field) applied per row as in sc_front.run_chunk_pass2 (phase += phi_a(r_prior), depth x f_a(r_prior))."""
    from simcode.hd import msolve_fast as MF
    timing = timing if timing is not None else {}
    t_ = time.time()
    cal = json.load(open(sc_cal)) if isinstance(sc_cal, str) else dict(sc_cal)
    opts = dict(solve_opts or {})
    keep = R['out'].astype(bool)
    X0, Y0 = info['X0'], info['Y0']
    ws = R['wstart'][keep]
    n = int(keep.sum())
    idx_all = np.nonzero(keep)[0]
    tt = lambda a: torch.tensor(np.asarray(a), device=dev, dtype=torch.float32)
    uw, inv = np.unique(ws, return_inverse=True)
    PW = pattern_frames(pattern, (uw[:, None] + np.arange(T)[None]).reshape(-1), X0, Y0, dev).reshape(len(uw), T, 5)
    G = {k: tt(R[k][idx_all]) for k in ('I', 'L', 'x', 'y', 'sx', 'sy', 'x_det', 'y_det')}
    binv = torch.tensor(inv, device=dev)
    acc = {k: [] for k in ('r', 's_cal', 'kin', 'p', 'q', 'dnll', 'r_map')}
    for c0 in range(0, n, solve_batch):
        sl = slice(c0, c0 + solve_batch)
        o = dict(I=G['I'][sl], L=G['L'][sl], x=G['x'][sl] - X0, y=G['y'][sl] - Y0, lsxy=torch.log(torch.stack([G['sx'][sl], G['sy'][sl]], 1)),
                 x_det=G['x_det'][sl] - X0, y_det=G['y_det'][sl] - Y0)
        if field is not None:
            from simcode.pipeline.pattern_field import field_at
            ii = idx_all[sl]
            dp, df = field_at(field, R['x'][ii].astype(np.float64), R['y'][ii].astype(np.float64))
            PWr = PW[binv[sl]].clone()                                                # [n, T, 5] per-row pattern
            ang = torch.tensor(((ws[sl][:, None] + np.arange(T)[None]) % 6) // 3, device=dev)
            PWr[..., 2] = PWr[..., 2] + torch.gather(tt(dp), 1, ang); PWr[..., 3] = (PWr[..., 3] * torch.gather(tt(df), 1, ang)).clamp(0, 1)
            so = MF.sc_solve_fast(o, PWr, torch.arange(len(ii), device=dev), cal, **opts)
        else:
            so = MF.sc_solve_fast(o, PW, binv[sl], cal, **opts)
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
    out = {k: np.asarray(v, np.float32) for k, v in out.items()}
    torch.cuda.synchronize(); timing['solve'] = timing.get('solve', 0) + time.time() - t_
    out.update(wstart=ws, frame=R['frame'][keep], p=R['p'][keep], I_net=R['I'][keep])
    return out


# ------------------------------------------------------------------------------------------------ pattern-aware pass 2
class Pass2P2Fast:
    """= p2.front.Pass2P2 (same constructor arguments) with the queued front-end and one chunk-wide pattern solve"""

    def __init__(self, calib_path, head=None, det=None, pf_head=None, dev='cuda', compile='off', channels_last=False, compile_head='off'):
        from simcode.hd import sc as SCM
        from simcode.p2 import head2 as H2, front as FR
        from simcode.sim.simulator import Simulator
        head = head or FR.HEAD_P2; det = det or FR.DET_P2
        self.SCM, self.H2 = SCM, H2
        self.det, self.needs_il = H2.load_det(det)
        self.head = H2.load(head) if ('p2_head' in head or H2.is_p2_head(head)) else SCM.load(head)
        self.pf_head = SCM.load(pf_head) if pf_head else None
        self.col = SCM.PSFCol(Simulator(calib_path, H=32, W=32, T=T, device=dev))
        self.dev = dev
        kind = 'p2' if getattr(self.head, 'p2', False) else 'sc'
        self.fronts = {}
        self._mk = lambda use_pf: QueuedFront(self.det, self.needs_il,
                                              dict(h=(self.head, kind), **({'pf': (self.pf_head, 'sc')} if use_pf else {})),
                                              self.col, dev, compile_head=compile_head, channels_last=channels_last)
        self.compile = compile          # pattern solve LM step

    @torch.no_grad()
    def run_chunk(self, movie: Movie, cal, f0, f1, pattern, sc_cal, crop=(5, 5, 110), margin=5, pad=12, det_batch=64, head_batch=1024,
                  streams=True, solve_batch=16384, timing=None, solve_opts=None, return_head=False):
        from simcode.hd import msolve_fast as MF
        dev = self.dev; SCM = self.SCM
        timing = timing if timing is not None else {}
        tw = time.time()
        scal = json.load(open(sc_cal)) if isinstance(sc_cal, str) else dict(sc_cal)
        prior = scal.get('p2_prior', 'own')
        use_pf = self.pf_head is not None and prior == 'pf'
        if use_pf not in self.fronts: self.fronts[use_pf] = self._mk(use_pf)
        front = self.fronts[use_pf]
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
        q = front.run(frames, starts, k_cand, PWall=PWall, det_batch=det_batch, head_batch=head_batch, streams=streams, outs=outs, ev=ev)
        oh = dict(q['heads']['h']); oh['x_det'] = q['x']; oh['y_det'] = q['y']
        if use_pf:
            pf = q['heads']['pf']; oh.update(pf_x=pf['x'], pf_y=pf['y'], pf_lsxy=pf['lsxy'])
        I = oh['I']
        ws = torch.tensor(starts, device=dev)[q['w']] + l0
        tc, fabs, keep = _flags(I, ws, oh['x'] + X0, oh['y'] + Y0, l0, starts[-1], crop, own0, own1, dev)
        k_ = torch.nonzero(keep)[:, 0]
        o = {kk: v[k_] for kk, v in oh.items()}
        if use_pf:
            o = dict(o, x=o['pf_x'], y=o['pf_y'], lsxy=o['pf_lsxy'])
        bw = q['w'][k_]
        done = ev.rec('solve')
        acc = {k: [] for k in ('r', 's_cal', 'q', 'kin')}
        for c0 in range(0, len(k_), solve_batch):
            sl = slice(c0, c0 + solve_batch)
            so = MF.sc_solve_fast({kk: v[sl] for kk, v in o.items()}, PWall, bw[sl], scal, **dict(dict(compile=self.compile), **(solve_opts or {})))
            for k in acc: acc[k].append(so[k])
        done()
        so = {k: torch.cat(v) if v else torch.zeros((0, 2) if k in ('r', 's_cal') else 0, device=dev) for k, v in acc.items()}
        sp = oh['lsxy'][k_].exp()
        ph = I[k_].sum(1)
        fdir = torch.ones_like(ph)
        if 'kap_bins_dir' in scal:
            fdir = SCM._binval(scal['kap_bins_dir'], 'kap_p', ph) * SCM._binval(scal['sig_ph_dir'], 'f', ph)
        po = torch.tensor(scal.get('prior_off_dir', [0.0, 0.0]), device=dev)
        p = torch.sigmoid(q['lg'])
        r = dict(wstart=ws[k_], x_hy0=so['r'][:, 0] + X0, y_hy0=so['r'][:, 1] + Y0, sx_cal=so['s_cal'][:, 0], sy_cal=so['s_cal'][:, 1],
                 sig_hy0=so['s_cal'].mean(1), q_dof=so['q'], kin_hy=so['kin'].float(), phot=ph, p=p[k_],
                 x_prior=oh['x'][k_] + X0, y_prior=oh['y'][k_] + Y0, x_det=oh['x_det'][k_] + X0, y_det=oh['y_det'][k_] + Y0,
                 x_dir=oh['x'][k_] - po[0] + X0, y_dir=oh['y'][k_] - po[1] + Y0, sx_dir=sp[:, 0] * fdir, sy_dir=sp[:, 1] * fdir,
                 frame=fabs[k_], I_net=I[k_])
        R = _np(r); R['wstart'] = R['wstart'].astype(np.int64)
        for k, v in ev.totals().items(): timing[k] = timing.get(k, 0) + v
        timing['wall'] = timing.get('wall', 0) + time.time() - tw
        if return_head:
            return R, dict(o={kk: v for kk, v in o.items()}, PWall=PWall, bw=bw, scal=scal)
        del q, oh, o, frames
        torch.cuda.empty_cache()
        return R
