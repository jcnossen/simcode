"""Per-stage position estimates of a SIMCODE v2 run for the pixel-bias diagnosis (no training).

For one run directory (simcode.run output with pass1_<f0>.npz, pass2_est_<f0>.npz, pass2_est2_<f0>.npz, pattern_est.pt):
  pass 1  the stored head outputs (pass1_<f0>.npz: I, L, x, y, sx, sy) -> production pattern solve (check against pass2_est)
          and a FRINGE-ONLY solve: same intensities, covariance, pattern and calibration, but the position prior made flat
          (sigma 1e3 px) and a single start at the production solution (nf = 0, box 0.5 px), i.e. the local optimum of the
          intensity likelihood alone (no pull of the network position prior); its Fisher sigma s_fringe.
  pass 2  the pass-2 networks are re-run on the movie (front_fast.Pass2FrontFast.run_chunk(return_head=True), same model / pattern /
          calibration as the run) -> production solve (checked against pass2_est2) and the same fringe-only solve.
Also stored: the prior sigma the solve uses (exp(lsxy) kap_p) and the production Fisher sigma, for the prior-weight decomposition.
Model files are read from the run's run_config.json; nothing in the run directory is written.

usage: CUDA_VISIBLE_DEVICES=2 python analysis/pixel_bias_simcode/stage_solves.py <run dir> <out.npz> [--model_dir DIR]
  --model_dir: use the networks / calibrations of another model folder (same file names as the run's, e.g. a longer-trained copy)
  --pass1_from: (with --model_dir) re-run pass 1 too is NOT done; pass 1 stays the run's stored rows"""
import argparse, glob, json, os, sys, time
import numpy as np, torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


def chunks_of(d, pat):
    fs = glob.glob(f'{d}/{pat}_*.npz')
    return sorted(int(f.rsplit('_', 1)[1][:-4]) for f in fs if f.rsplit('_', 1)[1][:-4].isdigit())


@torch.no_grad()
def fringe_only(o, PW, bw, cal, r_start):
    """local optimum of the intensity likelihood near r_start, flat position prior"""
    from simcode.solve import solve as SV
    I, Sig, r0, sp, ph = SV.solve_inputs(o, cal)
    dr = torch.zeros(len(I), 1, 2, device=I.device); flat = torch.full_like(sp, 1e3); pt = PW[bw]
    kw = dict(box=0.5, cmult=float(cal['cmult']), cang=float(cal['cmult_ang']))
    o0 = SV.pattern_solve_all(I, Sig, r_start, flat, pt, dr, kinetics=False, **kw)
    o1 = SV.pattern_solve_all(I, Sig, r_start, flat, pt, dr, kinetics=True, **kw)
    use = (o0['nll'][:, 0] - o1['nll'][:, 0]) > 9.21          # kinetic model selection as fringe_posterior
    r = torch.where(use[:, None], o1['r'][:, 0], o0['r'][:, 0]); s = torch.where(use[:, None], o1['s'][:, 0], o0['s'][:, 0])
    return r, s, sp


class PAggRecorder:
    """wraps simcode.detector.select.select_subpixel (read at call time by front_fast.QueuedFront.run) and records, per kept
    candidate, the AGGREGATED detection probability p_agg = p + the p of the pixels it suppresses in the sub-pixel NMS (same
    emitter: their point within the NMS radius, lower score), clipped to 1. The detector is trained on one target cell per
    emitter, so an emitter on a pixel edge splits its probability between two cells; p_agg puts it back together (the
    candidate set, the network inputs and every other output are unchanged)."""

    def __init__(self, win=2):
        import simcode.detector.select as SEL
        self.SEL = SEL; self.orig = SEL.select_subpixel; self.win = win; self.calls = []
        SEL.select_subpixel = self

    def restore(self): self.SEL.select_subpixel = self.orig

    def __call__(self, logit, dxy, k, p_min=0.05, radius=0.7, **kw):
        import torch.nn.functional as F
        c = self.orig(logit, dxy, k, p_min=p_min, radius=radius, **kw)
        B, H, W = logit.shape; w = self.win; dev = logit.device
        yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing='ij')
        X = xx + dxy[:, 0].float(); Y = yy + dxy[:, 1].float(); s = logit.float(); p = torch.sigmoid(s)
        p = torch.where(s > np.log(p_min / (1 - p_min)), p, torch.zeros_like(p))
        pad = lambda t, v: F.pad(t, (w, w, w, w), value=v)
        sP, XP, YP, pP = pad(s, -1e9), pad(X, 1e6), pad(Y, 1e6), pad(p, 0.0)
        kp = torch.isfinite(c['logit'])
        b, iy, ix = c['b'], c['iy'], c['ix']
        cx, cy, cs = c['x'], c['y'], torch.where(kp, c['logit'], torch.full_like(c['logit'], -1e9))
        agg = torch.sigmoid(cs) * kp; mx = torch.zeros_like(agg)
        for oy in range(-w, w + 1):
            for ox in range(-w, w + 1):
                if (oy, ox) == (0, 0): continue
                qy, qx = iy + w + oy, ix + w + ox
                qs, qX, qY, qp = sP[b, qy, qx], XP[b, qy, qx], YP[b, qy, qx], pP[b, qy, qx]
                m = kp & (qs <= cs) & ((qX - cx) ** 2 + (qY - cy) ** 2 < radius * radius)
                agg = agg + torch.where(m, qp, torch.zeros_like(qp)); mx = torch.maximum(mx, torch.where(m, qp, torch.zeros_like(qp)))
        self.calls.append(dict(b=b[kp], x=cx[kp], y=cy[kp], p_agg=agg[kp].clamp(max=1.0), p_max2=(torch.sigmoid(cs) + mx)[kp].clamp(max=1.0), p=torch.sigmoid(c['logit'][kp])))
        return c

    def table(self, det_batch):
        """all recorded candidates: window index (call j covers windows j * det_batch ...), x, y (region px), p_agg"""
        if not self.calls: return {}
        w = torch.cat([q['b'] + j * det_batch for j, q in enumerate(self.calls)])
        return dict(w=w.cpu().numpy(), x=torch.cat([q['x'] for q in self.calls]).cpu().numpy(), y=torch.cat([q['y'] for q in self.calls]).cpu().numpy(),
                    p_agg=torch.cat([q['p_agg'] for q in self.calls]).cpu().numpy(), p_max2=torch.cat([q['p_max2'] for q in self.calls]).cpu().numpy(), p=torch.cat([q['p'] for q in self.calls]).cpu().numpy())


def solve_both(o, PW, bw, cal, batch=16384):
    from simcode.solve import solve_fast as MF
    acc = {k: [] for k in ('r', 's', 'rf', 'sf', 'sp')}
    n = len(o['I'])
    for c0 in range(0, n, batch):
        sl = slice(c0, c0 + batch)
        oc = {k: v[sl] for k, v in o.items()}
        so = MF.solve_emitters_fast(oc, PW, bw[sl], cal, compile='off')
        rf, sf, sp = fringe_only(oc, PW, bw[sl], cal, so['r'])
        for k, v in (('r', so['r']), ('s', so['s']), ('rf', rf), ('sf', sf), ('sp', sp)): acc[k].append(v.float().cpu().numpy())
    return {k: np.concatenate(v) if v else np.zeros((0, 2), np.float32) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('out')
    ap.add_argument('--model_dir', default='', help='networks / calibrations from this folder (same file names) instead of the run\'s')
    ap.add_argument('--emitter_pass2', default='', help='pass-2 emitter network checkpoint instead of the run\'s (e.g. a longer-trained one)')
    ap.add_argument('--skip_pass1', type=int, default=0)
    ap.add_argument('--pass2_only_check', type=int, default=0)
    a = ap.parse_args()
    d = a.run
    RC = json.load(open(f'{d}/run_config.json'))['settings']
    if a.model_dir:
        for k in ('detector_pass2', 'emitter_pass2', 'cal_pass2', 'cal'):
            RC[k] = os.path.join(a.model_dir, os.path.relpath(RC[k], os.path.dirname(RC['detector'])))
    if a.emitter_pass2: RC['emitter_pass2'] = a.emitter_pass2
    print('settings', {k: RC[k] for k in ('detector_pass2', 'emitter_pass2', 'cal_pass2', 'cal')}, flush=True)
    from simcode.processing.common import load_calib, pattern_frames, region
    from simcode.processing.front_fast import Pass2FrontFast
    from simcode.io.movie import open_movie
    dev = 'cuda'
    Pt = torch.load(f'{d}/pattern_est.pt')
    cal1 = json.load(open(RC['cal'])); cal2 = json.load(open(RC['cal_pass2']))
    crop = tuple(int(v) for v in RC['crop'].split(',')) if RC['crop'] else None
    if crop is None:
        from simcode.table import run_crop
        crop = tuple(int(v) for v in run_crop(d))
    chunks = chunks_of(d, 'pass1')
    O = {}
    t0 = time.time()
    # ---------------- pass 1: stored head outputs
    if not a.skip_pass1:
        P1 = {k: [] for k in ('r', 's', 'rf', 'sf', 'sp')}; chk = []
        for f0 in chunks:
            R = dict(np.load(f'{d}/pass1_{f0}.npz')); E = np.load(f'{d}/pass2_est_{f0}.npz')
            X0, Y0 = int(R['X0']), int(R['Y0'])
            keep = R['out'].astype(bool); idx = np.nonzero(keep)[0]; ws = R['wstart'][keep]
            tt = lambda v: torch.tensor(np.asarray(v), device=dev, dtype=torch.float32)
            uw, inv = np.unique(ws, return_inverse=True)
            PW = pattern_frames(Pt, (uw[:, None] + np.arange(6)[None]).reshape(-1), X0, Y0, dev).reshape(len(uw), 6, 5)
            o = dict(I=tt(R['I'][idx]), L=tt(R['L'][idx]), x=tt(R['x'][idx]) - X0, y=tt(R['y'][idx]) - Y0,
                     lsxy=torch.log(torch.stack([tt(R['sx'][idx]), tt(R['sy'][idx])], 1)))
            S = solve_both(o, PW, torch.tensor(inv, device=dev), cal1)
            for k in ('r', 'rf'): S[k] = S[k] + np.array([X0, Y0], np.float32)
            chk.append(np.abs(S['r'][:, 0] - E['x_hy0']).max() if len(idx) else 0.0)
            for k in P1: P1[k].append(S[k])
            print(f'pass1 f0={f0}: {len(idx)} rows, max |x - stored| {chk[-1]:.2e} px ({time.time() - t0:.0f} s)', flush=True)
        for k, v in P1.items(): O['p1_' + k] = np.concatenate(v)
        O['p1_check_max_dx'] = np.array(chk)
    # ---------------- pass 2: re-run the networks
    cal = load_calib(RC['calib'])
    movie = open_movie(RC['movie'])
    p2 = Pass2FrontFast(RC['calib'], emitter=RC['emitter_pass2'], detector=RC['detector_pass2'], compile='off')
    P2 = {k: [] for k in ('r', 's', 'rf', 'sf', 'sp', 'wstart', 'x_prior', 'y_prior', 'x_det', 'y_det', 'p', 'phot', 'sx_cal', 'sy_cal', 'chunk', 'p_agg', 'p_max2')}
    rec = PAggRecorder()
    nchk = []
    for f0 in chunks:
        P1n = dict(np.load(f'{d}/pass1_{f0}.npz'))
        nf = movie.n if not RC.get('nframes') else min(movie.n, RC['nframes'])
        rec.calls = []
        R, H = p2.run_chunk(movie, cal, f0, min(f0 + RC['chunk'], nf), Pt, cal2, P1n, cal1, crop=crop, margin=5, return_head=True)
        o = H['o']; X0, Y0, _ = region(movie, crop, 5)
        # aggregated detection probability of each row (PAggRecorder): match (window, detector point)
        tb = rec.table(64); l0 = max(f0 - 12, 0) // 6 * 6
        key = {(int(w_), int(round(float(x_) * 1000)), int(round(float(y_) * 1000))): (pa, pm) for w_, x_, y_, pa, pm in zip(tb['w'], tb['x'], tb['y'], tb['p_agg'], tb['p_max2'])}
        wr = (R['wstart'] - l0) // 3
        pa = np.array([key.get((int(w_), int(round(float(x_ - X0) * 1000)), int(round(float(y_ - Y0) * 1000))), (np.nan, np.nan)) for w_, x_, y_ in zip(wr, R['x_det'], R['y_det'])]).reshape(-1, 2)
        if np.isnan(pa[:, 0]).mean() > 0.01: print(f'  WARNING p_agg unmatched {np.isnan(pa[:, 0]).mean():.3f}', flush=True)
        P2['p_agg'].append(pa[:, 0]); P2['p_max2'].append(pa[:, 1])
        oo = {k: o[k] for k in ('I', 'L', 'x', 'y', 'lsxy')}
        S = solve_both(oo, H['PWall'], H['bw'], cal2)
        for k in ('r', 'rf'): S[k] = S[k] + np.array([X0, Y0], np.float32)
        for k in ('r', 's', 'rf', 'sf', 'sp'): P2[k].append(S[k])
        for k in ('wstart', 'x_prior', 'y_prior', 'x_det', 'y_det', 'p', 'phot', 'sx_cal', 'sy_cal'): P2[k].append(R[k])
        P2['chunk'].append(np.full(len(R['wstart']), f0))
        E2 = np.load(f'{d}/pass2_est2_{f0}.npz')
        same = len(E2['x_hy0']) == len(R['x_hy0'])
        dx = np.abs(E2['x_hy0'] - S['r'][:, 0]).max() if same else np.nan
        nchk.append((len(E2['x_hy0']), len(R['x_hy0']), dx))
        print(f'pass2 f0={f0}: {len(R["x_hy0"])} rows (stored {len(E2["x_hy0"])}), max |x - stored| {dx:.2e} px ({time.time() - t0:.0f} s)', flush=True)
        if a.pass2_only_check: break
    for k, v in P2.items(): O['p2_' + k] = np.concatenate(v)
    O['p2_check'] = np.array(nchk, np.float64)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, **O)
    print('->', a.out, f'{time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
