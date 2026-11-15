"""Nanoruler ablation, part B2: the solve variants on a whole pipeline run (= analysis/ablation_cov/movie_abl.py 'solve' + table for this setup).

Same network outputs for every variant (pass 1: the run's pass1_<f0>.npz, which store I, L and the prior; pass 2: the captured head
outputs of abl_capture.py); per variant the pipeline's solve (solve_fast.solve_emitters_fast, batches of 16384) with the variant's covariance
and ITS OWN calibration (abl_gate.py; no '_real' multiplier of the other setup) -> <out>/pass2_<tag>_<f0>.npz in the pipeline format,
then simcode.table.build_table (strict crop = the run's crop) -> locs.npz with the extra sigma columns
    sx, sy          calibrated sigma (the solve's s_cal with the variant's per-photon-bin factor)
    sx_nb, sy_nb    the same without the per-photon-bin sigma factor f (still: kap_I2 / kap_p bins, chi2 inflation sig_a, sig_g, spread)
    sx_raw, sy_raw  UNCALIBRATED model sigma: a second solve with kap_I2 = kap_p = 1, sig_a = 1, sig_g = 0, no sigma factor
                    (= sqrt(posterior-mean Fisher variance + fringe spread) of the noise model as the network states it; variant E, which has no
                    network covariance, keeps its variance table); x_raw, y_raw = the positions of that solve
    sx_u, sy_u      calibrated sigma x g(u): the chain's selection-aware rescale (sigu_p<pass>_nr80.json), when that file exists
    sx_f, sy_f      Fisher sigma at the MAP of the calibrated solve (no inflation, no spread, no factor)
All rows of the run are solved (the per-tile sigma rank u of the rescale is defined on all good rows); the analysis uses the cycle-aligned rows only.
Options: --pattern <file.pt> --ptag <name> (part D: another pattern for the same rows; a dict with key 'pipeline' is unwrapped).

usage: CUDA_VISIBLE_DEVICES=g python analysis/nanoruler/abl_solve.py <set> [variants A,B,..|all] [--pattern f --ptag name] [--auto 0]
  -> out/nanoruler/ablation/runs/<set>[_<ptag>]/<variant>/"""
import glob, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abl_lib as A
import variants as V
T6 = 6


def raw_cal(cal, name):
    c = {k: v for k, v in cal.items() if k not in ('sig_ph',)}
    c['sig_a'] = 1.0; c['sig_g'] = 0.0
    if name != 'E':
        c.pop('kap_bins', None); c['kap_I2'] = 1.0; c['kap_p'] = 1.0
    else:
        c['kap_bins'] = dict(cal['kap_bins'], kap_p=[1.0] * len(cal['kap_bins']['kap_p']))
    return c


def chunks_p1(S, Pt):
    """pass-1 rows as run_chunk_pattern_solve_fast builds them"""
    from simcode.processing.common import pattern_frames
    tt = lambda a: torch.tensor(np.asarray(a), device='cuda', dtype=torch.float32)
    for f in sorted(glob.glob(f'{S["run"]}/pass1_*.npz'), key=lambda f: int(f.rsplit('_', 1)[1][:-4])):
        R = np.load(f); X0, Y0 = int(R['X0']), int(R['Y0']); k = np.nonzero(R['out'].astype(bool))[0]
        ws = R['wstart'][k]; uw, inv = np.unique(ws, return_inverse=True)
        PW = pattern_frames(Pt, (uw[:, None] + np.arange(T6)[None]).reshape(-1), X0, Y0, 'cuda').reshape(len(uw), T6, 5)
        o = dict(I=R['I'][k], L=R['L'][k], x=R['x'][k] - X0, y=R['y'][k] - Y0, lsxy=np.log(np.c_[R['sx'][k], R['sy'][k]]), x_det=R['x_det'][k] - X0, y_det=R['y_det'][k] - Y0)
        meta = dict(wstart=ws.astype(np.int64), frame=R['frame'][k], p=R['p'][k], phot=R['phot'][k].astype(np.float32), x_prior=R['x'][k], y_prior=R['y'][k], I_net=R['I'][k])
        yield int(f.rsplit('_', 1)[1][:-4]), o, PW, torch.tensor(inv, device='cuda'), X0, Y0, meta, tuple(int(v) for v in R['crop'])


def chunks_p2(S, Pt):
    """captured pass-2 head outputs (abl_capture.py); the pattern per window is rebuilt from Pt (so --pattern works for pass 2 too;
    note the pass-2 networks were FED the run's own pattern)"""
    from simcode.processing.common import pattern_frames
    for f in sorted(glob.glob(f'{S["head"]}/head_*.npz'), key=lambda f: int(f.rsplit('_', 1)[1][:-4])):
        H = np.load(f); X0, Y0 = int(H['X0']), int(H['Y0']); ws = H['wstart'].astype(np.int64); uw, inv = np.unique(ws, return_inverse=True)
        PW = pattern_frames(Pt, (uw[:, None] + np.arange(T6)[None]).reshape(-1), X0, Y0, 'cuda').reshape(len(uw), T6, 5)
        o = {k: H['o_' + k] for k in ('I', 'L', 'x', 'y', 'lsxy', 'x_det', 'y_det')}
        meta = {k: H[k] for k in ('wstart', 'frame', 'p', 'phot', 'x_prior', 'y_prior', 'I_net')}
        yield int(f.rsplit('_', 1)[1][:-4]), o, PW, torch.tensor(inv, device='cuda'), X0, Y0, meta, tuple(int(v) for v in H['crop'])


@torch.no_grad()
def solve(setn, name, v, cal, pattern=None, ptag='', log=print):
    from simcode.solve import solve_fast as MF
    from simcode.emitter import model as SCM
    from simcode.solve import solve as SV
    S = A.SETS[setn]; d = f'{A.OUT}/runs/{setn}{"_" + ptag if ptag else ""}/{name}'; os.makedirs(d, exist_ok=True)
    Pt = torch.load(pattern or f'{S["run"]}/pattern_est.pt', weights_only=False)
    if 'pipeline' in Pt: Pt = Pt['pipeline']
    tt = lambda a: torch.tensor(np.asarray(a), device='cuda', dtype=torch.float32)
    calr = raw_cal(cal, name); t0 = time.time(); crop = None; n = 0
    for f0, o, PW, bw, X0, Y0, meta, crop in (chunks_p2 if S['tag'] == 'est2' else chunks_p1)(S, Pt):
        fn = f'{d}/pass2_{S["tag"]}_{f0}.npz'
        if os.path.exists(fn): continue
        o = dict(o); o['L'] = V.L_variant(o['L'], v['L']); og = {k: tt(a) for k, a in o.items()}
        acc = {k: [] for k in ('r', 's_cal', 'q', 'kin', 's', 'dnll', 'r_raw', 's_raw', 'f')}
        for c0 in range(0, len(bw), 16384):
            sl = slice(c0, c0 + 16384); ob = {kk: vv[sl] for kk, vv in og.items()}
            so = MF.solve_emitters_fast(ob, PW, bw[sl], cal, compile='auto')
            for k in ('r', 's_cal', 'q', 'kin', 's'): acc[k].append(so[k].float().cpu().numpy())
            acc['dnll'].append(so['dnll'].float().cpu().numpy() if 'dnll' in so else np.full(len(so['q']), np.nan, np.float32))
            acc['f'].append(SV._binval(cal['sig_ph'], 'f', ob['I'].sum(1)).float().cpu().numpy())
            sr = MF.solve_emitters_fast(ob, PW, bw[sl], calr, compile='auto')
            acc['r_raw'].append(sr['r'].float().cpu().numpy()); acc['s_raw'].append(sr['s_cal'].float().cpu().numpy())
        so = {k: np.concatenate(a) for k, a in acc.items()}
        R = dict(meta); f = so['f'][:, None]
        R.update(x_hy0=so['r'][:, 0] + X0, y_hy0=so['r'][:, 1] + Y0, sx_cal=so['s_cal'][:, 0], sy_cal=so['s_cal'][:, 1], sig_hy0=so['s_cal'].mean(1),
                 q_dof=so['q'], kin_hy=so['kin'].astype(np.float32), dnll=so['dnll'], sx_nb=so['s_cal'][:, 0] / f[:, 0], sy_nb=so['s_cal'][:, 1] / f[:, 0],
                 sx_f=so['s'][:, 0], sy_f=so['s'][:, 1], sx_raw=so['s_raw'][:, 0], sy_raw=so['s_raw'][:, 1], x_raw=so['r_raw'][:, 0] + X0, y_raw=so['r_raw'][:, 1] + Y0)
        np.savez(fn, **R); n += len(bw)
    log(f'{setn} {ptag} {name}: solved {n} rows in {time.time() - t0:.0f}s')
    return d, crop


def table(setn, d, crop):
    import simcode.table as SP
    S = A.SETS[setn]; crop = tuple(float(c) for c in crop)
    SP.CROP = crop; SP.tile_of.__defaults__ = (crop,) + SP.tile_of.__defaults__[1:]; SP.tile_rank.__defaults__ = SP.tile_rank.__defaults__[:-1] + (crop,)
    T, info = SP.build_table(d, S['tag'], A.PX)
    R = SP.load_rows(d, S['tag'])
    for k in ('sx_nb', 'sy_nb', 'sx_f', 'sy_f', 'sx_raw', 'sy_raw', 'x_raw', 'y_raw'): T[k] = R[k].astype(np.float32 if k[0] == 's' else np.float64)
    # sx_u, sy_u: calibrated sigma x g(u), the chain's selection-aware rescale of this pass (out/nanoruler/models/calib/sigu_p<pass>_nr80.json,
    # fitted on nr_a with the PRODUCTION calibration = variant A; u = each variant's own per-tile sigma rank, as simcode.table.build_table)
    su = os.path.join(A.NR, 'models', 'calib', f'sigu_p{2 if S["tag"] == "est2" else 1}_nr80.json')
    if os.path.exists(su):
        Tu, _ = SP.build_table(d, S['tag'], A.PX, sig_u=su); T['sx_u'] = Tu['sx']; T['sy_u'] = Tu['sy']
    else:
        T['sx_u'] = T['sx'].copy(); T['sy_u'] = T['sy'].copy()
    SP.write_table(T, info, f'{d}/locs', ('npz',)); A.jdump({k: v for k, v in info.items() if k != 'columns'}, f'{d}/locs_info.json')
    T['crop'] = np.asarray(crop, np.float64)      # (for abl_lib.selection; not written to the table file)
    return T, info


def main():
    import abl_gate as G
    a = sys.argv[1:]; g = lambda k, dflt=None: a[a.index(k) + 1] if k in a else dflt
    setn = a[0]; names = a[1] if len(a) > 1 and not a[1].startswith('--') else 'all'
    pattern, ptag = g('--pattern'), g('--ptag', ''); A.title()
    S = A.SETS[setn]; auto = setn.replace('p2', 'p1') if int(g('--auto', 1)) else None      # 'auto' / 'autoS' use the theta of the pass-1 rows (as the pipeline would)
    if auto and not os.path.exists(f'{A.OUT}/autoc_{auto}.json'): auto = None
    VARS, order = G.variants(auto)
    if names != 'all': order = [n for n in order if n in names.split(',')]
    for name in order:
        v = VARS[name]; cal = G.fit(S['front'], name, v)
        d, crop = solve(setn, name, v, cal, pattern, ptag)
        if crop is None: crop = tuple(float(c) for c in json.load(open(f'{d}/locs_info.json'))['crop'])
        T, info = table(setn, d, crop); info['crop'] = list(crop); A.jdump({k: v for k, v in info.items() if k != 'columns'}, f'{d}/locs_info.json')
        m = A.selection(T, 'm12')
        print(f'{setn} {ptag} {name}: {info["n_rows"]} rows, good {info["n_good"]}, cycle-aligned max sigma <= 12 nm: {int(m.sum())}, median sigma '
              f'{np.median(0.5 * (T["sx"] + T["sy"])[m]) * A.PX:.2f} nm (no bin factor {np.median(0.5 * (T["sx_nb"] + T["sy_nb"])[m]) * A.PX:.2f}, uncalibrated '
              f'{np.median(0.5 * (T["sx_raw"] + T["sy_raw"])[m]) * A.PX:.2f}), photons {np.median(T["photons"][m]):.0f}', flush=True)


if __name__ == '__main__':
    main()
