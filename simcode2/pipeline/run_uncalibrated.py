"""
Expert entry point of the SIMCODE v2 pipeline (same flags and file layout as the development pipeline; the user-facing CLI
is `python -m simcode2.run`, which calls this module's stages with the production defaults).

  pass1    detector (det_n1) + per-candidate head (sc5nb): network per-frame intensities with covariance, per-frame bg,
           pattern-free position prior; candidates p >= cand_pmin are stored         (simcode2/pipeline/sc_front[_fast].py)
  pattern  pattern k, per-step phases, depth, relative intensities and the per-angle phase drift eps(t), from the pass-1
           network intensities alone                                                (simcode2/pipeline/pattern_est.py)
  pass2    --frontend sc:   the explicit intensity-domain pattern solve of the pass-1 rows (pass2_est_<f0>.npz)
           --frontend sc2p: + the pattern-aware pass 2 (pass-2 detector + head fed the estimated pattern, then the same
                            solve; pass2_est2_<f0>.npz, head-prior variant pass2_est2b_<f0>.npz); --p1_solve 0 skips the
                            pass-1 solve
Only the camera maps and the PSF are taken from --calib (in-situ calibration json). Optional --ref_pattern: the solve is
repeated with that pattern on the same detections (tag 'ref' / 'ref2'; for comparisons only).
usage: python -m simcode2.pipeline.run_uncalibrated --movie <tif> --calib <calib.json> --out <dir> [--frontend sc|sc2p]
       [--fast 1 --compile auto] [--stages pass1,pattern,pass2] [--nframes N] [--chunk 6000]
Outputs in --out: pass1_<f0>.npz, pattern_est.pt / pattern_est.json, pass2_<tag>_<f0>.npz, runtime.json, log.txt
"""
import argparse, glob, hashlib, json, os, time
import numpy as np, torch
from simcode2 import config


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()[:10]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--movie', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ref_pattern', default='')
    ap.add_argument('--stages', default='pass1,pattern,pass2')
    ap.add_argument('--crop', default='5,5,110')
    ap.add_argument('--margin', type=int, default=5)
    ap.add_argument('--chunk', type=int, default=6000)
    ap.add_argument('--nframes', type=int, default=0)
    ap.add_argument('--nn_min', type=float, default=3.0)
    ap.add_argument('--min_phot', type=float, default=300.0)
    ap.add_argument('--redo', type=int, default=0)
    # --pattern_field (2026-11-10, opt-in; default '' = unchanged): pattern field maps (simcode2/pipeline/pattern_field.py):
    # 'est' = estimate a 10 px phase map + Legendre-2 depth field from the pass-1 rows after the pattern stage, or a saved
    # field .pt. Adds an extra pass-2 tag 'estfld' (pass2_estfld_<f0>.npz; sc front-end); pass2_est unchanged.
    # 'est_depth' (2026-11-10): DEPTH-ONLY field (Legendre-2 depth, no phase map; pattern_field_depth.pt) -> tag 'estdep'; a saved
    # depth-only .pt (phase_on False) also gives tag 'estdep'. Since 2026-11-10 a field also runs on the --fast path (was: ignored).
    ap.add_argument('--pattern_field', default='')
    ap.add_argument('--frontend', default='sc', choices=['sc', 'sc2p'])   # sc: pass 1 + pattern + solve; sc2p: + pattern-aware pass 2
    ap.add_argument('--sc_head', default=config.HEAD)
    ap.add_argument('--sc_det', default=config.DET)
    ap.add_argument('--sc_cal', default=config.CAL_P1_REAL)
    # --frontend sc2p (simcode2/p2/front.py): passes 1 / pattern / pass-1 solve as 'sc', then pass 2 with the
    # pattern-aware networks (pass-2 detector + head fed the estimated pattern) and the explicit solve -> pass2_est2_<f0>.npz
    # (solve) and pass2_est2b_<f0>.npz (variant b: the pass-2 head prior); with --ref_pattern also est2 / est2b on that pattern ('ref2')
    ap.add_argument('--p2_det', default=config.P2_DET)
    ap.add_argument('--p2_head', default=config.P2_HEAD)
    ap.add_argument('--p2_cal', default=config.CAL_P2_REAL)
    ap.add_argument('--p2_pf_head', default=config.HEAD)   # only used by a pass-2 calibration with p2_prior 'pf'
    ap.add_argument('--chunk_list', default='')         # explicit chunk starts, e.g. '12000,60000' (default: all)
    # --fast 1 (default 0 here = the reference code paths; simcode2.run uses --fast 1): sc / sc2p front-ends via
    # simcode2/pipeline/sc_front_fast.py (queued fixed-size detector -> head batching, 2 CUDA streams, one CPU copy per chunk)
    # and the pattern solve via simcode2/hd/msolve_fast.py (exact restructuring; approximate options only via --solve_opts)
    ap.add_argument('--fast', type=int, default=0)
    ap.add_argument('--cand_pmin', type=float, default=0.5)   # (2026-11-10, opt-in) stored-candidate detection cut of the sc / sc2p
                                                              # front-ends (head + solve run on p >= cand_pmin); 0.5 = unchanged
    ap.add_argument('--compile', default='off')         # fast path: torch.compile of the pattern-solve LM step: auto | on | off
    ap.add_argument('--compile_head', default='off')    # fast path, opt-in, NOT bf16-noise equivalent: 'blk' compiles the head's conv stack
    ap.add_argument('--det_batch', type=int, default=64)
    ap.add_argument('--head_batch', type=int, default=4096)
    ap.add_argument('--solve_batch', type=int, default=16384)
    ap.add_argument('--solve_opts', default='{}')       # json, msolve_fast options, e.g. '{"prune_k": 0.2}'
    ap.add_argument('--p1_solve', type=int, default=1)  # sc2p: 0 skips the pass-1 pattern solve (pass2_est files; not needed by pass 2)
    a = ap.parse_args(argv)
    for k in ('calib', 'sc_head', 'sc_det', 'sc_cal', 'p2_det', 'p2_head', 'p2_cal', 'p2_pf_head', 'ref_pattern'):
        setattr(a, k, config.resolve(getattr(a, k)))
    a.movie = config.resolve(a.movie, 'data')
    if a.pattern_field and a.pattern_field not in ('est', 'est_depth'): a.pattern_field = config.resolve(a.pattern_field, 'out')
    if a.cand_pmin != 0.5:
        from simcode2.pipeline import sc_front as _scf, sc_front_fast as _scff
        from simcode2.p2 import front as _p2f
        _scf.CAND_PMIN = _scff.CAND_PMIN = _p2f.CAND_PMIN = a.cand_pmin
        print(f'stored-candidate cut p >= {a.cand_pmin}', flush=True)
    fast = bool(a.fast)
    solve_opts = dict(json.loads(a.solve_opts), compile=a.compile) if fast else {}
    from simcode2.pipeline.common import Movie, load_calib
    os.makedirs(a.out, exist_ok=True)
    crop = tuple(int(v) for v in a.crop.split(','))
    rt_path = f'{a.out}/runtime.json'
    RT = json.load(open(rt_path)) if os.path.exists(rt_path) else {}
    save_rt = lambda: json.dump(RT, open(rt_path, 'w'), indent=1)
    log = open(f'{a.out}/log.txt', 'a')
    def say(*s):
        msg = ' '.join(str(v) for v in s); print(msg, flush=True); log.write(msg + '\n'); log.flush()
    movie = Movie(a.movie)
    nf = min(movie.n, a.nframes) if a.nframes else movie.n
    chunks = list(range(0, nf, a.chunk))
    if a.chunk_list: chunks = [int(v) for v in a.chunk_list.split(',')]
    t0 = time.time(); cal = load_calib(a.calib); RT['setup_psf_s'] = time.time() - t0
    say(f'movie {a.movie}: {movie.n} frames ({len(movie.parts)} parts), processing {nf} in {len(chunks)} chunks')
    stages = a.stages.split(',')

    if 'pass1' in stages:
        t0 = time.time()
        if fast:
            from simcode2.pipeline.sc_front_fast import Pass1SCFast
            p1 = Pass1SCFast(a.calib, head=a.sc_head, det=a.sc_det, compile_head=a.compile_head)
            RT['sc'] = dict(head=a.sc_head, det=a.sc_det, cal=a.sc_cal)
            RT['fast'] = dict(compile=a.compile, compile_head=a.compile_head, det_batch=a.det_batch, head_batch=a.head_batch, solve_batch=a.solve_batch,
                              solve_opts=solve_opts, p1_solve=a.p1_solve)
        else:
            from simcode2.pipeline.sc_front import Pass1SC
            p1 = Pass1SC(a.calib, head=a.sc_head, det=a.sc_det)
            RT['sc'] = dict(head=a.sc_head, det=a.sc_det, cal=a.sc_cal)
        torch.cuda.synchronize(); RT['setup_nets_s'] = time.time() - t0
        RT.setdefault('pass1', {})
        for f0 in chunks:
            fn = f'{a.out}/pass1_{f0}.npz'
            if os.path.exists(fn) and not a.redo: continue
            tm = {}; t0 = time.time()
            if fast:
                R, info = p1.run_chunk(movie, cal, f0, min(f0 + a.chunk, nf), crop=crop, margin=a.margin, timing=tm,
                                       det_batch=a.det_batch, head_batch=a.head_batch)
            else:
                R, info = p1.run_chunk(movie, cal, f0, min(f0 + a.chunk, nf), crop=crop, margin=a.margin, timing=tm)
            tm['total'] = time.time() - t0
            np.savez(fn, **R, X0=info['X0'], Y0=info['Y0'], S=info['S'], own=np.array(info['own']),
                     loaded=np.array(info['loaded']), crop=np.array(crop))
            RT['pass1'][str(f0)] = dict(tm, n_det=int(len(R['x'])), n_out=int(R['out'].sum()), frames=int(min(f0 + a.chunk, nf) - f0))
            save_rt()
            say(f'pass1 f0={f0}: {len(R["x"])} detections, {int(R["out"].sum())} localizations; ' +
                ' '.join(f'{k} {v:.1f}s' for k, v in tm.items()))

    if 'pattern' in stages:
        from simcode2.pipeline.pattern_est import estimate_pattern
        t0 = time.time()
        cols = ('wstart', 'x', 'y', 'sx', 'sy', 'I', 'phot', 'tc')
        Ls = [np.load(f'{a.out}/pass1_{f0}.npz') for f0 in chunks]
        # windows loaded by two chunks (padding) are taken from one chunk only (start in [own0 - 3, own1 - 3));
        # 'out' recomputed globally = central half (time centroid) & inside the crop
        keep = np.concatenate([(L['wstart'] >= int(L['own'][0]) - 3) & ((L['wstart'] < int(L['own'][1]) - 3) | (f0 == chunks[-1]))
                               for L, f0 in zip(Ls, chunks)])
        R = {c: np.concatenate([L[c] for L in Ls])[keep] for c in cols}
        # the constant prior-position bias of the head (sc_cal 'prior_off', px) is removed before the pattern estimation too;
        # otherwise the estimated phases absorb it and it reappears in the final positions
        po = json.load(open(a.sc_cal)).get('prior_off')
        if po is not None:
            R['x'] = R['x'] - po[0]; R['y'] = R['y'] - po[1]; say(f'pattern: prior_off {po} px removed from the pass-1 priors')
        x0c, y0c, szc = crop
        R['out'] = (R['tc'] >= 1.0) & (R['tc'] < 4.0) & (R['x'] >= x0c) & (R['x'] < x0c + szc) & (R['y'] >= y0c) & (R['y'] < y0c + szc)
        t_load = time.time() - t0
        t0 = time.time()
        P, info = estimate_pattern(R, movie.shape, nn_min=a.nn_min, min_phot=a.min_phot, verbose=say)
        RT['pattern'] = dict(load_s=t_load, estimate_s=time.time() - t0)
        torch.save(P, f'{a.out}/pattern_est.pt')
        json.dump(info, open(f'{a.out}/pattern_est.json', 'w'), indent=1, default=float)
        save_rt(); say(f'pattern estimation {RT["pattern"]["estimate_s"]:.1f}s -> {a.out}/pattern_est.pt')

    if 'pass2' in stages and not (fast and a.frontend == 'sc2p' and not a.p1_solve):
        pats = [('est', f'{a.out}/pattern_est.pt')] + ([('ref', a.ref_pattern)] if a.ref_pattern else [])
        FLD = None
        if a.pattern_field:
            if a.pattern_field in ('est', 'est_depth'):
                from simcode2.pipeline.pattern_est import estimate_pattern  # noqa (same R as the pattern stage)
                from simcode2.pipeline.pattern_field import estimate_field
                Ls = [np.load(f'{a.out}/pass1_{f0}.npz') for f0 in chunks]
                cols = ('wstart', 'x', 'y', 'sx', 'sy', 'I', 'phot', 'tc')
                keep = np.concatenate([(L['wstart'] >= int(L['own'][0]) - 3) & ((L['wstart'] < int(L['own'][1]) - 3) | (f0 == chunks[-1]))
                                       for L, f0 in zip(Ls, chunks)])
                Rf = {c: np.concatenate([L[c] for L in Ls])[keep] for c in cols}
                po = json.load(open(a.sc_cal)).get('prior_off')
                if po is not None: Rf['x'] = Rf['x'] - po[0]; Rf['y'] = Rf['y'] - po[1]
                x0c, y0c, szc = crop
                Rf['out'] = (Rf['tc'] >= 1.0) & (Rf['tc'] < 4.0) & (Rf['x'] >= x0c) & (Rf['x'] < x0c + szc) & (Rf['y'] >= y0c) & (Rf['y'] < y0c + szc)
                dep = a.pattern_field == 'est_depth'
                FLD = estimate_field(Rf, torch.load(f'{a.out}/pattern_est.pt'), movie.shape, cell=10, mode='shrink', depth=True, verbose=say,
                                     phase=not dep)
                torch.save(FLD, f'{a.out}/pattern_field{"_depth" if dep else ""}.pt')
            else:
                FLD = torch.load(a.pattern_field, weights_only=False)
            ftag = 'estdep' if (a.pattern_field == 'est_depth' or not FLD.get('phase_on', True)) else 'estfld'
            pats = pats + [(ftag, f'{a.out}/pattern_est.pt')]
        for tag, pf in pats:
            Pt = torch.load(pf)
            RT.setdefault(f'pass2_{tag}', {})
            for f0 in chunks:
                fn = f'{a.out}/pass2_{tag}_{f0}.npz'
                if os.path.exists(fn) and not a.redo: continue
                L = dict(np.load(f'{a.out}/pass1_{f0}.npz'))
                info = dict(X0=int(L['X0']), Y0=int(L['Y0']), S=int(L['S']), loaded=tuple(int(v) for v in L['loaded']))
                tm = {}; t0 = time.time()
                if fast:
                    from simcode2.pipeline.sc_front_fast import run_chunk_pattern_solve_fast
                    O = run_chunk_pattern_solve_fast(L, info, Pt, a.sc_cal, timing=tm, solve_batch=a.solve_batch, solve_opts=solve_opts,
                                                     field=FLD if tag in ('estfld', 'estdep') else None)
                else:
                    from simcode2.pipeline.sc_front import run_chunk_pass2
                    O = run_chunk_pass2(L, info, Pt, a.sc_cal, timing=tm, field=FLD if tag in ('estfld', 'estdep') else None)
                tm['total'] = time.time() - t0
                np.savez(fn, **O, pattern=pf)
                RT[f'pass2_{tag}'][str(f0)] = tm; save_rt()
                say(f'pass2 [{tag}] f0={f0}: ' + ' '.join(f'{k} {v:.1f}s' for k, v in tm.items()))
    if 'pass2' in stages and a.frontend == 'sc2p':
        from simcode2.p2.front import Pass2P2, direct_view
        t0 = time.time()
        if fast:
            from simcode2.pipeline.sc_front_fast import Pass2P2Fast
            p2 = Pass2P2Fast(a.calib, head=a.p2_head, det=a.p2_det, pf_head=a.p2_pf_head, compile=a.compile, compile_head=a.compile_head)
        else:
            p2 = Pass2P2(a.calib, head=a.p2_head, det=a.p2_det, pf_head=a.p2_pf_head)
        torch.cuda.synchronize(); RT['setup_p2_s'] = time.time() - t0; RT['p2'] = dict(det=a.p2_det, head=a.p2_head, cal=a.p2_cal)
        pats = [('est2', f'{a.out}/pattern_est.pt')] + ([('ref2', a.ref_pattern)] if a.ref_pattern else [])
        for tag, pf in pats:
            Pt = torch.load(pf)
            RT.setdefault(f'pass2_{tag}', {})
            for f0 in chunks:
                fn = f'{a.out}/pass2_{tag}_{f0}.npz'
                if os.path.exists(fn) and not a.redo: continue
                tm = {}; t0 = time.time()
                if fast:
                    O = p2.run_chunk(movie, cal, f0, min(f0 + a.chunk, nf), Pt, a.p2_cal, crop=crop, margin=a.margin, timing=tm,
                                     det_batch=a.det_batch, head_batch=a.head_batch, solve_batch=a.solve_batch, solve_opts=solve_opts)
                else:
                    O = p2.run_chunk(movie, cal, f0, min(f0 + a.chunk, nf), Pt, a.p2_cal, crop=crop, margin=a.margin, timing=tm)
                tm['total'] = time.time() - t0
                np.savez(fn, **O, pattern=pf)
                np.savez(f'{a.out}/pass2_{tag}b_{f0}.npz', **direct_view(O), pattern=pf)
                RT[f'pass2_{tag}'][str(f0)] = tm; save_rt()
                say(f'pass2 [{tag}] f0={f0}: {len(O["x_hy0"])} localizations; ' + ' '.join(f'{k} {v:.1f}s' for k, v in tm.items()))
    say('done')


if __name__ == '__main__':
    main()
