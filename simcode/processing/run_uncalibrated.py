"""
Expert entry point of the SIMCODE v2 pipeline (same flags and file layout as the development pipeline; the user-facing CLI
is `python -m simcode.run`, which calls this module's stages with the production defaults).

  pass1    detector + per-candidate head: network per-frame intensities with covariance, per-frame bg,
           pattern-free position prior; candidates p >= cand_pmin are stored         (simcode/processing/pass1_inference.py, inference_fast.py)
  pattern  pattern k, per-step phases, depth, relative intensities and the per-angle phase drift eps(t), from the pass-1
           network intensities alone                                                (simcode/processing/pattern_estimation.py)
  pass1_solve  the explicit intensity-domain pattern solve of the pass-1 rows (pass2_est_<f0>.npz, table tag est)
  pass2    the pattern-aware pass 2 (pass-2 detector + emitter network fed the estimated pattern, then the same solve;
           pass2_est2_<f0>.npz, table tag est2; network-prior variant pass2_est2b_<f0>.npz)
The stages run in this order (--stages, default all); each uses the files of the earlier ones (pattern_est.pt: the pattern of
pass1_solve and pass2).
Only the camera maps and the PSF are taken from --calib (in-situ calibration json).
Output region --crop x0,y0,size (px): default the region of the run's pass-1 files if they exist, else the whole frame
(square sensors; otherwise the largest top-left square).

Code path: the fast front-ends (simcode/processing/inference_fast.py: queued detector -> head batching, 2 CUDA streams) and the
fast solve (simcode/solve/solve_fast.py, torch.compile of its LM step when available); the reference implementations
(pass1_inference.py, processing/pass2_inference.py, solve/solve.solve_emitters) are the test oracle only (tests/test_fast_paths.py).

Chunks are the units of the shared work driver (simcode/io/work.py): a finished chunk is skipped (--redo 1 repeats it),
every output is written atomically (a killed run leaves no truncated chunk file), the free disk space is checked before a
chunk starts (--min_free_gb), and each chunk leaves a small record (pass1_<f0>.json ...) from which runtime.json is merged.
With --claim 1 several copies of the same command (one per GPU, python -m simcode.io.launch) share the chunks of the
pass-1 and pass-2 stages; the pattern stage then runs in one of them, after all pass-1 chunks exist.

usage: python -m simcode.processing.run_uncalibrated --movie <tif> --calib <calib.json> --out <dir>
       [--stages pass1,pattern,pass1_solve,pass2] [--crop x0,y0,size] [--nframes N] [--chunk 6000] [--claim 1]
Outputs in --out: pass1_<f0>.npz, pattern_est.pt / pattern_est.json, pass2_<tag>_<f0>.npz, runtime.json, log.txt
"""
import argparse, glob, json, os, time
import numpy as np, torch
from simcode import config


COMPILE = 'auto'          # torch.compile of the solve's LM step when available (simcode/util/compile.py)


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--movie', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stages', default='pass1,pattern,pass1_solve,pass2')
    ap.add_argument('--crop', default='')
    ap.add_argument('--margin', type=int, default=5)
    ap.add_argument('--chunk', type=int, default=6000)
    ap.add_argument('--nframes', type=int, default=0)
    ap.add_argument('--nn_min', type=float, default=3.0)
    ap.add_argument('--min_phot', type=float, default=300.0)
    ap.add_argument('--redo', type=int, default=0)
    ap.add_argument('--claim', type=int, default=0)     # 1: claim each chunk (several workers on one output directory); no prefetch then
    ap.add_argument('--min_free_gb', type=float, default=2.0)   # a chunk is not started with less free space on the output file system
    ap.add_argument('--emitter', default=config.EMITTER)         # pass-1 networks and their solve calibration
    ap.add_argument('--detector', default=config.DETECTOR)
    ap.add_argument('--cal', default=config.CAL_SIM)
    # stage pass2 (simcode/processing/pass2_inference.py): pass 2 with the pattern-aware networks (pass-2 detector + emitter network
    # fed the estimated pattern) and the explicit solve -> pass2_est2_<f0>.npz (solve) and pass2_est2b_<f0>.npz (variant b: the
    # pass-2 network prior)
    ap.add_argument('--detector_pass2', default=config.DETECTOR_PASS2)
    ap.add_argument('--emitter_pass2', default=config.EMITTER_PASS2)
    ap.add_argument('--cal_pass2', default=config.CAL_SIM_PASS2)
    ap.add_argument('--chunk_list', default='')         # explicit chunk starts, e.g. '12000,60000' (default: all)
    ap.add_argument('--det_batch', type=int, default=64)
    ap.add_argument('--emitter_batch', type=int, default=1024)
    ap.add_argument('--solve_batch', type=int, default=16384)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    from simcode.util.proctitle import set_title; set_title('pipeline')     # recognizable name in nvidia-smi / ps
    for k in ('calib', 'emitter', 'detector', 'cal', 'detector_pass2', 'emitter_pass2', 'cal_pass2'):
        setattr(a, k, config.resolve(getattr(a, k)))
    a.movie = os.path.abspath(os.path.expanduser(a.movie))
    from simcode.processing.common import load_calib, load_pass1_rows, chunk_read_plan
    from simcode.io.movie import open_movie, Prefetch
    from simcode.io.work import Units, atomic_json, atomic_torch_save
    os.makedirs(a.out, exist_ok=True)
    rt_path = f'{a.out}/runtime.json'
    RT = json.load(open(rt_path)) if os.path.exists(rt_path) else {}
    log = open(f'{a.out}/log.txt', 'a')
    def say(*s):
        msg = ' '.join(str(v) for v in s); print(msg, flush=True); log.write(msg + '\n'); log.flush()
    units = lambda stage: Units(a.out, stage, claim=a.claim, redo=a.redo, min_free_gb=a.min_free_gb, log=say)
    def save_rt(stage=None, U=None):
        """runtime.json = the set-up entries of this process + the per-chunk records of the stage (merged, never overwritten by
        another worker's partial view)"""
        if U is not None: RT.setdefault(stage, {}).update({str(k): v for k, v in U.records(chunks).items()})
        atomic_json(rt_path, RT, indent=1)
    movie = open_movie(a.movie)
    if a.crop: crop = tuple(int(v) for v in a.crop.split(','))
    elif glob.glob(f'{a.out}/pass1_*.npz'):
        from simcode.table import run_crop
        crop = tuple(int(v) for v in run_crop(a.out))                       # the region of the run's pass-1 files
    else:
        crop = (0, 0, int(min(movie.shape)))                                 # the whole frame
    nf = min(movie.n, a.nframes) if a.nframes else movie.n
    chunks = list(range(0, nf, a.chunk))
    if a.chunk_list: chunks = [int(v) for v in a.chunk_list.split(',')]
    t0 = time.time(); cal = load_calib(a.calib); RT['setup_psf_s'] = time.time() - t0
    say(f'movie {a.movie}: {movie.n} frames ({len(movie.parts)} parts), processing {nf} in {len(chunks)} chunks')
    stages = a.stages.split(',')
    bad = set(stages) - {'pass1', 'pattern', 'pass1_solve', 'pass2'}
    if bad: raise SystemExit(f'unknown stages {sorted(bad)} (pass1, pattern, pass1_solve, pass2)')
    prefetch = not a.claim                       # the next chunk's frames are read in a background thread (io/movie.Prefetch)

    if 'pass1' in stages:
        t0 = time.time()
        from simcode.processing.inference_fast import Pass1FrontFast
        p1 = Pass1FrontFast(a.calib, emitter=a.emitter, detector=a.detector)
        RT['networks_pass1'] = dict(emitter=a.emitter, detector=a.detector, cal=a.cal)
        RT['fast'] = dict(compile=COMPILE, det_batch=a.det_batch, emitter_batch=a.emitter_batch, solve_batch=a.solve_batch)
        torch.cuda.synchronize(); RT['setup_nets_s'] = time.time() - t0
        RT.setdefault('pass1', {})
        U = units('pass1')
        mv1 = movie
        if prefetch:
            mv1 = Prefetch(movie, chunk_read_plan(movie, [f0 for f0 in chunks if not U.done(f0)], a.chunk, nf, crop, a.margin))
        for f0 in U.todo(chunks):
            tm = {}; t0 = time.time()
            R, info = p1.run_chunk(mv1, cal, f0, min(f0 + a.chunk, nf), crop=crop, margin=a.margin, timing=tm,
                                   det_batch=a.det_batch, emitter_batch=a.emitter_batch)
            tm['total'] = time.time() - t0
            U.save(f0, **R, X0=info['X0'], Y0=info['Y0'], S=info['S'], own=np.array(info['own']),
                   loaded=np.array(info['loaded']), crop=np.array(crop))
            U.record(f0, dict(tm, n_det=int(len(R['x'])), n_out=int(R['out'].sum()), frames=int(min(f0 + a.chunk, nf) - f0)))
            save_rt('pass1', U)
            say(f'pass1 f0={f0}: {len(R["x"])} detections, {int(R["out"].sum())} localizations; ' +
                ' '.join(f'{k} {v:.1f}s' for k, v in tm.items()))
        if prefetch: mv1.close()

    if 'pattern' in stages:
        # one unit: with --claim 1 only one worker estimates the pattern (the others wait for pattern_est.pt); without
        # claiming the stage always runs when it is requested (as before)
        UP = Units(a.out, 'pattern', pattern='pattern_est.pt', claim=a.claim, redo=not a.claim, log=say)
        for _ in UP.todo(['']):
            from simcode.processing.pattern_estimation import estimate_pattern
            t0 = time.time()
            # the constant prior-position bias of the head (cal 'prior_off', px) is removed before the pattern estimation too;
            # otherwise the estimated phases absorb it and it reappears in the final positions
            po = json.load(open(a.cal)).get('prior_off')
            R = load_pass1_rows(a.out, chunks, crop, po)
            if po is not None: say(f'pattern: prior_off {po} px removed from the pass-1 priors')
            t_load = time.time() - t0
            t0 = time.time()
            P, info = estimate_pattern(R, movie.shape, nn_min=a.nn_min, min_phot=a.min_phot, verbose=say)
            RT['pattern'] = dict(load_s=t_load, estimate_s=time.time() - t0)
            atomic_json(f'{a.out}/pattern_est.json', info, indent=1, default=float)
            atomic_torch_save(P, f'{a.out}/pattern_est.pt')
            save_rt(); say(f'pattern estimation {RT["pattern"]["estimate_s"]:.1f}s -> {a.out}/pattern_est.pt')

    if 'pass1_solve' in stages:
        # the pattern solve of the pass-1 rows (pass2_est)
        pats = [('est', f'{a.out}/pattern_est.pt')]
        for tag, pf in pats:
            Pt = torch.load(pf)
            RT.setdefault(f'pass2_{tag}', {})
            U = units(f'pass2_{tag}')
            for f0 in U.todo(chunks):
                L = dict(np.load(f'{a.out}/pass1_{f0}.npz'))
                info = dict(X0=int(L['X0']), Y0=int(L['Y0']), S=int(L['S']), loaded=tuple(int(v) for v in L['loaded']))
                tm = {}; t0 = time.time()
                from simcode.processing.inference_fast import run_chunk_pattern_solve_fast
                O = run_chunk_pattern_solve_fast(L, info, Pt, a.cal, timing=tm, solve_batch=a.solve_batch, compile=COMPILE)
                tm['total'] = time.time() - t0
                U.save(f0, **O, pattern=pf)
                U.record(f0, tm); save_rt(f'pass2_{tag}', U)
                say(f'pass2 [{tag}] f0={f0}: ' + ' '.join(f'{k} {v:.1f}s' for k, v in tm.items()))
    if 'pass2' in stages:
        from simcode.processing.pass2_inference import direct_view
        from simcode.processing.inference_fast import Pass2FrontFast
        t0 = time.time()
        p2 = Pass2FrontFast(a.calib, emitter=a.emitter_pass2, detector=a.detector_pass2, compile=COMPILE)
        torch.cuda.synchronize(); RT['setup_p2_s'] = time.time() - t0
        RT['networks_pass2'] = dict(detector=a.detector_pass2, emitter=a.emitter_pass2, cal=a.cal_pass2)
        pats = [('est2', f'{a.out}/pattern_est.pt')]
        for tag, pf in pats:
            Pt = torch.load(pf)
            RT.setdefault(f'pass2_{tag}', {})
            U = units(f'pass2_{tag}'); Ub = Units(a.out, f'pass2_{tag}b', log=say)
            mv2 = movie
            if prefetch:
                mv2 = Prefetch(movie, chunk_read_plan(movie, [f0 for f0 in chunks if not U.done(f0)], a.chunk, nf, crop, a.margin))
            for f0 in U.todo(chunks):
                tm = {}; t0 = time.time()
                P1 = dict(np.load(f'{a.out}/pass1_{f0}.npz'))       # z of the pass-2 rows from pass 1 (z_conditioning.pass1_z_columns)
                O = p2.run_chunk(mv2, cal, f0, min(f0 + a.chunk, nf), Pt, a.cal_pass2, P1, a.cal, crop=crop, margin=a.margin, timing=tm,
                                 det_batch=a.det_batch, emitter_batch=a.emitter_batch, solve_batch=a.solve_batch)
                tm['total'] = time.time() - t0
                Ub.save(f0, **direct_view(O), pattern=pf)       # the head-prior variant first: the chunk counts as done by the
                U.save(f0, **O, pattern=pf)                      # main file, so a run killed in between repeats both
                U.record(f0, tm); save_rt(f'pass2_{tag}', U)
                say(f'pass2 [{tag}] f0={f0}: {len(O["x_hy0"])} localizations ({int((O["p1_row"] < 0).sum())} without a pass-1 partner: z NaN); '
                    + ' '.join(f'{k} {v:.1f}s' for k, v in tm.items()))
            if prefetch: mv2.close()
    say('done')


if __name__ == '__main__':
    main()
