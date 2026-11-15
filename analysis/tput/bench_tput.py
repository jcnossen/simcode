"""End-to-end throughput, original vs fast, per stage, on one chunk (run on an otherwise idle GPU).
  pass 1        read, detector, head                    (front.Pass1Front  vs front_fast.Pass1FrontFast)
  pattern solve on the pass-1 rows                      (front.run_chunk_pattern_solve vs run_chunk_pattern_solve_fast [eager, compiled])
  pass 2 (p2)   read, detector, head, pattern solve     (pass2_front.Pass2Front vs front_fast.Pass2FrontFast [eager, compiled])
  head variants (fast pass 1): head_batch 8192, channels_last, cudnn.benchmark (+ their I / x difference vs the original)
Original-path stage times include its per-stage syncs (as logged by the pipeline); fast-path stage times are GPU busy times
from CUDA events (the stages overlap with 2 streams) and 'wall' is the chunk wall time.  same stages / timing code on the simcode modules. The
datasets are a json file (default analysis/tput/datasets.json; $TPUT_DATA overrides): name -> movie, calib (in-situ), f0, f1,
pattern (a pattern_est.pt of a full run of that movie, e.g. <out>/pattern_est.pt of simcode.run), sc_cal, p2_cal; relative
movie paths resolve against the simcode data root, calibrations against the models dir (simcode/config.py). Networks =
simcode defaults (det_n1, hd_sc5nb, p2_det, p2_head_long). Output: <simcode out>/tput/bench_<name>.json ($TPUT_OUT overrides).
usage: CUDA_VISIBLE_DEVICES=1 python analysis/tput/bench_tput.py <name> [variants: orig,fast,fastc,heads] [--pattern <pattern_est.pt>]
       (BENCH_NF=<frames>: short smoke chunk -> bench_<name>_smoke.json; MEM_FRAC: per-process GPU memory cap, default 0.45)"""
import sys, os, json, time
import numpy as np, torch
from simcode import config
from simcode.psf import TiffMovie
from simcode.processing.common import load_calib
from simcode.processing import front as SF, front_fast as SFF
from simcode.processing import pass2_front as FR

ARGS = sys.argv[1:]
PAT = None
if '--pattern' in ARGS:
    i = ARGS.index('--pattern'); PAT = ARGS[i + 1]; del ARGS[i:i + 2]
DATA = json.load(open(os.environ.get('TPUT_DATA') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets.json')))
R_ = lambda p: config.resolve(p)
HEAD, DET = R_(config.HEAD), R_(config.DET)
P2H, P2D, PFH = R_(config.P2_HEAD), R_(config.P2_DET), R_(config.HEAD)
name = ARGS[0]
variants = (ARGS[1] if len(ARGS) > 1 else 'orig,fast,fastc,heads').split(',')
D = {k: v for k, v in DATA[name].items() if not k.startswith('_')}
D['movie'] = config.resolve(D['movie'], 'data')
for k in ('calib', 'sc_cal', 'p2_cal'): D[k] = R_(D[k])
if PAT: D['pattern'] = PAT
OD = os.environ.get('TPUT_OUT') or config.out_dir('tput'); os.makedirs(OD, exist_ok=True)
out = os.path.join(OD, f'bench_{name}.json')
if os.environ.get('BENCH_NF'):      # smoke test: short chunk, separate output file
    D['f1'] = D['f0'] + int(os.environ['BENCH_NF']); out = os.path.join(OD, f'bench_{name}_smoke.json')
REP = json.load(open(out)) if os.path.exists(out) else {}
REP['gpu'] = torch.cuda.get_device_name(); REP['pid'] = os.getpid(); REP['cuda_visible'] = os.environ.get('CUDA_VISIBLE_DEVICES'); REP['frames'] = D['f1'] - D['f0']
movie = TiffMovie(D['movie']); cal = load_calib(D['calib']); Pt = torch.load(D['pattern'], weights_only=False)
r2 = lambda t: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items()}
save = lambda: json.dump(REP, open(out, 'w'), indent=1)


# shared GPUs: cap this process (env MEM_FRAC, default 0.45 of the card) so the other users' jobs keep their memory
torch.cuda.set_per_process_memory_fraction(float(os.environ.get('MEM_FRAC', 0.45)))
PEAK = {}


def timed(f):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time(); r = f(); torch.cuda.synchronize()
    PEAK['gb'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2); PEAK['t'] = [round(t0, 1), round(time.time(), 1)]
    return r, time.time() - t0


def diffs(Ra, Rb):
    if len(Ra['x']) != len(Rb['x']) or not np.array_equal(Ra['wstart'], Rb['wstart']):
        return dict(same_list=False, n=(len(Ra['x']), len(Rb['x'])))
    o = lambda R: np.lexsort((R['y_det'], R['x_det'], R['wstart']))
    ia, ib = o(Ra), o(Rb)
    dx = np.abs(np.c_[Ra['x'][ia] - Rb['x'][ib], Ra['y'][ia] - Rb['y'][ib]]).max(1); dI = np.abs(Ra['I'][ia] - Rb['I'][ib]).max(1)
    return dict(same_list=bool(np.abs(Ra['x_det'][ia] - Rb['x_det'][ib]).max() < 1e-3), dx_max=float(dx.max()), dx_med=float(np.median(dx)),
                dI_max=float(dI.max()), dI_med=float(np.median(dI)))


# warm-up (cuDNN / allocator / module load) on a short chunk, not timed
p1o = SF.Pass1Front(D['calib'], head=HEAD, det=DET); p1f = SFF.Pass1FrontFast(D['calib'], head=HEAD, det=DET)
p1f.det = p1o.det; p1f.front.det = p1o.det
p1o.run_chunk(movie, cal, D['f0'], D['f0'] + 300); p1f.run_chunk(movie, cal, D['f0'], D['f0'] + 300)
R1 = None
if 'orig' in variants:
    tm = {}; (R1, info), w = timed(lambda: p1o.run_chunk(movie, cal, D['f0'], D['f1'], timing=tm)); tm['wall'] = w; tm['peak_gb'] = PEAK['gb']; tm['epoch'] = PEAK['t']
    REP['orig_pass1'] = dict(r2(tm), n_det=int(len(R1['x'])), n_out=int(R1['out'].sum())); save(); print('orig pass1', REP['orig_pass1'], flush=True)
    tm = {}; _, w = timed(lambda: SF.run_chunk_pattern_solve(R1, info, Pt, D['sc_cal'], timing=tm)); tm['wall'] = w; tm['peak_gb'] = PEAK['gb']; tm['epoch'] = PEAK['t']
    REP['orig_solve'] = r2(tm); save(); print('orig pattern solve', REP['orig_solve'], flush=True)
    p2o = FR.Pass2Front(D['calib'], head=P2H, det=P2D, pf_head=PFH)
    p2o.run_chunk(movie, cal, D['f0'], D['f0'] + 300, Pt, D['p2_cal'])
    tm = {}; O, w = timed(lambda: p2o.run_chunk(movie, cal, D['f0'], D['f1'], Pt, D['p2_cal'], timing=tm)); tm['wall'] = w; tm['peak_gb'] = PEAK['gb']; tm['epoch'] = PEAK['t']
    REP['orig_pass2'] = dict(r2(tm), n=int(len(O['x_hy0']))); save(); print('orig pass2', REP['orig_pass2'], flush=True)
    del p2o
for v in ('fast', 'fastc'):
    if v not in variants: continue
    comp = 'on' if v == 'fastc' else 'off'
    tm = {}; (Rf, info), w = timed(lambda: p1f.run_chunk(movie, cal, D['f0'], D['f1'], timing=tm)); tm['wall'] = w; tm['peak_gb'] = PEAK['gb']; tm['epoch'] = PEAK['t']
    REP[f'{v}_pass1'] = dict(r2(tm), n_det=int(len(Rf['x'])), n_out=int(Rf['out'].sum()))
    if R1 is not None: REP[f'{v}_pass1']['vs_orig'] = diffs(R1, Rf)
    save(); print(v, 'pass1', REP[f'{v}_pass1'], flush=True)
    if comp == 'on':     # compile warm-up
        Lw = dict(Rf); Lw['out'] = Rf['out'] & (np.cumsum(Rf['out']) <= 20000); SFF.run_chunk_pattern_solve_fast(Lw, info, Pt, D['sc_cal'], compile='on')
    tm = {}; _, w = timed(lambda: SFF.run_chunk_pattern_solve_fast(Rf, info, Pt, D['sc_cal'], timing=tm, compile=comp)); tm['wall'] = w; tm['peak_gb'] = PEAK['gb']; tm['epoch'] = PEAK['t']
    REP[f'{v}_solve'] = r2(tm); save(); print(v, 'pattern solve', REP[f'{v}_solve'], flush=True)
    p2f = SFF.Pass2FrontFast(D['calib'], head=P2H, det=P2D, pf_head=PFH, compile=comp)
    p2f.run_chunk(movie, cal, D['f0'], D['f0'] + 600, Pt, D['p2_cal'])
    tm = {}; O, w = timed(lambda: p2f.run_chunk(movie, cal, D['f0'], D['f1'], Pt, D['p2_cal'], timing=tm)); tm['wall'] = w; tm['peak_gb'] = PEAK['gb']; tm['epoch'] = PEAK['t']
    REP[f'{v}_pass2'] = dict(r2(tm), n=int(len(O['x_hy0']))); save(); print(v, 'pass2', REP[f'{v}_pass2'], flush=True)
    del p2f
if 'heads' in variants:
    Rref = R1
    if Rref is None:
        Rref, _ = p1o.run_chunk(movie, cal, D['f0'], D['f0'] + 1500)
        f1h = D['f0'] + 1500
    else:
        f1h = D['f1']
    for hv, kw in (('hb4096', dict()), ('hb8192', dict(head_batch=8192)), ('hb2048', dict(head_batch=2048)), ('det128', dict(det_batch=128)),
                   ('nostreams', dict(streams=False))):
        tm = {}; (Rh, _), w = timed(lambda: p1f.run_chunk(movie, cal, D['f0'], f1h, timing=tm, **kw)); tm['wall'] = w; tm['peak_gb'] = PEAK['gb']; tm['epoch'] = PEAK['t']
        REP[f'head_{hv}'] = dict(r2(tm), frames=f1h - D['f0'], vs_orig=diffs(Rref, Rh)); save(); print(hv, REP[f'head_{hv}'], flush=True)
    torch.backends.cudnn.benchmark = True
    p1f.run_chunk(movie, cal, D['f0'], D['f0'] + 300)
    tm = {}; (Rh, _), w = timed(lambda: p1f.run_chunk(movie, cal, D['f0'], f1h, timing=tm)); tm['wall'] = w; tm['peak_gb'] = PEAK['gb']; tm['epoch'] = PEAK['t']
    REP['head_cudnn_benchmark'] = dict(r2(tm), frames=f1h - D['f0'], vs_orig=diffs(Rref, Rh)); save(); print('cudnn.benchmark', REP['head_cudnn_benchmark'], flush=True)
    torch.backends.cudnn.benchmark = False
    p1c = SFF.Pass1FrontFast(D['calib'], head=HEAD, det=DET, channels_last=True); p1c.det = p1o.det; p1c.front.det = p1o.det
    p1c.run_chunk(movie, cal, D['f0'], D['f0'] + 300)
    tm = {}; (Rh, _), w = timed(lambda: p1c.run_chunk(movie, cal, D['f0'], f1h, timing=tm)); tm['wall'] = w; tm['peak_gb'] = PEAK['gb']; tm['epoch'] = PEAK['t']
    REP['head_channels_last'] = dict(r2(tm), frames=f1h - D['f0'], vs_orig=diffs(Rref, Rh)); save(); print('channels_last', REP['head_channels_last'], flush=True)
print('BENCH DONE', name)
