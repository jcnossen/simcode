"""Pass-1 head variants on a real chunk: speed (back-to-back, alternating, repeated) and accuracy vs the production eager
head, with the noise reference = production head at head_batch 2048 (other cuDNN algorithms). Accuracy at the head (I, prior
x/y) and after the pattern solve (final x/y, kinetics decisions).
usage: python analysis/speed/bench_head.py <1809|20nM> <nf> <variants comma list> [reps]
variants: base | noise | cl | comp | comp_emu | blk | blk_emu | <any QueuedFront compile_head mode as mode:<name>>"""
import sys, os, time, json, copy
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, SC_CAL, OUT, gpu_state, UtilSampler, canon, qstats, dump
from simcode2 import config, fastutil
from simcode2.pipeline.common import Movie, load_calib
from simcode2.pipeline import sc_front_fast as SFF
name, nf, vs = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(',')
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 2
D = DATA[name]; f0 = D['f0']; f1 = f0 + nf
SFF.CAND_PMIN = 0.3
calib = config.resolve(D['calib']); sc_cal = config.resolve(SC_CAL)
movie = Movie(D['movie']); cal = load_calib(calib); Pt = torch.load(D['pattern'], weights_only=False)
base = SFF.Pass1SCFast(calib)
P = {}
def mk(v):
    import torch._inductor.config as ic
    ic.emulate_precision_casts = v.endswith('_emu')
    kw = {}
    if v in ('base', 'noise'): return base, ({} if v == 'base' else dict(head_batch=2048))
    if v == 'cl': p = SFF.Pass1SCFast(calib, channels_last=True)
    elif v.startswith('comp'): p = SFF.Pass1SCFast(calib, compile_head='on')
    elif v.startswith('blk'): p = SFF.Pass1SCFast(calib, compile_head='blk')
    elif v.startswith('mode:'): p = SFF.Pass1SCFast(calib, compile_head=v[5:])
    elif v.startswith('clmode:'): p = SFF.Pass1SCFast(calib, compile_head=v[7:], channels_last=True)
    else: raise ValueError(v)
    return p, kw
REP = dict(dataset=name, f0=f0, nf=nf, gpu=torch.cuda.get_device_name(), gpu_state_start=gpu_state(), runs=[])
out_f = f'{OUT}/bench_head_{name}_{nf}_{"_".join(v.replace(":", "-") for v in vs)}.json'
for v in vs:
    p, kw = mk(v); P[v] = (p, kw)
    t = time.time(); p.run_chunk(movie, cal, f0, f0 + 300, **kw); print(f'warm-up {v} {time.time() - t:.1f}s', flush=True)
Rs = {}
for r in range(reps):
    for v in (vs if r % 2 == 0 else vs[::-1]):
        p, kw = P[v]; tm = {}; gs = gpu_state()
        with UtilSampler() as us:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
            R, info = p.run_chunk(movie, cal, f0, f1, timing=tm, **kw); torch.cuda.synchronize(); w = time.time() - t
        REP['runs'].append(dict(variant=v, rep=r, wall=round(w, 3), head=round(tm['head'], 3), detector=round(tm['detector'], 3),
                                n=int(len(R['x'])), cand_per_s=round(len(R['x']) / w), util=us.summary(),
                                peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), gpu_state=gs))
        print(REP['runs'][-1], flush=True)
        if r == 0: Rs[v] = (canon(R), info)
        dump(REP, out_f)
# accuracy
comp = 'on' if fastutil.compile_available() else 'off'
Rb, info = Rs['base']
Lw = dict(Rb); Lw['out'] = Rb['out'] & (np.cumsum(Rb['out']) <= 20000)
SFF.run_chunk_pattern_solve_fast(Lw, info, Pt, sc_cal, solve_opts=dict(compile=comp))
REP['acc'] = {}
for v in vs:
    if v == 'base': continue
    Rv, _ = Rs[v]
    a = dict(same_cands=bool(len(Rv['x']) == len(Rb['x']) and np.array_equal(Rv['wstart'], Rb['wstart'])))
    if a['same_cands']:
        a['dxdet_max'] = float(np.abs(Rv['x_det'] - Rb['x_det']).max())
        a['head_dI'] = qstats(Rv['I'] - Rb['I']); a['head_dxy'] = qstats(np.c_[Rv['x'] - Rb['x'], Rv['y'] - Rb['y']].max(1) if False else np.abs(np.c_[Rv['x'] - Rb['x'], Rv['y'] - Rb['y']]).max(1))
        a['out_equal'] = float((Rv['out'] == Rb['out']).mean())
        if True:     # solve on the rows that are 'out' in both (the time-centroid flag flips on ~1e-4 of the rows)
            com = Rv['out'] & Rb['out']
            Ob = SFF.run_chunk_pattern_solve_fast(dict(Rb, out=com), info, Pt, sc_cal, solve_opts=dict(compile=comp))
            Ov = SFF.run_chunk_pattern_solve_fast(dict(Rv, out=com), info, Pt, sc_cal, solve_opts=dict(compile=comp))
            a['solve_dr'] = qstats(np.abs(np.c_[Ov['x_hy0'] - Ob['x_hy0'], Ov['y_hy0'] - Ob['y_hy0']]).max(1))
            a['solve_dsig'] = qstats(Ov['sig_hy0'] - Ob['sig_hy0'])
            a['kin_equal'] = float((Ov['kin_hy'] == Ob['kin_hy']).mean())
            a['n_solve'] = int(len(Ov['x_hy0']))
    REP['acc'][v] = a; print(v, json.dumps({k: a[k] for k in a if k in ('same_cands', 'out_equal', 'solve_dr', 'kin_equal', 'n_solve', 'head_dI')}), flush=True)
REP['gpu_state_end'] = gpu_state(); dump(REP, out_f); print('saved', out_f)
