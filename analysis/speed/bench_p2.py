"""Pass-2 (p2_det + p2_head_long + pattern solve) variants on a real chunk: speed (alternating, repeated) and accuracy vs
the production setting, noise reference = head_batch 2048. Rows matched on (window, detector point).
usage: python analysis/speed/bench_p2.py <1809|20nM> <nf> <variants: base,noise,trunk> [reps]"""
import sys, os, time, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, P2_CAL, OUT, gpu_state, UtilSampler, qstats, dump
from simcode2 import config, fastutil
from simcode2.pipeline.common import Movie, load_calib
from simcode2.pipeline import sc_front_fast as SFF
name, nf, vs = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(',')
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 2
D = DATA[name]; f0 = D['f0']; f1 = f0 + nf; SFF.CAND_PMIN = 0.3
calib = config.resolve(D['calib']); p2_cal = config.resolve(P2_CAL)
movie = Movie(D['movie']); cal = load_calib(calib); Pt = torch.load(D['pattern'], weights_only=False)
comp = 'on' if fastutil.compile_available() else 'off'
P = {}
for v in vs:
    p = SFF.Pass2P2Fast(calib, compile=comp, compile_head='trunk' if v == 'trunk' else 'off')
    kw = dict(head_batch=2048) if v == 'noise' else {}
    p.run_chunk(movie, cal, f0, f0 + 300, Pt, p2_cal, **kw); P[v] = (p, kw)
REP = dict(dataset=name, f0=f0, nf=nf, gpu=torch.cuda.get_device_name(), gpu_state_start=gpu_state(), runs=[], acc={})
out_f = f'{OUT}/bench_p2_{name}_{nf}_{"_".join(vs)}.json'; Rs = {}
for r in range(reps):
    for v in (vs if r % 2 == 0 else vs[::-1]):
        p, kw = P[v]; tm = {}; gs = gpu_state()
        with UtilSampler() as us:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
            R = p.run_chunk(movie, cal, f0, f1, Pt, p2_cal, timing=tm, **kw); torch.cuda.synchronize(); w = time.time() - t
        REP['runs'].append(dict(variant=v, rep=r, wall=round(w, 3), head=round(tm['head'], 3), detector=round(tm['detector'], 3), solve=round(tm['solve'], 3),
                                n=int(len(R['x_hy0'])), util=us.summary(), peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), gpu_state=gs))
        print(json.dumps({k: x for k, x in REP['runs'][-1].items() if k != 'gpu_state'}), flush=True)
        if r == 0: Rs[v] = R
        dump(REP, out_f)
key = lambda R: {(int(a), round(float(b), 4), round(float(c), 4)): i for i, (a, b, c) in enumerate(zip(R['wstart'], R['x_det'], R['y_det']))}
Kb = key(Rs['base'])
for v in vs:
    if v == 'base': continue
    Kv = key(Rs[v]); com = [k for k in Kv if k in Kb]
    ib = np.array([Kb[k] for k in com]); iv = np.array([Kv[k] for k in com])
    A, B = Rs['base'], Rs[v]
    a = dict(n_base=len(Kb), n_var=len(Kv), n_common=len(com),
             dI=qstats(B['I_net'][iv] - A['I_net'][ib]), dprior=qstats(np.abs(np.c_[B['x_prior'][iv] - A['x_prior'][ib], B['y_prior'][iv] - A['y_prior'][ib]]).max(1)),
             dsolve=qstats(np.abs(np.c_[B['x_hy0'][iv] - A['x_hy0'][ib], B['y_hy0'][iv] - A['y_hy0'][ib]]).max(1)),
             dsig=qstats(B['sig_hy0'][iv] - A['sig_hy0'][ib]), kin_equal=float((B['kin_hy'][iv] == A['kin_hy'][ib]).mean()))
    REP['acc'][v] = a; print(v, json.dumps(a), flush=True)
REP['gpu_state_end'] = gpu_state(); dump(REP, out_f); print('saved', out_f)
