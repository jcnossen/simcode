"""Batch-size sweep of the queued front-end (QueuedFront): head_batch x det_batch, pass 1 and pass 2, on a real chunk.
Per setting: wall per chunk, head GPU time, candidates/s, peak GPU memory, final-batch padding waste, and the output
difference vs the default (4096 / 64) setting (I, prior x/y). Settings alternate across repetitions (A B C .. C B A).
usage: python analysis/speed/batch_sweep.py <1809|20nM> <nf> [reps] [passes p1,p2]"""
import sys, os, time, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, P2_CAL, OUT, gpu_state, canon, qstats, dump
from simcode2 import config
from simcode2.pipeline.common import Movie, load_calib
from simcode2.pipeline import sc_front_fast as SFF
name, nf = sys.argv[1], int(sys.argv[2]); reps = int(sys.argv[3]) if len(sys.argv) > 3 else 2
passes = sys.argv[4].split(',') if len(sys.argv) > 4 else ['p1', 'p2']
D = DATA[name]; f0 = D['f0']; f1 = f0 + nf; SFF.CAND_PMIN = 0.3
calib = config.resolve(D['calib']); p2_cal = config.resolve(P2_CAL)
movie = Movie(D['movie']); cal = load_calib(calib); Pt = torch.load(D['pattern'], weights_only=False)
SET = [(hb, 64) for hb in (1024, 2048, 4096, 8192, 16384)] + [(4096, db) for db in (32, 128, 256)]
p1 = SFF.Pass1SCFast(calib); p2 = SFF.Pass2P2Fast(calib, compile='off')
REP = dict(dataset=name, f0=f0, nf=nf, gpu=torch.cuda.get_device_name(), gpu_state_start=gpu_state(), runs=[], acc={})
out_f = f'{OUT}/batch_sweep_{name}_{nf}.json'
def run(ps, hb, db):
    tm = {}
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); t = time.time()
    if ps == 'p1': R, _ = p1.run_chunk(movie, cal, f0, f1, timing=tm, head_batch=hb, det_batch=db); n = len(R['x'])
    else: R = p2.run_chunk(movie, cal, f0, f1, Pt, p2_cal, timing=tm, head_batch=hb, det_batch=db); n = int(tm.get('_n', 0)) or len(R['x_hy0'])
    torch.cuda.synchronize(); w = time.time() - t
    return R, dict(wall=round(w, 3), head=round(tm['head'], 3), detector=round(tm['detector'], 3), solve=round(tm.get('solve', 0), 3),
                   peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
# warm-up every shape once (cuDNN plans, allocator)
for ps in passes:
    for hb, db in SET:
        if ps == 'p1': p1.run_chunk(movie, cal, f0, f0 + 150, head_batch=hb, det_batch=db)
        else: p2.run_chunk(movie, cal, f0, f0 + 150, Pt, p2_cal, head_batch=hb, det_batch=db)
ref = {}; KEEP = {}
for r in range(reps):
    for ps in passes:
        for hb, db in (SET if r % 2 == 0 else SET[::-1]):
            R, m = run(ps, hb, db)
            ncand = len(R['x']) if ps == 'p1' else None
            if ps == 'p1':
                m['n_cand'] = ncand; m['cand_per_s_wall'] = round(ncand / m['wall']); m['cand_per_s_head'] = round(ncand / m['head'])
                m['pad_frac_last_batch'] = round(((-ncand) % hb) / max(ncand, 1), 4)
            else:
                m['n_rows'] = len(R['x_hy0'])
            m.update(pass_=ps, head_batch=hb, det_batch=db, rep=r, gpu_util_all=gpu_state())
            REP['runs'].append(m); print(json.dumps({k: v for k, v in m.items() if k != 'gpu_util_all'}), flush=True)
            if r == 0:
                Rc = canon(R) if ps == 'p1' else {k: v[np.lexsort((R['y_det'], R['x_det'], R['wstart']))] for k, v in R.items() if np.ndim(v) and len(v) == len(R['wstart'])}
                if (hb, db) == (4096, 64): ref[ps] = Rc
                else: KEEP[f'{ps}_{hb}_{db}'] = Rc
            dump(REP, out_f)
acc = {}
for k, Rc in KEEP.items():
    ps = k.split('_')[0]; Rr = ref[ps]
    if len(Rc['wstart']) != len(Rr['wstart']) or not np.array_equal(Rc['wstart'], Rr['wstart']):
        acc[k] = dict(same_rows=False, n=(len(Rc['wstart']), len(Rr['wstart']))); continue
    if ps == 'p1': acc[k] = dict(same_rows=True, dI=qstats(Rc['I'] - Rr['I']), dxy=qstats(np.abs(np.c_[Rc['x'] - Rr['x'], Rc['y'] - Rr['y']]).max(1)))
    else: acc[k] = dict(same_rows=True, dI=qstats(Rc['I_net'] - Rr['I_net']), dprior=qstats(np.abs(np.c_[Rc['x_prior'] - Rr['x_prior'], Rc['y_prior'] - Rr['y_prior']]).max(1)),
                        dsolve=qstats(np.abs(np.c_[Rc['x_hy0'] - Rr['x_hy0'], Rc['y_hy0'] - Rr['y_hy0']]).max(1)))
    print(k, json.dumps(acc[k]), flush=True)
REP['acc'] = acc
# per-drain torch.cat of the pending list: GPU time of one drain's cat at the typical sizes (6 fields, head_batch + one detector piece)
dev = 'cuda'; cats = {}
for hb in (1024, 4096, 16384):
    pend = [dict(w=torch.zeros(m_, dtype=torch.long, device=dev), iy=torch.zeros(m_, dtype=torch.long, device=dev), ix=torch.zeros(m_, dtype=torch.long, device=dev),
                 x=torch.zeros(m_, device=dev), y=torch.zeros(m_, device=dev), lg=torch.zeros(m_, device=dev)) for m_ in (hb // 2, hb // 2, 3000)]
    torch.cuda.synchronize(); t = time.time()
    for _ in range(200): cat = {k: torch.cat([p_[k] for p_ in pend]) for k in pend[0]}
    torch.cuda.synchronize(); cats[hb] = round((time.time() - t) / 200 * 1e3, 4)
REP['drain_cat_ms_per_drain'] = cats; print('drain cat ms per drain', cats)
REP['gpu_state_end'] = gpu_state(); dump(REP, out_f); print('saved', out_f)
