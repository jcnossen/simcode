"""Per-stage wall / GPU profile of the production fast path on one real chunk (pass 1, pattern solve, pass 2, outputs).
Every stage is timed with CUDA-synchronised wall clocks (+ the pipeline's own CUDA-event GPU busy times), and the GPU
utilisation (nvidia-smi, 100 ms) is sampled per stage. A torch.profiler trace of a pass-1 sub-chunk gives the GPU busy
fraction (union of kernel intervals) and the idle-gap histogram.
usage: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<g> python analysis/speed/prof_stages.py <1809|20nM> [nf] [--trace_nf N]
"""
import sys, os, time, json, io
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, SC_CAL, P2_CAL, OUT, gpu_state, UtilSampler, dump
from simcode import config
from simcode.pipeline.common import Movie, load_calib, region
from simcode.pipeline import sc_front_fast as SFF

A = sys.argv[1:]
trace_nf = 300
if '--trace_nf' in A: i = A.index('--trace_nf'); trace_nf = int(A[i + 1]); del A[i:i + 2]
name = A[0]; nf = int(A[1]) if len(A) > 1 else 3000
D = DATA[name]; f0 = D['f0']; f1 = f0 + nf
SFF.CAND_PMIN = 0.3                         # production stored-candidate cut (simcode.run default)
calib = config.resolve(D['calib']); sc_cal = config.resolve(SC_CAL); p2_cal = config.resolve(P2_CAL)
REP = dict(dataset=name, f0=f0, nf=nf, gpu=torch.cuda.get_device_name(), gpu_state_start=gpu_state(), cand_pmin=0.3)
S = {}
def sync(): torch.cuda.synchronize()

def stage(key, fn):
    with UtilSampler() as us:
        sync(); t = time.time(); r = fn(); sync(); w = time.time() - t
    S[key] = dict(wall=round(w, 3), util=us.summary()); print(key, S[key], flush=True); return r

movie = Movie(D['movie']); cal = load_calib(calib); Pt = torch.load(D['pattern'], weights_only=False)
X0, Y0, Sz = region(movie, (5, 5, 110), 5)
# --- 1. I/O pieces in isolation
l0 = max(f0 - 12, 0) // 6 * 6; l1 = f1 + 12
def rd():
    return movie.read(l0, l1, Y0, X0, Sz)
mov = stage('io_tiff_read_decode', rd)
def rd_pages():          # raw page decode only (full frames), to split decode vs crop
    j = 0; o = None
    for fi in range(l0, min(l0 + 600, l1)):
        jj = int(np.searchsorted(movie.bounds, fi, side='right') - 1); o = movie.tifs[jj].pages[int(fi - movie.bounds[jj])].asarray()
    return o
stage('io_tiff_600_full_pages', rd_pages)
off = torch.tensor(cal['offset'][Y0:Y0 + Sz, X0:X0 + Sz], device='cuda')
stage('io_h2d_camera_correction', lambda: (torch.tensor(mov, device='cuda') - off) * cal['gain'])
# --- 2. pass 1 (production path), warm-up first
p1 = SFF.Pass1SCFast(calib)
p1.run_chunk(movie, cal, f0, f0 + 300)
tm = {}
R, info = stage('pass1_total', lambda: p1.run_chunk(movie, cal, f0, f1, timing=tm))
S['pass1_total']['events'] = {k: round(v, 3) for k, v in tm.items()}; S['pass1_total']['n_det'] = int(len(R['x'])); S['pass1_total']['n_out'] = int(R['out'].sum())
# candidate count -> per-candidate rates
S['pass1_total']['head_cand_per_s'] = round(len(R['x']) / tm['head'], 0)
# --- 3. outputs: pass-1 npz write (as run_uncalibrated)
os.makedirs(f'{OUT}/prof_{name}', exist_ok=True)
stage('write_pass1_npz', lambda: np.savez(f'{OUT}/prof_{name}/pass1_{f0}.npz', **R, X0=info['X0'], Y0=info['Y0'], S=info['S'],
                                          own=np.array(info['own']), loaded=np.array(info['loaded']), crop=np.array((5, 5, 110))))
S['write_pass1_npz']['mb'] = round(os.path.getsize(f'{OUT}/prof_{name}/pass1_{f0}.npz') / 2 ** 20, 1)
stage('read_pass1_npz', lambda: dict(np.load(f'{OUT}/prof_{name}/pass1_{f0}.npz')))
# --- 4. pattern solve (compiled LM step as production, eager for reference), warm-up on a small subset
L = dict(R); Lw = dict(R); Lw['out'] = R['out'] & (np.cumsum(R['out']) <= 20000)
from simcode import fastutil
comp = 'on' if fastutil.compile_available() else 'off'
SFF.run_chunk_pattern_solve_fast(Lw, info, Pt, sc_cal, compile=comp)
SFF.run_chunk_pattern_solve_fast(Lw, info, Pt, sc_cal, compile='off')
O = stage('solve_p1_compiled', lambda: SFF.run_chunk_pattern_solve_fast(L, info, Pt, sc_cal, compile=comp))
stage('solve_p1_eager', lambda: SFF.run_chunk_pattern_solve_fast(L, info, Pt, sc_cal, compile='off'))
stage('write_pass2_est_npz', lambda: np.savez(f'{OUT}/prof_{name}/pass2_est_{f0}.npz', **O, pattern=D['pattern']))
# --- 5. pass 2
p2 = SFF.Pass2P2Fast(calib, compile=comp)
p2.run_chunk(movie, cal, f0, f0 + 600, Pt, p2_cal)
tm2 = {}
O2 = stage('pass2_total', lambda: p2.run_chunk(movie, cal, f0, f1, Pt, p2_cal, timing=tm2))
S['pass2_total']['events'] = {k: round(v, 3) for k, v in tm2.items()}; S['pass2_total']['n'] = int(len(O2['x_hy0']))
from simcode.p2.front import direct_view
stage('write_pass2_est2_npz', lambda: (np.savez(f'{OUT}/prof_{name}/pass2_est2_{f0}.npz', **O2, pattern=D['pattern']),
                                       np.savez(f'{OUT}/prof_{name}/pass2_est2b_{f0}.npz', **direct_view(O2), pattern=D['pattern'])))
# --- 6. table + strict product on this chunk (as simcode.run step 6)
from simcode import strict as SP
px = json.load(open(calib))['pixelsize_um'] * 1e3
def tab(tag, nm):
    T_, inf = SP.build_table(f'{OUT}/prof_{name}', tag, px)
    return SP.write_table(T_, inf, f'{OUT}/prof_{name}/{nm}', ['npz'])
stage('table_strict_p1', lambda: tab('est', 'locs_p1'))
stage('table_strict_p2', lambda: tab('est2', 'locs_p2'))
# --- 7. torch.profiler trace of a pass-1 sub-chunk: GPU busy fraction, idle gaps, sync points
from torch.profiler import profile, ProfilerActivity
sync()
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    t = time.time(); p1.run_chunk(movie, cal, f0, f0 + trace_nf); sync(); wall = time.time() - t
ka = prof.key_averages()
gpu_total = sum(k.self_device_time_total for k in ka) / 1e6
syncs = {k.key: round(k.cpu_time_total / 1e3, 1) for k in ka if k.key in ('aten::nonzero', 'aten::item', 'aten::_local_scalar_dense', 'aten::index_put_', 'cudaStreamSynchronize', 'cudaDeviceSynchronize', 'cudaMemcpyAsync')}
S['trace_pass1'] = dict(frames=trace_nf, wall_s=round(wall, 3), gpu_kernel_time_s=round(gpu_total, 3), kernel_time_over_wall=round(gpu_total / wall, 3),
                        cpu_ms_sync_ops=syncs, note='kernel time summed over the 2 streams (can exceed wall when they overlap)')
print('trace', S['trace_pass1'], flush=True)
s_ = io.StringIO()
REP['kernels_top'] = prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=25, max_name_column_width=60)
REP['stages'] = S; REP['gpu_state_end'] = gpu_state()
dump(REP, f'{OUT}/prof_stages_{name}_{nf}.json'); print('saved', f'{OUT}/prof_stages_{name}_{nf}.json')
