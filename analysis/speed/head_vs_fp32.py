"""Head numerics on real candidates: each variant vs an fp32 reference (no autocast, no TF32) of the same head on the same
inputs (captured from the production queue, first `nb` batches of a real chunk). Answers: is a variant's deviation from
the production bf16 head larger than the production bf16 head's own deviation from fp32?
usage: python analysis/speed/head_vs_fp32.py <1809|20nM> [nb]"""
import sys, os, copy, json, contextlib
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, OUT, qstats, dump, gpu_state
from simcode2 import config
from simcode2.pipeline.common import Movie, load_calib
from simcode2.pipeline import sc_front_fast as SFF
name = sys.argv[1]; nb = int(sys.argv[2]) if len(sys.argv) > 2 else 6
D = DATA[name]; SFF.CAND_PMIN = 0.3
calib = config.resolve(D['calib']); movie = Movie(D['movie']); cal = load_calib(calib)
p = SFF.Pass1SCFast(calib)
CAP = []
orig = SFF.QueuedFront._head_call
def cap(self, nm, pt, off, nbp, ilp):
    if len(CAP) < nb: CAP.append((pt.clone(), off.clone(), nbp.clone()))
    return orig(self, nm, pt, off, nbp, ilp)
SFF.QueuedFront._head_call = cap
p.run_chunk(movie, cal, D['f0'], D['f0'] + (600 if name == '20nM' else 2400))
SFF.QueuedFront._head_call = orig
print('captured', len(CAP), 'batches', flush=True)
h = p.head; col = p.col
def run(m, bs=4096):
    out = {k: [] for k in ('I', 'x', 'bg')}
    with torch.no_grad():
        for pt, off, nbp in CAP:
            for i in range(0, len(pt), bs):
                o = m(pt[i:i + bs], None, off[i:i + bs], col(off[i:i + bs]), nbp[i:i + bs])
                out['I'].append(o['I'].float()); out['x'].append(o['dxy'].float()); out['bg'].append(o['bg'].float())
    return {k: torch.cat(v).cpu().numpy() for k, v in out.items()}
# fp32 reference
h.bf16 = False
tf = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
ref = run(h)
torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf
h.bf16 = True
V = {'bf16_eager': run(h), 'bf16_eager_bs2048': run(h, 2048), 'bf16_eager_bs1000': run(h, 1000)}
hc = copy.deepcopy(h).to(memory_format=torch.channels_last); V['bf16_channels_last'] = run(hc)
import torch._inductor.config as ic
ic.emulate_precision_casts = False; V['bf16_compile'] = run(torch.compile(copy.deepcopy(h), dynamic=False))
modes = [m for m in sys.argv[3].split(',')] if len(sys.argv) > 3 else []
for m in modes:
    q = SFF.QueuedFront(h, False, {'h': (h, 'sc')}, col, compile_head=m)
    hc_, _, _, tf_ = q.heads['h']
    V['mode_' + m] = run(lambda pt, r, off, psf, nbp: hc_(pt, r, off, psf, nbp, **({} if tf_ is None else dict(trunk_fn=tf_))))
ph = ref['I'].sum(1)
REP = dict(dataset=name, n=int(len(ph)), gpu_state=gpu_state(), vs_fp32={}, vs_bf16_eager={})
B = V['bf16_eager']
for k, v in V.items():
    REP['vs_fp32'][k] = dict(dI=qstats(v['I'] - ref['I']), dIrel=qstats((v['I'] - ref['I']).sum(1) / np.maximum(ph, 50)),
                             dxy=qstats(np.abs(v['x'] - ref['x']).max(1)), dbg=qstats(v['bg'] - ref['bg']))
    if k != 'bf16_eager':
        REP['vs_bf16_eager'][k] = dict(dI=qstats(v['I'] - B['I']), dxy=qstats(np.abs(v['x'] - B['x']).max(1)))
for sec in ('vs_fp32', 'vs_bf16_eager'):
    print(sec)
    for k, v in REP[sec].items():
        print(f"  {k:22s} dI med {v['dI']['med']:.3g} p99 {v['dI']['p99']:.3g} p99.9 {v['dI']['p999']:.3g} max {v['dI']['max']:.3g} | dxy med {v['dxy']['med']:.2e} p99 {v['dxy']['p99']:.2e} max {v['dxy']['max']:.2e}")
dump(REP, f'{OUT}/head_vs_fp32_{name}.json')
