"""Speed of head variants on one synthetic 4096 batch (timing only): eager, channels_last, cudnn.benchmark, compile(+emulate casts)."""
import sys, time, copy, torch
from simcode2 import config
from simcode2.hd import sc as SCM
from simcode2.sim.simulator import Simulator
torch.manual_seed(0)
dev = 'cuda'
h = SCM.load(config.model(config.HEAD))
col = SCM.PSFCol(Simulator(config.model(config.INSITU_05), H=32, W=32, T=6, device=dev))
n = 4096
pt = 20 + 5 * torch.randn(n, 6, 21, 21, device=dev); pt[:, :, 8:13, 8:13] += 300
off = 0.5 * (torch.rand(n, 2, device=dev) - 0.5)
nb = torch.rand(n, 2, 21, 21, device=dev) * 0.2
psf = col(off)
def bench(m, tag, k=10):
    with torch.no_grad():
        for _ in range(3): o = m(pt, None, off, psf, nb)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(k): o = m(pt, None, off, psf, nb)
        torch.cuda.synchronize(); dt = (time.time() - t) / k
    print(f'{tag:28s} {dt*1e3:7.1f} ms  {n/dt:8.0f} cand/s', flush=True); return o
vs = sys.argv[1].split(',') if len(sys.argv) > 1 else ['eager', 'cl', 'bench', 'clbench', 'compile', 'compile_emu']
ref = bench(h, 'eager')
for v in vs:
    if v == 'eager': continue
    if v in ('cl', 'clbench'):
        hc = copy.deepcopy(h).to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = v == 'clbench'
        bench(hc, v); torch.backends.cudnn.benchmark = False
    if v == 'bench':
        torch.backends.cudnn.benchmark = True; bench(h, v); torch.backends.cudnn.benchmark = False
    if v.startswith('compile'):
        import torch._inductor.config as ic, torch._dynamo
        torch._dynamo.reset()
        ic.emulate_precision_casts = v == 'compile_emu'
        hc = torch.compile(copy.deepcopy(h), dynamic=False)
        o = bench(hc, v)
        print('   max|dI| vs eager', float((o['I'] - ref['I']).abs().max()), 'med', float((o['I'] - ref['I']).abs().median()))
