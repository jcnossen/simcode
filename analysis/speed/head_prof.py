"""Kernel-level profile of one pass-1 head batch (SCHead4 / hd_sc5nb, 4096 candidates) on synthetic inputs."""
import sys, time, torch
from simcode2 import config
from simcode2.hd import sc as SCM
from simcode2.sim.simulator import Simulator
torch.manual_seed(0)
dev = 'cuda'
h = SCM.load(config.model(config.HEAD))
col = SCM.PSFCol(Simulator(config.model(config.INSITU_05), H=32, W=32, T=6, device=dev))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
pt = 20 + 5 * torch.randn(n, 6, 21, 21, device=dev); pt[:, :, 8:13, 8:13] += 300
off = 0.5 * (torch.rand(n, 2, device=dev) - 0.5)
nb = torch.rand(n, 2, 21, 21, device=dev) * 0.2
def f():
    psf = col(off); return h(pt, None, off, psf, nb)
with torch.no_grad():
    for _ in range(3): f()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(10): f()
    torch.cuda.synchronize(); dt = (time.time() - t) / 10
    print(f'head batch {n}: {dt*1e3:.1f} ms -> {n/dt:.0f} cand/s', flush=True)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for _ in range(3): f()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25, max_name_column_width=70))
