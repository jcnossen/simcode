"""Train the 4Pi head (simcode/fourpi/head4pi.py) on on-the-fly simulated candidate windows (simcode/fourpi/sim4pi.py,
bead-fitted 4Pi PSF). Loss = (T, C) 2x2 Gaussian NLL per (frame, channel) + 3D prior NLL.

Usage: CUDA_VISIBLE_DEVICES=2 python analysis/fourpi/train_head4pi.py --tag v1 --steps 30000 [--batch 512]
Output: out/4pi/models/head4pi_<tag>.pt, log out/4pi/models/head4pi_<tag>.log
"""
import argparse, json, math, os, sys, time
import torch
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.config import root; P4 = root('fourpi')
from simcode.util.ckpt import save as safe_save   # atomic, disk-guarded
from simcode.fourpi.lut4pi import LUT4Pi
from simcode.simulate.fourpi_patches import FourPiPatchSim
from simcode.emitter.fourpi import FourPiPrior, nll_2x2, nll_prior

OUT = f'{P4}/models'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='v1'); p.add_argument('--steps', type=int, default=30000)
    p.add_argument('--batch', type=int, default=512); p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--ch', type=int, default=96); p.add_argument('--nblk', type=int, default=4)
    p.add_argument('--psf', default='NPC_f_dxy15'); p.add_argument('--seed', type=int, default=1)
    p.add_argument('--wprior', type=float, default=1.0)
    p.add_argument('--init', default='', help='checkpoint to continue from (same architecture)')
    p.add_argument('--template', default='gauss', help="'lut': 4Pi PSF template read-out (v7)")
    p.add_argument('--eps_range', type=float, default=0.3, help='pattern phase range at the patch centre (rad); pi = arbitrary (v3)')
    a = p.parse_args()
    from simcode.util.proctitle import set_title; set_title('4pi-train')     # recognizable name in nvidia-smi / ps
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(a.seed)
    st = json.load(open(f'{P4}/psf4pi/{a.psf}.json'))['psf']
    lut = LUT4Pi(st, P=15)
    sim = FourPiPatchSim(lut, seed=a.seed)
    net = FourPiPrior(ch=a.ch, nblk=a.nblk).cuda()
    if a.init:
        net.load_state_dict(torch.load(a.init if a.init.startswith('/') else f'{REPO}/{a.init}', weights_only=False)['state'])
    if a.template == 'lut':
        net.tlut = LUT4Pi(st, P=15, sub=1)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    log = open(f'{OUT}/head4pi_{a.tag}.log', 'a')
    t0 = time.time(); acc = []
    for it in range(a.steps):
        with torch.no_grad():
            S = sim.sample(a.batch, eps_range=a.eps_range)
        o = net(S['img'], S['doff'])
        lI = nll_2x2(S['T'], S['C'], o).sum((1, 2)) / 48
        lp = nll_prior(S['x'], S['y'], S['z'], o) / 3
        L = (lI + a.wprior * lp).mean()
        opt.zero_grad(); L.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step(); sch.step()
        acc.append([L.item(), lI.mean().item(), lp.mean().item()])
        if (it + 1) % 250 == 0:
            m = torch.tensor(acc).mean(0).tolist(); acc = []
            with torch.no_grad():
                eT = ((o['T'] - S['T']) / (S['T'].clamp(min=0) + 10).sqrt()).std().item()
                ez = (o['z'] - S['z']).abs().median().item() * 1000
            s = f'it {it + 1} loss {m[0]:.4f} nllI {m[1]:.4f} nllp {m[2]:.4f} Tres/sqrtT {eT:.3f} |dz| med {ez:.1f} nm lr {sch.get_last_lr()[0]:.2e} t {time.time() - t0:.0f}s'
            print(s, flush=True); log.write(s + '\n'); log.flush()
        if (it + 1) % 5000 == 0 or it + 1 == a.steps:
            safe_save(dict(state=net.state_dict(), args=vars(a), psf=st, it=it + 1), f'{OUT}/head4pi_{a.tag}.pt')


if __name__ == '__main__':
    main()
