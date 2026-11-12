"""Train the structured 4Pi head (simcode2/fourpi/head4pi_s.py, v8) on on-the-fly simulated candidate windows.
Loss = (T, C) 2x2 Gaussian NLL per (frame, channel) + 3D prior NLL + w_bg x Huber of the per-channel background.
Recipe otherwise as train_head4pi.py (v3): same simulator, arbitrary pattern phase, AdamW + OneCycle.

Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi/train_head4pi_s.py --tag v8 --steps 40000 --init_v3 out/4pi/models/head4pi_v3L.pt
Resume: --init out/4pi/models/head4pi_<tag>.pt --steps <remaining>   (checkpoint every 2500 steps, atomic)
Output: out/4pi/models/head4pi_<tag>.pt, log out/4pi/models/head4pi_<tag>.log
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F
REPO = '/home/jc/simcode2'
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode2.fourpi.ckpt import save as safe_save
from simcode2.fourpi.lut4pi import LUT4Pi
from simcode2.fourpi.sim4pi import Sim4Pi
from simcode2.fourpi.head4pi import nll_2x2, nll_prior
from simcode2.fourpi.head4pi_s import Head4PiS

OUT = f'{REPO}/out/4pi/models'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='v8'); p.add_argument('--steps', type=int, default=40000)
    p.add_argument('--batch', type=int, default=512); p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--ch', type=int, default=96); p.add_argument('--nblk', type=int, default=4)
    p.add_argument('--psf', default='NPC_f_dxy15'); p.add_argument('--seed', type=int, default=1)
    p.add_argument('--wprior', type=float, default=1.0); p.add_argument('--w_bg', type=float, default=0.1)
    p.add_argument('--init', default='', help='Head4PiS checkpoint to continue from')
    p.add_argument('--init_v3', default='', help='Head4Pi (v3-type) checkpoint: trunk + prior rows')
    p.add_argument('--eps_range', type=float, default=math.pi)
    p.add_argument('--pct', type=float, default=0.05)
    a = p.parse_args(); a.arch = 's'
    from simcode2.util.proctitle import auto; auto('4pi-train-s')
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(a.seed)
    st = json.load(open(f'{REPO}/out/4pi/psf4pi/{a.psf}.json'))['psf']
    lut = LUT4Pi(st, P=15)
    sim = Sim4Pi(lut, seed=a.seed)
    net = Head4PiS(ch=a.ch, nblk=a.nblk).cuda()
    ld = lambda f: torch.load(f if f.startswith('/') else f'{REPO}/{f}', weights_only=False)['state']
    if a.init: net.load_state_dict(ld(a.init))
    elif a.init_v3: net.load_v3(ld(a.init_v3))
    net.tlut = LUT4Pi(st, P=15, sub=1)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=a.pct)
    log = open(f'{OUT}/head4pi_{a.tag}.log', 'a')
    t0 = time.time(); acc = []
    for it in range(a.steps):
        with torch.no_grad():
            S = sim.sample(a.batch, eps_range=a.eps_range)
        o = net(S['img'], S['doff'])
        lI = nll_2x2(S['T'], S['C'], o).sum((1, 2)) / 48
        lp = nll_prior(S['x'], S['y'], S['z'], o) / 3
        bt = S['bg'][:, 0].mean((-1, -2))                                       # [n,4] true flat level per channel
        lb = F.huber_loss((o['bgc'] - bt) / 0.1, torch.zeros_like(bt), delta=3.0, reduction='none').mean(1)
        L = (lI + a.wprior * lp + a.w_bg * lb).mean()
        if not torch.isfinite(L):
            print(f'it {it + 1}: non-finite loss, step skipped', flush=True); opt.zero_grad(); sch.step(); continue
        opt.zero_grad(); L.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step(); sch.step()
        acc.append([L.item(), lI.mean().item(), lp.mean().item(), lb.mean().item()])
        if (it + 1) % 250 == 0:
            m = torch.tensor(acc).mean(0).tolist(); acc = []
            with torch.no_grad():
                iso = (S['nnb'] == 0) & (S['q'].min(1).values > 0.99)
                eT = ((o['T'] - S['T'])[iso] / (S['T'][iso].clamp(min=0) + 10).sqrt()).std().item()
                eb = (o['bgc'] - bt).pow(2).mean().sqrt().item()
            s = f'it {it + 1} loss {m[0]:.4f} nllI {m[1]:.4f} nllp {m[2]:.4f} lbg {m[3]:.3f} bg rms {eb:.3f} Tres/sqrtT iso {eT:.3f} lr {sch.get_last_lr()[0]:.2e} t {time.time() - t0:.0f}s'
            print(s, flush=True); log.write(s + '\n'); log.flush()
        if (it + 1) % 2500 == 0 or it + 1 == a.steps:
            safe_save(dict(state=net.state_dict(), args=vars(a), psf=st, it=it + 1), f'{OUT}/head4pi_{a.tag}.pt')


if __name__ == '__main__':
    main()
