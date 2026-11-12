"""Train head4pi v4: the pass-1 head architecture (simcode2/fourpi/head4pi_p1.Head4PiP1, low-rank covariance) with the
background per (angle, channel) and the sc4 anti-shrinkage terms, on candidate patches (simcode2/fourpi/sim4pi.py) with the
pattern phase uniform over 2 pi (v3 fix); neighbour / detector inputs zero (patch scenes). Fixes from the CRLB
decomposition (PROCESSING_STATUS.md 23:20): background error dominated the read-out noise, frame profile shrunk to 0.90.
Loss = low-rank NLL / 48 + prior NLL / 3 + w_mr x mean-removed + w_slope x slope penalty + w_bg x supervised background
(v3 had no background loss: its bg output was unconstrained and the learned correction carried the read-out). Init: v3 weights.
Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi/train_head4pi_v4.py --tag v4 --steps 60000
"""
import argparse, json, math, os, sys, time
import torch
REPO = '/home/jc/simcode2'
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode2.fourpi.ckpt import save as safe_save   # atomic, disk-guarded
from simcode2.fourpi.lut4pi import LUT4Pi
from simcode2.fourpi.sim4pi import Sim4Pi
from simcode2.fourpi.head4pi import nll_prior
from simcode2.fourpi.head4pi_p1 import Head4PiP1, nll_lowrank, mean_removed, slope_penalty

OUT = f'{REPO}/out/4pi/models'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='v4'); p.add_argument('--steps', type=int, default=60000)
    p.add_argument('--batch', type=int, default=512); p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--init', default='out/4pi/models/head4pi_v3.pt'); p.add_argument('--psf', default='NPC_f_dxy15')
    p.add_argument('--w_mr', type=float, default=2.0); p.add_argument('--w_slope', type=float, default=50.0)
    p.add_argument('--seed', type=int, default=4); p.add_argument('--w_bg', type=float, default=0.1)
    p.add_argument('--template', default='gauss', help="'gauss' (v4) or 'lut' (v5: 4Pi PSF template at the head's prior)")
    a = p.parse_args()
    from simcode2.util.proctitle import auto; auto('4pi-train4')
    torch.manual_seed(a.seed)
    st = json.load(open(f'{REPO}/out/4pi/psf4pi/{a.psf}.json'))['psf']
    lut = LUT4Pi(st, P=15); sim = Sim4Pi(lut, seed=a.seed)
    ck = torch.load(f'{REPO}/{a.init}', weights_only=False)
    net = Head4PiP1.from_v3(ck, bg_mode='angle').cuda()
    if a.template == 'lut':
        net.tlut = LUT4Pi(st, P=15, sub=1)   # template only: 1 px knots are enough (speed)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    log = open(f'{OUT}/head4pi_{a.tag}.log', 'a'); t0 = time.time(); acc = []
    for it in range(a.steps):
        with torch.no_grad():
            S = sim.sample(a.batch, eps_range=math.pi)
        o = net(S['img'], S['doff'])
        li = nll_lowrank(S['T'], S['C'], o) / 48
        lp = nll_prior(S['x'], S['y'], S['z'], o) / 3
        lmr = mean_removed(S['T'], o)
        lsl = slope_penalty(S['T'], o)
        # supervised background (as sc4): the true background at the patch centre, angle mean (bg is per (angle, channel))
        bgt = S['bg'][..., 7, 7]; bgt = bgt.reshape(-1, 2, 3, 4).mean(2)[:, [0, 0, 0, 1, 1, 1]]
        lbg = ((o['bg'] - bgt) ** 2 / (bgt + 1.0)).mean((1, 2))
        L = (li + lp + a.w_mr * lmr + a.w_bg * lbg).mean() + a.w_slope * lsl
        opt.zero_grad(); L.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sch.step()
        acc.append([L.item(), li.mean().item(), lp.mean().item(), lmr.mean().item(), lsl.item()])
        if (it + 1) % 250 == 0:
            m = torch.tensor(acc).mean(0).tolist(); acc = []
            with torch.no_grad():
                bge = (o['bg'] - S['bg'][..., 7, 7]).abs().median().item()
                eT = (o['T'] - S['T']).pow(2).mean().sqrt().item()
            s = (f'it {it + 1} loss {m[0]:.4f} nllI {m[1]:.4f} nllp {m[2]:.4f} mr {m[3]:.3f} slope {m[4]:.5f} |bg err| med {bge:.3f} '
                 f'T rms err {eT:.1f} lr {sch.get_last_lr()[0]:.2e} t {time.time() - t0:.0f}s')
            print(s, flush=True); log.write(s + '\n'); log.flush()
        if (it + 1) % 2500 == 0 or it + 1 == a.steps:
            safe_save(dict(state=net.state_dict(), args=dict(vars(a), ch=ck['args']['ch'], nblk=ck['args']['nblk'], arch='p1', bg_mode='angle', template=a.template),
                            psf=st, it=it + 1), f'{OUT}/head4pi_{a.tag}.pt')


if __name__ == '__main__':
    main()
