"""Train the learned-window 4Pi head (simcode2/fourpi/head4pi_w.py, v10: frame-symmetric, emitter-orthogonal window changes; frozen v3L prior) with the solve-aware loss
(simcode2/fourpi/solveloss4pi.py) on the patch simulator (neighbours p 0.5, kinetics, arbitrary pattern phase, as v3).

Loss = NLL of the linearised solve's (x, y, z) error under its reported covariance  + prior NLL  + w_bg x background
Huber  + w_ro x read-out NLL with the shared covariance scale (keeps the read-outs themselves and chi2 meaningful; small,
because the per-read-out NLL is what makes a free read-out mix frames).
Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi/train_head4pi_w.py --tag v10 --steps 20000 --init_v3 out/4pi/models/head4pi_v3L.pt
Resume: --init out/4pi/models/head4pi_<tag>.pt --steps <remaining>. Checkpoint every 2500 steps (atomic).
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F
REPO = '/home/jc/simcode2'
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode2.fourpi.ckpt import save as safe_save
from simcode2.fourpi.lut4pi import LUT4Pi
from simcode2.fourpi.sim4pi import Sim4Pi
from simcode2.fourpi.head4pi import nll_prior
from simcode2.fourpi.head4pi_k import WINS_K
from simcode2.fourpi.head4pi_w import Head4PiW
from simcode2.fourpi.solvek4pi import SolveK
from simcode2.fourpi.solveloss4pi import solve_loss

OUT = f'{REPO}/out/4pi/models'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='v10'); p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--batch', type=int, default=512); p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--ch', type=int, default=64); p.add_argument('--nblk', type=int, default=4)
    p.add_argument('--psf', default='NPC_f_dxy15'); p.add_argument('--seed', type=int, default=1)
    p.add_argument('--w_bg', type=float, default=0.1); p.add_argument('--w_ro', type=float, default=0.02)
    p.add_argument('--bound', type=float, default=1.0)
    p.add_argument('--init', default=''); p.add_argument('--init_v3', default='')
    p.add_argument('--p_nb', type=float, default=0.5)
    a = p.parse_args(); a.arch = 'w'; a.wins = [list(w) for w in WINS_K]
    from simcode2.util.proctitle import auto; auto('4pi-train-w')
    torch.manual_seed(a.seed)
    st = json.load(open(f'{REPO}/out/4pi/psf4pi/{a.psf}.json'))['psf']
    lut = LUT4Pi(st, P=15)
    sim = Sim4Pi(lut, seed=a.seed)
    net = Head4PiW(ch=a.ch, nblk=a.nblk, bound=a.bound).cuda(); net.lut = lut
    ld = lambda f: torch.load(f if f.startswith('/') else f'{REPO}/{f}', weights_only=False)['state']
    if a.init: net.load_state_dict(ld(a.init))
    elif a.init_v3: net.prior.load_state_dict(ld(a.init_v3))
    net.train()
    sv = SolveK(lut, WINS_K); sv.tab_lim = None
    opt = torch.optim.AdamW([q for q in net.parameters() if q.requires_grad], lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    log = open(f'{OUT}/head4pi_{a.tag}.log', 'a')
    t0 = time.time(); acc = []
    for it in range(a.steps):
        with torch.no_grad():
            S = sim.sample(a.batch, eps_range=math.pi, p_nb=a.p_nb)
        o = net(S['img'], S['doff'])
        Ystar = torch.einsum('nckuv,ntcuv->ntck', o['W'], S['mu0'])
        ls_, info = solve_loss(sv, o, S, Ystar, grad_model=True)
        lp = torch.zeros_like(ls_)      # prior frozen
        bt = S['bg'][:, 0].mean((-1, -2))
        lb = F.huber_loss((o['bgc'] - bt) / 0.1, torch.zeros_like(bt), delta=3.0, reduction='none').mean(1)
        lr_ = 0.5 * info['r2'].clamp(max=50.0) + torch.log(o['cs'])
        L = (ls_ / 3 + lp + a.w_bg * lb + a.w_ro * lr_).mean()
        if not torch.isfinite(L):
            print(f'it {it + 1}: non-finite loss, step skipped', flush=True); opt.zero_grad(); sch.step(); continue
        opt.zero_grad(); L.backward()
        torch.nn.utils.clip_grad_norm_([q for q in net.parameters() if q.requires_grad], 5.0)
        opt.step(); sch.step()
        with torch.no_grad():
            iso = (S['nnb'] == 0); u = info['u']
            acc.append([L.item(), ls_.mean().item() / 3, lp.mean().item(), lb.mean().item(),
                        u[iso].abs().median().item() * 1.4826, u[~iso].abs().median().item() * 1.4826,
                        (u[iso] ** 2).clamp(max=25).mean().sqrt().item(), (u[~iso] ** 2).clamp(max=25).mean().sqrt().item(),
                        o['cs'][iso].mean().item(), o['cs'][~iso].mean().item()])
        if (it + 1) % 250 == 0:
            m = torch.tensor(acc).mean(0).tolist(); acc = []
            s = (f'it {it + 1} loss {m[0]:.4f} solve {m[1]:.4f} nllp {m[2]:.4f} lbg {m[3]:.3f} | err/sigma robust iso {m[4]:.3f} nb {m[5]:.3f}'
                 f' rms5 iso {m[6]:.3f} nb {m[7]:.3f} | cs iso {m[8]:.3f} nb {m[9]:.3f} lr {sch.get_last_lr()[0]:.2e} t {time.time() - t0:.0f}s')
            print(s, flush=True); log.write(s + '\n'); log.flush()
        if (it + 1) % 2500 == 0 or it + 1 == a.steps:
            safe_save(dict(state=net.state_dict(), args=vars(a), psf=st, it=it + 1), f'{OUT}/head4pi_{a.tag}.pt')


if __name__ == '__main__':
    main()
