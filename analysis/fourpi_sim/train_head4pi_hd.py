"""COPY of analysis/fourpi/train_head4pi_w.py for the high-density head experiment (hd, 2026-11-15); differences:
  --sim hd: the dense-scene neighbour generator of analysis/fourpi_sim/sim4pi_hd.py (density 0.08-2 ON /um^2 per sample,
  microtubule-like lines, crossings, close pairs with z offsets); --sim v10: the original FourPiPatchSim rule.
  Outputs in $P4/models/head4pi_<tag>/ (checkpoint head4pi_<tag>.pt, log head4pi_<tag>.log); --log_every.
  Everything else (head, loss, optimiser, schedule) is the v10 trainer's.
Original docstring:
Train the learned-window 4Pi head (simcode/fourpi/head4pi_w.py, v10: frame-symmetric, emitter-orthogonal window changes; frozen v3L prior) with the solve-aware loss
(simcode/fourpi/solveloss4pi.py) on the patch simulator (neighbours p 0.5, kinetics, arbitrary pattern phase, as v3).

Loss = NLL of the linearised solve's (x, y, z) error under its reported covariance  + prior NLL  + w_bg x background
Huber  + w_ro x read-out NLL with the shared covariance scale (keeps the read-outs themselves and chi2 meaningful; small,
because the per-read-out NLL is what makes a free read-out mix frames).
Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi/train_head4pi_w.py --tag v10 --steps 20000 --init_v3 out/4pi/models/head4pi_v3L.pt
Resume: --init out/4pi/models/head4pi_<tag>.pt --steps <remaining>. Checkpoint every 2500 steps (atomic).
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.config import root; P4 = root('fourpi')
from simcode.psf import fourpi_state
from simcode.util.checkpoint import save as safe_save
from simcode.fourpi.spline_table import LUT4Pi
from simcode.simulate.fourpi_patches import FourPiPatchSim
from simcode.emitter.fourpi import nll_prior
from simcode.emitter.fourpi import WINDOWS_K
from simcode.emitter.fourpi import FourPiEmitterNet
from simcode.solve.fourpi import FourPiSolve
from simcode.emitter.fourpi import solve_loss

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from sim4pi_hd import Sim4PiHD


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='v10'); p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--batch', type=int, default=512); p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--ch', type=int, default=64); p.add_argument('--nblk', type=int, default=4)
    p.add_argument('--psf', default='NPC_f_dxy15'); p.add_argument('--seed', type=int, default=1)
    p.add_argument('--w_bg', type=float, default=0.1); p.add_argument('--w_ro', type=float, default=0.02)
    p.add_argument('--bound', type=float, default=1.0)
    p.add_argument('--init', default=''); p.add_argument('--init_v3', default='')
    p.add_argument('--p_nb', type=float, default=0.5); p.add_argument('--nb_max', type=int, default=3)
    p.add_argument('--ksum', default='1', help='comma list of K (summed windows): per step one K: background 0.2 K - 2 K e/px, read sd 1.14 sqrt(K), offset error 0.2 K')
    p.add_argument('--nbr_in', type=int, default=0, help='1: neighbour-candidate map input (v11)'); p.add_argument('--train_prior', type=int, default=0)
    p.add_argument('--sim', default='hd'); p.add_argument('--p_iso', type=float, default=0.2); p.add_argument('--p_close', type=float, default=0.3)
    p.add_argument('--rho_hi', type=float, default=2.0); p.add_argument('--log_every', type=int, default=250)
    p.add_argument('--init_v10', default='', help='v10-type checkpoint to start a neighbour-input head from')
    a = p.parse_args(); a.arch = 'w'; a.wins = [list(w) for w in WINDOWS_K]
    from simcode.util.proctitle import set_title; set_title('4pi-train-hd')
    OUT = f'{P4}/models/head4pi_{a.tag}'; os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(a.seed)
    st = fourpi_state(a.psf, P4)
    lut = LUT4Pi(st, P=15)
    sim = Sim4PiHD(lut, seed=a.seed, p_iso=a.p_iso, p_close=a.p_close, rho=(0.08, a.rho_hi)) if a.sim == 'hd' else FourPiPatchSim(lut, seed=a.seed)
    net = FourPiEmitterNet(ch=a.ch, nblk=a.nblk, bound=a.bound, nbr_in=bool(a.nbr_in), train_prior=bool(a.train_prior)).cuda(); net.lut = lut
    from simcode.emitter.fourpi import neighbour_map, init_from_v10
    ksums = [int(v) for v in a.ksum.split(',')]
    ld = lambda f: torch.load(f if f.startswith('/') else f'{REPO}/{f}', weights_only=False)['state']
    if a.init: net.load_state_dict(ld(a.init))
    elif a.init_v10: init_from_v10(net, ld(a.init_v10))
    elif a.init_v3: net.prior.load_state_dict(ld(a.init_v3))
    net.train()
    sv = FourPiSolve(lut, WINDOWS_K); sv.tab_lim = None
    opt = torch.optim.AdamW([q for q in net.parameters() if q.requires_grad], lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    log = open(f'{OUT}/head4pi_{a.tag}.log', 'a')
    t0 = time.time(); acc = []
    for it in range(a.steps):
        K = ksums[it % len(ksums)]
        with torch.no_grad():
            S = sim.sample(a.batch, eps_range=math.pi, p_nb=a.p_nb, nb_max=a.nb_max, bg=(0.2 * K, 2.0 * K), rn=1.14 * math.sqrt(K), offerr=0.2 * K)
            nbr = None
            if a.nbr_in:   # other candidates as the detector would give them: position jitter 0.3 px, 20 % missed
                dxn = torch.zeros(a.batch, a.nb_max, device='cuda'); dyn = torch.zeros_like(dxn); val = torch.zeros_like(dxn, dtype=torch.bool)
                for j, nb in enumerate(S['nb']):
                    kk = int(nb['sel'].sum())
                    dxn[nb['sel'], j] = nb['x'] + 0.3 * torch.randn(kk, device='cuda'); dyn[nb['sel'], j] = nb['y'] + 0.3 * torch.randn(kk, device='cuda')
                    val[nb['sel'], j] = (torch.rand(kk, device='cuda') > 0.2) & (nb['N'].sum(1) > 150)
                nbr = neighbour_map(dxn, dyn, val, S['doff'])
        net.vread = K * 1.14 ** 2 + (0.2 * K) ** 2
        o = net(S['img'], S['doff'], nbr=nbr)
        Ystar = torch.einsum('nckuv,ntcuv->ntck', o['W'], S['mu0'])
        ls_, info = solve_loss(sv, o, S, Ystar, grad_model=True, sb=0.2 * math.sqrt(K))
        lp = nll_prior(S['x'], S['y'], S['z'], o) / 3 if a.train_prior else torch.zeros_like(ls_)
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
                        o['cs'][iso].mean().item(), o['cs'][~iso].mean().item(), S['nnb'].float().mean().item()])
        if (it + 1) % a.log_every == 0:
            m = torch.tensor(acc).mean(0).tolist(); acc = []
            s = (f'it {it + 1} loss {m[0]:.4f} solve {m[1]:.4f} nllp {m[2]:.4f} lbg {m[3]:.3f} | err/sigma robust iso {m[4]:.3f} nb {m[5]:.3f}'
                 f' rms5 iso {m[6]:.3f} nb {m[7]:.3f} | cs iso {m[8]:.3f} nb {m[9]:.3f} nnb {m[10]:.2f} lr {sch.get_last_lr()[0]:.2e} t {time.time() - t0:.0f}s')
            print(s, flush=True); log.write(s + '\n'); log.flush()
        if (it + 1) % 2500 == 0 or it + 1 == a.steps:
            safe_save(dict(state=net.state_dict(), args=vars(a), psf=st, it=it + 1), f'{OUT}/head4pi_{a.tag}.pt')


if __name__ == '__main__':
    main()
