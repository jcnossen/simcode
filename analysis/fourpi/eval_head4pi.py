"""Stage-2 sim benchmark, network side: head4pi + explicit solve vs the classical estimators and the CRLB.

Set A (identical to bench_classical.py: Sim4Pi seed 11, 6000 isolated on-through windows): solve 'on'.
Set B (seed 12, realistic: neighbours p 0.5, up to 3 within 7 px; kinetics p_on 0.6): solve 'free' (z for every
window) and 'on' (x, y, z for GT on-through windows), and the classical pixel MLE (single emitter, 'on' model) on the
same windows for comparison.
Output: out/4pi/sim/eval_head4pi_<tag>.json + tables.
Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi/eval_head4pi.py --tag v1
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
REPO = '/home/jc/simcode2'
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode2.fourpi.lut4pi import LUT4Pi
from simcode2.fourpi.sim4pi import Sim4Pi
from simcode2.fourpi.head4pi import Head4Pi
from simcode2.fourpi.solve4pi import Solve4Pi, psi_from_eps, kin_starts, best_of

OUT = f'{REPO}/out/4pi/sim'
CREL = 0.0


def run_net(net, S, chunk=1024):
    outs = []
    with torch.no_grad():
        for i in range(0, S['img'].shape[0], chunk):
            outs.append(net(S['img'][i:i + chunk], S['doff'][i:i + chunk]))
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def solve_chunks(sv, o, S, mode, chunk=2000, sel=None, **kw):
    idx = torch.arange(S['img'].shape[0], device=o['T'].device) if sel is None else torch.nonzero(sel)[:, 0]
    res = []
    Phi_all = kw.pop('Phi0') if 'Phi0' in kw else S['Phi0']
    K0 = kw.pop('_k0', None)
    for i in range(0, len(idx), chunk):
        j = idx[i:i + chunk]
        oo = {k: v[j] for k, v in o.items()}
        Phi0 = Phi_all[j]
        if K0 is not None: kw['kin0'] = K0[j]
        res.append(sv.solve(oo, Phi0, S['doff'][j], S['m'][j], psi_from_eps(S['eps'][j]), mode=mode, crel=CREL, **kw))
    return idx, {k: torch.cat([r[k] for r in res]) for k in res[0]}


def stats(e, s):
    e = e[s]
    return dict(rms=float(torch.sqrt((e ** 2).mean())), robust=float(1.4826 * (e - e.median()).abs().median()))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', default='v1'); p.add_argument('--n', type=int, default=6000); p.add_argument('--crel', type=float, default=0.0)
    a = p.parse_args()
    global CREL; CREL = a.crel
    ck = torch.load(f'{REPO}/out/4pi/models/head4pi_{a.tag}.pt', weights_only=False)
    net = Head4Pi(ch=ck['args']['ch'], nblk=ck['args']['nblk']).cuda().eval(); net.load_state_dict(ck['state'])
    lut = LUT4Pi(ck['psf'], P=15)
    sv = Solve4Pi(lut); sv.build_table()
    edges = [300, 1000, 2000, 4000, 8000, 16000]
    out = dict(tag=a.tag, it=ck['it'])
    # ---- set A
    S = Sim4Pi(lut, seed=11).sample(6000, p_nb=0.0, p_on=1.0)
    bc = torch.load(f'{OUT}/bench_classical.pt', weights_only=False)
    assert torch.allclose(bc['S']['x'], S['x']), 'set A differs from bench_classical'
    t0 = time.time()
    o = run_net(net, S)
    _, r = solve_chunks(sv, o, S, 'on')
    print(f'set A solve {time.time() - t0:.0f}s', flush=True)
    Nw = S['Nt'].sum(1); tr = torch.stack([S['x'], S['y'], S['z']], 1)
    e_net = (r['th'][:, :3] - tr) * torch.tensor([128.0, 128.0, 1000.0], device='cuda')
    e_pix = (bc['th_p'][:, :3] - tr) * torch.tensor([128.0, 128.0, 1000.0], device='cuda')
    e_pri = (torch.stack([o['dxy'][:, 0], o['dxy'][:, 1], o['z']], 1) - tr) * torch.tensor([128.0, 128.0, 1000.0], device='cuda')
    cr = bc['cr_p'][:, :3] * torch.tensor([128.0, 128.0, 1000.0], device='cuda')
    LA = ['| window photons | n | net x / y / z RMS | net fringe err | net z RMS/CRLB (no fringe err) | pix x / y / z RMS | CRLB x / y / z | prior z RMS |', '|---|---|---|---|---|---|---|---|']
    out['A'] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (Nw >= lo) & (Nw < hi)
        if s.sum() < 50: continue
        fr_n = float((e_net[s, 2].abs() > 100).float().mean()); fr_p = float((e_pix[s, 2].abs() > 100).float().mean())
        gn = s & (e_net[:, 2].abs() <= 100); gp = s & (e_pix[:, 2].abs() <= 100)
        rm = lambda e, g, k: float(torch.sqrt((e[g, k] ** 2).mean()))
        crm = lambda g, k: float(torch.sqrt((cr[g, k] ** 2).mean()))
        b = dict(lo=lo, hi=hi, n=int(s.sum()), net=[rm(e_net, gn, k) for k in range(3)], net_fringe=fr_n,
                 pix=[rm(e_pix, gp, k) for k in range(3)], pix_fringe=fr_p, crlb=[crlb_ for crlb_ in (crm(s, 0), crm(s, 1), crm(s, 2))],
                 prior_z=rm(e_pri, s, 2), net_z_over_crlb=rm(e_net, gn, 2) / crm(gn, 2))
        out['A'].append(b)
        LA.append(f"| {lo}-{hi} | {b['n']} | {b['net'][0]:.2f} / {b['net'][1]:.2f} / {b['net'][2]:.2f} | {fr_n:.3f} | {b['net_z_over_crlb']:.2f} | "
                  f"{b['pix'][0]:.2f} / {b['pix'][1]:.2f} / {b['pix'][2]:.2f} (fringe {fr_p:.3f}) | {b['crlb'][0]:.2f} / {b['crlb'][1]:.2f} / {b['crlb'][2]:.2f} | {b['prior_z']:.1f} |")
    print('\nSet A (isolated, on-through; nm)\n' + '\n'.join(LA), flush=True)
    # ---- set B
    S = Sim4Pi(lut, seed=12).sample(a.n, p_nb=0.5, p_on=0.6)
    o = run_net(net, S)
    _, rf = solve_chunks(sv, o, S, 'free')
    # kinetics model for every window, started from the free solve; kept unless chi2 is much worse than free
    ks = [kin_starts({'th': rf['th'][i:i + 2000]}, S['m'][i:i + 2000], psi_from_eps(S['eps'][i:i + 2000]), sv) for i in range(0, a.n, 2000)]
    _, rk1 = solve_chunks(sv, o, S, 'kin', _k0=torch.cat([k[0] for k in ks]))
    _, rk2 = solve_chunks(sv, o, S, 'kin', _k0=torch.cat([k[1] for k in ks]))
    rk = best_of(rk1, rk2)
    use_k = (rk['chi2'] - rf['chi2']) < 7.81
    ek = torch.where(use_k[:, None], rk['th'][:, :3], rf['th'][:, :3])
    ek = (ek - torch.stack([S['x'], S['y'], S['z']], 1)) * torch.tensor([128.0, 128.0, 1000.0], device='cuda')
    onth = S['q'].min(1).values >= 0.999
    idx_on, ro = solve_chunks(sv, o, S, 'on', sel=onth)
    tr = torch.stack([S['x'], S['y'], S['z']], 1); sc = torch.tensor([128.0, 128.0, 1000.0], device='cuda')
    ef = (rf['th'][:, :3] - tr) * sc
    eo = (ro['th'][:, :3] - tr[idx_on]) * sc
    Nw = S['Nt'].sum(1)
    nb_close = torch.zeros_like(Nw, dtype=torch.bool)
    for nb in S['nb']:
        dd = torch.full_like(Nw, 99.0); dd[nb['sel']] = torch.hypot(nb['x'] - S['x'][nb['sel']], nb['y'] - S['y'][nb['sel']])
        nb_close |= dd < 2.0
    LB = ['| window photons | n | free: z RMS / robust / fringe | close nb (<2 px) z robust / fringe | on-through n | on: x / y / z RMS (fringe) | all, kin/free: x / y robust (nm), z robust, kin frac |', '|---|---|---|---|---|---|---|']
    out['B'] = []
    Nwo = Nw[idx_on]
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (Nw >= lo) & (Nw < hi)
        if s.sum() < 50: continue
        g = s & (ef[:, 2].abs() <= 100)
        sc_ = s & nb_close
        so = (Nwo >= lo) & (Nwo < hi); go = so & (eo[:, 2].abs() <= 100)
        b = dict(lo=lo, hi=hi, n=int(s.sum()), free_z_rms=float(torch.sqrt((ef[g, 2] ** 2).mean())), free_z_robust=stats(ef[:, 2], s)['robust'],
                 free_fringe=float((ef[s, 2].abs() > 100).float().mean()),
                 close_n=int(sc_.sum()), close_z_robust=stats(ef[:, 2], sc_)['robust'] if sc_.sum() > 20 else None,
                 close_fringe=float((ef[sc_, 2].abs() > 100).float().mean()) if sc_.sum() > 20 else None,
                 on_n=int(so.sum()), on=[float(torch.sqrt((eo[go, k] ** 2).mean())) for k in range(3)],
                 on_fringe=float((eo[so, 2].abs() > 100).float().mean()) if so.sum() else None)
        out['B'].append(b)
        cz = f"{b['close_z_robust']:.1f} / {b['close_fringe']:.3f}" if b['close_z_robust'] is not None else '-'
        LB.append(f"| {lo}-{hi} | {b['n']} | {b['free_z_rms']:.1f} / {b['free_z_robust']:.1f} / {b['free_fringe']:.3f} | {cz} (n {b['close_n']}) | {b['on_n']} | "
                  f"{b['on'][0]:.2f} / {b['on'][1]:.2f} / {b['on'][2]:.2f} ({b['on_fringe']:.3f}) | "
                  f"{stats(ek[:, 0], s)['robust']:.2f} / {stats(ek[:, 1], s)['robust']:.2f}, {stats(ek[:, 2], s)['robust']:.2f}, {float(use_k[s].float().mean()):.2f} |")
        b.update(kin_x_robust=stats(ek[:, 0], s)['robust'], kin_y_robust=stats(ek[:, 1], s)['robust'], kin_z_robust=stats(ek[:, 2], s)['robust'],
                 kin_frac=float(use_k[s].float().mean()))
    print('\nSet B (neighbours + kinetics; nm)\n' + '\n'.join(LB))
    # ---- phase measurement with z held by the prior (the input of the in-situ phase-field estimate)
    _, rp = solve_chunks(sv, o, S, 'free', fit_phi=True, Phi0=torch.zeros_like(S['Phi0']))
    dphi = torch.remainder(rp['th'][:, -1] - S['Phi0'] + math.pi, 2 * math.pi) - math.pi
    LP = ['| window photons | n | circ. sd of the Phi estimate (deg) | R | expected from prior z sd (deg) |', '|---|---|---|---|---|']
    out['phi'] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (Nw >= lo) & (Nw < hi)
        if s.sum() < 50: continue
        R = float(torch.abs(torch.exp(1j * dphi[s].double()).mean()))
        csd = math.degrees(math.sqrt(-2 * math.log(max(R, 1e-9))))
        zs = float(torch.exp(o['ls'][s, 2]).median()) * 1000
        out['phi'].append(dict(lo=lo, hi=hi, n=int(s.sum()), R=R, csd_deg=csd, prior_z_sd_nm=zs))
        LP.append(f"| {lo}-{hi} | {int(s.sum())} | {csd:.1f} | {R:.3f} | {360 * zs / 246:.1f} |")
    print('\nPhi measurement (z by prior, set B)\n' + '\n'.join(LP))
    out['tables'] = dict(A=LA, B=LB, phi=LP)
    json.dump(out, open(f'{OUT}/eval_head4pi_{a.tag}' + (f'_c{a.crel:g}' if a.crel else '') + '.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
