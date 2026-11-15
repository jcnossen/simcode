"""Outputs of the 4Pi code path with 6-frame windows (4Pi-SIMFLUX) on fixed inputs, on the CPU: the multi-channel scene
simulator (patterns on / off), the candidate-patch simulator, the 4Pi priors / learned-window networks (v3L, v10, v11)
and the 4Pi detector, the explicit solve (modes on / kin / free, fit_phi, window model, table model) and the solve-aware
training loss. Used to check that the generalisation to T frames per window (2026-11-15, plain 4Pi) leaves the T = 6
outputs bit-identical: run once with the code before the change (--save), once after (--ref). Only APIs present in both
versions are used.

  PYTHONPATH=<code> python tests/check_fourpi_T.py --save ref.pt
  PYTHONPATH=<code> python tests/check_fourpi_T.py --ref ref.pt        (exit code 1 on any difference)
"""
import argparse, json, math, os, sys
import torch

dev = 'cpu'


def run():
    from simcode.config import root
    from simcode.fourpi.lut4pi import LUT4Pi
    from simcode.simulate.multichannel import MultiChannelAcquisitionSim
    from simcode.simulate.fourpi_patches import FourPiPatchSim
    from simcode.emitter.fourpi import FourPiPrior, FourPiEmitterNet, WINDOWS_K, solve_loss, linear_readout, border_bg
    from simcode.solve.fourpi import FourPiSolve, kin_starts, best_of
    from simcode.detector.model import build
    P4 = root('fourpi')
    torch.set_num_threads(16)
    R = {}
    st = json.load(open(f'{P4}/psf4pi/NPC_f_dxy15.json'))['psf']
    # ---- scene simulator
    for pat in (True, False):
        sim = MultiChannelAcquisitionSim(st, H=32, W=32, n_psf=2, dev=dev, seed=3, patterns=pat, sub=1)
        S = sim.sample(B=3, dens=(0.5, 1.5))
        for k in ('img', 'mu', 'origin', 'det_in', 'bg'):
            R[f'mcs{int(pat)}|{k}'] = S[k]
        for k, v in S['gt'].items(): R[f'mcs{int(pat)}|gt|{k}'] = v
    sim = MultiChannelAcquisitionSim(st, H=32, W=32, n_psf=2, dev=dev, seed=4, reg_shift_sd=0.1, reg_lin_sd=0.05, sub=1)
    S = sim.sample(B=2, ksum=4)
    for k in ('img', 'det_in'): R[f'mcs_reg|{k}'] = S[k]
    del sim
    # ---- patch simulator + networks
    lut = LUT4Pi(st, P=15, dev=dev)
    ps = FourPiPatchSim(lut, dev=dev, seed=5)
    S = ps.sample(96, eps_range=math.pi)
    for k in ('img', 'mu', 'mu0', 'bg', 'x', 'y', 'z', 'Phi0', 'Nt', 'q', 'F', 'm', 'eps', 'doff', 'T', 'C', 'nnb'):
        R[f'ps|{k}'] = S[k]
    nets = {}
    for tag in ('v3L', 'v10', 'v11'):
        ck = torch.load(f'{P4}/models/head4pi_{tag}.pt', weights_only=False, map_location=dev); a = ck['args']
        if a.get('arch') == 'w':
            net = FourPiEmitterNet(ch=a['ch'], nblk=a['nblk'], wins=tuple(tuple(w) for w in a['wins']), bound=a.get('bound', 1.0), nbr_in=a.get('nbr_in', False))
            net.load_state_dict(ck['state']); net.lut = lut
        else:
            net = FourPiPrior(ch=a['ch'], nblk=a['nblk']); net.load_state_dict(ck['state'])
        net.eval(); nets[tag] = net
        with torch.no_grad():
            o = net(S['img'], S['doff'])
        for k, v in o.items(): R[f'net|{tag}|{k}'] = v
    ck = torch.load(f'{P4}/models/det4pi/final.pt', weights_only=False, map_location=dev)
    det = build(ck['args']); det.load_state_dict(ck['model']); det.eval()
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        do = det(torch.poisson(torch.full((2, 6, 40, 40), 30.0) + 400 * torch.rand(2, 6, 40, 40, generator=g), generator=g))
    for k, v in do.items(): R[f'det|{k}'] = v
    # ---- fixed-window read-out, background
    Y, Lb, Ws = linear_readout(S['img'], S['doff'], lut, WINDOWS_K, torch.stack([S['x'], S['y']], 1), S['z'], border_bg(S['img']))
    R['lin|Y'], R['lin|Lb'], R['lin|Ws'] = Y, Lb, Ws
    # ---- solve (window model) on v10
    net = nets['v10']
    with torch.no_grad():
        o = net(S['img'], S['doff'])
    sv = FourPiSolve(lut, net.wins, dev=dev); sv.tab_lim = None
    n = len(S['x'])
    from simcode.solve.fourpi import psi_from_eps
    m = S['m'][:, None].expand(-1, 6).contiguous(); psi = psi_from_eps(S['eps'])
    oo = dict(Y=o['Y'], Lb=o['Lb'], dxy=o['dxy'], z=o['z'], ls=o['ls'], W=o['W'], Wsum=o['Wsum'], sb=torch.full((n, 4), 0.2))
    ron = sv.solve(oo, S['Phi0'], S['doff'], m, psi, mode='on', iters=8)
    for k, v in ron.items(): R[f'solve|on|{k}'] = v
    rp = sv.solve(oo, torch.zeros(n), S['doff'], m, psi, mode='free', fit_phi=True, iters=8)
    for k, v in rp.items(): R[f'solve|freephi|{k}'] = v
    o2 = dict(oo, z0=ron['th'][:, 2], xy0=ron['th'][:, :2])
    k1, k2 = kin_starts({'th': torch.cat([ron['th'][:, :3], oo['Y'][..., 0].sum(2).clamp(min=0) / 0.85], 1)}, m, psi, sv)
    rk = best_of(sv.solve(o2, S['Phi0'], S['doff'], m, psi, mode='kin', kin0=k1, iters=8, zstarts=(0.0,)),
                 sv.solve(o2, S['Phi0'], S['doff'], m, psi, mode='kin', kin0=k2, iters=8, zstarts=(0.0,)))
    for k, v in rk.items(): R[f'solve|kin|{k}'] = v
    sv.k2 = ((3.83, 0.01), (-0.02, 3.85))
    rk2 = sv.solve(oo, S['Phi0'], S['doff'], m, psi, mode='on', iters=6)
    for k, v in rk2.items(): R[f'solve|k2|{k}'] = v
    # ---- solve (table model, T / C windows) on v3L
    svt = FourPiSolve(lut, dev=dev); svt.build_table(xr=2.0, hx=0.25, zr=(-0.6, 0.6), hz=0.05)
    R['tab'] = svt.tab
    op = nets['v3L']
    with torch.no_grad():
        q = op(S['img'], S['doff'])
    Yt = torch.stack([q['T'], q['C']], -1)
    ot = dict(Y=Yt, Lb=q['L'], dxy=q['dxy'], z=q['z'], ls=q['ls'], p0=torch.cat([q['dxy'], q['z'][:, None]], 1))
    rt = svt.solve(ot, S['Phi0'], S['doff'], m, psi, mode='on', iters=6)
    for k, v in rt.items(): R[f'solvetab|{k}'] = v
    # ---- solve-aware loss
    sv2 = FourPiSolve(lut, WINDOWS_K, dev=dev); sv2.tab_lim = None
    Ystar = torch.einsum('nckuv,ntcuv->ntck', o['W'], S['mu0'])
    nll, info = solve_loss(sv2, o, S, Ystar)
    R['loss|nll'] = nll
    for k, v in info.items(): R[f'loss|{k}'] = v
    return {k: v.detach().float().cpu() if v.dtype.is_floating_point else v.detach().cpu() for k, v in R.items() if torch.is_tensor(v)}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--save'); p.add_argument('--ref')
    a = p.parse_args()
    R = run()
    if a.save:
        torch.save(R, a.save); print(f'{len(R)} outputs -> {a.save}')
    if a.ref:
        ref = torch.load(a.ref)
        bad = [k for k in ref if k not in R or R[k].shape != ref[k].shape or not torch.equal(R[k], ref[k])]
        miss = [k for k in R if k not in ref]
        for k in bad:
            d = (R[k].float() - ref[k].float()).abs().max().item() if k in R and R[k].shape == ref[k].shape else 'shape/missing'
            print(f'DIFF {k}: {d}')
        print(f'{len(ref)} reference outputs: {len(ref) - len(bad)} bit-identical, {len(bad)} different; {len(miss)} new keys')
        sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
