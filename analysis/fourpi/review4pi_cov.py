"""Review, part 2: is the solve's covariance model (independent 2 x 2 blocks per (frame, channel)) the limit?

Measures the correlation structure of the network read-out errors on a calibration set (Sim4Pi seed 12, isolated,
on-through, arbitrary pattern phase), fits a parametric full 48 x 48 covariance
    Sigma = D Sigma_net D + c_g^2 mu mu^T + c_a^2 sum_a mu_a mu_a^T + c_f^2 sum_t mu_t mu_t^T + c_c^2 sum_c mu_c mu_c^T
(D = diag(s_T, s_C); mu = the network read-outs; gain errors common to all / per angle / per frame / per channel; this is
the main pipeline's autoc form  s Sigma_net + c^2 diag + c_a^2 mu mu^T  extended by the 4Pi groupings) by maximum likelihood
per photon bin, and re-solves the gate emitters (seed 11) with it. Also: frame-sum / channel-ratio swaps.
Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi/review4pi_cov.py --head v3L
"""
import argparse, json, math, os, sys
import numpy as np, torch
REPO = '/home/jc/simcode2'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode2.fourpi.lut4pi import LUT4Pi
from simcode2.fourpi.sim4pi import Sim4Pi
from simcode2.fourpi.solve4pi import Solve4Pi
from eval_head4pi import run_net
from review4pi import solve_chunks, load_head, EDGES, V

dev = 'cuda'


def vec(T, C):            # [n,6,4] x2 -> [n,48] in the solve's order (t, c, ro)
    return torch.stack([T, C], -1).reshape(T.shape[0], 48)


def groups():
    t = torch.arange(6).repeat_interleave(8); c = torch.arange(4).repeat_interleave(2).repeat(6); ro = torch.arange(2).repeat(24)
    G = dict(g=[torch.ones(48)], a=[(t // 3 == a).float() for a in range(2)], f=[(t == i).float() for i in range(6)],
             c=[(c == i).float() for i in range(4)], cT=[((c == i) & (ro == 0)).float() for i in range(4)])
    return {k: torch.stack(v).to(dev) for k, v in G.items()}, ro.to(dev)


def build_cov(o, par, G, ro):
    n = o['T'].shape[0]
    a = o['L'][..., 0, 0]; b = o['L'][..., 1, 0]; d = o['L'][..., 1, 1]
    sT, sC = torch.exp(par['ls'][0]), torch.exp(par['ls'][1])
    S = torch.zeros(n, 24, 2, 2, device=dev)
    S[:, :, 0, 0] = (a ** 2).reshape(n, 24) * sT ** 2; S[:, :, 0, 1] = S[:, :, 1, 0] = (a * b).reshape(n, 24) * sT * sC
    S[:, :, 1, 1] = (b ** 2 + d ** 2).reshape(n, 24) * sC ** 2
    Sig = torch.zeros(n, 48, 48, device=dev)
    idx = torch.arange(24, device=dev)
    for i in range(2):
        for j in range(2):
            Sig[:, 2 * idx + i, 2 * idx + j] = S[:, :, i, j]
    mu = vec(o['T'], o['C']).clamp(min=0).detach()
    for k in G:
        cc = torch.exp(par['lc'][k]) ** 2
        for gm in G[k]:
            v = mu * gm[None]
            Sig = Sig + cc * v[:, :, None] * v[:, None, :]
    return Sig + 1e-3 * torch.eye(48, device=dev)


def fit(o, e, G, ro, steps=400):
    par = dict(ls=torch.zeros(2, device=dev, requires_grad=True), lc={k: torch.tensor(-4.0, device=dev, requires_grad=True) for k in G})
    opt = torch.optim.Adam([par['ls']] + list(par['lc'].values()), lr=0.05)
    for it in range(steps):
        Sig = build_cov(o, par, G, ro)
        L = torch.linalg.cholesky(Sig)
        u = torch.linalg.solve_triangular(L, e[..., None], upper=False)[..., 0]
        nll = (0.5 * (u ** 2).sum(1) + torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)).mean()
        opt.zero_grad(); nll.backward(); opt.step()
    return par, float(nll)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--head', default='v3L'); a = ap.parse_args()
    st = json.load(open(f'{REPO}/out/4pi/psf4pi/NPC_f_dxy15.json'))['psf']
    lut = LUT4Pi(st, P=15, dev=dev)
    net, ck = load_head(a.head)
    sv = Solve4Pi(lut); sv.build_table()
    bc = torch.load(f'{REPO}/out/4pi/sim/bench_classical_epspi.pt', weights_only=False)
    sc = torch.tensor([128.0, 128.0, 1000.0], device=dev)
    cr = (bc['cr_p'][:, :3] * sc).cpu().numpy()
    G, ro = groups()
    # ---- calibration set
    Sc = Sim4Pi(lut, seed=12).sample(12000, p_nb=0.0, p_on=1.0, eps_range=math.pi)
    oc = run_net(net, Sc); ec = vec(oc['T'] - Sc['T'], oc['C'] - Sc['C']); Nc = Sc['Nt'].sum(1)
    sd = torch.sqrt(vec(oc['L'][..., 0, 0] ** 2, oc['L'][..., 1, 0] ** 2 + oc['L'][..., 1, 1] ** 2))
    out = dict(head=a.head, corr=[], fit=[], rows={})
    print('== correlation of the whitened read-out errors (e / reported sd), mean over pairs')
    print('bin | T-T same frame, other channel | T-T same channel, other frame same angle | T-T same channel other angle | T-T other frame other channel | T-C same (t,c) | sd T | sd C')
    tt = torch.arange(6).repeat_interleave(4); cc = torch.arange(4).repeat(6)
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        s = (Nc >= lo) & (Nc < hi); u = (ec / sd)[s]
        Cm = (u.T @ u / u.shape[0]).cpu()
        CT = Cm[0::2, 0::2]; dT = CT.diag().mean().sqrt(); R = CT / CT.diag().mean()
        sf = (tt[:, None] == tt[None]) & (cc[:, None] != cc[None]); scs = (cc[:, None] == cc[None]) & (tt[:, None] != tt[None]) & ((tt[:, None] // 3) == (tt[None] // 3))
        sca = (cc[:, None] == cc[None]) & ((tt[:, None] // 3) != (tt[None] // 3)); oth = (cc[:, None] != cc[None]) & (tt[:, None] != tt[None])
        tc = (Cm[0::2, 1::2].diag() / (Cm[0::2, 0::2].diag() * Cm[1::2, 1::2].diag()).sqrt()).mean()
        d = dict(lo=lo, hi=hi, same_frame=float(R[sf].mean()), same_ch_same_angle=float(R[scs].mean()), same_ch_other_angle=float(R[sca].mean()),
                 other=float(R[oth].mean()), T_C=float(tc), sdT=float(dT), sdC=float(Cm[1::2, 1::2].diag().mean().sqrt()))
        out['corr'].append(d)
        print(f"{lo}-{hi} | {d['same_frame']:+.2f} | {d['same_ch_same_angle']:+.2f} | {d['same_ch_other_angle']:+.2f} | {d['other']:+.2f} | {d['T_C']:+.2f} | {d['sdT']:.2f} | {d['sdC']:.2f}")
    # ---- gate set
    S = Sim4Pi(lut, seed=11).sample(6000, p_nb=0.0, p_on=1.0, eps_range=math.pi)
    assert torch.allclose(bc['S']['x'], S['x'])
    o = run_net(net, S); n = 6000; Nw = S['Nt'].sum(1)
    tr = torch.stack([S['x'], S['y'], S['z']], 1)
    netp = dict(dxy=o['dxy'], z=o['z'], ls=o['ls'])
    rows = {}
    r0 = solve_chunks(sv, dict(T=o['T'], C=o['C'], L=o['L'], **netp), S)
    rows['production (2 x 2 blocks)'] = ((r0['th'][:, :3] - tr) * sc).cpu().numpy()
    # prior sigma recalibrated per bin (whitened sd of the prior on the calibration set)
    epc = torch.stack([oc['dxy'][:, 0] - Sc['x'], oc['dxy'][:, 1] - Sc['y'], oc['z'] - Sc['z']], 1) / torch.exp(oc['ls'])
    ls2 = o['ls'].clone()
    Lf = torch.zeros(n, 48, 48, device=dev); Lf_d = torch.zeros(n, 48, 48, device=dev)
    print('\n== fitted covariance model per bin: s_T, s_C, c_global, c_angle, c_frame, c_channel(T+C), c_channel(T only); NLL/48 before -> after')
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        s = (Nc >= lo) & (Nc < hi); idx = torch.nonzero(s)[:, 0][:3000]
        osub = {k: oc[k][idx] for k in ('T', 'C', 'L')}
        par0 = dict(ls=torch.zeros(2, device=dev), lc={k: torch.tensor(-20.0, device=dev) for k in G})
        with torch.no_grad():
            L0 = torch.linalg.cholesky(build_cov(osub, par0, G, ro)); u0 = torch.linalg.solve_triangular(L0, ec[idx][..., None], upper=False)[..., 0]
            nll0 = float((0.5 * (u0 ** 2).sum(1) + torch.log(torch.diagonal(L0, dim1=1, dim2=2)).sum(1)).mean())
        par, nll = fit(osub, ec[idx], G, ro)
        pv = [float(torch.exp(par['ls'][0])), float(torch.exp(par['ls'][1]))] + [float(torch.exp(par['lc'][k])) for k in G]
        out['fit'].append(dict(lo=lo, hi=hi, par=pv, nll0=nll0 / 48, nll=nll / 48))
        print(f'{lo}-{hi}: ' + ' '.join(f'{v:.4f}' for v in pv) + f'   {nll0 / 48:.4f} -> {nll / 48:.4f}')
        sg = (Nw >= lo) & (Nw < hi)
        with torch.no_grad():
            Lf[sg] = torch.linalg.cholesky(build_cov({k: o[k][sg] for k in ('T', 'C', 'L')}, par, G, ro))
            pd = dict(ls=par['ls'], lc={k: torch.tensor(-20.0, device=dev) for k in G})
            Lf_d[sg] = torch.linalg.cholesky(build_cov({k: o[k][sg] for k in ('T', 'C', 'L')}, pd, G, ro))
            kk = 1.4826 * epc[s].abs().median(0).values
            ls2[sg] = o['ls'][sg] + torch.log(kk)[None]
    sgall = (Nw >= EDGES[0]) & (Nw < EDGES[-1])
    Lf[~sgall] = torch.eye(48, device=dev) * 1e3; Lf_d[~sgall] = torch.eye(48, device=dev) * 1e3
    netp2 = dict(dxy=o['dxy'], z=o['z'], ls=ls2)
    for nm, oo in (('production + prior sigma recalibrated', dict(T=o['T'], C=o['C'], L=o['L'], **netp2)),
                   ('rescaled blocks only (s_T, s_C) + recal prior', dict(T=o['T'], C=o['C'], L=o['L'], Lfull=Lf_d, **netp2)),
                   ('fitted full covariance + network prior', dict(T=o['T'], C=o['C'], L=o['L'], Lfull=Lf, **netp)),
                   ('fitted full covariance + recal prior', dict(T=o['T'], C=o['C'], L=o['L'], Lfull=Lf, **netp2))):
        r = solve_chunks(sv, oo, S); rows[nm] = ((r['th'][:, :3] - tr) * sc).cpu().numpy()
        if nm.startswith('fitted full covariance + recal'):
            sig = (r['sig'][:, :3] * sc).cpu().numpy(); rfin = rows[nm]
    # swaps: frame sums vs channel ratios
    fs = o['T'].sum(2, keepdim=True); fst = S['T'].sum(2, keepdim=True)
    Ta = fs * S['T'] / fst.clamp(min=1e-6); Tb = fst * o['T'] / fs.clamp(min=1e-6)
    Ca = o['C'] * (Ta / o['T'].clamp(min=1e-6)); Cb = o['C'] * (Tb / o['T'].clamp(min=1e-6))
    for nm, T_, C_ in (('net frame sums, true channel ratios (C scaled alike)', Ta, Ca), ('true frame sums, net channel ratios', Tb, Cb)):
        r = solve_chunks(sv, dict(T=T_, C=C_, L=o['L'], **netp), S); rows[nm] = ((r['th'][:, :3] - tr) * sc).cpu().numpy()
    Nn = Nw.cpu().numpy()
    print('\n== RMS (5-sigma trim) / pixel CRLB, x / y / z')
    for nm, e in rows.items():
        cells = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            s = (Nn >= lo) & (Nn < hi); u = e[s] / cr[s]
            rr = [float(np.sqrt(np.mean(u[np.abs(u[:, k]) < 5, k] ** 2))) for k in range(3)]
            out['rows'].setdefault(nm, []).append(rr); cells.append(' / '.join(f'{v:.2f}' for v in rr))
        print(f'| {nm} | ' + ' | '.join(cells) + ' |')
    print('\n== reported sigma calibration of the full-covariance row: robust sd of error / sigma (x / y / z)')
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        s = (Nn >= lo) & (Nn < hi); u = rfin[s] / sig[s]
        print(f'{lo}-{hi}: ' + ' / '.join(f'{1.4826 * np.median(np.abs(u[:, k])):.2f}' for k in range(3)))
    json.dump(out, open(f'{REPO}/out/4pi/sim/review_cov_{a.head}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
