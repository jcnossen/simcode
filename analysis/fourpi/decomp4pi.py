"""Isolated-emitter CRLB decomposition of the 4Pi network pipeline (user request 23:40).

Emitters: the bench set (Sim4Pi seed 11, 6000 isolated on-through windows, ARBITRARY pattern phase (eps_range pi),
unknown-but-given 4Pi phase), identical to bench_classical.py --eps_range pi --suffix _epspi (pixel MLE + pixel CRLB),
so all rows are compared on the same emitters. Pattern and 4Pi phase field are given (true) in every row.

Rows (all solved with the production Solve4Pi, 'on' model unless stated):
  pixel MLE        : 24-image pixel MLE with the true model (classical optimum), from the bench
  oracle read-out  : the two read-outs computed linearly from the simulated pixels at the TRUE position with the true
                     background: T = per-frame matched filter (unbiased for the patch photons), C = Gaussian-window sum
                     (the training definition); exact 2 x 2 covariance. Weak prior (1 px, 0.2 um) -> what the
                     representation + solve can reach
  net, weak prior  : network read-outs and predicted covariance, weak prior
  net, recal cov   : network read-outs, covariance rescaled per read-out type to the observed whitened spread
  net + net prior  : production 'on' solve (network prior)
  production       : kinetics / free selection as in the pipeline (on-through emitters: should pick kinetics)
Plus: network read-out errors vs the oracle read-out noise (ratio), bias (shrinkage), and covariance calibration
(std of the whitened residuals: 1 = calibrated).
Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi/decomp4pi.py --head v3
"""
import argparse, json, math, os, sys
import numpy as np, torch
REPO = '/home/jc/simcode2'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode2.fourpi.lut4pi import LUT4Pi
from simcode2.fourpi.sim4pi import Sim4Pi
from simcode2.fourpi.head4pi import Head4Pi
from simcode2.fourpi.solve4pi import Solve4Pi, psi_from_eps, kin_starts, best_of
from eval_head4pi import run_net

EDGES = (1000, 2000, 4000, 8000, 16000)
V = 1.14 ** 2 + 0.2 ** 2


def solve_chunks(sv, o, S, mode, **kw):
    n = o['T'].shape[0]; res = []
    for i in range(0, n, 2000):
        j = slice(i, i + 2000)
        oo = {k: v[j] for k, v in o.items()}
        extra = {k: (v[j] if torch.is_tensor(v) else v) for k, v in kw.items()}
        res.append(sv.solve(oo, S['Phi0'][j], S['doff'][j], S['m'][j], psi_from_eps(S['eps'][j]), mode=mode, **extra))
    return {k: torch.cat([r[k] for r in res]) for k in res[0]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--head', default='v3'); ap.add_argument('--suffix', default='_epspi')
    a = ap.parse_args()
    dev = 'cuda'
    st = json.load(open(f'{REPO}/out/4pi/psf4pi/NPC_f_dxy15.json'))['psf']
    lut = LUT4Pi(st, P=15, dev=dev)
    S = Sim4Pi(lut, seed=11).sample(6000, p_nb=0.0, p_on=1.0, eps_range=math.pi)
    bc = torch.load(f'{REPO}/out/4pi/sim/bench_classical{a.suffix}.pt', weights_only=False)
    assert torch.allclose(bc['S']['x'], S['x']), 'not the bench emitters'
    n = len(S['x']); tr = torch.stack([S['x'], S['y'], S['z']], 1); sc = torch.tensor([128.0, 128.0, 1000.0], device=dev)
    cr = (bc['cr_p'][:, :3] * sc).cpu().numpy()
    Nw = S['Nt'].sum(1).cpu().numpy()
    sv = Solve4Pi(lut); sv.build_table()
    rows = {}
    # ---- pixel MLE (bench)
    rows['pixel MLE (24 images)'] = ((bc['th_p'][:, :3] - tr) * sc).cpu().numpy()
    # ---- oracle linear read-outs at the truth
    img, mu, bg = S['img'], S['mu'], S['bg']                                  # [n,6,4,P,P]
    P = img.shape[-1]; o_ = torch.arange(P, device=dev, dtype=torch.float32) - (P - 1) / 2
    oy, ox = torch.meshgrid(o_, o_, indexing='ij')
    g = mu / mu.sum((-1, -2), keepdim=True).clamp(min=1e-6)                   # shape per (t, c)
    var = mu + bg + V
    w1 = g / var; w1 = w1 / (w1 * g).sum((-1, -2), keepdim=True)              # unbiased for the patch photons
    pc = torch.stack([S['x'][:, None] + S['doff'][:, :, 0], S['y'][:, None] + S['doff'][:, :, 1]], -1)   # [n,4,2]
    w2 = torch.exp(-((ox[None, None] - pc[:, :, 0, None, None]) ** 2 + (oy[None, None] - pc[:, :, 1, None, None]) ** 2) / 2.0)
    w2 = w2[:, None].expand(-1, 6, -1, -1, -1)
    y = img - bg
    To = (w1 * y).sum((-1, -2)); Co = (w2 * y).sum((-1, -2))
    v11 = (w1 * w1 * var).sum((-1, -2)); v22 = (w2 * w2 * var).sum((-1, -2)); v12 = (w1 * w2 * var).sum((-1, -2))
    L = torch.zeros(n, 6, 4, 2, 2, device=dev)
    L[..., 0, 0] = v11.sqrt(); L[..., 1, 0] = v12 / v11.sqrt(); L[..., 1, 1] = (v22 - v12 ** 2 / v11).clamp(min=1e-6).sqrt()
    weak = dict(dxy=torch.stack([S['x'], S['y']], 1), z=S['z'], ls=torch.log(torch.tensor([1.0, 1.0, 0.2], device=dev)).expand(n, 3))
    oO = dict(T=To, C=Co, L=L, **weak)
    rO = solve_chunks(sv, oO, S, 'on')
    rows['oracle read-out, weak prior'] = ((rO['th'][:, :3] - tr) * sc).cpu().numpy()
    # ---- network (loaded early: its prior position / z define the knowledge-limited oracles below)
    from simcode2.fourpi.loadhead import load_head     # all head architectures (v2..v8)
    net, ck = load_head(a.head)
    o = run_net(net, S)
    # ---- knowledge-limited oracles: matched filters from the MODEL shape at the network prior (x, y, z), with the true 4Pi
    # phase (Phi known) or with the phase-averaged (incoherent) shape (Phi unknown, what a pass-1 head can know); the
    # C read-out is measured directly (Gaussian window at the prior position); exact covariance; weak prior
    for tag_, use_phi in (('true Phi', True), ('Phi unknown', False)):
        Tk = torch.zeros(n, 6, 4, device=dev); Ck = torch.zeros(n, 6, 4, device=dev); Lk = torch.zeros(n, 6, 4, 2, 2, device=dev)
        for c in range(4):
            pos = torch.stack([o['dxy'][:, 0] + S['doff'][:, c, 0], o['dxy'][:, 1] + S['doff'][:, c, 1], o['z']], 1)
            ga = [lut.evaluate_ch(pos, S['Phi0'] + (math.radians(-12.0) if aa else 0.0), c) if use_phi else lut.luts[0][c].evaluate(pos)
                  for aa in range(2)]
            wg = torch.exp(-((ox[None] - pos[:, 0, None, None]) ** 2 + (oy[None] - pos[:, 1, None, None]) ** 2) / 2.0)
            for t in range(6):
                gk = ga[t // 3]; gk = gk / gk.sum((-1, -2), keepdim=True)
                vv = var[:, t, c]
                ww = gk / vv; ww = ww / (ww * gk).sum((-1, -2), keepdim=True)
                Tk[:, t, c] = (ww * y[:, t, c]).sum((-1, -2)); Ck[:, t, c] = (wg * y[:, t, c]).sum((-1, -2))
                a11 = (ww * ww * vv).sum((-1, -2)); a22 = (wg * wg * vv).sum((-1, -2)); a12 = (ww * wg * vv).sum((-1, -2))
                Lk[:, t, c, 0, 0] = a11.sqrt(); Lk[:, t, c, 1, 0] = a12 / a11.sqrt(); Lk[:, t, c, 1, 1] = (a22 - a12 ** 2 / a11).clamp(min=1e-6).sqrt()
        # targets are defined at the truth: the knowledge-limited read-outs also carry the shape-mismatch bias
        rK = solve_chunks(sv, dict(T=Tk, C=Ck, L=Lk, **weak), S, 'on')
        rows[f'oracle matched filter at the prior position, {tag_}'] = ((rK['th'][:, :3] - tr) * sc).cpu().numpy()
    ow = dict(T=o['T'], C=o['C'], L=o['L'], dxy=o['dxy'], z=o['z'], ls=torch.log(torch.tensor([1.0, 1.0, 0.2], device=dev)).expand(n, 3))
    if o.get('U') is not None: ow['U'] = o['U']
    rW = solve_chunks(sv, ow, S, 'on')
    rows['network read-outs, weak prior'] = ((rW['th'][:, :3] - tr) * sc).cpu().numpy()
    # covariance calibration (whitened residuals vs the GT read-out targets)
    eT = o['T'] - S['T']; eC = o['C'] - S['C']
    a_ = o['L'][..., 0, 0]; b_ = o['L'][..., 1, 0]; d_ = o['L'][..., 1, 1]
    u1 = eT / a_; u2 = (eC - b_ * u1) / d_
    kT = float(1.4826 * u1.abs().median()); kC = float(1.4826 * u2.abs().median())
    oc = dict(o, L=o['L'] * torch.tensor([[kT, 0.0], [kC, kC]], device=dev))
    if o.get('U') is not None: oc['U'] = o['U'] * math.sqrt(kT * kC)
    rC = solve_chunks(sv, oc, S, 'on')
    rows['network, recalibrated covariance'] = ((rC['th'][:, :3] - tr) * sc).cpu().numpy()
    rP = solve_chunks(sv, o, S, 'on')
    rows['network + network prior (on)'] = ((rP['th'][:, :3] - tr) * sc).cpu().numpy()
    rf = solve_chunks(sv, o, S, 'free')
    m_ = S['m']; psi = psi_from_eps(S['eps'])
    k1, k2 = kin_starts(rf, m_, psi, sv)
    r1 = solve_chunks(sv, o, S, 'kin', kin0=k1); r2 = solve_chunks(sv, o, S, 'kin', kin0=k2)
    rk = best_of(r1, r2); use = (rk['chi2'] - rf['chi2'] < 7.81)
    thp = torch.where(use[:, None], rk['th'][:, :3], rf['th'][:, :3])
    rows['production v1 (kinetics / free fallback)'] = ((thp - tr) * sc).cpu().numpy()
    # main-pipeline selection: on-through vs kinetics, kinetics kept only if it lowers the NLL by > 9.21; no free fallback
    use2 = (rP['chi2'] - rk['chi2']) > 9.21
    th2 = torch.where(use2[:, None], rk['th'][:, :3], rP['th'][:, :3])
    rows['production v2 (on-through vs kinetics, dNLL 9.21)'] = ((th2 - tr) * sc).cpu().numpy()
    usek = use.cpu().numpy()
    # ---- read-out diagnostics vs the oracle read-out
    ratT = float(torch.sqrt((eT ** 2).mean() / ((To - S['T']) ** 2).mean())); ratC = float(torch.sqrt((eC ** 2).mean() / ((Co - S['C']) ** 2).mean()))
    shrinkT = float(((o['T'] * S['T']).sum() / (S['T'] ** 2).sum())); shrinkC = float(((o['C'] * S['C']).sum() / (S['C'] ** 2).sum()))
    prof = (o['T'] - o['T'].mean(1, keepdim=True)); proft = (S['T'] - S['T'].mean(1, keepdim=True))
    mod_slope = float((prof * proft).sum() / (proft * proft).sum())
    # per-bin table
    out = dict(head=a.head, rows={}, readout=dict(err_ratio_T_vs_oracle=ratT, err_ratio_C_vs_oracle=ratC, gain_T=shrinkT, gain_C=shrinkC,
                                                  frame_modulation_slope_T=mod_slope, whitened_sd_T=kT, whitened_sd_C=kC,
                                                  production_kin_frac=float(usek.mean())))
    L_ = ['| row | ' + ' | '.join(f'{lo}-{hi} e: x / y / z RMS/CRLB (fringe)' for lo, hi in zip(EDGES[:-1], EDGES[1:])) + ' |',
          '|---|' + '---|' * (len(EDGES) - 1)]
    for nm, e in rows.items():
        cells = []; out['rows'][nm] = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            s = (Nw >= lo) & (Nw < hi)
            ok = s & (np.abs(e[:, 2]) <= 100) & (np.abs(e[:, :2]) <= 60).all(1)
            rr = [float(np.sqrt(np.mean(e[ok, k] ** 2)) / np.sqrt(np.mean(cr[ok, k] ** 2))) for k in range(3)]
            fr = float(np.mean(np.abs(e[s, 2]) > 100)); hop = float(np.mean((np.abs(e[s, :2]) > 60).any(1)))
            out['rows'][nm].append(dict(lo=lo, hi=hi, n=int(s.sum()), rms_over_crlb=rr, fringe=fr, lateral_hop=hop))
            cells.append(f'{rr[0]:.2f} / {rr[1]:.2f} / {rr[2]:.2f} ({fr:.3f}{"" if hop < 0.0005 else f", hop {hop:.3f}"})')
        L_.append(f'| {nm} | ' + ' | '.join(cells) + ' |')
    out['table'] = L_
    print('\n'.join(L_))
    print('\nread-outs:', json.dumps(out['readout']))
    json.dump(out, open(f'{REPO}/out/4pi/sim/decomp_{a.head}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
