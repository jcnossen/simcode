"""Critical review diagnostics of the 4Pi head on the isolated-emitter gate set (2026-11-12, successor worker).

Same emitters as decomp4pi.py / bench_classical.py (Sim4Pi seed 11, 6000 isolated on-through windows, arbitrary pattern
phase, pattern and Phi0 given). Adds to the decomposition:
  A. statistic check: RMS/CRLB with the decomp trim (60 nm / 100 nm), with a 5-sigma-of-CRLB trim, and robust (MAD)
  B. bounds: oracle read-outs with a weak prior, with an efficient summed-image (SMLM-CRLB) prior, and with the network
     prior; network prior error vs the summed-image CRLB
  C. swaps: which network output costs what (T, C, covariance, prior)
  D. read-out errors per photon bin vs the oracle noise: total, channel-common gain part, frame part, per-frame channel sum
  E. bias of the final position: slope vs the true sub-pixel offset, harmonics vs the pattern phase, vs z
Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi/review4pi.py --head v3L   -> out/4pi/sim/review_<head>.json
"""
import argparse, json, math, os, sys
import numpy as np, torch
REPO = '/home/jc/simcode2'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode2.fourpi.lut4pi import LUT4Pi
from simcode2.fourpi.sim4pi import Sim4Pi, K_PAT
from simcode2.fourpi.head4pi import Head4Pi
from simcode2.fourpi.solve4pi import Solve4Pi, psi_from_eps
from eval_head4pi import run_net

EDGES = (1000, 2000, 4000, 8000, 16000)
V = 1.14 ** 2 + 0.2 ** 2


def solve_chunks(sv, o, S, mode='on', **kw):
    n = o['T'].shape[0]; res = []
    for i in range(0, n, 2000):
        j = slice(i, i + 2000)
        oo = {k: v[j] for k, v in o.items()}
        res.append(sv.solve(oo, S['Phi0'][j], S['doff'][j], S['m'][j], psi_from_eps(S['eps'][j]), mode=mode, **kw))
    return {k: torch.cat([r[k] for r in res]) for k in res[0]}


from simcode2.fourpi.loadhead import load_head     # all head architectures


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--head', default='v3L'); ap.add_argument('--suffix', default='_epspi')
    a = ap.parse_args()
    dev = 'cuda'
    st = json.load(open(f'{REPO}/out/4pi/psf4pi/NPC_f_dxy15.json'))['psf']
    lut = LUT4Pi(st, P=15, dev=dev)
    S = Sim4Pi(lut, seed=11).sample(6000, p_nb=0.0, p_on=1.0, eps_range=math.pi)
    bc = torch.load(f'{REPO}/out/4pi/sim/bench_classical{a.suffix}.pt', weights_only=False)
    assert torch.allclose(bc['S']['x'], S['x'])
    n = len(S['x']); tr = torch.stack([S['x'], S['y'], S['z']], 1); sc = torch.tensor([128.0, 128.0, 1000.0], device=dev)
    cr = (bc['cr_p'][:, :3] * sc).cpu().numpy(); crs = (bc['cr_s'][:, :3] * sc)
    Nw = S['Nt'].sum(1).cpu().numpy()
    sv = Solve4Pi(lut); sv.build_table()
    img, mu, bg = S['img'], S['mu'], S['bg']
    P = img.shape[-1]; o_ = torch.arange(P, device=dev, dtype=torch.float32) - (P - 1) / 2
    oy, ox = torch.meshgrid(o_, o_, indexing='ij')
    g = mu / mu.sum((-1, -2), keepdim=True).clamp(min=1e-6)
    var = mu + bg + V
    w1 = g / var; w1 = w1 / (w1 * g).sum((-1, -2), keepdim=True)
    pc = torch.stack([S['x'][:, None] + S['doff'][:, :, 0], S['y'][:, None] + S['doff'][:, :, 1]], -1)
    w2 = torch.exp(-((ox[None, None] - pc[:, :, 0, None, None]) ** 2 + (oy[None, None] - pc[:, :, 1, None, None]) ** 2) / 2.0)
    w2 = w2[:, None].expand(-1, 6, -1, -1, -1)
    y = img - bg
    To = (w1 * y).sum((-1, -2)); Co = (w2 * y).sum((-1, -2))
    v11 = (w1 * w1 * var).sum((-1, -2)); v22 = (w2 * w2 * var).sum((-1, -2)); v12 = (w1 * w2 * var).sum((-1, -2))
    Lo = torch.zeros(n, 6, 4, 2, 2, device=dev)
    Lo[..., 0, 0] = v11.sqrt(); Lo[..., 1, 0] = v12 / v11.sqrt(); Lo[..., 1, 1] = (v22 - v12 ** 2 / v11).clamp(min=1e-6).sqrt()
    weak = dict(dxy=torch.stack([S['x'], S['y']], 1), z=S['z'], ls=torch.log(torch.tensor([1.0, 1.0, 0.2], device=dev)).expand(n, 3))
    # efficient summed-image prior: truth + Gaussian noise at the summed-image (SMLM) CRLB, independent of the read-outs
    gen = torch.Generator(device=dev).manual_seed(5)
    sp = bc['cr_s'][:, :3].to(dev).clamp(min=1e-4)
    pe = torch.randn(n, 3, device=dev, generator=gen) * sp
    smlm = dict(dxy=torch.stack([S['x'] + pe[:, 0], S['y'] + pe[:, 1]], 1), z=(S['z'] + pe[:, 2]).clamp(-0.58, 0.58), ls=torch.log(sp))
    net, ck = load_head(a.head)
    o = run_net(net, S)
    netp = dict(dxy=o['dxy'], z=o['z'], ls=o['ls'])
    rows = {}

    def add(name, oo):
        r = solve_chunks(sv, oo, S)
        rows[name] = ((r['th'][:, :3] - tr) * sc).cpu().numpy()
        return r
    rows['pixel MLE'] = ((bc['th_p'][:, :3] - tr) * sc).cpu().numpy()
    add('oracle T,C,cov + weak prior', dict(T=To, C=Co, L=Lo, **weak))
    add('oracle T,C,cov + efficient summed-image prior', dict(T=To, C=Co, L=Lo, **smlm))
    add('oracle T,C,cov + network prior', dict(T=To, C=Co, L=Lo, **netp))
    add('net T,C,cov + weak prior', dict(T=o['T'], C=o['C'], L=o['L'], **weak))
    rN = add('net T,C,cov + network prior (production)', dict(T=o['T'], C=o['C'], L=o['L'], **netp))
    add('net T,C,cov + efficient summed-image prior', dict(T=o['T'], C=o['C'], L=o['L'], **smlm))
    add('net T, oracle C, oracle cov + weak prior', dict(T=o['T'], C=Co, L=Lo, **weak))
    add('oracle T, net C, oracle cov + weak prior', dict(T=To, C=o['C'], L=Lo, **weak))
    add('net T,C, oracle cov + weak prior', dict(T=o['T'], C=o['C'], L=Lo, **weak))
    add('oracle T,C, net cov + weak prior', dict(T=To, C=Co, L=o['L'], **weak))
    # T only / C only information (other read-out with a huge variance)
    big = Lo.clone(); big[..., 1, 0] = 0; big[..., 1, 1] = 1e4
    add('oracle T only + weak prior', dict(T=To, C=Co, L=big, **weak))
    # linear part of the network read-out (before the nonlinear correction)
    if 'T_lin' in o:
        add('net linear read-outs (T_lin, C_lin), net cov + weak prior', dict(T=o['T_lin'], C=o['C_lin'], L=o['L'], **weak))
    out = dict(head=a.head, it=ck.get('it'), rows={}, tables={})
    # ---- A/B/C table: three statistics
    bins = list(zip(EDGES[:-1], EDGES[1:]))
    hdr = '| row | ' + ' | '.join(f'{lo}-{hi}' for lo, hi in bins) + ' |\n|---|' + '---|' * len(bins)
    for stat in ('rms_trim60', 'rms_trim5s', 'robust'):
        L_ = [hdr]
        for nm, e in rows.items():
            cells = []
            for lo, hi in bins:
                s = (Nw >= lo) & (Nw < hi); u = e[s] / cr[s]
                if stat == 'rms_trim60':
                    ok = (np.abs(e[s, 2]) <= 100) & (np.abs(e[s, :2]) <= 60).all(1)
                    rr = [float(np.sqrt(np.mean(e[s][ok, k] ** 2) / np.mean(cr[s][ok, k] ** 2))) for k in range(3)]
                elif stat == 'rms_trim5s':
                    rr = [float(np.sqrt(np.mean(u[np.abs(u[:, k]) < 5, k] ** 2))) for k in range(3)]
                else:
                    rr = [float(1.4826 * np.median(np.abs(u[:, k] - np.median(u[:, k])))) for k in range(3)]
                out['rows'].setdefault(nm, {}).setdefault(stat, []).append(rr)
                cells.append(' / '.join(f'{v:.2f}' for v in rr))
            L_.append(f'| {nm} | ' + ' | '.join(cells) + ' |')
        out['tables'][stat] = L_
        print(f'\n== {stat}: x / y / z error over pixel CRLB'); print('\n'.join(L_))
    # ---- tails: fraction beyond 3 and 5 CRLB sigma (production row vs pixel MLE vs oracle)
    print('\n== tails: fraction |e| > 3 CRLB (x, z)')
    for nm in ('pixel MLE', 'oracle T,C,cov + weak prior', 'net T,C,cov + network prior (production)'):
        e = rows[nm]; cells = []
        for lo, hi in bins:
            s = (Nw >= lo) & (Nw < hi); u = np.abs(e[s] / cr[s])
            cells.append(f'{np.mean(u[:, 0] > 3):.3f} / {np.mean(u[:, 2] > 3):.3f}')
        print(f'| {nm} | ' + ' | '.join(cells) + ' |')
    # ---- B: prior quality
    print('\n== network prior error / summed-image CRLB (x / y / z, RMS trimmed 5 sigma), and reported sigma / actual')
    ep = (torch.stack([o['dxy'][:, 0], o['dxy'][:, 1], o['z']], 1) - tr) * sc
    up = (ep / crs).cpu().numpy(); us = (ep / (torch.exp(o['ls']) * sc)).cpu().numpy(); out['prior'] = []
    for lo, hi in bins:
        s = (Nw >= lo) & (Nw < hi)
        r1 = [float(np.sqrt(np.mean(up[s][np.abs(up[s][:, k]) < 5, k] ** 2))) for k in range(3)]
        r2 = [float(1.4826 * np.median(np.abs(us[s][:, k]))) for k in range(3)]
        out['prior'].append(dict(lo=lo, hi=hi, err_over_smlm_crlb=r1, whitened_sd=r2))
        print(f'{lo}-{hi}: err/CRLB_sum ' + ' / '.join(f'{v:.2f}' for v in r1) + '   whitened sd ' + ' / '.join(f'{v:.2f}' for v in r2))
    # ---- D: read-out errors per bin
    print('\n== read-out errors vs oracle noise per bin (RMS ratios)')
    print('bin | T total | T channel-gain part removed | T channel-sum per frame | T frame-contrast (mean-removed over frames, channel sum) | C total | gain T | modulation slope | whitened sd T / C')
    eT = o['T'] - S['T']; eC = o['C'] - S['C']; eTo = To - S['T']; eCo = Co - S['C']
    out['readout'] = []
    for lo, hi in bins:
        s = torch.tensor((Nw >= lo) & (Nw < hi), device=dev)
        def rat(a_, b_): return float(torch.sqrt((a_[s] ** 2).mean() / (b_[s] ** 2).mean()))
        def degain(e_):   # remove the best per-(sample, channel) gain error: e_tc - g_c T_tc
            gch = (e_ * S['T']).sum(1, keepdim=True) / (S['T'] ** 2).sum(1, keepdim=True)
            return e_ - gch * S['T']
        def fcon(e_):     # channel sum per frame, then remove the part proportional to the true frame profile per angle
            es = e_.sum(2).reshape(-1, 2, 3); ts = S['T'].sum(2).reshape(-1, 2, 3)
            gg = (es * ts).sum(2, keepdim=True) / (ts ** 2).sum(2, keepdim=True)
            return (es - gg * ts).reshape(-1, 6)
        Ts = S['T'][s]; gain = float((o['T'][s] * Ts).sum() / (Ts ** 2).sum())
        prof = (o['T'] - o['T'].mean(1, keepdim=True))[s]; proft = (S['T'] - S['T'].mean(1, keepdim=True))[s]
        slope = float((prof * proft).sum() / (proft * proft).sum())
        a__ = o['L'][..., 0, 0]; b__ = o['L'][..., 1, 0]; d__ = o['L'][..., 1, 1]
        u1 = (eT / a__)[s]; u2 = ((eC - b__ * eT / a__) / d__)[s]
        d = dict(lo=lo, hi=hi, T=rat(eT, eTo), T_degain=rat(degain(eT), degain(eTo)), T_chsum=rat(eT.sum(2), eTo.sum(2)),
                 T_framecontrast=rat(fcon(eT), fcon(eTo)), C=rat(eC, eCo), gain_T=gain, mod_slope=slope,
                 wsd_T=float(u1.std()), wsd_C=float(u2.std()),
                 lin_T=rat(o['T_lin'] - S['T'], eTo) if 'T_lin' in o else None,
                 bias_T_rel=float((eT[s].sum((1, 2)) / S['T'][s].sum((1, 2))).mean()),
                 bg_err=float((o['bg'][s] - S['bg'][s][:, :, :, 0, 0].mean()).abs().mean()) if 'bg' in o else None)
        out['readout'].append(d)
        print(f"{lo}-{hi} | {d['T']:.2f} | {d['T_degain']:.2f} | {d['T_chsum']:.2f} | {d['T_framecontrast']:.2f} | {d['C']:.2f} | {d['gain_T']:.3f} | {d['mod_slope']:.3f} | {d['wsd_T']:.2f} / {d['wsd_C']:.2f} | linear-part T {d['lin_T']}")
    # ---- E: bias of the production position
    print('\n== bias of the production row (nm): slope vs true offset (per px -> fraction), pattern-phase harmonics, vs z')
    e = torch.tensor(rows['net T,C,cov + network prior (production)'], device=dev)
    eo = torch.tensor(rows['oracle T,C,cov + weak prior'], device=dev)
    out['bias'] = []
    for lo, hi in bins:
        s = torch.tensor((Nw >= lo) & (Nw < hi), device=dev)
        d = dict(lo=lo, hi=hi)
        for nm_, ee in (('net', e), ('oracle', eo)):
            okk = s & (ee[:, :2].abs() < 30).all(1) & (ee[:, 2].abs() < 60)
            for k, nmk in ((0, 'x'), (1, 'y')):
                t = tr[okk, k] * 128; v = ee[okk, k]
                sl = float(((t - t.mean()) * (v - v.mean())).sum() / ((t - t.mean()) ** 2).sum())
                ph = K_PAT[k] * tr[okk, k] - S['eps'][okk, k]
                h = {}
                for q in (1, 2, 3):
                    h[q] = float(torch.sqrt((2 * (v * torch.cos(q * ph)).mean()) ** 2 + (2 * (v * torch.sin(q * ph)).mean()) ** 2))
                d[f'{nm_}_{nmk}'] = dict(mean=float(v.mean()), slope=sl, harm=h, sd=float(v.std()))
            v = ee[okk, 2]; zz = tr[okk, 2]
            zb = [float(v[(zz >= z0) & (zz < z0 + 0.15)].mean()) for z0 in (-0.45, -0.3, -0.15, 0.0, 0.15, 0.3)]
            zs_ = [float(v[(zz >= z0) & (zz < z0 + 0.15)].std()) for z0 in (-0.45, -0.3, -0.15, 0.0, 0.15, 0.3)]
            ph4 = S['Phi0'][okk] + 4 * math.pi * zz / 0.246 * 0   # 4Pi phase at the emitter is Phi0 + 2 kz z; harmonic vs Phi0 only
            h4 = float(torch.sqrt((2 * (v * torch.cos(S['Phi0'][okk])).mean()) ** 2 + (2 * (v * torch.sin(S['Phi0'][okk])).mean()) ** 2))
            d[f'{nm_}_z'] = dict(mean=float(v.mean()), by_z=zb, sd_by_z=zs_, harm_Phi0=h4)
        out['bias'].append(d)
        print(f"{lo}-{hi}: net x mean {d['net_x']['mean']:+.2f} slope {d['net_x']['slope']:+.4f} harm1-3 " + ' '.join(f"{d['net_x']['harm'][q]:.2f}" for q in (1, 2, 3))
              + f" (sd {d['net_x']['sd']:.2f}) | y mean {d['net_y']['mean']:+.2f} slope {d['net_y']['slope']:+.4f} harm " + ' '.join(f"{d['net_y']['harm'][q]:.2f}" for q in (1, 2, 3))
              + f" | oracle x slope {d['oracle_x']['slope']:+.4f} harm " + ' '.join(f"{d['oracle_x']['harm'][q]:.2f}" for q in (1, 2, 3)))
        print(f"      z net mean {d['net_z']['mean']:+.2f} by z " + ' '.join(f'{v:+.2f}' for v in d['net_z']['by_z']) + ' sd ' + ' '.join(f'{v:.2f}' for v in d['net_z']['sd_by_z'])
              + f" | oracle sd by z " + ' '.join(f'{v:.2f}' for v in d['oracle_z']['sd_by_z']))
    # CRLB by z for reference
    zz = S['z'].cpu().numpy()
    print('\n== z error / CRLB by true z (2000-8000 e), trimmed 5 sigma: pixel MLE | oracle weak | production')
    s0 = (Nw >= 2000) & (Nw < 8000); out['z_by_z'] = []
    for z0 in (-0.45, -0.3, -0.15, 0.0, 0.15, 0.3):
        s = s0 & (zz >= z0) & (zz < z0 + 0.15); c = []
        for nm in ('pixel MLE', 'oracle T,C,cov + weak prior', 'net T,C,cov + network prior (production)'):
            u = rows[nm][s, 2] / cr[s, 2]; c.append(float(np.sqrt(np.mean(u[np.abs(u) < 5] ** 2))))
        out['z_by_z'].append(dict(z0=z0, vals=c)); print(f'z {z0:+.2f}..{z0 + 0.15:+.2f}: ' + ' | '.join(f'{v:.2f}' for v in c))
    hn = os.path.basename(a.head).replace('head4pi_', '').replace('.pt', '')
    json.dump(out, open(f'{REPO}/out/4pi/sim/review_{hn}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
