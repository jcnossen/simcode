"""Bounds of the linear fixed-window read-out + K-read-out solve on the gate emitters, WITHOUT any learned read-out:
windows placed (1) at the network prior (pass 1), (2) at the pass-1 solved position (re-read-out), (3) at the truth
(bound of the window set). Network prior from --head (v3L). Background: frame-mean border median per channel, or true.
Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi/linbound4pi.py [--sets 0 1 2 ...]
"""
import argparse, json, math, os, sys
import numpy as np, torch
REPO = '/home/jc/simcode2'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode2.fourpi.lut4pi import LUT4Pi
from simcode2.fourpi.sim4pi import Sim4Pi
from simcode2.fourpi.solve4pi import psi_from_eps
from simcode2.fourpi.solvek4pi import SolveK
from simcode2.fourpi.linread4pi import linear_readout, border_bg
from simcode2.fourpi.loadhead import load_head
from eval_head4pi import run_net

EDGES = (1000, 2000, 4000, 8000, 16000)
SETS = [(('mf', 0.003), ('gauss', 1.0)),
        (('mf', 0.003), ('gauss', 0.7)),
        (('mf', 0.003), ('gauss', 1.4)),
        (('mf', 0.003), ('inc', 0.0), ('gauss', 1.0)),
        (('mf', 0.001), ('mf', 0.03), ('gauss', 1.0)),
        (('mf', 0.003), ('gauss', 0.7), ('gauss', 1.5)),
        (('mf', 0.003), ('gauss', 1.0), ('flat', 0.0)),
        (('mf', 0.003), ('cr', 0.003), ('ci', 0.003)),
        (('mf', 0.003), ('cr', 0.003), ('ci', 0.003), ('flat', 0.0)),
        (('mf', 0.003), ('cr', 0.02), ('ci', 0.02), ('flat', 0.0)),
        (('mf', 0.003), ('gauss', 1.0), ('cr', 0.003), ('ci', 0.003), ('flat', 0.0))]
dev = 'cuda'


def solve(sv, Y, Lb, pr, S, p0=None, Ws=None, sb=None, W=None):
    res = []
    for i in range(0, Y.shape[0], 2000):
        j = slice(i, i + 2000)
        oo = dict(Y=Y[j], Lb=Lb[j], dxy=pr['dxy'][j], z=pr['z'][j], ls=pr['ls'][j])
        if p0 is not None: oo['p0'] = p0[j]
        if W is not None: oo['W'] = W[j]
        if Ws is not None: oo['Wsum'] = Ws[j]; oo['sb'] = torch.full((Y[j].shape[0], 4), sb, device=Y.device)
        res.append(sv.solve(oo, S['Phi0'][j], S['doff'][j], S['m'][j], psi_from_eps(S['eps'][j])))
    return {k: torch.cat([r[k] for r in res]) for k in res[0]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--head', default='v3L'); ap.add_argument('--sets', type=int, nargs='*', default=list(range(len(SETS)))); ap.add_argument('--bound_only', type=int, default=0); ap.add_argument('--exact_only', type=int, default=0)
    a = ap.parse_args()
    st = json.load(open(f'{REPO}/out/4pi/psf4pi/NPC_f_dxy15.json'))['psf']
    lut = LUT4Pi(st, P=15, dev=dev)
    S = Sim4Pi(lut, seed=11).sample(6000, p_nb=0.0, p_on=1.0, eps_range=math.pi)
    bc = torch.load(f'{REPO}/out/4pi/sim/bench_classical_epspi.pt', weights_only=False)
    assert torch.allclose(bc['S']['x'], S['x'])
    sc = torch.tensor([128.0, 128.0, 1000.0], device=dev); cr = (bc['cr_p'][:, :3] * sc).cpu().numpy(); Nw = S['Nt'].sum(1).cpu().numpy()
    tr = torch.stack([S['x'], S['y'], S['z']], 1)
    net, _ = load_head(a.head); o = run_net(net, S)
    pr = dict(dxy=o['dxy'], z=o['z'], ls=o['ls'])
    img = S['img']; bm = border_bg(img); bt = S['bg'][:, 0].mean((-1, -2))
    out = {}

    def cells(th):
        e = ((th[:, :3] - tr) * sc).cpu().numpy(); c = []; v = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            s = (Nw >= lo) & (Nw < hi); u = e[s] / cr[s]
            q = [float(np.sqrt(np.mean(u[np.abs(u[:, k]) < 5, k] ** 2))) for k in range(3)]
            v.append(q); c.append(' / '.join(f'{x:.2f}' for x in q))
        return ' | '.join(c), v

    print('RMS (5-sigma trim) / pixel CRLB, x / y / z; bins 1-2k | 2-4k | 4-8k | 8-16k e')
    for si in a.sets:
        wins = SETS[si]; sv = SolveK(lut, wins); sv.build_table()
        print(f'\n== windows {wins}', flush=True)
        pt = torch.cat([o['dxy'], o['z'][:, None]], 1)
        if not a.exact_only:
            # NOTE: windows at the TRUTH leak the true position into the read-out (0.16 x CRLB in z for the cr / ci windows):
            # not a bound; kept only as a consistency check of the tables
            Y, Lb, Ws = linear_readout(img, S['doff'], lut, wins, tr[:, :2], tr[:, 2], bt)
            r = solve(sv, Y, Lb, pr, S); c, v = cells(r['th']); out[f'{si} truth true-bg'] = v
            print(f'| (leaky, not a bound) windows at the truth, true bg | {c} |', flush=True)
            Y, Lb, Ws = linear_readout(img, S['doff'], lut, wins, o['dxy'], o['z'], bt)
            r = solve(sv, Y, Lb, pr, S, p0=pt); c, v = cells(r['th']); out[f'{si} prior true-bg p0'] = v
            print(f'| windows at the network prior, true bg, first-order placement model | {c} |', flush=True)
            Y, Lb, Ws = linear_readout(img, S['doff'], lut, wins, o['dxy'], o['z'], bm)
            r = solve(sv, Y, Lb, pr, S); c, v = cells(r['th']); out[f'{si} prior border-bg plain'] = v
            print(f'| windows at the network prior, border bg, plain model | {c} |', flush=True)
            r = solve(sv, Y, Lb, pr, S, p0=pt); c, v = cells(r['th']); out[f'{si} prior border-bg p0'] = v
            print(f'| + first-order placement model | {c} |', flush=True)
            r = solve(sv, Y, Lb, pr, S, p0=pt, Ws=Ws, sb=0.2); c, v = cells(r['th']); out[f'{si} prior border-bg p0 bgnuis0.2'] = v
            print(f'| + first-order placement model + background nuisance (prior sd 0.2 e/px) | {c} |', flush=True)
            th = r['th']
            Y, Lb, Ws = linear_readout(img, S['doff'], lut, wins, th[:, :2], th[:, 2], bm + th[:, -4:])
            r2 = solve(sv, Y, Lb, pr, S, p0=th[:, :3], Ws=Ws, sb=0.2); c, v = cells(r2['th']); out[f'{si} reread'] = v
            print(f'| re-read-out at the solved position and background | {c} |', flush=True)
        # exact window model (the head's actual windows in the solve)
        Y, Lb, Ws, W = linear_readout(img, S['doff'], lut, wins, o['dxy'], o['z'], bm, return_w=True)
        rx = solve(sv, Y, Lb, pr, S, Ws=Ws, sb=0.2, W=W); c, v = cells(rx['th']); out[f'{si} prior border-bg exact bgnuis0.2'] = v
        print(f'| EXACT window model, windows at the network prior, border bg + nuisance (0.2) | {c} |', flush=True)
        berr = (rx['th'][:, -4:] + bm - bt); print(f'   background error: border {float((bm - bt).pow(2).mean().sqrt()):.3f} -> after the solve {float(berr.pow(2).mean().sqrt()):.3f} e/px')
        sig = rx['sig'][:, :3] * sc; e = (rx['th'][:, :3] - tr) * sc; u = (e / sig).cpu().numpy()
        print('   reported sigma calibration (robust sd of error / sigma, x / y / z): ' + ' | '.join(' / '.join(f'{1.4826 * np.median(np.abs(u[(Nw >= lo) & (Nw < hi), k])):.2f}' for k in range(3)) for lo, hi in zip(EDGES[:-1], EDGES[1:])))
        fr = (e[:, 2].abs() > 100).cpu().numpy(); print('   4Pi fringe errors: ' + ' | '.join(f'{fr[(Nw >= lo) & (Nw < hi)].mean():.4f}' for lo, hi in zip(EDGES[:-1], EDGES[1:])))
        en = e.cpu().numpy(); cc = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):   # the handover statistic (60 nm / 100 nm trim)
            s_ = (Nw >= lo) & (Nw < hi); ok = s_ & (np.abs(en[:, 2]) <= 100) & (np.abs(en[:, :2]) <= 60).all(1)
            cc.append(' / '.join(f'{np.sqrt(np.mean(en[ok, k] ** 2) / np.mean(cr[ok, k] ** 2)):.2f}' for k in range(3)))
        print('   handover statistic (60 / 100 nm trim): ' + ' | '.join(cc))
        Y, Lb, Ws, W = linear_readout(img, S['doff'], lut, wins, o['dxy'], o['z'], bt, return_w=True)
        rx = solve(sv, Y, Lb, pr, S, W=W); c, v = cells(rx['th']); out[f'{si} prior true-bg exact'] = v
        print(f'| EXACT window model, windows at the network prior, true bg | {c} |', flush=True)
    json.dump(out, open(f"{REPO}/out/4pi/sim/linbound_{'_'.join(str(i) for i in a.sets)}.json", 'w'), indent=1)


if __name__ == '__main__':
    main()
