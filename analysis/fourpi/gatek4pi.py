"""Gate table of a learned K-read-out head (Head4PiK): the head's intensities + explicit exact-window solve vs the FIXED
linear read-out (same windows, same prior: the reference / teacher; forward with learned=False) and the pixel MLE.
  set A: the gate emitters (6000 isolated on-through, arbitrary pattern phase, pattern and Phi0 given): RMS / pixel CRLB
  set B: emitters with 1-3 neighbours within 0.6-7 px (seed 12, on-through): robust and RMS error in nm, fixed vs learned
Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi/gatek4pi.py --work <4Pi work folder> --head v9   -> <work>/sim/gatek_<head>.json
"""
import argparse, json, math, os, sys
import numpy as np, torch
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.simulate.fourpi_patches import FourPiPatchSim
from simcode.solve.fourpi import psi_from_eps
from simcode.solve.fourpi import FourPiSolve
from simcode.emitter.fourpi import load as load_head

EDGES = (1000, 2000, 4000, 8000, 16000)
dev = 'cuda'


def run(net, sv, S, learned, nbr=None, sb=0.2):
    res = []; cs = []
    for i in range(0, S['img'].shape[0], 1000):
        j = slice(i, i + 1000)
        with torch.no_grad():
            o = net(S['img'][j], S['doff'][j], learned=learned, **({'nbr': nbr[j]} if nbr is not None else {}))
            oo = dict(Y=o['Y'], Lb=o['Lb'], dxy=o['dxy'], z=o['z'], ls=o['ls'], W=o['W'], Wsum=o['Wsum'],
                      sb=torch.full((o['Y'].shape[0], 4), sb, device=dev))
            res.append(sv.solve(oo, S['Phi0'][j], S['doff'][j], S['m'][j], psi_from_eps(S['eps'][j]))); cs.append(o['cs'])
    r = {k: torch.cat([q[k] for q in res]) for k in res[0]}; r['cs'] = torch.cat(cs)
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--head', default='v9')
    ap.add_argument('--ksum', type=int, default=1, help='K summed windows: background 0.4 K - 1.2 K e/px (the real range), read sd 1.14 sqrt(K), offset error 0.2 K; reference = the fixed read-out (no pixel-MLE row for K > 1)')
    ap.add_argument('--nb_max', type=int, default=3)
    ap.add_argument('--work', required=True, help='4Pi work folder (head tags: <work>/models/head4pi_<tag>.pt)')
    a = ap.parse_args(); K = a.ksum; P4 = a.work
    net, ck = load_head(a.head, work=P4); lut = net.lut
    sv = FourPiSolve(lut, net.wins); sv.tab_lim = None
    sc = torch.tensor([128.0, 128.0, 1000.0], device=dev)
    out = dict(head=a.head, it=ck.get('it'))
    # ---- set A
    kw = dict(bg=(0.4 * K, 1.2 * K), rn=1.14 * math.sqrt(K), offerr=0.2 * K) if K > 1 else {}
    sbk = 0.2 * math.sqrt(K)
    if hasattr(net, 'vread'): net.vread = K * 1.14 ** 2 + (0.2 * K) ** 2
    S = FourPiPatchSim(lut, seed=11).sample(6000, p_nb=0.0, p_on=1.0, eps_range=math.pi, **kw)
    Nw = S['Nt'].sum(1).cpu().numpy(); tr = torch.stack([S['x'], S['y'], S['z']], 1)
    if K == 1:
        bc = torch.load(f'{P4}/sim/bench_classical_epspi.pt', weights_only=False)
        assert torch.allclose(bc['S']['x'], S['x'])
        cr = (bc['cr_p'][:, :3] * sc).cpu().numpy()
        rows = {'pixel MLE': ((bc['th_p'][:, :3] - tr) * sc).cpu().numpy()}
    else:   # reference sigma = the fixed read-out's reported sigma (calibrated, ~1.05 x CRLB at K = 1)
        rows = {}; cr = (run(net, sv, S, False, sb=sbk)['sig'][:, :3] * sc).cpu().numpy()
        print(f'K = {K}: reference = reported sigma of the fixed read-out; median x / z at 2-4k e: {np.median(cr[(Nw >= 2000) & (Nw < 4000), 0]):.2f} / {np.median(cr[(Nw >= 2000) & (Nw < 4000), 2]):.2f} nm')
    R = {}
    for nm, lr in (('fixed linear read-out (reference)', False), (f'learned head {a.head}', True)):
        R[nm] = run(net, sv, S, lr, sb=sbk); rows[nm] = ((R[nm]['th'][:, :3] - tr) * sc).cpu().numpy()
    print(f'== set A (isolated gate emitters), head {a.head} it {ck.get("it")}: RMS / pixel CRLB x / y / z, handover statistic (60 / 100 nm trim) [robust]')
    for nm, e in rows.items():
        cells = []; out[nm] = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            s = (Nw >= lo) & (Nw < hi); ok = s & (np.abs(e[:, 2]) <= 100) & (np.abs(e[:, :2]) <= 60).all(1)
            rr = [float(np.sqrt(np.mean(e[ok, k] ** 2) / np.mean(cr[ok, k] ** 2))) for k in range(3)]
            u = e[s] / cr[s]; rb = [float(1.4826 * np.median(np.abs(u[:, k] - np.median(u[:, k])))) for k in range(3)]
            out[nm].append(dict(lo=lo, hi=hi, rms=rr, robust=rb, fringe=float(np.mean(np.abs(e[s, 2]) > 100))))
            cells.append(' / '.join(f'{v:.2f}' for v in rr) + ' [' + ' / '.join(f'{v:.2f}' for v in rb) + ']')
        print(f'| {nm} | ' + ' | '.join(cells) + ' |')
    for nm in list(R):
        sig = (R[nm]['sig'][:, :3] * sc).cpu().numpy(); u = rows[nm] / sig
        print(f'   {nm}: error / reported sigma (robust) ' + ' | '.join(' / '.join(f'{1.4826 * np.median(np.abs(u[(Nw >= lo) & (Nw < hi), k])):.2f}' for k in range(3)) for lo, hi in zip(EDGES[:-1], EDGES[1:]))
              + f' ; mean covariance scale {float(R[nm]["cs"].mean()):.2f}')
    # ---- set B: neighbours
    S = FourPiPatchSim(lut, seed=12).sample(6000, p_nb=1.0, p_on=1.0, eps_range=math.pi, nb_max=a.nb_max, **kw)
    nbr = None
    if getattr(net, 'nbr_in', False):   # the other candidates as a detector would give them (0.3 px jitter, 20 % missed), fixed seed
        from simcode.emitter.fourpi import neighbour_map
        gg = torch.Generator(device=dev).manual_seed(3)
        dxn = torch.zeros(6000, a.nb_max, device=dev); dyn = torch.zeros_like(dxn); val = torch.zeros_like(dxn, dtype=torch.bool)
        for j, nb in enumerate(S['nb']):
            kk = int(nb['sel'].sum())
            dxn[nb['sel'], j] = nb['x'] + 0.3 * torch.randn(kk, device=dev, generator=gg); dyn[nb['sel'], j] = nb['y'] + 0.3 * torch.randn(kk, device=dev, generator=gg)
            val[nb['sel'], j] = (torch.rand(kk, device=dev, generator=gg) > 0.2) & (nb['N'].sum(1) > 150)
        nbr = neighbour_map(dxn, dyn, val, S['doff'])
    Nw = S['Nt'].sum(1).cpu().numpy(); tr = torch.stack([S['x'], S['y'], S['z']], 1)
    print(f'== set B (1-{a.nb_max} neighbours within 7 px, on-through): robust error nm x / y / z [RMS with the 60 / 100 nm trim; kept fraction]')
    for nm, lr in (('fixed linear read-out (reference)', False), (f'learned head {a.head}', True)):
        r = run(net, sv, S, lr, nbr=nbr if lr else None, sb=sbk); e = ((r['th'][:, :3] - tr) * sc).cpu().numpy(); cells = []; out['B ' + nm] = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            s = (Nw >= lo) & (Nw < hi); ok = s & (np.abs(e[:, 2]) <= 100) & (np.abs(e[:, :2]) <= 60).all(1)
            rb = [float(1.4826 * np.median(np.abs(e[s, k] - np.median(e[s, k])))) for k in range(3)]
            rr = [float(np.sqrt(np.mean(e[ok, k] ** 2))) for k in range(3)]
            out['B ' + nm].append(dict(lo=lo, hi=hi, robust=rb, rms=rr, kept=float(ok.sum() / s.sum())))
            cells.append(' / '.join(f'{v:.1f}' for v in rb) + ' [' + ' / '.join(f'{v:.1f}' for v in rr) + f'; {ok.sum() / s.sum():.2f}]')
        sig = (r['sig'][:, :3] * sc).cpu().numpy(); u = e / sig
        print(f'| {nm} | ' + ' | '.join(cells) + ' |')
        print('   error / reported sigma (robust): ' + ' | '.join(' / '.join(f'{1.4826 * np.median(np.abs(u[(Nw >= lo) & (Nw < hi), k])):.2f}' for k in range(3)) for lo, hi in zip(EDGES[:-1], EDGES[1:])))
    hn = os.path.basename(a.head).replace('head4pi_', '').replace('.pt', '')
    json.dump(out, open(f'{P4}/sim/gatek_{hn}' + (f'_k{K}' if K > 1 else '') + '.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
