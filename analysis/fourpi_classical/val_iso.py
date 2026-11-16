"""Validation of the classical 4Pi-SIMFLUX fitter (cl4pi.py) on isolated on-through emitters (simcode/fourpi/sim4pi.py,
bead-fitted 4Pi PSF, pattern phase uniform over 2 pi at the patch centre, no neighbours), known pattern and 4Pi phase
offset Phi0 (the movie tests estimate them). Reports RMS / CRLB per photon bin for
  summed fit (x, y, z_ast), published-style (SIMFLUX two-step x, y; 4Pi moment phase z unwrapped against z_ast),
  pixel MLE (24 images), and the fringe-error rates (|dz| > 100 nm, lateral |dx| > 50 nm).
CRLB = pixel Fisher of the 24-image on-through model at the truth.
Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi_classical/val_iso.py [--n 6000]
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p_ in (REPO, os.path.dirname(os.path.abspath(__file__))):
    if p_ not in sys.path: sys.path.insert(0, p_)
from simcode.config import root; P4 = root('fourpi')
from simcode.psf import fourpi_state
from simcode.fourpi.spline_table import LUT4Pi
from simcode.simulate.fourpi_patches import FourPiPatchSim, DELTA_A, K_PAT
from pixel_lm import fisher_j, crlb
from cl4pi import Classic4Pi, run_chunk

OUT = f'{P4}/classical'
EDGES = [300, 1000, 2000, 4000, 8000, 16000]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=6000); ap.add_argument('--seed', type=int, default=21)
    ap.add_argument('--chunk', type=int, default=1500)
    a = ap.parse_args()
    from simcode.util.proctitle import set_title; set_title('4pi-classic')
    dev = 'cuda'
    st = fourpi_state('NPC_f_dxy15', P4)
    lut = LUT4Pi(st, P=15, dev=dev)
    c4 = Classic4Pi(lut, delta=DELTA_A, dev=dev)
    print(f'calibration: fringe {c4.fringe_um * 1000:.0f} nm, monotone {c4.monotone}, tau {c4.tau.cpu().numpy().round(3)}, '
          f'|e| {c4.e.abs().cpu().numpy().round(3)}, arg e {np.degrees(np.angle(c4.e.cpu().numpy())).round(0)}', flush=True)
    sim = FourPiPatchSim(lut, dev=dev, seed=a.seed)
    k = torch.tensor([[K_PAT[0], 0.0], [0.0, K_PAT[1]]], device=dev)
    R = {}; T = {}
    t0 = time.time()
    for i in range(0, a.n, a.chunk):
        S = sim.sample(a.chunk, p_nb=0.0, p_on=1.0, eps_range=math.pi)
        psi = torch.stack([2 * math.pi * s / 3 + S['eps'][:, aa] for aa in range(2) for s in range(3)], 1)
        o = run_chunk(c4, S['img'], S['doff'], k, psi, S['m'], Phi0=S['Phi0'])
        # CRLB at the truth: reuse the pixel model
        tt = torch.cat([torch.stack([S['x'], S['y'], S['z'], S['F']], 1), S['bg'][:, 0, :, 0, 0]], 1)
        cr_list = []
        for j in range(0, len(tt), 500):
            sl = slice(j, j + 500)
            pj = c4.pixel_mle(S['img'][sl], S['doff'][sl], tt[sl, :3], S['m'][sl], psi[sl], k, S['Phi0'][sl], iters=0, ret_fisher=True)
            cr_list.append(crlb(fisher_j(pj['fj'], tt[sl], 1.14 ** 2 + 0.2 ** 2)))
        o['crlb_true'] = torch.cat(cr_list)
        for kk, v in o.items(): R.setdefault(kk, []).append(v.float().cpu())
        for kk in ('x', 'y', 'z', 'Nt', 'm', 'Phi0'): T.setdefault(kk, []).append(S[kk].float().cpu())
        print(f'{i + a.chunk}/{a.n} {time.time() - t0:.0f}s', flush=True)
    R = {k_: torch.cat(v).numpy() for k_, v in R.items()}; T = {k_: torch.cat(v).numpy() for k_, v in T.items()}
    Nw = T['Nt'].sum(1)
    px = 128.0
    est = {'summed (x0, y0, z_ast)': (R['x0'], R['y0'], R['z_ast']),
           'published-style (SIMFLUX two-step x, y; 4Pi phase z)': (R['xs'], R['ys'], R['z4']),
           'pixel MLE 24 images': (R['xp'], R['yp'], R['zp'])}
    res = dict(n=a.n, fringe_nm=c4.fringe_um * 1000, methods={})
    L = []
    for nm, (x, y, z) in est.items():
        ex, ey, ez = (x - T['x']) * px, (y - T['y']) * px, (z - T['z']) * 1000
        rows = []
        L.append(f'\n{nm}')
        L.append('| window e | n | x RMS / CRLB | y RMS / CRLB | z RMS / CRLB | x, y robust | z robust | z fringe err | lat. hops |')
        L.append('|---|---|---|---|---|---|---|---|---|')
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            s = (Nw >= lo) & (Nw < hi)
            if s.sum() < 50: continue
            okz = s & (np.abs(ez) <= 100); okl = s & (np.abs(ex) <= 50) & (np.abs(ey) <= 50)
            cx_, cy_, cz_ = (float(np.sqrt(np.mean((R['crlb_true'][s, q] * sc) ** 2))) for q, sc in ((0, px), (1, px), (2, 1000.)))
            rx = float(np.sqrt(np.mean(ex[okl] ** 2))); ry = float(np.sqrt(np.mean(ey[okl] ** 2))); rz = float(np.sqrt(np.mean(ez[okz] ** 2)))
            rb = lambda e: float(1.4826 * np.median(np.abs(e[s] - np.median(e[s]))))
            r = dict(lo=lo, hi=hi, n=int(s.sum()), x_rms=rx, y_rms=ry, z_rms=rz, crlb=[cx_, cy_, cz_], x_rob=rb(ex), y_rob=rb(ey), z_rob=rb(ez),
                     fringe=float(np.mean(np.abs(ez[s]) > 100)), hops=float(np.mean(~okl[s])),
                     bias=[float(np.median(ex[s])), float(np.median(ey[s])), float(np.median(ez[s]))])
            rows.append(r)
            L.append(f"| {lo}-{hi} | {r['n']} | {rx:.2f} / {cx_:.2f} ({rx / cx_:.2f}) | {ry:.2f} / {cy_:.2f} ({ry / cy_:.2f}) | {rz:.2f} / {cz_:.2f} ({rz / cz_:.2f}) "
                     f"| {r['x_rob']:.2f}, {r['y_rob']:.2f} | {r['z_rob']:.2f} | {r['fringe']:.3f} | {r['hops']:.4f} |")
        res['methods'][nm] = rows
    # filter statistics on this set (all should pass: on-through, isolated)
    fs = dict(R_gt_0p2=float(np.mean(R['R'] > 0.2)), MD_out=float(np.mean(((R['MD'] < 0.6) | (R['MD'] > 1.2)).any(1))),
              C_med=float(np.median(R['C'])), chi2px_med=float(np.median(R['chi2px'])), llr_med=float(np.median(R['llr'])),
              conv=float(np.mean(R['conv'])), bound=float(np.mean(R['at_bound'])), convp=float(np.mean(R['convp'])), boundp=float(np.mean(R['boundp'])),
              chi2p_med=float(np.median(R['chi2p'])),
              sphi_over_emp=None)
    res['filters'] = fs
    res['table'] = L
    L.append('\nfilter stats: ' + json.dumps(fs))
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f'{OUT}/val_iso.json', 'w'), indent=1)
    open(f'{OUT}/val_iso.txt', 'w').write('\n'.join(L) + '\n')
    np.savez(f'{OUT}/val_iso.npz', **{('est_' + k_): v for k_, v in R.items()}, **{('gt_' + k_): v for k_, v in T.items()})
    print('\n'.join(L))


if __name__ == '__main__':
    main()
