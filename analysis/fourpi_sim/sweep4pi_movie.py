"""4Pi-SIMCODE density sweep: simulated movie of a microtubule cell field (mt_cells.py, webSMLM / inSiliScope model).

Everything except the sample is analysis/fourpi/showcase4pi_movie.py (imported, run unchanged through its main()):
bead-fitted coherent 4Pi PSF NPC_f_dxy15, real channel layout / registration / camera, SIM pattern of the MT data
(m 0.9), drifting 4Pi phase field (simmovie4pi.phi0_field), DNA-PAINT kinetics (ton 10 frames, log-normal photons, median
450 e/frame), measured background with +-20 % structure, 60 x 60 px (7.68 um) per channel.
Sample: mt_cells.simulate(seed, fov 7.68 um, offset, focus, z_range None) -> to_showcase_sites (x px, y px, z um from the
focus, microtubule id); its make_sites() is replaced by these labels (webSMLM defaults: 70 % of the 13_3 lattice sites,
12 nm binder + 2-5 nm linker, i.e. labels ~24.6 nm from the axis). gt_axis.npz = the centrelines resampled at 2 nm (x, y,
z um, unit tangent, microtubule id), as showcase4pi_eval.py expects.
Default field: seed 1, start view + (0, 3.5) um = flat lamella below the nucleus (labels 0-300 nm above the coverslip),
focus 150 nm above the coverslip -> labels within about -0.155..+0.15 um of the focus.
Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi_sim/sweep4pi_movie.py --tag mtc_d03 --dens 0.3 --frames 30000
       (--dry 1: print the structure statistics only)
"""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, f'{REPO}/analysis/fourpi')
if REPO not in sys.path: sys.path.insert(0, REPO)
import mt_cells as MC

PX_NM = 128.0
FOV_UM = 60 * PX_NM / 1000


def structure(seed=1, offset=(0.0, 3.5), focus_nm=150.0, eff=0.7):
    r = MC.simulate(seed=seed, fov_um=FOV_UM, offset_um=offset, focus_nm=focus_nm, z_range_nm=None, label_efficiency=eff)
    S = MC.to_showcase_sites(r, px_nm=PX_NM)
    ok = (S[:, 0] >= 0) & (S[:, 0] < 60) & (S[:, 1] >= 0) & (S[:, 1] < 60)
    S = S[ok]
    ax = []
    for i, c in enumerate(r['centrelines']):
        if len(c) < 2: continue
        seg = np.linalg.norm(np.diff(c, axis=0), axis=1); s = np.r_[0, np.cumsum(seg)]
        q = np.arange(0, s[-1], 2.0)
        P = np.stack([np.interp(q, s, c[:, k]) for k in range(3)], 1)
        T = np.gradient(P, axis=0); T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
        inside = (P[:, 0] > -500) & (P[:, 0] < FOV_UM * 1000 + 500) & (P[:, 1] > -500) & (P[:, 1] < FOV_UM * 1000 + 500)
        if not inside.any(): continue
        P, T = P[inside], T[inside]
        ax.append(np.c_[P[:, 0] / 1000, P[:, 1] / 1000, (P[:, 2] - focus_nm) / 1000, T, np.full(len(P), i)])
    AX = np.concatenate(ax)
    return S, AX, r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cell_seed', type=int, default=1); p.add_argument('--dx', type=float, default=0.0)
    p.add_argument('--dy', type=float, default=3.5); p.add_argument('--focus_nm', type=float, default=150.0)
    p.add_argument('--dry', type=int, default=0)
    a, rest = p.parse_known_args()
    S, AX, r = structure(a.cell_seed, (a.dx, a.dy), a.focus_nm)
    z = S[:, 2] * 1000
    print(f'structure: seed {a.cell_seed}, centre {r["center_um"]} um, {len(S)} labels in the FOV, '
          f'{len(np.unique(S[:, 3]))} microtubules, tube {len(S) / (1625 * 0.7):.0f} um ({len(S) / (1625 * 0.7) / FOV_UM ** 2:.2f} um/um^2), '
          f'z from focus p0/1/50/99/100 {np.percentile(z, [0, 1, 50, 99, 100]).round(0)} nm', flush=True)
    if a.dry: return
    import showcase4pi_movie as SM
    SM.make_sites = lambda rng, r_=None, step=None, jit=None: (S, AX)
    sys.argv = [sys.argv[0]] + rest
    SM.main()


if __name__ == '__main__':
    main()
