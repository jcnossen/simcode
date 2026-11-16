"""Consistency test of the explicit 4Pi solve (simcode/solve/fourpi.py) against the simulator: feed the ground-truth
read-out targets (T = patch sum, C = Gaussian window sd 1 px at the emitter) without noise and a loose prior around the
truth, as the two windows ('flat', 'gauss' 1.0) of the solve (tables, co-moving windows); the solve must return the truth,
and the phase mode must return Phi0. GPU, ~1 min; needs the 4Pi PSF model <4Pi work folder>/psf4pi/NPC_f_dxy15.json (skipped without it).
Run: CUDA_VISIBLE_DEVICES=<gpu> python tests/test_solve4pi_oracle.py <4Pi work folder>  (or pytest tests --work4pi <4Pi work folder>)

Measured 2026-11-12 (2000 isolated on-through emitters, seed 5): mode 'on' p99 |error| 0.0026 / 0.0026 px, 5.1 nm in z;
mode 'free' (lateral position = the prior, by construction) z p99 5.1 nm; phase mode |dPhi| median 6.3 deg."""
import json, math, os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path: sys.path.insert(0, REPO)


def test_solve_oracle(work4pi):
    import torch
    f = f'{work4pi}/psf4pi/NPC_f_dxy15.json' if work4pi else 'the 4Pi work folder (--work4pi)'
    if not os.path.exists(f) or not torch.cuda.is_available():
        print(f'  skip (needs a GPU and {f})'); return
    from simcode.fourpi.spline_table import LUT4Pi
    from simcode.simulate.fourpi_patches import FourPiPatchSim
    from simcode.solve.fourpi import FourPiSolve, psi_from_eps
    lut = LUT4Pi(json.load(open(f))['psf'], P=15)
    n = 2000
    S = FourPiPatchSim(lut, seed=5).sample(n, p_nb=0.0, p_on=1.0)
    g = torch.Generator(device='cuda').manual_seed(0)
    o = dict(T=S['T'], C=S['C'], L=torch.zeros(n, 6, 4, 2, 2, device='cuda'),
             dxy=torch.stack([S['x'], S['y']], 1) + 0.1 * torch.randn(n, 2, device='cuda', generator=g),
             z=S['z'] + 0.02 * torch.randn(n, device='cuda', generator=g), ls=torch.log(torch.tensor([0.3, 0.3, 0.04], device='cuda')).expand(n, 3))
    o['L'][..., 0, 0] = (S['T'] + 5).sqrt(); o['L'][..., 1, 1] = (S['C'] + 2).sqrt()
    o['Y'] = torch.stack([S['T'], S['C']], -1); o['Lb'] = o.pop('L')
    sv = FourPiSolve(lut, wins=(('flat', 0.0), ('gauss', 1.0))); sv.build_table()
    psi = psi_from_eps(S['eps'])
    truth = torch.stack([S['x'], S['y'], S['z']], 1)
    p99 = {}
    for mode in ('on', 'free'):
        r = sv.solve(o, S['Phi0'], S['doff'], S['m'], psi, mode=mode)
        e = (r['th'][:, :3] - truth).abs(); p99[mode] = e.quantile(0.99, 0).tolist()
        print(f'  {mode}: max |err| x y (px), z (um): {[round(v, 4) for v in e.max(0).values.tolist()]}, p99 {[round(v, 4) for v in p99[mode]]}')
    r = sv.solve(o, torch.zeros_like(S['Phi0']), S['doff'], S['m'], psi, mode='free', fit_phi=True)
    dphi = torch.remainder(r['th'][:, -1] - S['Phi0'] + math.pi, 2 * math.pi) - math.pi
    med = math.degrees(dphi.abs().median())
    print(f'  phi mode: |dPhi| p50 / p99 (deg) {med:.1f} / {math.degrees(dphi.abs().quantile(0.99)):.1f}')
    assert p99['on'][0] < 0.01 and p99['on'][1] < 0.01 and p99['on'][2] < 0.015, p99['on']       # px, px, um
    assert p99['free'][2] < 0.015, p99['free']
    assert med < 12.0, med
    print('  ok 4Pi solve returns the simulator truth')


if __name__ == '__main__':
    print('test_solve_oracle', flush=True); test_solve_oracle(sys.argv[1] if len(sys.argv) > 1 else None)
    print('ALL OK')
