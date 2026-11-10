"""summary table of check1_<ds>.npz: tile-to-tile spread of the pattern-vs-prior offset and widths, real vs sim null."""
import os, json, numpy as np, sys
if any(a in ('-h', '--help') for a in sys.argv[1:]): print(__doc__); sys.exit()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simcode2 import config
from check1_offsets import spread, PB

OUT = config.out_dir('field')
L = []
pr = lambda s='': (print(s), L.append(s))
J = {}
for sub in ('all', 'iso'):
    for nt in (4, 8):
        pr(f'\n### subset {sub}, {nt}x{nt} tiles: spread of tile means about the FOV mean (rms / noise rms / chi2 per dof / excess rms, nm)')
        pr('| data | photons | n/tile med | mean dx: rms / noise / chi2 / excess | mean dy: rms / noise / chi2 / excess | sd dx excess (rel) | z-width x / y: tile range (FOV mean) | q_dof p50 range | zcal range |')
        pr('|---|---|---|---|---|---|---|---|---|')
        for ds in ('real05', 'sim05', 'real25', 'sim25'):
            R = np.load(f'{OUT}/check1_{ds}.npz')
            for bi, (lo, hi) in enumerate(PB):
                g = lambda k: R[f'{sub}_{nt}_{k}'][..., bi]
                n = g('n')
                if np.nanmedian(n) < 30: continue
                sx_, sy_ = spread(g('mx'), g('mx_se')), spread(g('my'), g('my_se'))
                # width: relative spread of sdx / zx across tiles
                zx, zxs = g('zx'), g('zx_se'); zy = g('zy')
                wsp = spread(zx, zxs)
                q = g('q50'); zc = R[f'{sub}_{nt}_zcal'][..., bi]
                lab = f'{lo}-{hi:.0f}' if hi < 1e8 else (f'>{lo}' if lo > 200 else 'all')
                pr(f'| {ds} | {lab} | {np.nanmedian(n):.0f} | {sx_["rms"]:.2f} / {sx_["noise"]:.2f} / {sx_["chi2"]:.1f} / **{sx_["excess"]:.2f}** | '
                   f'{sy_["rms"]:.2f} / {sy_["noise"]:.2f} / {sy_["chi2"]:.1f} / **{sy_["excess"]:.2f}** | '
                   f'{wsp["excess"] / np.nanmean(zx):.3f} | {np.nanmin(zx):.2f}-{np.nanmax(zx):.2f} ({np.nanmean(zx):.2f}) / {np.nanmin(zy):.2f}-{np.nanmax(zy):.2f} ({np.nanmean(zy):.2f}) | '
                   f'{np.nanmin(q):.2f}-{np.nanmax(q):.2f} | {np.nanmin(zc):.2f}-{np.nanmax(zc):.2f} |')
                J[f'{ds}_{sub}_{nt}_{lab}'] = dict(mx=sx_, my=sy_, zx=wsp, zx_mean=float(np.nanmean(zx)), zy_mean=float(np.nanmean(zy)),
                                                   q50=[float(np.nanmin(q)), float(np.nanmax(q))], zcal=[float(np.nanmin(zc)), float(np.nanmax(zc))])
        # sim truth: spread of the true final / prior bias per tile
        pr(f'\nsim truth ({sub}, {nt}x{nt}): spread of the tile-mean TRUE error (final | prior), x / y excess rms nm')
        for ds in ('sim05', 'sim25'):
            R = np.load(f'{OUT}/check1_{ds}.npz')
            for bi, (lo, hi) in enumerate(PB):
                g = lambda k: R[f'{sub}_{nt}_{k}'][..., bi]
                if np.nanmedian(g('n')) < 30: continue
                s = [spread(g(k), g(k + '_se')) for k in ('efx', 'efy', 'epx', 'epy')]
                pr(f'  {ds} {lo}-{hi:.0g}: final {s[0]["excess"]:.2f} / {s[1]["excess"]:.2f} (rms {s[0]["rms"]:.2f} / {s[1]["rms"]:.2f}, noise {s[0]["noise"]:.2f}) | '
                   f'prior {s[2]["excess"]:.2f} / {s[3]["excess"]:.2f} (rms {s[2]["rms"]:.2f} / {s[3]["rms"]:.2f})')
open(f'{OUT}/check1_summary.txt', 'w').write('\n'.join(L) + '\n')
json.dump(J, open(f'{OUT}/check1_summary.json', 'w'), indent=1)
