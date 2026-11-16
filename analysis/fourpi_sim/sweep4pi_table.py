"""Tables of the 4Pi-SIMCODE density sweep (markdown) from <work>/sweep4pi/<tag>/eval.json -> out/fourpi_sim/sweep/tables.md
Usage: python analysis/fourpi_sim/sweep4pi_table.py --work <4Pi work folder>
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path: sys.path.insert(0, REPO)
OUT = os.path.expanduser('~/simcode2/out/fourpi_sim/sweep')
XY_T, XZ_T, FR_MAX, GH_MAX, CV_MAX = 25, 35, 0.02, 0.07, 0.2
TAGS = ['mtc_d01', 'mtc_d03', 'mtc_d05', 'mtc_d075', 'mtc_d15']
ROWS = ['net_all', 'net_base', 'net_chi16', 'net_chi13', 'net_sig0', 'net_sig1', 'net_sig2', 'net_chi16_sig1', 'net_bound_base', 'cl_pub', 'cl_pix', 'gt_units']
LAB = {'net_all': 'network, all candidates', 'net_base': 'network, chi2 sel. OFF (base filters)', 'net_chi16': 'network, chi2/dof < 1.6 (showcase default)',
       'net_chi13': 'network, chi2/dof < 1.3 (stricter)', 'net_sig0': 'network, σxy < 7 nm & σz < 16 nm', 'net_sig1': 'network, σxy < 5 nm & σz < 12 nm', 'net_sig2': 'network, σxy < 3.5 nm & σz < 8 nm',
       'net_chi16_sig1': 'network, chi2 < 1.6 + σxy < 5 & σz < 12 nm', 'net_bound_base': '(network, TRUE pattern + field, base filters)',
       'cl_pub': 'classical published-style', 'cl_pix': '(classical pixel MLE)', 'gt_units': '(GT units ≥ 300 e, exact: sampling limit)'}


def f(v, fmt='{:.1f}', none='–'):
    if v is None: return none
    try:
        return none if not np.isfinite(v) else fmt.format(v)
    except TypeError:
        return none


def ft(d, k):
    v = (d or {}).get(k)
    return '> T' if v is None else f'{v / 1000:.1f}k'


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--work', required=True, help='4Pi work folder')
    P4 = ap.parse_args().work
    Es = {t: json.load(open(f'{P4}/sweep4pi/{t}/eval.json')) for t in TAGS if os.path.exists(f'{P4}/sweep4pi/{t}/eval.json')}
    L = ['## Movies', '', '| movie | ON /µm²/frame (FOV) | ON /µm²/frame (eval. region) | other ON emitters in the 15 px patch (same frame) | frames | GT units ≥ 300 e | per µm tube | GT dip in-plane / axial |', '|---|---|---|---|---|---|---|---|']
    for t, E in Es.items():
        g = E['gt_sites_tube']
        L.append(f"| {t} | {E['dens_fov']:.2f} | {E['dens_region']:.2f} | {E['crowd_frame_mean']:.2f} | {E['T']} | {E['gt_units']} | {E['gt_units_per_um']:.0f} | {g['dip_lat']:.2f} / {g['dip_ax']:.2f} |")
    L += ['', '## Per density x selection', '',
          'GT-FRC = last 1/7 crossing (first crossing of gtfrc.py in brackets; it stops at the hollow-tube spectral zeros at 1/64 and 1/28 nm), '
          'x-y projection and pooled x-z sections, all 30 000 frames, 5 nm pixels, 10 nm = Nyquist. frames to: first N on the sqrt(2) grid '
          '(log-interpolated) with GT-FRC (last crossing) at or below the target; > T = not within 30 000 frames. coverage CV over 250 nm '
          'segments (Poisson floor in brackets). false positive: no emitter ON in the own window within 3 own σ (3D) / within a fixed 30/30/75 nm gate. '
          'wrong fringe: matched localizations with |Δz| > 100 nm. errors: robust sd over all matched (2–4k e in brackets); axial without fringe errors. '
          'dip < 0.81 = hollow tube resolved.', '']
    hdr = ('| ON/µm²/frame | method / selection | locs | recall | precision | GT-FRC x-y (nm) | GT-FRC x-z (nm) | frames to x-y 25 / 20 nm | '
           'frames to x-z 40 / 35 nm | coverage CV | false pos. σ / fixed | wrong fringe | σ-Jaccard | lateral (nm) | axial (nm) | dip in-plane / axial |')
    L += [hdr, '|' + '---|' * (hdr.count('|') - 1)]
    for t, E in Es.items():
        for nm in ROWS:
            if nm not in E['methods']: continue
            r = E['methods'][nm]; cv = r.get('coverage', {}); gh = r.get('ghost', {}); gf = r.get('ghost_fixed', {}); sj = r.get('sigma_jaccard', {})
            L.append(f"| {E['dens_fov']:.2f} | {LAB[nm]} | {r['n_locs']} | {r['recall']:.2f} | {r['precision']:.3f} | "
                     f"{f(r['gtfrc_last_xy'], '{:.0f}')} ({f(r['gtfrc_xy'], '{:.0f}')}) | {f(r['gtfrc_last_xz'], '{:.0f}')} ({f(r['gtfrc_xz'], '{:.0f}')}) | "
                     f"{ft(r.get('frames_to_last_xy'), '25')} / {ft(r.get('frames_to_last_xy'), '20')} | {ft(r.get('frames_to_last_xz'), '40')} / {ft(r.get('frames_to_last_xz'), '35')} | "
                     f"{f(cv.get('cv'), '{:.2f}')} ({f(cv.get('poisson_cv'), '{:.2f}')}) | {f(gh.get('ghost_fraction'), '{:.2f}')} / {f(gf.get('ghost_fraction'), '{:.3f}')} | "
                     f"{f(r['wrong_fringe'], '{:.3f}')} | {f(sj.get('jaccard'), '{:.2f}')} | {f(r['lat_rob'])} ({f(r['lat_rob_24'])}) | "
                     f"{f(r['ax_rob'])} ({f(r['ax_rob_24'])}) | {f(r['tube']['dip_lat'], '{:.2f}')} / {f(r['tube']['dip_ax'], '{:.2f}')} |")
    # ---- time to a faithful image per density x selection
    L += ['', '## Time to a faithful image', '',
          f'T_fid = frames until GT-FRC x-y <= {XY_T} nm AND x-z sections <= {XZ_T} nm (last crossing, log-interpolated on the grid), '
          f'counted only if the selection is faithful at 30 000 frames: wrong fringe <= {FR_MAX:.0%}, fixed-gate false positives <= {GH_MAX:.0%}, '
          f'coverage excess CV (sqrt(CV^2 - Poisson^2)) <= {CV_MAX}. Frames x 6 ms = acquisition time on the 4Pi-SIMFLUX setup.', '',
          '| ON/µm²/frame | selection | frames to x-y | frames to x-z | T_fid | wrong fringe | fixed false positives | coverage excess | faithful |', '|---|---|---|---|---|---|---|---|---|']
    best = {}
    for t, E in Es.items():
        for nm in ROWS:
            r = E['methods'].get(nm)
            if r is None or nm.startswith('gt') or 'frames_to_last_xy' not in r: continue
            a_, b_ = r['frames_to_last_xy'].get(str(XY_T)), r['frames_to_last_xz'].get(str(XZ_T))
            tf = None if (a_ is None or b_ is None) else max(a_, b_)
            fr = r['wrong_fringe']; gh = r.get('ghost_fixed', {}).get('ghost_fraction', np.nan); cv = r.get('coverage', {}).get('excess_cv', np.nan)
            ok = (fr <= FR_MAX) and (gh <= GH_MAX) and (cv <= CV_MAX)
            if ok and tf is not None and nm.startswith('net') and (t not in best or tf < best[t][1]): best[t] = (nm, tf)
            L.append(f"| {E['dens_fov']:.2f} | {LAB[nm]} | {ft(r['frames_to_last_xy'], str(XY_T))} | {ft(r['frames_to_last_xz'], str(XZ_T))} | "
                     f"{'> T' if tf is None else f'{tf / 1000:.1f}k ({tf * 5.575e-3:.0f} s)'} | {fr:.3f} | {gh:.3f} | {cv:.2f} | {'yes' if ok else 'no'} |")
    L += ['', 'Best faithful network selection per density: ' + '; '.join(f"{Es[t]['dens_fov']:.2f}: {LAB[b[0]]} ({b[1] / 1000:.1f}k frames)" for t, b in best.items()), '']
    json.dump({t: dict(selection=b[0], frames=b[1]) for t, b in best.items()}, open(f'{OUT}/best.json', 'w'), indent=1)
    L += ['', '## GT-FRC (last crossing) vs frames, x-y | x-z (nm)', '']
    for t, E in Es.items():
        g = E['ttf_grid']
        L += [f"### {t} ({E['dens_fov']:.2f} /µm²/frame)", '', '| method | ' + ' | '.join(str(n) for n in g) + ' |', '|' + '---|' * (len(g) + 1)]
        for nm in ROWS:
            r = E['methods'].get(nm)
            if r is None or 'ttf_last_xy' not in r: continue
            L.append(f'| {LAB[nm]} | ' + ' | '.join(f"{f(a, '{:.0f}')} \\| {f(b, '{:.0f}')}" for a, b in zip(r['ttf_last_xy'], r['ttf_last_xz'])) + ' |')
        L.append('')
    open(f'{OUT}/tables.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L[:60]))


if __name__ == '__main__':
    main()
