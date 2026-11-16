"""Markdown tables of the 4Pi-SIMFLUX high-density showcase (from <work>/showcase/<tag>/eval.json) -> stdout and
<work>/showcase/tables.md. Usage: python analysis/fourpi/showcase4pi_table.py --work <4Pi work folder>"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
import json, os
import numpy as np
TAGS = ['sc_d01', 'sc_d03', 'sc_d1', 'sc_d2', 'sc_d4', 'sc_d03_r125']
NAMES = {'net': 'network, in-situ pattern + field', 'net_bound': 'network, true pattern + field (bound)',
         'net_nochi2': 'network, no chi2 selection', 'net_all': 'network, all candidates',
         'cl_pub': 'classical published-style, filtered', 'cl_pix': 'classical pixel MLE, filtered',
         'cl_pub_nofilt': 'classical published-style, unfiltered', 'astig': 'astigmatism-only z, filtered', 'gt': 'ground truth'}
ORDER = ['net', 'net_bound', 'net_nochi2', 'net_all', 'cl_pub', 'cl_pix', 'cl_pub_nofilt', 'astig']


def f(v, d=1):
    return '–' if v is None or (isinstance(v, float) and not np.isfinite(v)) else f'{v:.{d}f}'


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--work', required=True, help='4Pi work folder')
    OUT = f'{ap.parse_args().work}/showcase'
    E = {t: json.load(open(f'{OUT}/{t}/eval.json')) for t in TAGS if os.path.exists(f'{OUT}/{t}/eval.json')}
    L = []
    L.append('### Crowding and sampling\n')
    L.append('| movie | ON emitters /µm²/frame | frames | ON emitters per frame (FOV) | other emitters in a 15 px patch, same frame | same 6-frame window | GT units ≥ 300 e | GT units per µm tube |')
    L.append('|---|---|---|---|---|---|---|---|')
    for t, e in E.items():
        L.append(f"| {t} | {e['dens']:.2f} | {e['T']} | {e['on_per_frame']:.0f} | {e['crowd_frame_mean']:.2f} | {e['crowd_window_mean']:.2f} | {e['gt_units']} | {e['gt_units_per_um']:.0f} |")
    L.append('\n### Summary (2–4k e per window bin for errors; recall over all GT units ≥ 300 e)\n')
    L.append('| movie | method | recall ≥300 e | recall 2–4k | precision | lateral robust / RMS (nm) | axial robust / RMS (nm) | CRLB lat / ax (nm) | fringe err 2–4k | correct locs per µm |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    for t, e in E.items():
        for m in ORDER:
            r = e['methods'][m]; q = [x for x in r['rows'] if x['lo'] == 2000][0]
            lat = (0.5 * (q['x_rob'] + q['y_rob']), 0.5 * (q['x_rms'] + q['y_rms'])) if 'x_rob' in q else (np.nan, np.nan)
            ax = (q['z_rob'], q['z_rms']) if 'z_rob' in q else (np.nan, np.nan)
            cr = (0.5 * (q['crlb'][0] + q['crlb'][1]), q['crlb'][2]) if 'crlb' in q else (np.nan, np.nan)
            L.append(f"| {t} ({e['dens']:.2f}) | {NAMES[m]} | {f(r['recall'], 2)} | {f(q['recall'], 2)} | {f(r['precision'], 3)} | "
                     f"{f(lat[0])} / {f(lat[1])} | {f(ax[0])} / {f(ax[1])} | {f(cr[0])} / {f(cr[1])} | {f(q.get('fringe', np.nan), 3)} | {f(r['correct_locs_per_um'], 0)} |")
    L.append('\n### Per photon bin (robust x / y / z nm, recall), main rows\n')
    L.append('| movie | method | 300–1k | 1–2k | 2–4k | 4–8k | ≥ 8k |')
    L.append('|---|---|---|---|---|---|---|')
    for t, e in E.items():
        for m in ('net', 'net_bound', 'cl_pub', 'cl_pix', 'astig'):
            cells = []
            for q in e['methods'][m]['rows']:
                if q['lo'] == 1000 and q['hi'] == 30000: continue
                cells.append(f"{f(q.get('x_rob'))}/{f(q.get('y_rob'))}/{f(q.get('z_rob'))} ({f(q['recall'], 2)})" if 'x_rob' in q else f"– ({f(q['recall'], 2)})")
            L.append(f"| {t} | {NAMES[m]} | " + ' | '.join(cells) + ' |')
    L.append('\n### Tube cross-section (localizations within 120 nm of a GT tube axis, > 300 nm from crossings) and FRC\n')
    L.append('dip = profile value at the axis / mean of the two peaks (4 nm bins, label radius from the GT); < 0.81 (Rayleigh-like) = hollow tube resolved in that direction. ring radius = median distance from the axis. FRC 1/7 threshold, halves = alternating 20-window blocks (sampling-limited here: see text).\n')
    L.append('| movie | method | n in profile | dip in-plane | dip axial | ring radius (nm) | radial spread rsd (nm) | FRC x-y (nm) | FRC x-z (nm) |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for t, e in E.items():
        for m in ['gt'] + ORDER:
            r = e['methods'][m]; tb = r['tube']; fr = r.get('frc_nm', {})
            L.append(f"| {t} | {NAMES[m]} | {tb['n']} | {f(tb['dip_lat'], 2)} | {f(tb['dip_ax'], 2)} | {f(tb['radius_med'])} | {f(tb['radius_rsd'])} | {f(fr.get('xy', np.nan), 0)} | {f(fr.get('xz', np.nan), 0)} |")
    s = '\n'.join(L) + '\n'
    open(f'{OUT}/tables.md', 'w').write(s); print(s)


if __name__ == '__main__':
    main()
