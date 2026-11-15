"""hd head experiment: per density x selection, head v10 (sweep eval.json, $P4/sweep4pi/<tag>/) vs the retrained heads
($P4/models/head4pi_<name>/sweep/<tag>/eval.json, or $P4/models/head4pi_hdref/sweep/<tag>/ for existing reference heads).
Same metrics and faithfulness rule as sweep4pi_table.py (T_fid: x-y <= 25 nm AND x-z <= 35 nm, last crossing; faithful = wrong
fringe <= 2 %, fixed-gate ghosts <= 7 %, coverage excess CV <= 0.2). -> out/fourpi_sim/hd_head/tables.md, best.json
Usage: python analysis/fourpi_sim/hd_table.py hd1 hd0 [v11b]
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path: sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from simcode.fourpi.setup import P4
from sweep4pi_table import XY_T, XZ_T, FR_MAX, GH_MAX, CV_MAX, TAGS, f, ft
OUT = os.path.expanduser('~/simcode2/out/fourpi_sim/hd_head')
SELS = ['net_all', 'net_base', 'net_chi16', 'net_sig0', 'net_sig1', 'net_sig2', 'net_bound_base']
LAB = {'net_all': 'all', 'net_base': 'chi2 off', 'net_chi16': 'chi2 < 1.6', 'net_sig0': 'σ 7/16', 'net_sig1': 'σ 5/12', 'net_sig2': 'σ 3.5/8',
       'net_bound_base': 'TRUE pattern + field, chi2 off'}


REF = {'v11b': 'hdref'}   # existing heads evaluated into $P4/models/head4pi_hdref/sweep


def load(head, t):
    if head != 'v10':
        p = f'{P4}/models/head4pi_{REF.get(head, head)}/sweep/{t}/eval.json'
        return json.load(open(p)) if os.path.exists(p) else None
    p = f'{P4}/sweep4pi/{t}/eval.json'
    return json.load(open(p)) if os.path.exists(p) else None


def row(r):
    a_, b_ = r['frames_to_last_xy'].get(str(XY_T)), r['frames_to_last_xz'].get(str(XZ_T))
    tf = None if (a_ is None or b_ is None) else max(a_, b_)
    fr = r['wrong_fringe']; gh = r.get('ghost_fixed', {}).get('ghost_fraction', np.nan); cv = r.get('coverage', {}).get('excess_cv', np.nan)
    ok = bool((fr <= FR_MAX) and (gh <= GH_MAX) and (cv <= CV_MAX))
    return tf, fr, gh, cv, ok


def main():
    heads = ['v10'] + sys.argv[1:]
    os.makedirs(OUT, exist_ok=True)
    hdr = ('| ON/µm²/frame | head | selection | recall | precision | GT-FRC x-y / x-z (nm) | frames to x-y 25 / x-z 35 | T_fid | wrong fringe (2-4k e) | '
           'fixed ghosts | coverage CV (Poisson) / excess | lateral / axial robust (nm) | dip in-plane / axial | faithful |')
    L = ['## Per density x selection x head (30 000 frames)', '', hdr, '|' + '---|' * (hdr.count('|') - 1)]
    best = {}; summ = {}
    for t in TAGS:
        for h in heads:
            E = load(h, t)
            if E is None: continue
            for nm in SELS:
                r = E['methods'].get(nm)
                if r is None or 'frames_to_last_xy' not in r: continue
                tf, fr, gh, cv, ok = row(r); c = r.get('coverage', {})
                if ok and tf is not None and nm != 'net_bound_base' and ((t, h) not in best or tf < best[(t, h)][1]): best[(t, h)] = (nm, tf)
                summ.setdefault(t, {}).setdefault(h, {})[nm] = dict(recall=r['recall'], precision=r['precision'], xy=r['gtfrc_last_xy'], xz=r['gtfrc_last_xz'],
                                                                     T_fid=tf, fringe=fr, ghost_fixed=gh, cov_excess=cv, faithful=ok)
                L.append(f"| {E['dens_fov']:.2f} | {h} | {LAB[nm]} | {r['recall']:.2f} | {r['precision']:.3f} | {f(r['gtfrc_last_xy'], '{:.0f}')} / {f(r['gtfrc_last_xz'], '{:.0f}')} | "
                         f"{ft(r['frames_to_last_xy'], str(XY_T))} / {ft(r['frames_to_last_xz'], str(XZ_T))} | {'> T' if tf is None else f'{tf / 1000:.1f}k ({tf * 0.006:.0f} s)'} | "
                         f"{fr * 100:.1f} % ({f(r['wrong_fringe_24'] * 100, '{:.1f}')} %) | {gh * 100:.1f} % | {f(c.get('cv'), '{:.2f}')} ({f(c.get('poisson_cv'), '{:.2f}')}) / {cv:.2f} | "
                         f"{f(r['lat_rob'])} / {f(r['ax_rob'])} | {f(r['tube']['dip_lat'], '{:.2f}')} / {f(r['tube']['dip_ax'], '{:.2f}')} | {'yes' if ok else 'no'} |")
    L += ['', '## Best faithful selection per density and head (T_fid, frames)', '', '| movie | ' + ' | '.join(heads) + ' |', '|' + '---|' * (len(heads) + 1)]
    for t in TAGS:
        cells = []
        for h in heads:
            b = best.get((t, h)); cells.append('–' if load(h, t) is None else ('not faithful' if b is None else f'{LAB[b[0]]}: {b[1] / 1000:.1f}k ({b[1] * 0.006:.0f} s)'))
        L.append(f'| {t} | ' + ' | '.join(cells) + ' |')
    open(f'{OUT}/tables.md', 'w').write('\n'.join(L) + '\n')
    json.dump(dict(best={f'{t}|{h}': b for (t, h), b in best.items()}, rows=summ), open(f'{OUT}/best.json', 'w'), indent=1, default=float)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
