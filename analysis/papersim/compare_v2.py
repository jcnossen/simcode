"""SIMCODE v2 of two runs on the same papersim movies (e.g. the 3D-head networks of 2026-11-15 vs the 2D networks of 13 Nov):
the v2 row (v2sig, sigma <= 12 nm) of <movie>/full/metrics_<old>.json vs metrics_<new>.json per movie: FRC (fair protocol),
Jaccard (10 nm, end of the movie), localizations (all / FRC set), GT RMS x, y of the FRC set, recall (>= 150 photons).
usage: python analysis/papersim/compare_v2.py --movies D0.5,D1,D2,D4 --old m1809 --new m1809_3d [--name D_m1809_3d]
output: out/papersim/compare_v2_<name>.md / .json (name default: --new)"""
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C


def row(md, model):
    f = os.path.join(C.OUT, md, 'full', f'metrics_{model}.json')
    if not os.path.exists(f): return None
    r = json.load(open(f))['methods'].get('v2sig')
    if r is None: return None
    g = r['gt_match']
    return dict(frc_nm=r.get('frc_nm'), jaccard=r['jaccard_end'], n_all=r['n_all'], n_frc=r['n_frc'], rms_nm=g.get('rms_nm'),
                recall=g.get('recall_gt_ph150'), err_gt30=g.get('frac_err_gt30'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--movies', required=True); ap.add_argument('--old', required=True); ap.add_argument('--new', required=True)
    ap.add_argument('--name', default=None)
    a = ap.parse_args()
    name = a.name or a.new
    out = []; L = [f'# SIMCODE v2 (sigma <= 12 nm): {a.new} vs {a.old} (13 Nov), 60 000 frames', '',
                   'FRC fair protocol (nm); Jaccard 10 nm at the end of the movie; n = localizations all / FRC set; GT RMS x, y of the FRC set (nm); recall of GT emitters >= 150 photons; err > 30 nm = share of the FRC set', '',
                   '| movie | run | FRC nm | Jaccard | n all | n FRC set | GT RMS x,y nm | recall | err > 30 nm |', '|---|---|---|---|---|---|---|---|---|']
    fmt = lambda v, s='.1f': 'n/a' if v is None else format(v, s)
    for md in a.movies.split(','):
        O, N = row(md, a.old), row(md, a.new)
        out.append(dict(movie=md, old=O, new=N))
        for lab, r in ((a.old, O), (a.new, N)):
            if r is None: L.append(f'| {md} | {lab} | (missing) | | | | | | |'); continue
            L.append(f"| {md} | {lab} | {fmt(r['frc_nm'])} | {fmt(r['jaccard'], '.3f')} | {r['n_all']} | {r['n_frc']} | "
                     f"{','.join(f'{v:.1f}' for v in r['rms_nm'] or [])} | {fmt(r['recall'], '.3f')} | {fmt(r['err_gt30'], '.3f')} |")
        if O and N:
            d = lambda k: (N[k] - O[k]) if N[k] is not None and O[k] is not None else None
            L.append(f"| {md} | change | {fmt(d('frc_nm'), '+.1f')} | {fmt(d('jaccard'), '+.3f')} | {N['n_all'] / O['n_all'] - 1:+.1%} | {N['n_frc'] / O['n_frc'] - 1:+.1%} | "
                     f"{','.join(f'{n - o:+.1f}' for n, o in zip(N['rms_nm'] or [], O['rms_nm'] or []))} | {fmt(d('recall'), '+.3f')} | {fmt(d('err_gt30'), '+.3f')} |")
    fn = os.path.join(C.OUT, f'compare_v2_{name}')
    json.dump(out, open(fn + '.json', 'w'), indent=1); open(fn + '.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L)); print('written', fn + '.md')


if __name__ == '__main__':
    main()
