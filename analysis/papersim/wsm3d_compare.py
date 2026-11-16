"""Old (_wsm: v2-rule heights drawn per label) vs new (_wsm3d: the labels' own webSMLM 3D heights) per method and density, at the
common selection rule (common_rule.py: own filters + max(sx, sy) < 0.15 px; SIMCODE p >= 0.63) and SIMCODE's previous 12 nm cut.
usage: python analysis/papersim/wsm3d_compare.py [--prefix Dexp] [--densities 0.5,1,2,4]
output: out/papersim/common_rule/compare_<prefix>_wsm_vs_wsm3d.{json,md}"""
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common_rule as CR
import gtfrc as G

KEYS = [('smlm', 'SMLM'), ('sf', 'SIMFLUX'), ('decode', 'DECODE'), ('v2', 'SIMCODE'), ('v2_12nm', 'SIMCODE 12 nm (prev. cut)'),
        ('sf_gtp', 'SIMFLUX GT pattern (internal)'), ('v1', 'SIMCODE preprint (internal)')]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--prefix', default='Dexp'); ap.add_argument('--densities', default='0.5,1,2,4')
    a = ap.parse_args()
    out = dict(rows=[]); md = [f'# {a.prefix}<d>_v2psf_wsm (v2-rule heights) vs _wsm3d (webSMLM 3D heights), common rule', '',
                               '| density | method | n old | n new | GT-FRC old (nm) | GT-FRC new (nm) | delta | rms TP old / new (nm) | FP old / new | frames to 30 nm old / new |',
                               '|---|---|---|---|---|---|---|---|---|---|']
    f = lambda v, p=1: '-' if v is None else f'{v:.{p}f}'
    fk = lambda v: 'n.r.' if v is None else f'{v / 1e3:.1f}k'
    for d in a.densities.split(','):
        so, sn = f'{a.prefix}{d}_v2psf_wsm', f'{a.prefix}{d}_v2psf_wsm3d'
        if not all(os.path.exists(os.path.join(CR.OUTD, f'{s}.json')) for s in (so, sn)): continue
        O = json.load(open(os.path.join(CR.OUTD, f'{so}.json')))['methods']; N = json.load(open(os.path.join(CR.OUTD, f'{sn}.json')))['methods']
        for k, lk in KEYS:
            if k not in O or k not in N: continue
            o, n = O[k], N[k]
            r = dict(density=float(d), method=k, n_old=o['n'], n_new=n['n'], gt_old=o.get(G.GT_KEY), gt_new=n.get(G.GT_KEY),
                     gt_first_old=o.get('gt_first'), gt_first_new=n.get('gt_first'), rms_old=o.get('rms_tp_nm'), rms_new=n.get('rms_tp_nm'),
                     fp_old=o.get('fp'), fp_new=n.get('fp'), f30_old=o['ttf']['frames_to'].get('30'), f30_new=n['ttf']['frames_to'].get('30'),
                     hs_old=o.get(G.HS_KEY), hs_new=n.get(G.HS_KEY), rule=G.RULE)
            out['rows'].append(r)
            dl = None if r['gt_old'] is None or r['gt_new'] is None else r['gt_new'] - r['gt_old']
            md.append(f"| {d} | {lk} | {r['n_old']} | {r['n_new']} | {f(r['gt_old'])} | {f(r['gt_new'])} | {'-' if dl is None else f'{dl:+.1f}'} | "
                      f"{f(r['rms_old'])} / {f(r['rms_new'])} | {f(r['fp_old'], 3)} / {f(r['fp_new'], 3)} | {fk(r['f30_old'])} / {fk(r['f30_new'])} |")
    fn = os.path.join(CR.OUTD, f'compare_{a.prefix}_wsm_vs_wsm3d')
    json.dump(out, open(fn + '.json', 'w'), indent=1); open(fn + '.md', 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))


if __name__ == '__main__':
    main()
