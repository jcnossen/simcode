"""Real-data pixel locking, old vs new DECODE table of a paper movie: |c1| x / y of the sub-pixel histogram on the fair-FRC row set
(frc_fair masks: disjoint 6-frame windows, max(sigma) < 0.15 px on DECODE's predicted sigma, crop [5, 115) px) and on all rows,
by photon tercile (each table's own photons) and by predicted-sigma bin; row counts. Optional SIMCODE / SMLM / SIMFLUX tables
(kind:path) as references.
usage: python analysis/decode_checks/v3/real_pixlock.py <ds> [name=kind:path ...] -> out/decode_checks/v3/real_pixlock_<ds>.json/.md"""
import os, sys, json
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (REPO, os.path.join(REPO, 'analysis', 'paper'), os.path.join(REPO, 'analysis', 'pixbias')):
    sys.path.insert(0, p)
import frc_fair as FF, pblib as B
from simcode.dataset import decode_table
OUT = os.path.join(REPO, 'out', 'decode_checks', 'v3')


def c1(v):
    u = B.ustats(B.sub(v)); return u.get('c1_abs', np.nan)


def analyse(T):
    disj, filt = FF.masks(T)
    R = {}
    for nm, m in (('all_in_crop', FF.in_crop(T['x'], T['y']) & (T['f'] % 6 == 0)), ('fair_frc_set', filt)):
        x, y, ph = T['x'][m], T['y'][m], T['photons'][m]; s = np.maximum(T['sx'][m], T['sy'][m])
        q = np.quantile(ph, [1 / 3, 2 / 3])
        r = dict(n=int(m.sum()), c1_x=c1(x), c1_y=c1(y), floor=1 / np.sqrt(max(m.sum(), 1)), ph_terciles=q.tolist(), median_sigma_nm=float(np.median(s) * FF.PX))
        for i, b in enumerate([ph < q[0], (ph >= q[0]) & (ph < q[1]), ph >= q[1]]):
            r[f'T{i + 1}'] = dict(n=int(b.sum()), c1_x=c1(x[b]), c1_y=c1(y[b]))
        sb = [0, 0.05, 0.08, 0.11, 0.15, 10]
        r['sigma_bins_px'] = {f'{sb[i]}-{sb[i + 1]}': dict(n=int(((s >= sb[i]) & (s < sb[i + 1])).sum()), c1_x=c1(x[(s >= sb[i]) & (s < sb[i + 1])]), c1_y=c1(y[(s >= sb[i]) & (s < sb[i + 1])]))
                              for i in range(len(sb) - 1)}
        R[nm] = r
    return R


def main():
    ds = sys.argv[1]
    d = os.path.join(REPO, 'out', 'paper2', ds, 'decode')
    tabs = {'DECODE old (dec_%s)' % ds: ('decode', decode_table(os.path.join(d, 'locs.npz'))), 'DECODE v3': ('decode', os.path.join(d, 'locs_v3.npz'))}
    for s in sys.argv[2:]:
        n, kp = s.split('=', 1); k, p = kp.split(':', 1); tabs[n] = (k, p)
    R = {}
    for n, (k, p) in tabs.items():
        if not os.path.exists(p): print('missing', p); continue
        T = FF.load(k, p); R[n] = dict(path=p, **analyse(T))
        a, f = R[n]['all_in_crop'], R[n]['fair_frc_set']
        print(n, 'all n %d |c1| %.3f %.3f | fair n %d |c1| %.3f %.3f  T1 %.3f %.3f  T3 %.3f %.3f' % (a['n'], a['c1_x'], a['c1_y'], f['n'], f['c1_x'], f['c1_y'],
              f['T1']['c1_x'], f['T1']['c1_y'], f['T3']['c1_x'], f['T3']['c1_y']), flush=True)
    json.dump(R, open(os.path.join(OUT, f'real_pixlock_{ds}.json'), 'w'), indent=1, default=float)
    L = ['| table | set | n | median sigma nm | |c1| x / y | dim tercile x / y | mid x / y | bright x / y | sigma < 0.05 px x / y | 0.11-0.15 px x / y |', '|---|---|---|---|---|---|---|---|---|---|']
    for n, r in R.items():
        for s in ('all_in_crop', 'fair_frc_set'):
            q = r[s]; sb = q['sigma_bins_px']
            L.append(f"| {n} | {s} | {q['n']} | {q['median_sigma_nm']:.1f} | {q['c1_x']:.3f} / {q['c1_y']:.3f} | {q['T1']['c1_x']:.3f} / {q['T1']['c1_y']:.3f} | "
                     f"{q['T2']['c1_x']:.3f} / {q['T2']['c1_y']:.3f} | {q['T3']['c1_x']:.3f} / {q['T3']['c1_y']:.3f} | {sb['0-0.05']['c1_x']:.3f} / {sb['0-0.05']['c1_y']:.3f} | "
                     f"{sb['0.11-0.15']['c1_x']:.3f} / {sb['0.11-0.15']['c1_y']:.3f} |")
    open(os.path.join(OUT, f'real_pixlock_{ds}.md'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
