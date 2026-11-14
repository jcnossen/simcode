"""Merged events, GT-free estimator from co-ON statistics (2026-11-13 evening). CPU only.
Two sites of one structure ON in the same window give ONE row (photon-weighted mean) when they are ~20 nm apart, but TWO rows when they are
far apart. So the probability that a site has a partner site ON in the same 6-frame cycle can be measured on FAR pairs of the same structure
(1.8-3.5 d0 apart, always resolved) and transferred to the ADJACENT pairs (0.65-1.35 d0), where the co-ON windows become merged rows:
  rows are assigned to the nearest site of the list within 0.5 d0 (one global offset per table, nra_lib.global_shift); S_i = cycles with a
  row at site i; p_co(class) = sum |S_i & S_j| / sum |S_i| over ordered site pairs of a structure in that distance class;
  predicted merged fraction = DUTY-CYCLE prediction: per row at site i, the summed duty cycle (cycles with a row / all cycles) of the other
  populated sites of its structure, averaged over rows (independent sites). p_co(far) turned out ~0 for SIMCODE: sites 36-70 nm apart that are
  ON together also give one row (the detector merges up to ~1 px), so the far-pair transfer idea does not work; p_co is kept as a diagnostic;
  deficit = p_co(adjacent) / p_co(far) (1 = no merging; 0 = every co-ON adjacent window merged into one row).
Calibrated on simulations with truth (true mixed fraction of the table rows, f2 >= 0.05, r24_simeval json), then applied to the real tables.
Assumptions: sites of a structure bind independently with the same kinetics at all distances; the selection keeps a merged row and a single
row alike (the simulations measure the net factor).
usage: python analysis/nanoruler/nr_coon.py [out prefix, default out/nanoruler/nr20/merged/coon]"""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nra_lib as L
from simcode.util import proctitle
proctitle.set_title('nr-coon')
NR = '/home/jc/simcode2/out/nanoruler'; O = f'{NR}/nr20'; D0 = 20.0; MINL = 20
out = sys.argv[1] if len(sys.argv) > 1 else f'{O}/merged/coon'
rng = np.random.default_rng(1)


def coon(tab, pairs, ncyc, nboot=300):
    D = json.load(open(pairs)); sc = np.array(D['sites']); su = np.array(D['site_unit']); T = L.load_table(tab)
    sh = L.global_shift(T['x_nm'], T['y_nm'], sc, rmax=0.5 * D0); x = T['x_nm'] - sh[0]; y = T['y_nm'] - sh[1]; cyc = T['frame'].astype(np.int64)
    d, j = cKDTree(sc).query(np.c_[x, y], distance_upper_bound=0.5 * D0); ok = np.isfinite(d)
    S = [set() for _ in sc]
    for c, s in zip(cyc[ok], j[ok]): S[s].add(int(c))
    n = np.array([len(s) for s in S]); good = n >= MINL
    units = np.unique(su[good]); acc = {}      # unit -> [co_adj, base_adj, co_far, base_far, rows, rows x n_adj, rows x duty of the other sites]
    for u in units:
        idx = np.nonzero((su == u) & good)[0]; a = np.zeros(7)
        for i in idx:
            nadj = 0
            for k in idx:
                if k == i: continue
                r = np.hypot(*(sc[i] - sc[k])) / D0
                if 0.65 <= r <= 1.35: a[0] += len(S[i] & S[k]); a[1] += n[i]; nadj += 1
                elif 1.8 <= r <= 3.5: a[2] += len(S[i] & S[k]); a[3] += n[i]
            a[4] += n[i]; a[5] += n[i] * nadj; a[6] += n[i] * (n[idx].sum() - n[i]) / ncyc
        acc[u] = a
    A = np.array(list(acc.values()))
    def stat(w):
        s = (A * w[:, None]).sum(0); pa = s[0] / max(s[1], 1); pf = s[2] / max(s[3], 1); na = s[5] / max(s[4], 1)
        return np.array([pf, pa, pa / pf if pf > 0 else np.nan, s[6] / max(s[4], 1), na])
    e = stat(np.ones(len(A))); bs = np.array([stat(np.bincount(rng.integers(0, len(A), len(A)), minlength=len(A)).astype(float)) for _ in range(nboot)])
    ci = np.nanpercentile(bs, [2.5, 97.5], 0)
    return dict(n_units=int(len(A)), n_rows_assigned=int(ok.sum()), p_far=e[0], p_adj=e[1], deficit=e[2], pred=e[3], n_adj=e[4],
                ci={k: ci[:, i].tolist() for i, k in enumerate(('p_far', 'p_adj', 'deficit', 'pred', 'n_adj'))})


def mixed(fn, sel): return float(json.load(open(fn))['sel'][sel]['est']['mixed'])


S = f'{O}/simeval'; cal = []
for s in ('s20g_a', 's20g_c0'):
    for m in ('m12', 'sel'):
        for pl in ('truth', 'auto'):
            cal.append((f'{s} SIMCODE {m} ({pl} sites)', f'{S}/tabs/{s}_p1_{m}.npz', f'{S}/pairs_{pl}_s20g.json', mixed(f'{S}/gt_{s}_p1m12.json', m)))
for m in ('sel',):
    cal.append((f's20_a (linear) SIMCODE {m} (truth sites)', f'{S}/tabs/s20_a_p1_{m}.npz', f'{S}/pairs_truth_s20.json', mixed(f'{S}/gt_s20_a_p1.json', m)))
rows = []
for lab, tab, pl, mix in cal:
    if not os.path.exists(tab): print('missing', tab); continue
    r = coon(tab, pl, 8000); r.update(label=lab, mixed=mix, ratio=mix / r['pred']); rows.append(r); print(lab, f"pred {r['pred']:.4f} true {mix:.4f} deficit {r['deficit']:.3f}", flush=True)
# calibration: the selections (m12 / sel) with the automatic site list (as on the real data)
G = [r for r in rows if r['label'].startswith('s20g') and 'auto sites' in r['label']]; k = float(np.mean([r['ratio'] for r in G])); ksd = float(np.std([r['ratio'] for r in G], ddof=1))
REF = f'{O}/ref'; T1 = f'{O}/methods_p1/tabs'; real = []
for pl in ('auto_sf', 'auto_sc', 'picks_g2d'):
    for m, tab in (('published SMLM', f'{REF}/tab_pub_smlm.npz'), ('published SIMFLUX', f'{REF}/tab_pub_simflux.npz'), ('re-implemented SIMFLUX', f'{T1}/bl_simflux.npz'),
                   ('SIMCODE v2 pass 1, m12', f'{T1}/simcode_m12.npz')):
        r = coon(tab, f'{O}/pairs_{pl}.json', 16666); r.update(pairs=pl, method=m, est=k * r['pred'], est_ci=[k * v for v in r['ci']['pred']]); real.append(r)
        print(pl, m, f"pred {r['pred']:.4f} est {r['est']:.4f} deficit {r['deficit']:.3f}", flush=True)
Lz = ['# Merged events on the real 20 nm movie: co-ON estimator (GT-free), calibrated on the grid simulations', '',
      f'rows assigned to the nearest listed site within {0.5 * D0:g} nm; sites with >= {MINL} rows; adjacent = 0.65-1.35 d0, far = 1.8-3.5 d0 apart in one structure; '
      f'predicted merged fraction = summed duty cycle of the other populated sites of the structure per row (independent sites). Calibration factor true / predicted on the grid simulations: '
      f'{k:.2f} +- {ksd:.2f} (sd over the {len(G)} m12 / sel tables with the automatic site list; truth sites give 0.93-1.04). The picks_g2d rows are shown uncalibrated-in-kind.', '',
      '| simulation table | structures | p_co far | p_co adjacent | deficit adj / far | adjacent sites per row | predicted [95 % CI] | true mixed fraction | true / predicted |', '|---|---|---|---|---|---|---|---|---|']
for r in rows:
    Lz.append(f"| {r['label']} | {r['n_units']} | {r['p_far']:.4f} | {r['p_adj']:.4f} | {r['deficit']:.3f} [{r['ci']['deficit'][0]:.3f}, {r['ci']['deficit'][1]:.3f}] | {r['n_adj']:.2f} | "
              f"{r['pred']:.4f} [{r['ci']['pred'][0]:.4f}, {r['ci']['pred'][1]:.4f}] | {r['mixed']:.4f} | {r['ratio']:.2f} |")
Lz += ['', '## Real nr20 movie', '', '| site list | method | structures | rows assigned | p_co far | p_co adjacent | deficit adj / far [95 % CI] | adjacent sites per row | '
       'predicted | estimated merged fraction (x calibration) [95 % CI] |', '|---|---|---|---|---|---|---|---|---|---|']
for r in real:
    Lz.append(f"| {r['pairs']} | {r['method']} | {r['n_units']} | {r['n_rows_assigned']} | {r['p_far']:.4f} | {r['p_adj']:.4f} | {r['deficit']:.3f} [{r['ci']['deficit'][0]:.3f}, {r['ci']['deficit'][1]:.3f}] | "
              f"{r['n_adj']:.2f} | {r['pred']:.4f} | {r['est']:.4f} [{r['est_ci'][0]:.4f}, {r['est_ci'][1]:.4f}] |")
open(out + '.md', 'w').write('\n'.join(Lz) + '\n'); json.dump(dict(k=k, ksd=ksd, cal=rows, real=real), open(out + '.json', 'w'), indent=1, default=float)
print('\n'.join(Lz))
