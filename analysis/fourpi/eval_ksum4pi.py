"""Real-data ground-truth evaluation of a K-summed run (real dense data made by summing K source windows; real_w4pi.py
--ksum K): the sparse K = 1 reconstruction of the source windows is the reference.

Reference emitters of summed window j of block b = the reference rows of the raw windows b K S + j + k S (k = 0..K-1),
i.e. w_sum = (w // (K S)) S + (w % S). Rows of the same summed window within 0.5 px (an emitter present in two sources)
are merged (brightest kept, photons added). Reference tables: the K = 1 network table (v10 kept rows, >= 1000 e, sigma
cuts) or the classical kept rows (published-style or pixel MLE).
Matching: nearest reference emitter within 0.5 px in the same summed window (one to one, greedy by distance).
Reported per reference photon bin and per number of other reference emitters (>= 300 e, any quality) within 7 px: recall,
lateral and axial error vs the K = 1 position (robust sd and RMS within 60 / 100 nm; the reference's own error, ~3 nm
lateral and ~3.5 nm axial at 2-4k e, is included), 4Pi fringe errors (|dz - median| > 100 nm). Precision = matched /
kept summed rows with >= 1000 e: a LOWER bound (unmatched rows may be real emitters that the K = 1 selection dropped).
The z offset between the two runs' phase-field gauges is removed as one median per file.
Usage: python analysis/fourpi/eval_ksum4pi.py --cell NPC --files 10 --tag v10_k4 --K 4 [--ref net|cl|clp] [--cl_tag v10_k4cl]
"""
import argparse, json, os, sys
import numpy as np
from scipy.spatial import cKDTree
REPO = '/home/jc/simcode2'
PX = 128.0
EDGES = (1000, 2000, 4000, 8000)
NBINS = ((0, 0), (1, 1), (2, 3), (4, 99))


def load_ref(cell, fi, ref, head='v10'):
    if ref == 'net':
        L = np.load(f'{REPO}/out/4pi/net/locs_{cell}_{fi:03d}_{head}.npz')
        allr = L['good'] & (L['N'] > 300)
        ok = allr & (L['N'] >= 1000) & (L['sz'] < 0.025) & (L['sx'] < 0.12) & (L['sy'] < 0.12)
        zmed = np.median(L['z'][ok & (L['N'] >= 2000)]); ok &= np.abs(L['z'] - zmed) <= 0.3
        return dict(w=L['w'], x=L['x'], y=L['y'], z=L['z'], N=L['N'], ok=ok, any=allr)
    C = np.load(f'{REPO}/out/4pi/classical/real/cl_{cell}_{fi:03d}_wv.npz')
    if ref == 'cl':
        return dict(w=C['w'], x=C['x'], y=C['y'], z=C['z'], N=C['N'], ok=C['keep'], any=np.isfinite(C['x0']) & (C['N'] > 300))
    return dict(w=C['w'], x=np.where(np.isfinite(C['xp']), C['xp'], C['x0']), y=np.where(np.isfinite(C['yp']), C['yp'], C['y0']), z=C['zp'], N=C['N'],
                ok=C['keep_pix'] & np.isfinite(C['xp']) & np.isfinite(C['zp']), any=np.isfinite(C['x0']) & (C['N'] > 300))


def to_sum(R, K, S, nblk):
    m = R['w'] < nblk * K * S
    ws = (R['w'] // (K * S)) * S + (R['w'] % S)
    return {k: v[m] for k, v in R.items()}, ws[m]


def merge(R, ws):
    """merge reference rows of the same summed window within 0.5 px (an emitter present in two sources)"""
    keep = np.ones(len(ws), bool); N = R['N'].copy()
    order = np.argsort(ws, kind='stable'); u, st = np.unique(ws[order], return_index=True); st = np.r_[st, len(ws)]
    for i in range(len(u)):
        a = order[st[i]:st[i + 1]]
        if len(a) < 2: continue
        pr = cKDTree(np.c_[R['x'][a], R['y'][a]]).query_pairs(0.5)
        for p, q in pr:
            p, q = a[p], a[q]
            if not (keep[p] and keep[q]): continue
            hi, lo = (p, q) if N[p] >= N[q] else (q, p)
            N[hi] += N[lo]; keep[lo] = False
    out = {k: v[keep] for k, v in R.items()}; out['N'] = N[keep]
    return out, ws[keep], int((~keep).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', default='NPC'); ap.add_argument('--files', type=int, nargs='+', default=[10])
    ap.add_argument('--tag', default='v10_k4', help='summed table = locs_<cell>_<file>_<tag>.npz')
    ap.add_argument('--K', type=int, default=4); ap.add_argument('--S', type=int, default=25)
    ap.add_argument('--ref', default='net'); ap.add_argument('--out', default='')
    ap.add_argument('--regions', default='', help='json with boxes (x0, x1, y0, y1; px): reference and summed rows only inside the boxes shrunk by 2 px (box-restricted runs)')
    ap.add_argument('--table', default='', help='alternative summed table pattern with {cell} {file} (e.g. a classical run on the summed data); columns x, y, z, w, N, ok')
    a = ap.parse_args()
    acc = []
    for fi in a.files:
        if a.table:
            D = dict(np.load(a.table.format(cell=a.cell, file=fi))); kept = D['ok' if 'ok' in D else 'keep'].astype(bool) & np.isfinite(D['x']) & np.isfinite(D['z'])
        else:
            D = dict(np.load(f'{REPO}/out/4pi/net/locs_{a.cell}_{fi:03d}_{a.tag}.npz'))
            kept = D['good'] & (D['N'] >= 1000) & (D['sz'] < 0.025 * np.sqrt(a.K)) & (D['sx'] < 0.12) & (D['sy'] < 0.12)
        nblk = (int(D['w'].max()) + 1 + a.S - 1) // a.S
        Rr = load_ref(a.cell, fi, a.ref)
        if a.regions:
            bx = [b for b in json.load(open(a.regions))['boxes'] if b.get('save', True)]
            inb = lambda x, y: np.any([(x >= b['x0'] + 2) & (x < b['x1'] - 2) & (y >= b['y0'] + 2) & (y < b['y1'] - 2) for b in bx], 0)
            mr = inb(Rr['x'], Rr['y']); Rr = {k: v[mr] for k, v in Rr.items()}
            kept = kept & inb(D['x'], D['y'])
        R, ws = to_sum(Rr, a.K, a.S, nblk)
        Rall = {k: v[R['any']] for k, v in R.items()}; wall = ws[R['any']]
        G, wg, nrep = merge({k: v[R['ok']] for k, v in R.items()}, ws[R['ok']])
        # neighbours: other reference emitters (any quality, >= 300 e) of the summed window within 7 px
        nnb = np.zeros(len(wg), int); mi = np.full(len(wg), -1); md = np.full(len(wg), 9.0)
        oa = np.argsort(wall, kind='stable'); wa = wall[oa]; ok_ = np.nonzero(kept)[0]; od = ok_[np.argsort(D['w'][ok_], kind='stable')]; wd = D['w'][od]
        og = np.argsort(wg, kind='stable'); ug, sg = np.unique(wg[og], return_index=True); sg = np.r_[sg, len(wg)]
        used = np.zeros(len(D['w']), bool)
        for i, wv in enumerate(ug):
            g = og[sg[i]:sg[i + 1]]
            aa = oa[np.searchsorted(wa, wv):np.searchsorted(wa, wv, 'right')]
            if len(aa):
                cnt = cKDTree(np.c_[Rall['x'][aa], Rall['y'][aa]]).query_ball_point(np.c_[G['x'][g], G['y'][g]], 7.0, return_length=True)
                far = cKDTree(np.c_[Rall['x'][aa], Rall['y'][aa]]).query_ball_point(np.c_[G['x'][g], G['y'][g]], 0.5, return_length=True)
                nnb[g] = np.maximum(cnt - far, 0)
            dd = od[np.searchsorted(wd, wv):np.searchsorted(wd, wv, 'right')]
            if len(dd) == 0: continue
            dist, k = cKDTree(np.c_[D['x'][dd], D['y'][dd]]).query(np.c_[G['x'][g], G['y'][g]], distance_upper_bound=0.5)
            for j in np.argsort(dist):
                if dist[j] >= 0.5: break
                if used[dd[k[j]]]: continue
                used[dd[k[j]]] = True; mi[g[j]] = dd[k[j]]; md[g[j]] = dist[j]
        m = mi >= 0
        dz0 = np.median((D['z'][mi[m]] - G['z'][m])) if m.any() else 0.0
        acc.append(dict(N=G['N'], nnb=nnb, m=m, zref=(G['z'] - np.median(G['z'])) * 1000, dx=np.where(m, (D['x'][mi] - G['x']) * PX, np.nan), dy=np.where(m, (D['y'][mi] - G['y']) * PX, np.nan),
                        dz=np.where(m, (D['z'][mi] - G['z'] - dz0) * 1000, np.nan), n_kept=int(kept.sum()), n_used=int(used.sum()), nwin=int(D['w'].max()) + 1,
                        nrep=nrep, n_ref=len(wg), n_all=int(len(wall))))
    A = {k: np.concatenate([x[k] for x in acc]) for k in ('N', 'nnb', 'm', 'dx', 'dy', 'dz', 'zref')}
    nk = sum(x['n_kept'] for x in acc); nu = sum(x['n_used'] for x in acc); nw = sum(x['nwin'] for x in acc)
    res = dict(cell=a.cell, files=a.files, tag=a.table or a.tag, K=a.K, ref=a.ref, kept_rows=nk, matched=nu, precision_lower_bound=nu / max(nk, 1),
               ref_emitters=int(len(A['N'])), ref_per_summed_window=len(A['N']) / nw, ref_any_per_summed_window=sum(x['n_all'] for x in acc) / nw,
               kept_per_summed_window=nk / nw, repeats_merged=sum(x['nrep'] for x in acc), rows=[])
    print(f"## {a.cell} files {a.files}, summed table {res['tag']}, K = {a.K}, reference = K = 1 {a.ref}")
    print(f"summed windows {nw}; reference emitters {res['ref_per_summed_window']:.1f} / window (all >= 300 e candidates {res['ref_any_per_summed_window']:.1f}); kept rows {res['kept_per_summed_window']:.1f} / window; "
          f"matched {nu} of {nk} kept rows (precision lower bound {res['precision_lower_bound']:.2f}); repeats merged {res['repeats_merged']}")
    print('| reference photons | neighbours within 7 px | reference emitters | recall | robust dx / dy / dz nm | RMS dx / dy / dz nm | fringe errors |'); print('|---|---|---|---|---|---|---|')

    def stat(s):
        mm = s & A['m']; n = int(s.sum())
        if n < 20 or mm.sum() < 20: return None
        rb = [float(1.4826 * np.nanmedian(np.abs(A[k][mm] - np.nanmedian(A[k][mm])))) for k in ('dx', 'dy', 'dz')]
        okr = mm & (np.abs(A['dx']) < 60) & (np.abs(A['dy']) < 60) & (np.abs(A['dz']) < 100)
        rms = [float(np.sqrt(np.mean(A[k][okr] ** 2))) for k in ('dx', 'dy', 'dz')]
        return dict(n=n, recall=float(mm.sum() / n), robust=rb, rms=rms, fringe=float(np.mean(np.abs(A['dz'][mm]) > 100)))
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        for nlo, nhi in ((0, 99),) + NBINS:
            s = (A['N'] >= lo) & (A['N'] < hi) & (A['nnb'] >= nlo) & (A['nnb'] <= nhi); r = stat(s)
            if r is None: continue
            nm = 'all' if (nlo, nhi) == (0, 99) else (str(nlo) if nlo == nhi else (f'{nlo}-{nhi}' if nhi < 99 else f'>= {nlo}'))
            res['rows'].append(dict(lo=lo, hi=hi, nb=nm, **r))
            print(f"| {lo}-{hi} | {nm} | {r['n']} | {r['recall']:.2f} | " + ' / '.join(f'{v:.1f}' for v in r['robust']) + ' | ' + ' / '.join(f'{v:.1f}' for v in r['rms']) + f" | {r['fringe']:.3f} |")
    # direction of the fringe errors (z of the summed run minus the K = 1 z): a biased hop direction moves emitters between structures asymmetrically
    mm = A['m'] & (A['N'] >= 1000)
    fp = float(np.mean(A['dz'][mm] > 100)); fn = float(np.mean(A['dz'][mm] < -100))
    hi = mm & (np.abs(A['dz']) > 100)
    res['fringe_up'] = fp; res['fringe_down'] = fn; res['fringe_abs_median_nm'] = float(np.median(np.abs(A['dz'][hi]))) if hi.any() else None
    print(f"fringe errors by direction (matched, >= 1000 e): dz > +100 nm {fp:.3f}, dz < -100 nm {fn:.3f}; median |dz| of those {res['fringe_abs_median_nm']} nm; "
          f"central part: median dz {np.median(A['dz'][mm & (np.abs(A['dz']) < 60)]):+.2f} nm")
    # recall and fringe errors vs the reference z (relative to the file median): a z-dependent recall changes the relative height of structures
    s24 = (A['N'] >= 2000) & (A['N'] < 4000); res['by_z'] = []
    print('2-4k e by reference z (nm, relative to the file median): ' + ' | '.join(
        f"{lo:+d}..{hi:+d}: recall {np.mean(A['m'][s24 & (A['zref'] >= lo) & (A['zref'] < hi)]):.2f}, fringe {np.mean(np.abs(A['dz'][s24 & A['m'] & (A['zref'] >= lo) & (A['zref'] < hi)]) > 100):.3f}"
        for lo, hi in ((-200, -100), (-100, -50), (-50, 0), (0, 50), (50, 100), (100, 200), (200, 300))))
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
