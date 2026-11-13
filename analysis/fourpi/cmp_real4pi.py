"""Real-data comparison of the network pipeline (real_w4pi.py tables) with the classical 4Pi-SIMFLUX tables of the same
file (out/4pi/classical/real/cl_<cell>_<file>_wv.npz; read only): adjacent-window precision (DAFL) on the SAME emitters.

Same emitter = network 'good' row and classical kept row in the same window within 64 nm. DAFL: the same emitter in two
consecutive windows (w, w + 1; within 0.35 px, unique), robust sd of the difference / sqrt 2, per photon bin (classical N).
Also: kept fractions, median offset network - classical, 4Pi fringe agreement, reported sigma vs DAFL.
DAFL cancels what is common to adjacent windows (phase field, pattern gauge): it measures precision, not accuracy.
Usage: python analysis/fourpi/cmp_real4pi.py --cell NPC --files 10 --head v10 [--tag _t3600] [--wmax 600]
"""
import argparse, json, os, sys
import numpy as np
from scipy.spatial import cKDTree
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.fourpi.setup import P4
from simcode.fourpi.setup import PX_NM as PX
EDGES = (1000, 2000, 4000, 8000, 20000)


def pairs(w, x, y, r=0.35):
    """indices (i, j) of the same emitter in windows w and w + 1"""
    I, J = [], []
    order = np.argsort(w, kind='stable'); ws = w[order]
    uw, st = np.unique(ws, return_index=True); st = np.r_[st, len(ws)]; pos = {k: i for i, k in enumerate(uw)}
    for i, k in enumerate(uw):
        j = pos.get(k + 1)
        if j is None: continue
        a = order[st[i]:st[i + 1]]; b = order[st[j]:st[j + 1]]
        dd, kk = cKDTree(np.c_[x[b], y[b]]).query(np.c_[x[a], y[a]], k=min(2, len(b)), distance_upper_bound=2 * r)
        if dd.ndim == 1: dd = dd[:, None]; kk = kk[:, None]
        ok = (dd[:, 0] < r) & (~(dd[:, 1] < 2 * r) if dd.shape[1] > 1 else True)
        I.append(a[ok]); J.append(b[kk[ok, 0]])
    return (np.concatenate(I), np.concatenate(J)) if I else (np.zeros(0, int), np.zeros(0, int))


def dafl(T, N, sel, cols):
    """T: dict x, y (px), z (um), w; rows sel; -> per bin [x, y, z nm, n pairs, sigma ratio x, y, z]"""
    idx = np.nonzero(sel)[0]
    I, J = pairs(T['w'][idx], T['x'][idx], T['y'][idx]); I, J = idx[I], idx[J]
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        Np = 0.5 * (N[I] + N[J]); s = (Np >= lo) & (Np < hi); row = []
        for v, sc, sg in (('x', PX, cols[0]), ('y', PX, cols[1]), ('z', 1000.0, cols[2])):
            if s.sum() < 30: row.append((np.nan, np.nan)); continue
            d = (T[v][I[s]] - T[v][J[s]]) * sc
            emp = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
            rep = np.sqrt(np.median(0.5 * (T[sg][I[s]] ** 2 + T[sg][J[s]] ** 2))) * sc
            row.append((float(emp), float(emp / rep)))
        out.append(dict(lo=lo, hi=hi, n=int(s.sum()), nm=[r[0] for r in row], ratio=[r[1] for r in row]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cell', default='NPC'); p.add_argument('--files', type=int, nargs='+', default=[10]); p.add_argument('--head', default='v10')
    p.add_argument('--tag', default=''); p.add_argument('--wmax', type=int, default=0); p.add_argument('--sz_max', type=float, default=0.025)
    p.add_argument('--sxy_max', type=float, default=0.12); p.add_argument('--out', default='')
    a = p.parse_args()
    res = {}
    for fi in a.files:
        L = dict(np.load(f'{P4}/net/locs_{a.cell}_{fi:03d}_{a.head}{a.tag}.npz'))
        C = dict(np.load(f'{P4}/classical/real/cl_{a.cell}_{fi:03d}_wv.npz'))
        wmax = a.wmax or int(L['w'].max()) + 1
        cs = C['w'] < wmax
        C = {k: (v[cs] if getattr(v, 'shape', ())[:1] == cs.shape else v) for k, v in C.items()}
        good = L['good'] & (L['sz'] < a.sz_max) & (L['sx'] < a.sxy_max) & (L['sy'] < a.sxy_max) & (L['N'] > 300)
        print(f'## {a.cell} {fi} ({a.head}{a.tag}), windows < {wmax}')
        print(f'network: {len(L["w"])} candidates ({len(L["w"]) / wmax:.0f} / window), good {good.sum()} ({good.mean():.3f}); >= 1000 e good {int((good & (L["N"] >= 1000)).sum())}')
        print(f'classical: {len(C["w"])} candidates ({len(C["w"]) / wmax:.0f} / window), kept {int(C["keep"].sum())} ({C["keep"].mean():.3f}), pixel MLE kept {int(C["keep_pix"].sum())}')
        # same emitters
        mi = np.full(len(C['w']), -1)
        oL = np.argsort(L['w'], kind='stable'); wl = L['w'][oL]
        for wv in np.unique(C['w']):
            c = np.nonzero((C['w'] == wv) & C['keep'])[0]
            l = oL[np.searchsorted(wl, wv):np.searchsorted(wl, wv, 'right')]; l = l[good[l]]
            if len(c) == 0 or len(l) == 0: continue
            d, k = cKDTree(np.c_[L['x'][l], L['y'][l]]).query(np.c_[C['x'][c], C['y'][c]], distance_upper_bound=0.5)
            ok = d < 0.5; mi[c[ok]] = l[k[ok]]
        m = mi >= 0
        dx = (L['x'][mi[m]] - C['x'][m]) * PX; dy = (L['y'][mi[m]] - C['y'][m]) * PX; dz = (L['z'][mi[m]] - C['z'][m]) * 1000
        dzp = (L['z'][mi[m]] - C['zp'][m]) * 1000
        print(f'same emitters: {int(m.sum())} ({m.sum() / max(C["keep"].sum(), 1):.2f} of classical kept, {m.sum() / max(good.sum(), 1):.2f} of network good)')
        print(f'network - classical (published): median dx {np.median(dx):+.1f}, dy {np.median(dy):+.1f}, dz {np.median(dz):+.1f} nm; robust sd dx {1.4826 * np.median(np.abs(dx - np.median(dx))):.1f}, dy {1.4826 * np.median(np.abs(dy - np.median(dy))):.1f}, dz {1.4826 * np.median(np.abs(dz - np.median(dz))):.1f} nm; '
              f'fringe disagreement (|dz - median| > 100 nm) {np.mean(np.abs(dz - np.median(dz)) > 100):.3f}; vs pixel MLE dz robust sd {1.4826 * np.median(np.abs(dzp - np.median(dzp))):.1f} nm')
        # DAFL on the same emitters, binned by the classical N
        selC = m.copy(); Ncl = C['N']
        selL = np.zeros(len(L['w']), bool); selL[mi[m]] = True
        NL = np.zeros(len(L['w'])); NL[mi[m]] = C['N'][m]
        rows = {'network ' + a.head: dafl(L, NL, selL, ('sx', 'sy', 'sz')),
                'classical published-style': dafl(C, Ncl, selC, ('sx', 'sy', 'sz')),
                'classical pixel MLE': dafl(dict(x=C['xp'], y=C['yp'], z=C['zp'], w=C['w'], sxp=C['sxp'], syp=C['syp'], szp=C['szp']), Ncl, selC & C['keep_pix'], ('sxp', 'syp', 'szp')),
                'network ' + a.head + ', all good rows (own N)': dafl(L, L['N'], good, ('sx', 'sy', 'sz'))}
        print('| rows | ' + ' | '.join(f'{lo}-{hi} e: DAFL x / y / z nm (pairs) [DAFL / reported sigma]' for lo, hi in zip(EDGES[:-1], EDGES[1:])) + ' |')
        print('|---|' + '---|' * (len(EDGES) - 1))
        for nm, r in rows.items():
            print(f'| {nm} | ' + ' | '.join(' / '.join(f'{v:.2f}' for v in b['nm']) + f' ({b["n"]}) [' + ' / '.join(f'{v:.2f}' for v in b['ratio']) + ']' for b in r) + ' |')
        res[fi] = dict(n_same=int(m.sum()), rows=rows, offset=[float(np.median(dx)), float(np.median(dy)), float(np.median(dz))])
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
