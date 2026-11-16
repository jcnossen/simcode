"""Referee check (i): why the SMLM render of the 2.5 nM zoom 2 (Fig 4 F, box (85, 40) + 15 px) is a broad smeared band.

Rows: the figure's fair-FRC sets (cache_real.py). For the SMLM rows inside the box vs the whole crop:
  * crowding: number of SIMCODE localizations (its figure set, a lower bound on the emitters ON) in the SAME 6-frame window
    within 4 px (the half width of the 8 px SMLM ROI) and within 2 px of the SMLM localization; the SMLM 6-frame sum fits ONE
    emitter per ROI;
  * distance of each SMLM localization to the nearest SIMCODE localization of its window, and the fraction that sits between
    two SIMCODE localizations (both within 4 px and the SMLM point within 0.35 x their separation of their midpoint);
  * the SMLM fit diagnostics of those rows (smlm.npy: photons, background, chi^2 per pixel, CRLB);
  * the emitter heights there (SIMCODE pass-1 z of its figure set) vs the SMLM / SIMFLUX fit plane (median of the field).
Also for the refit at fit plane 0.318 um (out/paper2/2.5nM_2/baselines_psfv2_z318) when --z318 is given.
usage: python analysis/refcheck/i_smlmzoom.py [--z318]
output: out/refcheck/i_smlmzoom.json"""
import os, sys, json, argparse
import numpy as np
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'analysis', 'paper')]
CACHE = os.path.join(REPO, 'out', 'refcheck', 'cache')
DS = '2.5nM_2'; F = 180000
BOXES = {'zoom2_F': (85, 40, 15), 'zoom1_B': (50, 15, 30)}
BIG = 1e4


def inbox(x, y, b):
    x0, y0, w = b; return (x >= x0) & (x < x0 + w) & (y >= y0) & (y < y0 + w)


def smlm_table(path):
    """smlm.npy columns in frc_fair.load row order + the fair masks -> dict of the disjoint, crop rows"""
    from smlmtorch.simflux.dataset import SFDataset
    import frc_fair as FF
    d = SFDataset.load(path)
    x, y = d.pos[:, 0].astype(np.float64), d.pos[:, 1].astype(np.float64); f = np.asarray(d.frame).astype(np.int64)
    sx, sy = d.crlb.pos[:, 0], d.crlb.pos[:, 1]
    keep = f + 6 <= F
    T = dict(x=x, y=y, f=f, sx=sx, sy=sy, ph=d.photons, bg=d.background, chisq=d.chisq / 64.0)
    T = {k: np.asarray(v)[keep] for k, v in T.items()}
    disj, filt = FF.masks(T)
    i = np.nonzero(disj)[0]
    return {k: v[i] for k, v in T.items()}, filt[i]


def crowd(S, V):
    """per SMLM row: SIMCODE rows of the same window within 4 / 2 px, distance to the nearest (nm), 'between two' flag"""
    tree = cKDTree(np.c_[V['x'], V['y'], V['f'] * BIG])
    q = np.c_[S['x'], S['y'], S['f'] * BIG]
    n4 = np.array([len(v) for v in tree.query_ball_point(q, 4.0)]); n2 = np.array([len(v) for v in tree.query_ball_point(q, 2.0)])
    d, _ = tree.query(q, k=2, distance_upper_bound=4.0)
    between = np.zeros(len(q), bool)
    two = np.isfinite(d[:, 1])
    if two.any():
        _, j = tree.query(q[two], k=2)
        a = tree.data[j[:, 0], :2]; b = tree.data[j[:, 1], :2]; mid = 0.5 * (a + b); sep = np.linalg.norm(a - b, axis=1)
        between[np.nonzero(two)[0]] = (np.linalg.norm(q[two, :2] - mid, axis=1) < 0.35 * sep) & (sep > 0.5)
    return n4, n2, np.where(np.isfinite(d[:, 0]), d[:, 0] * 108.33, np.nan), between


def summarize(S, m, n4, n2, dnn, between):
    q = lambda v: [float(np.nanpercentile(v, p)) for p in (25, 50, 75)]
    return dict(n=int(m.sum()), frac_crowded_4px=float(np.mean(n4[m] >= 2)), frac_crowded_2px=float(np.mean(n2[m] >= 2)),
                mean_simcode_within_4px=float(np.mean(n4[m])), frac_between_two=float(np.mean(between[m])),
                frac_no_simcode_within_4px=float(np.mean(n4[m] == 0)), dist_nn_nm_q25_50_75=q(dnn[m]),
                frac_dist_gt_50nm=float(np.nanmean(dnn[m] > 50)), photons_q=q(S['ph'][m]), bg_q=q(S['bg'][m]),
                chisq_per_px_q=q(S['chisq'][m]), crlb_nm_q=q(np.maximum(S['sx'][m], S['sy'][m]) * 108.33))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--z318', action='store_true'); a = ap.parse_args()
    Z = np.load(os.path.join(CACHE, f'{DS}_v2.npz')); vm = Z['filt']
    V = dict(x=Z['x'][vm], y=Z['y'][vm], f=Z['f'][vm])            # uncorrected positions: same frame as the raw SMLM rows
    # SIMCODE heights of the figure set (pass-1 z, um above the focal-plane reference of the calib)
    from simcode.dataset import Dataset
    import frc_fair as FF
    L = Dataset.load('/mnt/data/dmdsf/bn_dnapaint/dnapaint/2.5nM/2/runs/simcode_v2_psfv2_final').select(12.0)
    c = L.columns(); keep = c['frame'] + 6 <= F
    T = dict(x=c['x'][keep].astype(np.float64), y=c['y'][keep].astype(np.float64), f=c['frame'][keep].astype(np.int64),
             sx=c['sx'][keep].astype(np.float64), sy=c['sy'][keep].astype(np.float64))
    disj, filt = FF.masks(T); i = np.nonzero(disj)[0]
    assert np.allclose(T['x'][i], Z['x']), 'v2 row order'
    z = np.asarray(c['z'])[keep][i][vm] if 'z' in c else None
    res = dict(boxes=BOXES, sets={})
    tabs = {'figure (fit plane 0.086 um)': os.path.join(REPO, 'out', 'paper2', DS, 'baselines_psfv2', 'smlm', 'smlm.npy')}
    if a.z318: tabs['refit (fit plane 0.318 um)'] = os.path.join(REPO, 'out', 'paper2', DS, 'baselines_psfv2_z318', 'smlm', 'smlm.npy')
    for name, p in tabs.items():
        S, filt = smlm_table(p)
        S = {k: v[filt] for k, v in S.items()}
        n4, n2, dnn, btw = crowd(S, V)
        r = dict(path=p, field=summarize(S, np.ones(len(S['x']), bool), n4, n2, dnn, btw))
        for bn, b in BOXES.items(): r[bn] = summarize(S, inbox(S['x'], S['y'], b), n4, n2, dnn, btw)
        res['sets'][name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k != 'path'}), flush=True)
    if z is not None:
        zz = lambda m: [float(np.nanpercentile(z[m], p)) for p in (10, 25, 50, 75, 90)]
        res['simcode_z_um_q10_25_50_75_90'] = dict(field=zz(np.ones(len(z), bool)), **{bn: zz(inbox(V['x'], V['y'], b)) for bn, b in BOXES.items()})
        res['simcode_z_note'] = 'pass-1 z column of the SIMCODE table (table units); the SMLM / SIMFLUX fit plane = the field median'
    # SIMCODE density per window in the boxes (emitters ON per um^2 per 6-frame window, figure set)
    nwin = F // 6
    res['simcode_locs_per_um2_per_window'] = {bn: float(inbox(V['x'], V['y'], b).sum() / nwin / (b[2] * 0.10833) ** 2) for bn, b in BOXES.items()}
    res['simcode_locs_per_um2_per_window']['field'] = float(len(V['x']) / nwin / (110 * 0.10833) ** 2)
    json.dump(res, open(os.path.join(REPO, 'out', 'refcheck', 'i_smlmzoom.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'sets'}))


if __name__ == '__main__':
    main()
