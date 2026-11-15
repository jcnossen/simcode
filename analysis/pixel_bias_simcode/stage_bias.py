"""Where does SIMCODE v2's pixel bias come from? Per-stage pixel-bias statistics on the isolated-emitter movie and the
v2-PSF density movies (needs the per-stage solves of stage_solves.py in <out>/stages/<movie>.npz).

Stages of one final (pass-2) localization, all on the SAME rows (the paper's selection of locs_p2: p >= 0.63, in the crop,
calibrated max(sx, sy) <= 12 nm, GT-matched as analysis/pixbias/sim.gt_rows: nearest ON emitter of the window within 1 px, no
other ON emitter within 2 px, >= 100 window photons):
  P1 det      pass-1 detector point (pass1 npz x_det of the linked pass-1 candidate, p1_row)
  P1 prior    pass-1 emitter-network position prior (pass1 x, y)
  P1 fringe   pass-1 intensities alone (stage_solves fringe-only solve: flat prior, local optimum at the pass-1 solve)
  P1 solve    pass-1 explicit pattern solve (pass2_est x_hy0) = locs_p1
  P2 det / P2 prior / P2 fringe / P2 solve   the same for pass 2 (P2 solve = locs_p2 x, y = the SIMCODE v2 output)
Per stage and axis: |c1| of the estimate's own sub-pixel histogram (pblib.ustats), A_bias +- SE (sim.bias_curve: e = -A sin(2 pi
f_gt) + ..., A > 0 = pull to the pixel centre), overall, by photon tercile (network window photons of the final row, terciles of
the matched rows of the movie) and the GT-matched |c1| (the sub-pixel histogram of the matched TRUE positions = the recall part).
Prior-pull decomposition per pass: w = s_fringe^2 / (s_fringe^2 + s_prior^2) (the weight of the prior in a Gaussian product;
s_prior = the prior sigma the solve uses, s_fringe = the Fisher sigma of the fringe-only solve); predicted solve error
w e_prior + (1 - w) e_fringe -> its A, and the prior term's share A(w e_prior).
Recall vs GT sub-pixel position (the edge dip): GT units = isolated ON emitters (>= 300 window photons, >= 2 px from other ON
emitters, 1 px inside the field) of the disjoint 6-frame windows (start % 6 == 0); found by a row of the same window within 1 px:
  P1 cand (all stored pass-1 candidates, p >= 0.3) / P1 rows (pass-1 table) / P2 rows (all) / P2 p >= 0.63 / P2 sel (paper)
-> recall in 10 bins of f_gt (x, y) and the edge ratio recall(|f| > 0.4) / recall(|f| < 0.3).

usage: python analysis/pixel_bias_simcode/stage_bias.py [movies] [--stages DIR] [--tag NAME]
       -> <out>/stage_bias<tag>.json, stage_bias<tag>.md, fig_stage_bias<tag>.png"""
import argparse, glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.join(REPO, 'analysis', 'papersim'), os.path.join(REPO, 'analysis', 'paper'), os.path.join(REPO, 'analysis', 'pixbias')):
    sys.path.insert(0, p)
import pblib as B
import sim as PS
import metrics as M
from simcode import table as SP
from simcode.dataset import Dataset

OUT = os.path.join(REPO, 'out', 'pixel_bias_simcode')
MOV = {'iso': 'out/decode_checks/iso_v2psf', 'D0.5': 'out/papersim/D0.5_v2psf', 'D1': 'out/papersim/D1_v2psf', 'D2': 'out/papersim/D2_v2psf',
       'D4': 'out/papersim/D4_v2psf', 'Dexp1': 'out/papersim/Dexp1_v2psf'}
STAGES = ['P1 det', 'P1 prior', 'P1 fringe', 'P1 solve', 'P2 det', 'P2 prior', 'P2 fringe', 'P2 solve']


def chunks_of(d, pat):
    return sorted(int(f.rsplit('_', 1)[1][:-4]) for f in glob.glob(f'{d}/{pat}_*.npz') if f.rsplit('_', 1)[1][:-4].isdigit())


def load_run(d, Z):
    """per final row: every stage's (x, y); plus the selection, photons, window and the pass-1 candidate tables"""
    ch = chunks_of(d, 'pass1')
    P1 = {k: [] for k in ('wstart', 'x', 'y', 'x_det', 'y_det', 'out', 'p')}; off = {}; n = 0
    for f0 in ch:
        z = np.load(f'{d}/pass1_{f0}.npz'); off[f0] = n; n += len(z['x'])
        for k in P1: P1[k].append(z[k])
    P1 = {k: np.concatenate(v) for k, v in P1.items()}
    est_idx = np.cumsum(P1['out'].astype(bool)) - 1
    E1 = B.load_rows(d, 'est', ['wstart', 'x_hy0', 'y_hy0', 'x_prior', 'y_prior', 'phot', 'p'])
    E2 = {k: [] for k in ('wstart', 'x_hy0', 'y_hy0', 'x_prior', 'y_prior', 'x_det', 'y_det', 'phot', 'p', 'p1_row')}
    for f0 in ch:
        z = np.load(f'{d}/pass2_est2_{f0}.npz')
        for k in E2:
            v = z[k]
            if k == 'p1_row': v = np.where(v >= 0, v + off[f0], -1)
            E2[k].append(v)
    E2 = {k: np.concatenate(v) for k, v in E2.items()}
    T2 = Dataset.from_run(d, pass_=2, info=False).columns(); T1 = Dataset.from_run(d, pass_=1, info=False).columns()
    assert len(T2['x']) == len(E2['x_hy0']) and np.allclose(T2['x'], E2['x_hy0']), 'locs_p2 != pass2_est2 rows'
    assert len(T1['x']) == len(E1['x_hy0'])
    crop = SP.run_crop(d)
    sel2 = SP.select(T2, crop=crop); sel1 = SP.select(T1, crop=crop)
    # stored rows vs the re-run of stage_solves (pass 2 networks re-run: same row count per chunk is required)
    assert len(Z['p2_r']) == len(E2['x_hy0']), (len(Z['p2_r']), len(E2['x_hy0']))
    g = E2['p1_row']; has = g >= 0
    e1 = np.full(len(g), -1); e1[has] = np.where(P1['out'][g[has]].astype(bool), est_idx[g[has]], -1)
    nanv = lambda a, i: np.where(i >= 0, a[np.clip(i, 0, None)], np.nan)
    st = {
        'P1 det': (nanv(P1['x_det'], g), nanv(P1['y_det'], g)),
        'P1 prior': (nanv(P1['x'], g), nanv(P1['y'], g)),
        'P1 fringe': (nanv(Z['p1_rf'][:, 0], e1), nanv(Z['p1_rf'][:, 1], e1)),
        'P1 solve': (nanv(E1['x_hy0'], e1), nanv(E1['y_hy0'], e1)),
        'P2 det': (E2['x_det'], E2['y_det']), 'P2 prior': (E2['x_prior'], E2['y_prior']),
        'P2 fringe': (Z['p2_rf'][:, 0], Z['p2_rf'][:, 1]), 'P2 solve': (E2['x_hy0'], E2['y_hy0'])}
    st = {k: (np.asarray(a, np.float64), np.asarray(b, np.float64)) for k, (a, b) in st.items()}
    w = dict(P1=(nanv(Z['p1_sf'][:, 0] ** 2 / (Z['p1_sf'][:, 0] ** 2 + Z['p1_sp'][:, 0] ** 2), e1), nanv(Z['p1_sf'][:, 1] ** 2 / (Z['p1_sf'][:, 1] ** 2 + Z['p1_sp'][:, 1] ** 2), e1)),
             P2=(Z['p2_sf'][:, 0] ** 2 / (Z['p2_sf'][:, 0] ** 2 + Z['p2_sp'][:, 0] ** 2), Z['p2_sf'][:, 1] ** 2 / (Z['p2_sf'][:, 1] ** 2 + Z['p2_sp'][:, 1] ** 2)))
    return dict(run=d, st=st, w=w, sel2=sel2, good2=T2['p'] >= SP.P_GOOD, w2=E2['wstart'].astype(np.int64), ph=E2['phot'].astype(np.float64),
                P1=P1, E1=E1, sel1=sel1, E2=E2, crop=crop)


def stage_stats(R, gt, nf, iso_only=False):
    st = R['st']; xf, yf = st['P2 solve']
    tx, ty, tph = PS.gt_rows(gt, nf, R['w2'], xf, yf)
    m = R['sel2'] & np.isfinite(tx)
    if iso_only: m &= (R['w2'] % 12 == 0)
    for k in STAGES: m &= np.isfinite(st[k][0]) & np.isfinite(st[k][1])
    ph = R['ph']; q = np.quantile(ph[m], [1 / 3, 2 / 3])
    terc = dict(T1=ph < q[0], T2=(ph >= q[0]) & (ph < q[1]), T3=ph >= q[1])
    out = dict(n=int(m.sum()), n_sel=int(R['sel2'].sum()), ph_terciles=[float(v) for v in q], stages={})
    for k in STAGES:
        r = {}
        for ax, i, t in (('x', 0, tx), ('y', 1, ty)):
            v = st[k][i]; e = v[m] - t[m]; fg = B.sub(t[m])
            r[ax] = dict(hist=B.ustats(B.sub(v[m])), gt_hist=B.ustats(fg), bias=PS.bias_curve(e, fg),
                         terc={tn: dict(bias=PS.bias_curve(e[b[m]], fg[b[m]]), hist=B.ustats(B.sub(v[m][b[m]])), gt_hist=B.ustats(fg[b[m]]))
                               for tn, b in terc.items()},
                         h40=np.histogram(B.sub(v[m]), 40, (-0.5, 0.5))[0].tolist())
        out['stages'][k] = r
    out['gt_h40'] = dict(x=np.histogram(B.sub(tx[m]), 40, (-0.5, 0.5))[0].tolist(), y=np.histogram(B.sub(ty[m]), 40, (-0.5, 0.5))[0].tolist())
    # prior-pull decomposition
    dec = {}
    for ps in ('P1', 'P2'):
        dec[ps] = {}
        for ax, i, t in (('x', 0, tx), ('y', 1, ty)):
            ep = st[f'{ps} prior'][i] - t; ef = st[f'{ps} fringe'][i] - t; es = st[f'{ps} solve'][i] - t; ww = R['w'][ps][i]
            k = m & np.isfinite(ww) & (np.abs(ep) < 0.5) & (np.abs(ef) < 0.5) & (np.abs(es) < 0.5)
            fg = B.sub(t[k])
            pred = ww[k] * ep[k] + (1 - ww[k]) * ef[k]
            d = dict(n=int(k.sum()), w_mean=float(np.mean(ww[k])), w_median=float(np.median(ww[k])),
                     A_solve=PS.bias_curve(es[k], fg)['A_nm'], A_prior=PS.bias_curve(ep[k], fg)['A_nm'], A_fringe=PS.bias_curve(ef[k], fg)['A_nm'],
                     A_pred=PS.bias_curve(pred, fg)['A_nm'], A_prior_term=PS.bias_curve(ww[k] * ep[k], fg)['A_nm'],
                     A_fringe_term=PS.bias_curve((1 - ww[k]) * ef[k], fg)['A_nm'],
                     corr_prior_fringe=float(np.corrcoef(ep[k], ef[k])[0, 1]),
                     rms_prior_nm=float(np.sqrt(np.mean(ep[k] ** 2)) * B.PX), rms_fringe_nm=float(np.sqrt(np.mean(ef[k] ** 2)) * B.PX),
                     rms_solve_nm=float(np.sqrt(np.mean(es[k] ** 2)) * B.PX), rms_pred_nm=float(np.sqrt(np.mean(pred ** 2)) * B.PX))
            d['terc'] = {}
            for tn, b in terc.items():
                bb = b[k]
                if bb.sum() < 200: continue
                d['terc'][tn] = dict(w_mean=float(np.mean(ww[k][bb])), A_solve=PS.bias_curve(es[k][bb], fg[bb])['A_nm'], A_prior=PS.bias_curve(ep[k][bb], fg[bb])['A_nm'],
                                     A_fringe=PS.bias_curve(ef[k][bb], fg[bb])['A_nm'], A_prior_term=PS.bias_curve(ww[k][bb] * ep[k][bb], fg[bb])['A_nm'])
            dec[ps][ax] = d
    out['decomp'] = dec
    return out


def recall_edge(R, gt, nf, W, iso=False, ph_min=300.0, only=None):
    starts = np.arange(0, nf - 5, 12 if iso else 6)
    GW = M.gt_windows(gt, nf, starts)
    gx, gy, gw = [], [], []
    for f, (x, y, ph) in GW.items():
        if not len(x): continue
        dd = np.hypot(x[:, None] - x[None], y[:, None] - y[None]); np.fill_diagonal(dd, np.inf)
        k = (dd.min(1) > 2.0) & (ph >= ph_min) & (x > 1) & (x < W - 1) & (y > 1) & (y < W - 1)
        gx.append(x[k]); gy.append(y[k]); gw.append(np.full(k.sum(), f))
    gx, gy, gw = np.concatenate(gx), np.concatenate(gy), np.concatenate(gw)
    P1, E1, E2 = R['P1'], R['E1'], R['E2']
    sets = {'P1 cand': (P1['wstart'], P1['x_det'], P1['y_det'], np.ones(len(P1['x']), bool)),
            'P1 rows': (E1['wstart'], E1['x_hy0'], E1['y_hy0'], np.ones(len(E1['x_hy0']), bool)),
            'P1 sel': (E1['wstart'], E1['x_hy0'], E1['y_hy0'], R['sel1']),
            'P2 rows': (E2['wstart'], E2['x_hy0'], E2['y_hy0'], np.ones(len(E2['x_hy0']), bool)),
            'P2 p>=0.63': (E2['wstart'], E2['x_hy0'], E2['y_hy0'], R['good2']),
            'P2 sel': (E2['wstart'], E2['x_hy0'], E2['y_hy0'], R['sel2'])}
    out = dict(n_gt=int(len(gx)))
    fxg, fyg = B.sub(gx), B.sub(gy)
    for nm, (w, x, y, k) in sets.items():
        if only and nm not in only: continue
        w, x, y = w[k].astype(np.int64), x[k], y[k]
        o = np.argsort(w, kind='stable'); ws = w[o]
        found = np.zeros(len(gx), bool)
        go = np.argsort(gw, kind='stable')
        for f in np.unique(gw):
            a, b = np.searchsorted(ws, [f, f + 1])
            if a == b: continue
            ii = o[a:b]; jj = np.nonzero(gw == f)[0]
            d = np.hypot(x[ii][None] - gx[jj, None], y[ii][None] - gy[jj, None])
            found[jj] = d.min(1) < 1.0
        r = {}
        for ax, fg in (('x', fxg), ('y', fyg)):
            ib = np.clip(((fg + 0.5) * 10).astype(int), 0, 9)
            rc = [float(found[ib == i].mean()) for i in range(10)]
            edge = np.abs(fg) > 0.4; cen = np.abs(fg) < 0.3
            r[ax] = dict(recall_bins=rc, edge_ratio=float(found[edge].mean() / max(found[cen].mean(), 1e-9)))
        r['recall'] = float(found.mean())
        out[nm] = r
    return out


def sel_variants(R, Z, gt, nf, iso=False):
    """the stage biases under other row selections (is the bias in the estimates, or made by the selection?):
    paper (p >= 0.63, sigma <= 12 nm, crop), no_p (sigma + crop only), p_agg (p_agg >= 0.63 + sigma + crop), p63_nosig (p + crop),
    all_rows (crop only). Per selection and stage: A_bias x / y, n; plus |c1| of the estimate and of the matched GT (P2 solve)."""
    T = Dataset.from_run(R['run'], pass_=2, info=False).columns()
    xf, yf = R['st']['P2 solve']; tx, ty, _ = PS.gt_rows(gt, nf, R['w2'], xf, yf)
    sels = dict(paper=R['sel2'], no_p=SP.select(T, min_score=None, crop=R['crop']),
                p63_nosig=SP.select(T, max_precision_nm=None, crop=R['crop']), all_rows=SP.select(T, min_score=None, max_precision_nm=None, crop=R['crop']))
    if 'p2_p_agg' in Z.files: sels['p_agg'] = sels['no_p'] & (np.nan_to_num(Z['p2_p_agg']) >= SP.P_GOOD)
    out = {}
    for sn, m0 in sels.items():
        m = m0 & np.isfinite(tx)
        if iso: m &= (R['w2'] % 12 == 0)
        r = dict(n_rows=int(m0.sum()), n=int(m.sum()), c1_x=B.ustats(B.sub(xf[m0]))['c1_abs'], c1_y=B.ustats(B.sub(yf[m0]))['c1_abs'],
                 c1_x_matched=B.ustats(B.sub(xf[m]))['c1_abs'], gt_c1_x=B.ustats(B.sub(tx[m]))['c1_abs'], gt_c1_y=B.ustats(B.sub(ty[m]))['c1_abs'], A={})
        for k in STAGES:
            v = R['st'][k]; mm = m & np.isfinite(v[0])
            bx = PS.bias_curve(v[0][mm] - tx[mm], B.sub(tx[mm])); by = PS.bias_curve(v[1][mm] - ty[mm], B.sub(ty[mm]))
            r['A'][k] = dict(x=bx['A_nm'], y=by['A_nm'], se=bx['A_se_nm'], n=int(mm.sum()))
        out[sn] = r
    return out


def p_selection(R, Z, gt, nf, W, iso=False):
    """the paper's selection with the detection probability replaced by the aggregated one (stage_solves.PAggRecorder: p_agg = p
    + the suppressed same-emitter cells; p_max2 = p + the largest of them), at the threshold 0.63 and at the threshold that keeps
    the production row count. Per variant: rows, GT-matched fraction of the rows (1 px, any ON emitter of the window), recall
    edge ratio x / y (GT units of recall_edge), |c1| x / y of the selected rows, A_bias x / y (matched isolated rows)."""
    if 'p2_p_agg' not in Z.files: return None
    E2 = R['E2']; sel = R['sel2']
    T = Dataset.from_run(R['run'], pass_=2, info=False).columns()
    base = SP.select(T, min_score=None, crop=R['crop'])
    out = {}
    n0 = int(sel.sum())
    xf, yf = E2['x_hy0'], E2['y_hy0']; w2 = R['w2']
    tx, ty, _ = PS.gt_rows(gt, nf, w2, xf, yf)
    # any-emitter match (precision): nearest ON emitter of the window within 1 px, no isolation / photon condition
    GW = M.gt_windows(gt, nf, np.unique(w2)); anym = np.zeros(len(xf), bool)
    o = np.argsort(w2, kind='stable'); ws = w2[o]; st = np.r_[0, np.nonzero(np.diff(ws))[0] + 1, len(ws)]
    for a_, b_ in zip(st[:-1], st[1:]):
        gx, gy, gph = GW[ws[a_]]
        if not len(gx): continue
        ii = o[a_:b_]; d = np.hypot(xf[ii, None] - gx[None], yf[ii, None] - gy[None]); anym[ii] = d.min(1) < 1.0
    for name, pv in (('p', Z['p2_p']), ('p_agg', Z['p2_p_agg']), ('p_max2', Z['p2_p_max2'])):
        pv = np.nan_to_num(pv, nan=0.0)
        ts = [0.63]
        srt = np.sort(pv[base])[::-1]
        if n0 <= len(srt): ts.append(float(srt[n0 - 1]))
        for t in ts:
            m = base & (pv >= t)
            r = dict(thr=float(t), n=int(m.sum()), frac_matched=float(anym[m].mean()))
            Rm = dict(R, sel2=m)
            re = recall_edge(Rm, gt, nf, W, iso=iso, only=('P2 sel',))
            r['recall'] = re['P2 sel']['recall']; r['edge_x'] = re['P2 sel']['x']['edge_ratio']; r['edge_y'] = re['P2 sel']['y']['edge_ratio']
            r['recall_bins_x'] = re['P2 sel']['x']['recall_bins']
            r['c1_x'] = B.ustats(B.sub(xf[m]))['c1_abs']; r['c1_y'] = B.ustats(B.sub(yf[m]))['c1_abs']
            g = m & np.isfinite(tx)
            if iso: g &= (w2 % 12 == 0)
            bx = PS.bias_curve(xf[g] - tx[g], B.sub(tx[g])); by = PS.bias_curve(yf[g] - ty[g], B.sub(ty[g]))
            r['A_x'], r['A_y'] = bx['A_nm'], by['A_nm']; r['A_se'] = bx['A_se_nm']
            r['gt_c1_x'] = B.ustats(B.sub(tx[g]))['c1_abs']; r['gt_c1_y'] = B.ustats(B.sub(ty[g]))['c1_abs']
            e = np.r_[xf[g] - tx[g], yf[g] - ty[g]]; e = e[np.abs(e) < 0.5]
            r['rms_nm'] = float(np.sqrt(np.mean(e ** 2)) * B.PX)
            out[f'{name}@{t:.3f}'] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('movies', nargs='*', default=list(MOV))
    ap.add_argument('--stages', default=os.path.join(OUT, 'stages'))
    ap.add_argument('--tag', default='')
    ap.add_argument('--run', default='v2_m1809v2', help='run sub-directory under <movie>/full')
    a = ap.parse_args()
    RES = {}
    for mv in a.movies:
        md = os.path.join(REPO, MOV[mv]); d = os.path.join(md, 'full', a.run)
        zf = os.path.join(a.stages, mv + '.npz')
        if not os.path.exists(zf): print('missing', zf); continue
        gt = dict(np.load(os.path.join(md, 'gt.npz'))); nf = int(gt['nframes']); W = int(gt['W'])
        R = load_run(d, np.load(zf))
        S = stage_stats(R, gt, nf, iso_only=(mv == 'iso'))
        S['recall_edge'] = recall_edge(R, gt, nf, W, iso=(mv == 'iso'))
        S['p_selection'] = p_selection(R, np.load(zf), gt, nf, W, iso=(mv == 'iso'))
        S['sel_variants'] = sel_variants(R, np.load(zf), gt, nf, iso=(mv == 'iso'))
        for sn, v in S['sel_variants'].items():
            print(f"  selvar {sn:10s} n {v['n']:7d} |c1| {v['c1_x']:.4f}/{v['c1_y']:.4f} GT {v['gt_c1_x']:.4f}/{v['gt_c1_y']:.4f} " +
                  ' '.join(f"{k} {v['A'][k]['x']:+.2f}/{v['A'][k]['y']:+.2f}" for k in ('P1 prior', 'P1 fringe', 'P1 solve', 'P2 prior', 'P2 fringe', 'P2 solve')))
        for k, v in (S['p_selection'] or {}).items():
            print(f"  sel {k:14s} n {v['n']} matched {v['frac_matched']:.3f} recall {v['recall']:.3f} edge {v['edge_x']:.3f}/{v['edge_y']:.3f} "
                  f"|c1| {v['c1_x']:.4f}/{v['c1_y']:.4f} (GT {v['gt_c1_x']:.4f}/{v['gt_c1_y']:.4f}) A {v['A_x']:+.2f}/{v['A_y']:+.2f} rms {v['rms_nm']:.2f}")
        RES[mv] = S
        print(f'== {mv}: n {S["n"]}, photon terciles {S["ph_terciles"]}')
        for k in STAGES:
            r = S['stages'][k]
            print('  %-10s |c1| %.4f %.4f (GT-matched %.4f %.4f)  A %+.2f +- %.2f / %+.2f +- %.2f nm  T1/T2/T3 x %s' % (
                k, r['x']['hist']['c1_abs'], r['y']['hist']['c1_abs'], r['x']['gt_hist']['c1_abs'], r['y']['gt_hist']['c1_abs'],
                r['x']['bias']['A_nm'], r['x']['bias']['A_se_nm'], r['y']['bias']['A_nm'], r['y']['bias']['A_se_nm'],
                ' '.join('%+.2f' % r['x']['terc'][t]['bias']['A_nm'] for t in ('T1', 'T2', 'T3'))))
        for ps in ('P1', 'P2'):
            for ax in ('x', 'y'):
                dd = S['decomp'][ps][ax]
                print(f'  decomp {ps} {ax}: w {dd["w_mean"]:.2f} A solve {dd["A_solve"]:+.2f} prior {dd["A_prior"]:+.2f} fringe {dd["A_fringe"]:+.2f} '
                      f'pred {dd["A_pred"]:+.2f} (prior term {dd["A_prior_term"]:+.2f}, fringe term {dd["A_fringe_term"]:+.2f}) corr {dd["corr_prior_fringe"]:.2f} '
                      f'rms prior/fringe/solve {dd["rms_prior_nm"]:.1f}/{dd["rms_fringe_nm"]:.1f}/{dd["rms_solve_nm"]:.1f}')
        re = S['recall_edge']
        print('  recall edge ratio x/y: ' + '  '.join(f'{k} {re[k]["x"]["edge_ratio"]:.3f}/{re[k]["y"]["edge_ratio"]:.3f} (rec {re[k]["recall"]:.3f})' for k in re if k != 'n_gt'), flush=True)
    json.dump(RES, open(os.path.join(OUT, f'stage_bias{a.tag}.json'), 'w'), indent=1)
    write_md(RES, os.path.join(OUT, f'stage_bias{a.tag}.md'))
    fig(RES, os.path.join(OUT, f'fig_stage_bias{a.tag}.png'))


def write_md(RES, fn):
    L = ['| movie | stage | n | |c1| x / y | GT-matched |c1| x / y | A_bias x / y (nm +- SE) | A_bias x by photon tercile T1 / T2 / T3 | A_bias y T1 / T2 / T3 |', '|---|---|---|---|---|---|---|---|']
    for mv, S in RES.items():
        for k in STAGES:
            r = S['stages'][k]
            tx = ' / '.join('%+.2f' % r['x']['terc'][t]['bias']['A_nm'] for t in ('T1', 'T2', 'T3'))
            ty = ' / '.join('%+.2f' % r['y']['terc'][t]['bias']['A_nm'] for t in ('T1', 'T2', 'T3'))
            L.append(f"| {mv} | {k} | {S['n']} | {r['x']['hist']['c1_abs']:.4f} / {r['y']['hist']['c1_abs']:.4f} | {r['x']['gt_hist']['c1_abs']:.4f} / {r['y']['gt_hist']['c1_abs']:.4f} | "
                     f"{r['x']['bias']['A_nm']:+.2f} +- {r['x']['bias']['A_se_nm']:.2f} / {r['y']['bias']['A_nm']:+.2f} +- {r['y']['bias']['A_se_nm']:.2f} | {tx} | {ty} |")
    L += ['', '### prior-pull decomposition (w = prior weight s_f^2 / (s_f^2 + s_p^2); A in nm)', '',
          '| movie | pass | axis | mean w | A solve | A prior | A fringe-only | A predicted | prior term | fringe term | corr(e_prior, e_fringe) | RMS prior / fringe / solve nm |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for mv, S in RES.items():
        for ps in ('P1', 'P2'):
            for ax in ('x', 'y'):
                d = S['decomp'][ps][ax]
                L.append(f"| {mv} | {ps} | {ax} | {d['w_mean']:.2f} | {d['A_solve']:+.2f} | {d['A_prior']:+.2f} | {d['A_fringe']:+.2f} | {d['A_pred']:+.2f} | {d['A_prior_term']:+.2f} | "
                         f"{d['A_fringe_term']:+.2f} | {d['corr_prior_fringe']:.2f} | {d['rms_prior_nm']:.1f} / {d['rms_fringe_nm']:.1f} / {d['rms_solve_nm']:.1f} |")
    L += ['', '### stage A_bias x / y (nm) under other row selections (paper = p >= 0.63 + sigma <= 12 nm + crop; no_p = sigma + crop; p_agg = aggregated p >= 0.63 + sigma + crop; p63_nosig = p + crop; all_rows = crop)', '',
          '| movie | selection | n matched | |c1| x / y (all rows of the selection) | GT-matched |c1| x / y | ' + ' | '.join(STAGES) + ' |', '|---|---|---|---|---|' + '---|' * len(STAGES)]
    for mv, S in RES.items():
        for sn, v in S.get('sel_variants', {}).items():
            L.append(f"| {mv} | {sn} | {v['n']} | {v['c1_x']:.4f} / {v['c1_y']:.4f} | {v['gt_c1_x']:.4f} / {v['gt_c1_y']:.4f} | " + ' | '.join(f"{v['A'][k]['x']:+.2f} / {v['A'][k]['y']:+.2f}" for k in STAGES) + ' |')
    L += ['', '### selection on the aggregated detection probability (P2 solve; thresholds 0.63 and count-matched)', '',
          '| movie | variant @ threshold | rows | GT-matched fraction of rows | recall (isolated GT) | edge ratio x / y | |c1| x / y | GT-matched |c1| x / y | A_bias x / y nm | RMS nm |', '|---|---|---|---|---|---|---|---|---|---|']
    for mv, S in RES.items():
        for k, v in (S.get('p_selection') or {}).items():
            L.append(f"| {mv} | {k} | {v['n']} | {v['frac_matched']:.3f} | {v['recall']:.3f} | {v['edge_x']:.3f} / {v['edge_y']:.3f} | {v['c1_x']:.4f} / {v['c1_y']:.4f} | "
                     f"{v['gt_c1_x']:.4f} / {v['gt_c1_y']:.4f} | {v['A_x']:+.2f} / {v['A_y']:+.2f} | {v['rms_nm']:.2f} |")
    L += ['', '### recall vs GT sub-pixel position: edge ratio recall(|f| > 0.4) / recall(|f| < 0.3), x / y (overall recall)', '',
          '| movie | n GT | ' + ' | '.join(['P1 cand', 'P1 rows', 'P1 sel', 'P2 rows', 'P2 p>=0.63', 'P2 sel']) + ' |', '|---|---|' + '---|' * 6]
    for mv, S in RES.items():
        re = S['recall_edge']
        L.append(f"| {mv} | {re['n_gt']} | " + ' | '.join(f"{re[k]['x']['edge_ratio']:.3f} / {re[k]['y']['edge_ratio']:.3f} ({re[k]['recall']:.3f})"
                                                       for k in ['P1 cand', 'P1 rows', 'P1 sel', 'P2 rows', 'P2 p>=0.63', 'P2 sel']) + ' |')
    open(fn, 'w').write('\n'.join(L) + '\n')


def fig(RES, fn):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    mvs = list(RES)
    fig, axs = plt.subplots(len(mvs), 3, figsize=(15, 2.6 * len(mvs)), squeeze=False)
    col = {'P1 det': '0.75', 'P1 prior': 'tab:cyan', 'P1 fringe': 'tab:olive', 'P1 solve': 'tab:blue', 'P2 det': '0.4', 'P2 prior': 'tab:orange',
           'P2 fringe': 'tab:green', 'P2 solve': 'tab:red'}
    fc = (np.arange(10) + 0.5) / 10 - 0.5
    for i, mv in enumerate(mvs):
        S = RES[mv]
        for k in STAGES:
            b = S['stages'][k]['x']['bias']
            axs[i, 0].plot(fc, b['mu_nm'], 'o-', ms=3, color=col[k], label=k)
            h = np.array(S['stages'][k]['x']['h40'], float); axs[i, 1].step(np.linspace(-0.5, 0.5, 41)[:-1] + 0.0125, h / h.mean(), where='mid', color=col[k], lw=1)
        g = np.array(S['gt_h40']['x'], float); axs[i, 1].step(np.linspace(-0.5, 0.5, 41)[:-1] + 0.0125, g / g.mean(), where='mid', color='k', lw=1.2, ls='--', label='matched GT')
        re = S['recall_edge']
        for k, c in (('P1 cand', '0.6'), ('P1 sel', 'tab:blue'), ('P2 rows', 'tab:orange'), ('P2 p>=0.63', 'tab:purple'), ('P2 sel', 'tab:red')):
            axs[i, 2].plot(fc, re[k]['x']['recall_bins'], 'o-', ms=3, color=c, label=k)
        axs[i, 0].axhline(0, color='k', lw=0.5); axs[i, 0].set_ylabel(f'{mv}\n<est - GT> x (nm)')
        axs[i, 1].axhline(1, color='k', lw=0.5); axs[i, 1].set_ylim(0, 1.6)
        axs[i, 2].set_ylim(0, 1.02)
        if i == 0:
            axs[i, 0].legend(fontsize=6, ncol=2); axs[i, 1].legend(fontsize=6); axs[i, 2].legend(fontsize=6)
            axs[i, 0].set_title('bias curve x vs GT sub-pixel position', fontsize=9); axs[i, 1].set_title('sub-pixel histogram of the estimate (x)', fontsize=9)
            axs[i, 2].set_title('recall vs GT sub-pixel position (x)', fontsize=9)
    for ax in axs[-1]: ax.set_xlabel('sub-pixel position (px, 0 = pixel centre)')
    fig.tight_layout(); fig.savefig(fn, dpi=130); print(fn)


if __name__ == '__main__':
    main()
