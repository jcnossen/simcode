"""Pixel bias, precision and recall of the final SIMCODE v2 output for several pass-2 emitter networks (production vs longer
training), from the pass-2 re-runs of stage_solves.py (<stages dir>/<movie>.npz; one directory per network).

All variants are scored the same way, on their own re-run rows (so the production network is re-scored here too):
  selection  p >= 0.63, inside the crop, max(sx_cal, sy_cal) x 108.33 nm <= 12 nm (the paper's rule WITHOUT the model's
             selection-aware sigma rescale sig_u, which is fitted to the production networks) ['sel'], and p_agg >= 0.63 ['sel_pagg']
  bias       |c1| x / y of the selected rows, GT-matched |c1|, A_bias x / y +- SE (sim.gt_rows / bias_curve), by photon tercile
             (terciles of the production variant's matched rows, reused for every variant)
  precision  RMS of the matched errors (|e| < 0.5 px), nm; iso movie: RMS / CRLB (SIMFLUX CRLB of out/decode_checks/iso_crlb.npz,
             per GT emitter, windows start % 12 == 0, as iso_eval.py) and recall of the GT emitters
  recall     isolated GT emitters (stage_bias.recall_edge rule) found by the selected rows; edge ratio x / y
usage: python analysis/pixel_bias_simcode/ckpt_eval.py --variants name=dir,name=dir ... [--movies iso,D1,Dexp1] [--tag T]
       -> <out>/ckpt_eval<T>.json / .md"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import stage_bias as SB
B, PS, M = SB.B, SB.PS, SB.M
from simcode import table as SP


def score(Z, gt, nf, W, crop, mv, terc_q=None, CR=None):
    x, y = Z['p2_r'][:, 0].astype(np.float64), Z['p2_r'][:, 1].astype(np.float64)
    w = Z['p2_wstart'].astype(np.int64); ph = Z['p2_phot'].astype(np.float64)
    sig = np.maximum(Z['p2_sx_cal'], Z['p2_sy_cal']) * SP.PX_NM_DEFAULT
    reg = SP.in_crop(x, y, crop) & (sig <= SP.SIGMA_MAX_NM)
    out = {}
    tx, ty, _ = PS.gt_rows(gt, nf, w, x, y)
    for sn, pv in (('sel', Z['p2_p']), ('sel_pagg', np.nan_to_num(Z['p2_p_agg'], nan=0.0) if 'p2_p_agg' in Z.files else None)):
        if pv is None: continue
        m = reg & (pv >= SP.P_GOOD)
        g = m & np.isfinite(tx)
        if mv == 'iso': g &= (w % 12 == 0)
        q = terc_q if terc_q is not None else np.quantile(ph[g], [1 / 3, 2 / 3])
        terc = dict(T1=ph < q[0], T2=(ph >= q[0]) & (ph < q[1]), T3=ph >= q[1])
        r = dict(n=int(m.sum()), n_matched=int(g.sum()), ph_terciles=[float(v) for v in q])
        for ax, v, t in (('x', x, tx), ('y', y, ty)):
            e = v[g] - t[g]; fg = B.sub(t[g])
            r[ax] = dict(c1=B.ustats(B.sub(v[m]))['c1_abs'], gt_c1=B.ustats(fg)['c1_abs'], bias=PS.bias_curve(e, fg),
                         terc={tn: dict(bias=PS.bias_curve(e[b[g]], fg[b[g]]), c1=B.ustats(B.sub(v[m & b]))['c1_abs']) for tn, b in terc.items()})
        e = np.r_[x[g] - tx[g], y[g] - ty[g]]; e = e[np.abs(e) < 0.5]; r['rms_nm'] = float(np.sqrt(np.mean(e ** 2)) * B.PX)
        # recall of isolated GT emitters + edge ratio
        R = dict(P1=None, E1=None, E2=dict(wstart=w, x_hy0=x, y_hy0=y), sel2=m)
        re = recall_sel(R, gt, nf, W, iso=(mv == 'iso'))
        r.update(recall=re['recall'], edge_x=re['x']['edge_ratio'], edge_y=re['y']['edge_ratio'])
        if CR is not None:          # iso: per GT emitter, nearest selected row of its window within 1 px
            sl = {k: v[m] for k, v in dict(f=w, x=x, y=y).items()}
            j = match_iso(sl, gt); ok = j >= 0
            ex = (sl['x'][j[ok]] - gt['ev_x'][ok]) * B.PX; ey = (sl['y'][j[ok]] - gt['ev_y'][ok]) * B.PX
            cr = CR['sf'][ok]
            r['iso'] = dict(recall=float(ok.mean()), rms_nm=float(np.sqrt(np.mean(np.r_[ex, ey] ** 2))),
                            crlb_nm=float(np.sqrt(np.mean(np.r_[cr[:, 0], cr[:, 1]] ** 2))))
            r['iso']['rms_over_crlb'] = r['iso']['rms_nm'] / r['iso']['crlb_nm']
            lev = gt['ev_I6']
            r['iso']['by_ph'] = {}
            for L in np.unique(lev):
                k = ok & (lev == L); kk = k[ok]
                if kk.sum() < 20: continue
                rm = float(np.sqrt(np.mean(np.r_[ex[kk], ey[kk]] ** 2))); cb = float(np.sqrt(np.mean(np.r_[cr[kk, 0], cr[kk, 1]] ** 2)))
                r['iso']['by_ph'][f'{L:g}'] = dict(recall=float(k.sum() / (lev == L).sum()), rms_over_crlb=rm / cb)
        out[sn] = r
    return out


def recall_sel(R, gt, nf, W, iso=False, ph_min=300.0):
    starts = np.arange(0, nf - 5, 12 if iso else 6)
    GW = M.gt_windows(gt, nf, starts)
    gx, gy, gw = [], [], []
    for f, (x, y, ph) in GW.items():
        if not len(x): continue
        dd = np.hypot(x[:, None] - x[None], y[:, None] - y[None]); np.fill_diagonal(dd, np.inf)
        k = (dd.min(1) > 2.0) & (ph >= ph_min) & (x > 1) & (x < W - 1) & (y > 1) & (y < W - 1)
        gx.append(x[k]); gy.append(y[k]); gw.append(np.full(k.sum(), f))
    gx, gy, gw = np.concatenate(gx), np.concatenate(gy), np.concatenate(gw)
    E = R['E2']; k = R['sel2']
    w, x, y = E['wstart'][k], E['x_hy0'][k], E['y_hy0'][k]
    o = np.argsort(w, kind='stable'); ws = w[o]; found = np.zeros(len(gx), bool)
    for f in np.unique(gw):
        a, b = np.searchsorted(ws, [f, f + 1])
        if a == b: continue
        ii = o[a:b]; jj = np.nonzero(gw == f)[0]
        found[jj] = np.hypot(x[ii][None] - gx[jj, None], y[ii][None] - gy[jj, None]).min(1) < 1.0
    r = dict(recall=float(found.mean()))
    for ax, fg in (('x', B.sub(gx)), ('y', B.sub(gy))):
        r[ax] = dict(edge_ratio=float(found[np.abs(fg) > 0.4].mean() / max(found[np.abs(fg) < 0.3].mean(), 1e-9)))
    return r


def match_iso(T, gt):
    f0 = gt['ev_t0'].astype(np.int64); gx, gy = gt['ev_x'], gt['ev_y']
    j = np.full(len(gx), -1); o = np.argsort(T['f'], kind='stable'); fs = T['f'][o]
    for w in np.unique(f0):
        e = np.nonzero(f0 == w)[0]; a, b = np.searchsorted(fs, [w, w + 1]); r = o[a:b]
        if not len(r): continue
        d = np.hypot(T['x'][r][None] - gx[e, None], T['y'][r][None] - gy[e, None]); k = d.argmin(1); ok = d[np.arange(len(e)), k] < 1.0
        j[e[ok]] = r[k[ok]]
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', required=True, help='name=stages_dir,name=stages_dir (first = reference for the terciles)')
    ap.add_argument('--movies', default='iso,D1,Dexp1')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    V = [v.split('=', 1) for v in a.variants.split(',')]
    RES = {}
    for mv in a.movies.split(','):
        md = os.path.join(SB.REPO, SB.MOV[mv]); gt = dict(np.load(os.path.join(md, 'gt.npz'))); nf = int(gt['nframes']); W = int(gt['W'])
        crop = SP.run_crop(os.path.join(md, 'full', 'v2_m1809v2'))
        CR = dict(np.load(os.path.join(SB.REPO, 'out', 'decode_checks', 'iso_crlb.npz'))) if mv == 'iso' else None
        RES[mv] = {}; q = None
        for name, dd in V:
            zf = os.path.join(dd, mv + '.npz')
            if not os.path.exists(zf): print('missing', zf); continue
            r = score(np.load(zf), gt, nf, W, crop, mv, q, CR)
            if q is None: q = r['sel']['ph_terciles']
            RES[mv][name] = r
            s = r['sel']
            print(f"{mv:6s} {name:14s} n {s['n']} |c1| {s['x']['c1']:.4f}/{s['y']['c1']:.4f} (GT {s['x']['gt_c1']:.4f}/{s['y']['gt_c1']:.4f}) "
                  f"A {s['x']['bias']['A_nm']:+.2f}+-{s['x']['bias']['A_se_nm']:.2f}/{s['y']['bias']['A_nm']:+.2f}+-{s['y']['bias']['A_se_nm']:.2f} "
                  f"T x {' '.join('%+.2f' % s['x']['terc'][t]['bias']['A_nm'] for t in ('T1', 'T2', 'T3'))} rms {s['rms_nm']:.2f} recall {s['recall']:.3f} "
                  f"edge {s['edge_x']:.3f}/{s['edge_y']:.3f}" + (f" iso RMS/CRLB {s['iso']['rms_over_crlb']:.3f} recall {s['iso']['recall']:.3f}" if 'iso' in s else ''), flush=True)
    json.dump(RES, open(os.path.join(SB.OUT, f'ckpt_eval{a.tag}.json'), 'w'), indent=1)
    L = ['| movie | network | selection | n | |c1| x / y | GT-matched |c1| x / y | A_bias x / y (nm +- SE) | A_bias x T1 / T2 / T3 | A_bias y T1 / T2 / T3 | RMS nm | recall | edge ratio x / y | iso RMS/CRLB |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for mv, D in RES.items():
        for name, r in D.items():
            for sn, s in r.items():
                tx = ' / '.join('%+.2f' % s['x']['terc'][t]['bias']['A_nm'] for t in ('T1', 'T2', 'T3'))
                ty = ' / '.join('%+.2f' % s['y']['terc'][t]['bias']['A_nm'] for t in ('T1', 'T2', 'T3'))
                iso = f"{s['iso']['rms_over_crlb']:.3f}" if 'iso' in s else ''
                L.append(f"| {mv} | {name} | {sn} | {s['n']} | {s['x']['c1']:.4f} / {s['y']['c1']:.4f} | {s['x']['gt_c1']:.4f} / {s['y']['gt_c1']:.4f} | "
                         f"{s['x']['bias']['A_nm']:+.2f} +- {s['x']['bias']['A_se_nm']:.2f} / {s['y']['bias']['A_nm']:+.2f} +- {s['y']['bias']['A_se_nm']:.2f} | "
                         f"{tx} | {ty} | {s['rms_nm']:.2f} | {s['recall']:.3f} | {s['edge_x']:.3f} / {s['edge_y']:.3f} | {iso} |")
    open(os.path.join(SB.OUT, f'ckpt_eval{a.tag}.md'), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
