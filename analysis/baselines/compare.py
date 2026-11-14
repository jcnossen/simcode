"""Like-for-like comparison of the SIMCODE paper pipeline (run_simcode.py) and our pipeline (run_uncalibrated --frontend sc)
on a simulated movie (sc6/sim_movie.py) against the GT.
Common protocol:
  windows   W3 = 6-frame windows starting at frame w = 0 mod 3 (our pass-1 grid). SIMCODE emits a localization list for every
            window start (MovieProcessor batch overlap 5; frame = window start, I_t <-> frame w + t, verified on the GT:
            corr 0.92 at shift 0 vs < 0.25 at +-1..3); only its W3 windows are evaluated (its other windows are
            near-duplicates). recall_any additionally credits SIMCODE with a match in any window w-2..w+2.
  targets   (event, window) with >= 1 on-frame, inside the crop (5..115 px), photon time-centroid in [1, 4) (our attribution
            rule); other GT events of the window are 'don't care'.
  matching  per window, one-to-one greedy within 1 px on the DETECTION position (ours x_prior, SIMCODE network x, y; the
            same for its NDI / merged estimates, whose rows are a subset of the network rows); unmatched outputs = FP.
  offset    a global (x, y) offset per method is pre-fitted (median error, matched targets 400-2000 label ph) and removed;
            bias is reported before removal.
  CRLB      no per-target CRLB on the movie: reference = gate CRLB_SF_multi (gt_sf_m, rms over axes) at matched photons,
            log-binned median vs GT window photons of the matching gate scene (rs05pm / rs25pm), log-log interpolated,
            extrapolated as N^-1/2 beyond its range.
  metrics   per GT-photon bin (label units; and SIMCODE units = label / kappa_sim, kappa_sim = median GT / SIMCODE-network
            photons on this movie): n, recall, RMS nm / axis, RMS / CRLB_ref, trimmed RMS (radial |e| / CRLB_ref <= 5),
            median |e| (radial) nm, bias, zcal = 1.4826 med |e / sigma_reported| (per axis), gross = |e| > 4 sigma_rep
            (radial) or |e| > 50 nm, photon ratio (median method / GT) and slope (LSQ through origin).
            Detection (targets >= 200 label ph): TP, FN, FP (unmatched outputs), FP per window, JI = TP / (TP + FP + FN).
  headline (2026-11-09, critic 32 #2) -- three fairness views, label-unit bins 200-700 / 700-2000 / 2000-4000 / > 4000 / 200-2000:
    paired          the same targets matched by both (SIMCODE spec vs ours; *_gt / mapGT specs vs ours with the true pattern
                    'ours_ref'): n, RMS/C, trim/C for both.
    matched recall  per bin n = SIMCODE's matched targets; ours = its top-n matched targets of the bin ranked by the
                    GT-free score sigma_cal * sqrt(max(q_dof, 0.3)) (sigma_cal = hypot(sx_cal, sy_cal); lower = better),
                    i.e. ours filtered to SIMCODE's recall. Also ours unfiltered.
    matched FP      ours ranked by detection p: recall (targets >= 200 label ph) of the largest p-prefix whose FP fraction
                    <= SIMCODE-net's, and our FP fraction at SIMCODE-net's recall; for SIMCODE's filtered (NDI / merged)
                    outputs (FP ~ 0): our recall of the largest p-prefix with FP count <= SIMCODE's ('n/a' if none).
  caveat    SIMCODE's NDI uses the paper's FIXED depth 0.9 (fix_depth); its own estimated depths are 0.79-0.83 (0.5 nM
            movies) / 0.53-0.60 (2.5 nM) -- part of its NDI error is this depth mismatch (as in the paper processing).
usage: python compare.py <sim_dir> <gate_scene> <out_prefix> name=kind:path [...]   kind = ours (run dir, tag est/ref:
       'ours_est:<dir>') | sc_net / sc_ndi / sc_mrg (run_simcode npz)  same protocol and code; the only change is where
crlb_ref() finds the gate scenes (gate npz with the GT CRLB column gt_sf_m, any front-end tag; the CRLB does not depend on
the method): <gate_scene> may be a glob ('.../rs05pm__s30[123]_*.npz'); a bare scene name is looked up as
$SIMCODE_GATE_DIR/<scene>__s30[123]<tag>.npz with tag $SIMCODE_GATE_TAG (default '_hd_sc5nb15pm', the dev tag; the
analysis/gate/run_gate.sh tag is '_sc5nb_b10'), SIMCODE_GATE_DIR default <simcode out>/gate.
Also the library of analysis/simreal/compare_ci_r.py, analysis/hd_compare/score.py, analysis/psfshape/evalgt.py and
analysis/evalproto/{valid,filt}.py.
"""
import glob, json, os, sys
import numpy as np

PX = 108.33
BINS = [(200, 400), (400, 700), (700, 1200), (1200, 2000), (2000, 4000), (4000, 1e9), (200, 2000), (200, 1e9)]
CROP = (5, 5, 110)


def gt_windows(G, wst):
    x, y, br, t0, t1, ea, es, patt = (G[k] for k in ('x', 'y', 'bright', 't0', 't1', 'ea', 'es', 'patt'))
    rows = []
    for w in wst:
        f = np.arange(w, w + 6)
        a = np.nonzero((t1 > w) & (t0 < w + 6))[0]
        if len(a) == 0: continue
        onf = np.clip(np.minimum(t1[a, None], f[None] + 1) - np.maximum(t0[a, None], f[None]), 0, 1)
        pt = patt[f]
        il = pt[None, :, 4] * (1 + pt[None, :, 3] * np.cos(pt[None, :, 0] * x[a, None] + pt[None, :, 1] * y[a, None] - pt[None, :, 2]))
        s = f % 6
        ph = br[a, None] * il * onf * ea[a][:, s // 3] * es[a][:, s]
        P = ph.sum(1); tc = (ph * np.arange(6)).sum(1) / np.maximum(P, 1e-6)
        ins = (x[a] >= CROP[0]) & (x[a] < CROP[0] + CROP[2]) & (y[a] >= CROP[1]) & (y[a] < CROP[1] + CROP[2])
        tgt = ins & (tc >= 1.0) & (tc < 4.0) & (onf.sum(1) >= 1)
        rows.append(np.c_[np.full(len(a), w), a, x[a], y[a], P, tgt])
    return np.concatenate(rows)


def match(GW, wo, xo, yo):
    """greedy one-to-one within 1 px per window; returns gi_of [n_out] (-1 unmatched)"""
    gi_of = np.full(len(xo), -1)
    order_o = np.argsort(wo, kind='stable'); wo_s = wo[order_o]
    order_g = np.argsort(GW[:, 0], kind='stable'); wg_s = GW[order_g, 0]
    for w in np.unique(wo):
        io = order_o[np.searchsorted(wo_s, w):np.searchsorted(wo_s, w, 'right')]
        ig = order_g[np.searchsorted(wg_s, w):np.searchsorted(wg_s, w, 'right')]
        if len(io) == 0 or len(ig) == 0: continue
        D = np.hypot(xo[io, None] - GW[ig, 2][None], yo[io, None] - GW[ig, 3][None])
        pr = np.argwhere(D < 1.0); pr = pr[np.argsort(D[pr[:, 0], pr[:, 1]])]
        uo, ug = set(), set()
        for i, j in pr:
            if i in uo or j in ug: continue
            uo.add(i); ug.add(j); gi_of[io[i]] = ig[j]
    return gi_of


def crlb_ref(scene):
    if '/' in scene or '*' in scene:
        fs = sorted(glob.glob(scene))
    else:
        gd = os.environ.get('SIMCODE_GATE_DIR')
        if not gd:
            from simcode import config
            gd = config.out_dir('gate', make=False)
        fs = sorted(glob.glob(f'{gd}/{scene}__s30[123]{(os.environ.get("SIMCODE_GATE_TAG") or "_hd_sc5nb15pm")}.npz'))
    if not fs: raise FileNotFoundError(f'crlb_ref: no gate npz for {scene!r} (set SIMCODE_GATE_DIR / SIMCODE_GATE_TAG or pass a glob)')
    ph, c = [], []
    for f in fs:
        d = np.load(f); m = (d['gt_target'] > 0) & np.isfinite(d['gt_sf_m']).all(1)
        ph.append(d['gt_phot'][m]); c.append(np.sqrt((d['gt_sf_m'][m] ** 2).mean(1)))
    ph, c = np.concatenate(ph), np.concatenate(c)
    e = np.logspace(np.log10(150), np.log10(8000), 25)
    lx, ly = [], []
    for a, b in zip(e[:-1], e[1:]):
        m = (ph >= a) & (ph < b)
        if m.sum() >= 30: lx.append(np.log(np.median(ph[m]))); ly.append(np.log(np.median(c[m])))
    lx, ly = np.array(lx), np.array(ly)
    def f(p):
        lp = np.log(np.maximum(p, 1.0))
        out = np.interp(lp, lx, ly)
        out = np.where(lp > lx[-1], ly[-1] - 0.5 * (lp - lx[-1]), out)
        out = np.where(lp < lx[0], ly[0] - 0.5 * (lp - lx[0]), out)
        return np.exp(out)
    return f, dict(n=int(len(ph)), ph_range=[float(np.exp(lx[0])), float(np.exp(lx[-1]))])


def load_method(kind, path, NF):
    """-> dict(w, xd, yd, x, y, sx, sy, phot) on the W3 windows"""
    if kind.startswith('ours'):
        tag = kind.split('_')[1]
        Q = {}
        for f in sorted(glob.glob(f'{path}/pass2_{tag}_*.npz')):
            d = np.load(f)
            for k in d.files:
                if d[k].ndim: Q.setdefault(k, []).append(d[k])
        Q = {k: np.concatenate(v) for k, v in Q.items()}
        ok = np.isfinite(Q['x_hy0'])
        return dict(w=Q['wstart'][ok].astype(int), xd=Q['x_prior'][ok], yd=Q['y_prior'][ok], x=Q['x_hy0'][ok], y=Q['y_hy0'][ok],
                    sx=Q['sx_cal'][ok], sy=Q['sy_cal'][ok], phot=Q['phot'][ok], q_dof=Q['q_dof'][ok], p=Q['p'][ok])
    R = np.load(path)
    fr = R['frame'].astype(int)
    if kind == 'sc_net':
        rows = np.arange(len(fr)); x, y, sx, sy, ph = R['x'], R['y'], R['sx'], R['sy'], R['photons']
    else:
        rows = R['ndi_idx']; p = 'ndi' if kind == 'sc_ndi' else 'mrg'
        x, y, sx, sy = R[f'{p}_x'], R[f'{p}_y'], R[f'{p}_sx'], R[f'{p}_sy']
        ph = R['ndi_photons'] if kind == 'sc_ndi' else R['photons'][rows]
    D = dict(w=fr[rows], xd=R['x'][rows], yd=R['y'][rows], x=x, y=y, sx=sx, sy=sy, phot=ph)
    return D


def evaluate(G, M, GW, crf, kappa, simcode_like):
    NF = int(G['nf'])
    w3 = (M['w'] % 3 == 0) & (M['w'] <= NF - 6)
    Mw = {k: v[w3] for k, v in M.items()}
    gi = match(GW, Mw['w'], Mw['xd'], Mw['yd'])
    tg = GW[:, 5] > 0
    mt = gi >= 0
    mtg = mt & tg[np.maximum(gi, 0)]
    P_all = GW[:, 4]
    rec = np.zeros(len(GW), bool); rec[gi[mtg]] = True
    rec_any = rec.copy()
    if simcode_like:     # credit any window start w-2..w+2 (SIMCODE's stride-1 output)
        for sh in (-2, -1, 1, 2):
            m2 = ((M['w'] + sh) % 3 == 0) & (M['w'] + sh >= 0)
            if not m2.any(): continue
            idx = np.nonzero(tg & ~rec_any)[0]
            g2 = match(GW[idx], M['w'][m2] + sh, M['xd'][m2], M['yd'][m2])
            rec_any[idx[g2[g2 >= 0]]] = True
    g = gi[mtg]
    e = np.c_[Mw['x'][mtg] - GW[g, 2], Mw['y'][mtg] - GW[g, 3]]
    s = np.c_[Mw['sx'][mtg], Mw['sy'][mtg]]
    P = P_all[g]; phm = Mw['phot'][mtg]
    sel = (P >= 400) & (P < 2000)
    off = np.median(e[sel], 0) if sel.sum() > 20 else np.zeros(2)
    bias_raw = e.copy()
    e = e - off
    C = crf(P)
    out = dict(offset_px=off.tolist(), n_out=int(len(Mw['w'])), n_windows=int(len(np.unique(GW[:, 0]))))
    rs = (np.hypot(Mw['sx'], Mw['sy']) * np.sqrt(np.maximum(Mw['q_dof'], 0.3))) if 'q_dof' in Mw else np.zeros(len(Mw['w']))
    dp = Mw['p'] if 'p' in Mw else np.ones(len(Mw['w']))
    t200o = mt & tg[np.maximum(gi, 0)] & (P_all[np.maximum(gi, 0)] >= 200)
    out['_rows'] = dict(g=g, e=e, s=s, C=C, P=P, rank=rs[mtg], det_p=dp, det_tp=t200o, det_fp=~mt,
                        n_t200=int((tg & (P_all >= 200)).sum()))
    # detection (targets >= 200 label photons; non-target matches = don't care; FP = unmatched)
    t200 = tg & (P_all >= 200)
    m_t200 = mtg & (P_all[np.maximum(gi, 0)] >= 200)
    TP = int(rec[t200].sum()); FN = int(t200.sum() - TP); FP = int((~mt).sum())
    out['det'] = dict(TP=TP, FN=FN, FP=FP, recall=TP / max(TP + FN, 1), recall_any=float(rec_any[t200].mean()),
                      FP_per_window=FP / max(out['n_windows'], 1), FP_frac=FP / max(TP + FP, 1), JI=TP / max(TP + FP + FN, 1),
                      dontcare_matches=int((mt & ~mtg).sum()))
    def stats(m):
        if m.sum() < 15: return None
        ee, ss, cc = e[m], s[m], C[m]
        er = np.hypot(*ee.T); zc = er / (np.sqrt(2) * cc)
        r = dict(n=int(m.sum()), rms_nm=float(np.sqrt(np.mean(ee ** 2)) * PX), crlb_nm=float(np.sqrt(np.mean(cc ** 2)) * PX))
        r['rms_over_crlb'] = r['rms_nm'] / r['crlb_nm']
        k = (er / cc) <= 5       # radial |e| / per-axis CRLB_ref <= 5 (as gate_report_xdet: hypot(e / c) <= 5)
        r['trim_over_crlb'] = float(np.sqrt(np.mean(ee[k] ** 2)) / np.sqrt(np.mean(cc[k] ** 2)))
        r['med_abs_nm'] = float(np.median(er) * PX)
        r['bias_nm'] = (np.mean(bias_raw[m], 0) * PX).tolist()
        r['zcal'] = float(1.4826 * np.median(np.abs(ee / ss)))
        r['gross_pct'] = float(100 * np.mean((np.hypot(*(ee / ss).T) > 4) | (er * PX > 50)))
        r['gross4s_pct'] = float(100 * np.mean(np.hypot(*(ee / ss).T) > 4))
        r['gross50_pct'] = float(100 * np.mean(er * PX > 50))
        r['phot_ratio'] = float(np.median(phm[m] / P[m]))
        r['phot_slope'] = float((phm[m] * P[m]).sum() / (P[m] ** 2).sum())
        return r
    def rec_bin(lo, hi):
        mm = tg & (P_all >= lo) & (P_all < hi)
        return float(rec[mm].mean()) if mm.sum() else np.nan, float(rec_any[mm].mean()) if mm.sum() else np.nan, int(mm.sum())
    for unit, kap in (('label', 1.0), ('simcode', kappa)):
        B = {}
        for lo, hi in (BINS + [b_ for b_ in HBINS if b_ not in BINS] if unit == 'label' else BINS):
            r = stats((P >= lo * kap) & (P < hi * kap))
            rc, ra, nt = rec_bin(lo * kap, hi * kap)
            B[f'{lo:.0f}-{hi:.0f}'] = dict(r or {}, recall=rc, recall_any=ra, n_targets=nt)
        out[unit] = B
    return out


HBINS = [(200, 700), (700, 2000), (2000, 4000), (4000, 1e9), (200, 2000)]


def _rc(R_, ii):
    """RMS/C and trim/C (radial |e| / C <= 5) over rows ii"""
    ee, cc = R_['e'][ii], R_['C'][ii]
    if len(ee) < 15: return np.nan, np.nan
    k = np.hypot(*ee.T) / cc <= 5
    return (float(np.sqrt(np.mean(ee ** 2)) / np.sqrt(np.mean(cc ** 2))), float(np.sqrt(np.mean(ee[k] ** 2)) / np.sqrt(np.mean(cc[k] ** 2))))


def headline(REP, Ms, PATHS, ours, ours_ref):
    """paired / matched-recall / matched-FP headline tables (critic 32 #2); fills REP[name]['headline'], returns text lines"""
    L = ['\n== HEADLINE (critic 32 #2): paired / matched recall / matched FP; label bins; ours = ' + (ours[0] if ours else '-') +
         ' (ours_ref = ' + (ours_ref[0] if ours_ref else '-') + ' for true-pattern SIMCODE specs)']
    if not ours: return L
    bn = lambda lo, hi: f'{lo:.0f}-{hi:.0f}' if hi < 1e8 else f'>{lo:.0f}'
    L.append(f'{"SIMCODE spec":14s} {"vs":14s} {"bin":>9s} | paired: {"n":>6s} {"SC rms":>6s} {"ours":>6s} {"SC trm":>6s} {"ours":>6s} | '
             f'matched-recall: {"n":>6s} {"SC rms":>6s} {"ours":>6s} {"SC trm":>6s} {"ours":>6s} {"unf":>6s} {"rec SC":>6s} {"ours":>6s}')
    for name, (kind, _) in Ms.items():
        if not kind.startswith('sc'): continue
        o0 = ours_ref[0] if ('_gt' in PATHS[name] and ours_ref) else ours[0]
        A, B_ = REP[o0]['_rows'], REP[name]['_rows']
        com, ia, ib = np.intersect1d(A['g'], B_['g'], return_indices=True)
        H = {}
        for lo, hi in HBINS:
            b = bn(lo, hi); h = {}
            m = (A['P'][ia] >= lo) & (A['P'][ia] < hi)
            pa_s, pa_o = _rc(B_, ib[m]), _rc(A, ia[m])
            h['paired'] = dict(n=int(m.sum()), sc_rms=pa_s[0], ours_rms=pa_o[0], sc_trim=pa_s[1], ours_trim=pa_o[1])
            ms = np.nonzero((B_['P'] >= lo) & (B_['P'] < hi))[0]; mo = np.nonzero((A['P'] >= lo) & (A['P'] < hi))[0]
            n = len(ms)
            q = REP[name]['label'].get(f'{lo:.0f}-{hi:.0f}', {}); qo = REP[o0]['label'].get(f'{lo:.0f}-{hi:.0f}', {})
            if n >= 15 and len(mo) >= n:
                keep = mo[np.argsort(A['rank'][mo], kind='stable')[:n]]
                sr, orr, ou = _rc(B_, ms), _rc(A, keep), _rc(A, mo)
                h['matched_recall'] = dict(n=n, sc_rms=sr[0], ours_rms=orr[0], sc_trim=sr[1], ours_trim=orr[1], ours_unf_rms=ou[0],
                                           ours_unf_trim=ou[1], sc_recall=q.get('recall', np.nan), ours_recall=qo.get('recall', np.nan))
            else:
                h['matched_recall'] = dict(n=n)
            H[b] = h
            p, r = h['paired'], h['matched_recall']
            f = lambda v: f'{v:6.2f}' if np.isfinite(v) else f'{"-":>6s}'
            line = f'{name:14s} {o0:14s} {b:>9s} | paired: {p["n"]:6d} {f(p["sc_rms"])} {f(p["ours_rms"])} {f(p["sc_trim"])} {f(p["ours_trim"])} | '
            if 'sc_rms' in r:
                line += (f'matched-recall: {r["n"]:6d} {f(r["sc_rms"])} {f(r["ours_rms"])} {f(r["sc_trim"])} {f(r["ours_trim"])} '
                         f'{f(r["ours_unf_rms"])} {f(r["sc_recall"])} {f(r["ours_recall"])}')
            else:
                line += f'matched-recall: {r["n"]:6d} (too few)'
            L.append(line)
        REP[name]['headline'] = H
        L.append('')
    # matched FP: ours ranked by detection p
    L.append('== matched FP (targets >= 200 label ph; ours ranked by detection p; FP = unmatched W3 outputs)')
    for o0 in ours + ours_ref:
        R_ = REP[o0]['_rows']; o = np.argsort(-R_['det_p'], kind='stable')
        ctp = np.cumsum(R_['det_tp'][o]); cfp = np.cumsum(R_['det_fp'][o]); nt = R_['n_t200']
        fpf = cfp / np.maximum(ctp + cfp, 1)
        REP[o0]['matched_fp'] = {}
        for name, (kind, _) in Ms.items():
            if not kind.startswith('sc'): continue
            d = REP[name]['det']; S_ = REP[name]['_rows']
            sc_rec, sc_fpf, sc_fp = d['recall'], d['FP_frac'], d['FP']
            if kind == 'sc_net':
                ok = np.nonzero(fpf <= sc_fpf)[0]
                i = ok[-1] if len(ok) else None
                j = min(np.searchsorted(ctp, d['TP']), len(fpf) - 1)
                res = dict(sc_recall=sc_rec, sc_recall_any=d['recall_any'], sc_fp_frac=sc_fpf,
                           ours_recall_at_sc_fpfrac=float(ctp[i] / nt) if i is not None else None,
                           p_thr=float(R_['det_p'][o][i]) if i is not None else None, ours_fpfrac_at_sc_recall=float(fpf[j]),
                           p_thr_rec=float(R_['det_p'][o][j]))
                L.append(f'{o0:14s} @ {name:14s}: SC recall {sc_rec:.3f} (any {d["recall_any"]:.3f}) FP frac {sc_fpf:.4f} | ours recall at SC FP frac '
                         + (f'{res["ours_recall_at_sc_fpfrac"]:.3f} (p > {res["p_thr"]:.2f})' if i is not None else 'n/a')
                         + f' | ours FP frac at SC recall {res["ours_fpfrac_at_sc_recall"]:.4f} (p > {res["p_thr_rec"]:.2f}) | ours all: recall '
                         f'{REP[o0]["det"]["recall"]:.3f} FP frac {REP[o0]["det"]["FP_frac"]:.4f}')
            else:
                ok = np.nonzero(cfp <= sc_fp)[0]
                i = ok[-1] if len(ok) else None
                res = dict(sc_recall=sc_rec, sc_recall_any=d['recall_any'], sc_fp=sc_fp,
                           ours_recall_at_sc_fp=float(ctp[i] / nt) if i is not None else None,
                           ours_fp_frac_there=float(fpf[i]) if i is not None else None)
                L.append(f'{o0:14s} @ {name:14s}: SC recall {sc_rec:.3f} (any {d["recall_any"]:.3f}) FP {sc_fp} | ours recall at FP <= {sc_fp}: '
                         + (f'{res["ours_recall_at_sc_fp"]:.3f} (FP frac {res["ours_fp_frac_there"]:.4f}, p > {float(R_["det_p"][o][i]):.3f})' if i is not None else 'n/a'))
            REP[o0]['matched_fp'][name] = res
    return L


def main():
    sd, scene, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    G = dict(np.load(f'{sd}/sim_gt.npz'))
    NF = int(G['nf'])
    wst = np.arange(0, NF - 5, 3)
    GW = gt_windows(G, wst)
    crf, cinfo = crlb_ref(scene)
    specs = [a.split('=', 1) for a in sys.argv[4:]]
    # kappa_sim from the SIMCODE network output (any sc_* spec)
    kappa = np.nan
    REP = dict(scene_crlb=scene, crlb_ref=cinfo, sim=sd, n_targets=int((GW[:, 5] > 0).sum()))
    Ms = {}
    for name, ks in specs:
        kind, path = ks.split(':', 1)
        Ms[name] = (kind, load_method(kind, path, NF))
    for name, (kind, M) in Ms.items():
        if kind == 'sc_net':
            w3 = M['w'] % 3 == 0
            gi = match(GW, M['w'][w3], M['xd'][w3], M['yd'][w3]); m = gi >= 0
            P = GW[gi[m], 4]; s = (P >= 300) & (P < 3000)
            kappa = float(np.median(P[s] / M['phot'][w3][m][s])); break
    REP['kappa_sim'] = kappa
    kap = kappa if np.isfinite(kappa) else 2.0
    for name, (kind, M) in Ms.items():
        REP[name] = evaluate(G, M, GW, crf, kap, kind.startswith('sc'))
        REP[name]['kind'] = kind
    # paired: each SIMCODE output vs the first 'ours' spec on the targets matched by both
    ours = [n for n, (k, _) in Ms.items() if k == 'ours_est']
    ours_ref = [n for n, (k, _) in Ms.items() if k == 'ours_ref']
    PATHS = {n: a.split(':', 1)[1] for n, a in specs}
    PL = []
    if ours:
        PL.append(f'\n== paired: same targets matched by both (SIMCODE output vs ours; true-pattern SIMCODE runs vs ours with the true pattern); label-unit bins')
        PL.append(f'{"pair":24s} {"bin":>10s} {"n":>6s} | {"rms/C":>6s} {"trim/C":>6s} {"med|e|":>6s} {"zcal":>5s} {"gross%":>6s} | ours: {"rms/C":>6s} {"trim/C":>6s} {"med|e|":>6s} {"zcal":>5s} {"gross%":>6s}')
        for name, (kind, _) in Ms.items():
            if not kind.startswith('sc'): continue
            o0 = ours_ref[0] if ('_gt' in PATHS[name] and ours_ref) else ours[0]
            A = REP[o0]['_rows']
            B_ = REP[name]['_rows']
            com, ia, ib = np.intersect1d(A['g'], B_['g'], return_indices=True)
            REP[name]['paired'] = {}
            for lo, hi in BINS:
                m = (A['P'][ia] >= lo) & (A['P'][ia] < hi)
                if m.sum() < 15: continue
                def st(R_, ii):
                    ee, ss, cc = R_['e'][ii][m], R_['s'][ii][m], R_['C'][ii][m]; er = np.hypot(*ee.T); k = er / cc <= 5
                    return dict(rms=float(np.sqrt(np.mean(ee ** 2)) / np.sqrt(np.mean(cc ** 2))),
                                trim=float(np.sqrt(np.mean(ee[k] ** 2)) / np.sqrt(np.mean(cc[k] ** 2))), med=float(np.median(er) * PX),
                                zcal=float(1.4826 * np.median(np.abs(ee / ss))),
                                gross=float(100 * np.mean((np.hypot(*(ee / ss).T) > 4) | (er * PX > 50))))
                a_, b_ = st(B_, ib), st(A, ia)
                REP[name]['paired'][f'{lo:.0f}-{hi:.0f}'] = dict(n=int(m.sum()), simcode=a_, ours=b_, ours_name=o0)
                PL.append(f'{name + " vs " + o0:24s} {f"{lo:.0f}-{hi:.0f}":>10s} {int(m.sum()):6d} | {a_["rms"]:6.2f} {a_["trim"]:6.2f} {a_["med"]:6.2f} {a_["zcal"]:5.2f} {a_["gross"]:6.2f} | '
                          f'      {b_["rms"]:6.2f} {b_["trim"]:6.2f} {b_["med"]:6.2f} {b_["zcal"]:5.2f} {b_["gross"]:6.2f}')
    HL = headline(REP, Ms, PATHS, ours, ours_ref)
    for name in Ms: REP[name].pop('_rows', None)
    json.dump(REP, open(outp + '.json', 'w'), indent=1)
    L = [f'# {sd}  (CRLB ref: gate {scene}; kappa_sim = GT label / SIMCODE-net photons = {kappa:.3f}; targets {REP["n_targets"]})']
    for unit in ('label', 'simcode'):
        L.append(f'\n== photon bins in {unit} units' + (f' (label edges = x {kap:.2f})' if unit == 'simcode' else ''))
        L.append(f'{"method":14s} {"bin":>10s} {"n":>6s} {"recall":>6s} {"rec_any":>7s} {"rms nm":>7s} {"CRLB":>6s} {"rms/C":>6s} {"trim/C":>6s} '
                 f'{"med|e|":>6s} {"bias x,y nm":>13s} {"zcal":>5s} {"gross%":>6s} {"(4s/50)":>11s} {"ph/GT":>6s} {"slope":>6s}')
        for lo, hi in BINS:
            b = f'{lo:.0f}-{hi:.0f}'
            for name in Ms:
                q = REP[name][unit][b]
                if 'n' not in q:
                    L.append(f'{name:14s} {b:>10s} {"-":>6s} {q["recall"]:6.3f} {q["recall_any"]:7.3f}'); continue
                L.append(f'{name:14s} {b:>10s} {q["n"]:6d} {q["recall"]:6.3f} {q["recall_any"]:7.3f} {q["rms_nm"]:7.2f} {q["crlb_nm"]:6.2f} '
                         f'{q["rms_over_crlb"]:6.2f} {q["trim_over_crlb"]:6.2f} {q["med_abs_nm"]:6.2f} {q["bias_nm"][0]:+6.2f},{q["bias_nm"][1]:+6.2f} '
                         f'{q["zcal"]:5.2f} {q["gross_pct"]:6.2f} ({q["gross4s_pct"]:4.1f}/{q["gross50_pct"]:4.1f}) {q["phot_ratio"]:6.3f} {q["phot_slope"]:6.3f}')
            L.append('')
    L.append('== detection (targets >= 200 label ph; FP = unmatched outputs on the W3 windows)')
    for name in Ms:
        d = REP[name]['det']
        L.append(f'{name:14s} TP {d["TP"]:6d} FN {d["FN"]:6d} FP {d["FP"]:6d} recall {d["recall"]:.3f} (any {d["recall_any"]:.3f}) '
                 f'FP/window {d["FP_per_window"]:.2f} FP frac {d["FP_frac"]:.3f} JI {d["JI"]:.3f}; outputs {REP[name]["n_out"]}, '
                 f'dontcare matches {d["dontcare_matches"]}, offset {np.array(REP[name]["offset_px"]) * PX} nm')
    L += PL
    L += HL
    open(outp + '.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
