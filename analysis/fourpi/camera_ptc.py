"""Camera gain of the 4Pi-SIMFLUX movies by photon transfer on the raw movie frames (no dark / flat recordings exist).

Per movie: K consecutive raw frames (default 1200 = 200 six-frame windows) of one file (MT 005, NPC 010), raw sensor
coordinates (rows 1-199; row 0 carries embedded header pixels).

  offset   dark strip left of channel 0 (raw columns 20-200): per-pixel temporal mean; global offset = their mean;
           offset map = global offset + column FPN (as analysis/fourpi/camera.py: column means of L - median_filter(L, 1x15)).
  read     dark strip: pattern-matched lag-6 difference variance / 2 per pixel (median, mean -> rn^2).
  gain     photon transfer with pattern-matched frame pairs: frame (w, s) and (w + 1, s) (same SIM pattern step s, 6 frames
           = 33 ms apart) have the same expected signal when the pixel's signal is stable over the windows w-1 .. w+2
           (background, or a DNA-PAINT emitter that stays bound). Then var(d) / 2 = G * L + rn^2 with d = a(w+1, s) - a(w, s),
           L = pair mean - offset, G = ADU per photo-electron. Stability is tested on the 6-frame window sums of w-1 and
           w+2 and on S_w + S_(w+1) only (the sum of the pair is independent of its difference for equal variances, so the
           selection does not bias var(d)); pairs are binned by the reference level of frames (w-1, s), (w+2, s) (noise
           independent of the pair) with the bin mean of L as abscissa, mean of d^2/2 per bin, weighted fit var = rn^2 + G L
           (+ c L^2 to test for emitter-brightness fluctuations). Uncertainty: the fit repeated on 5 disjoint time blocks.
  bg-PTC   classical background photon transfer for comparison: per lit pixel, its clipped temporal level vs the
           lag-6 difference variance, binned across pixels (small range: 0.4-1.2 e/px/frame).
  Result (2026-11-15): pooled MT 4.26 / NPC 4.61 ADU/e; per channel (8-80 ADU) 3.4-5.0 ADU/e with the same channel pattern in both
           movies (a channel systematic of the non-shot variance, not camera gain) -> G = 4.3 +- 0.4 ADU/e = 0.23 +- 0.02 e/ADU, i.e.
           the 4Pi-SIMFLUX code constant 4.35 ADU/e is confirmed. Read noise (dark strip) median 4.7 ADU, rms 5.0 ADU; offset 100.5 ADU.
usage: python camera_ptc.py [out_dir] --work <4Pi work folder> --data <raw 4Pi movie folder>  -> camera_ptc.json, camera_ptc.png
       (default out_dir <work>/camera_ptc)
"""
import os, sys, json, argparse
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.ndimage import median_filter
from dc import Dcimg, movie_files, X0, CW

G0 = 4.35                       # ADU/e prior, only used in the stability tolerance
SETS = (('MT', 5, 600), ('NPC', 10, 600))
XLIT = (55, 305)                # lit columns within each 360-px crop (processing xlim)


def lit_columns():
    m = np.zeros(2304, bool)
    for x0 in X0:
        m[x0 + XLIT[0]:x0 + XLIT[1]] = True          # mirroring does not matter for a column mask (55..305 ~ symmetric)
    return m


def ptc_pairs(a, off, rn2, cols, rel=(0.03, 0.05), nsig=2.5):
    """a [K, 200, 2304] float32 raw. Stable pattern-matched pairs (w, s) / (w + 1, s) in the given columns.
    Returns L (pair mean - offset), v (= d^2 / 2), R (reference level = mean of frames (w - 1, s) and (w + 2, s) - offset:
    independent of the pair's noise, used for BINNING so that the bin means of L and v are unbiased: E[v | bin] =
    G E[mu | bin] + rn^2 and E[L | bin] = E[mu | bin]; binning on L itself would select on L's noise and flatten the
    slope at low levels), and the window index."""
    K = a.shape[0] // 6 * 6
    b = (a[:K, 1:, :][..., cols] - off[1:, cols][None]).reshape(K // 6, 6, 199, -1)        # [W, 6, y, x]
    S = b.sum(1)                                                                            # window sums
    W = S.shape[0]
    Ls, vs, Rs, ws = [], [], [], []
    for w in range(1, W - 2):
        Sm = 0.5 * (S[w - 1] + S[w + 2])
        m = np.maximum(Sm / 6, 0)
        sdS = np.sqrt(6 * (G0 * m + rn2))
        ok = (np.abs(S[w - 1] - S[w + 2]) < nsig * np.sqrt(2) * sdS + rel[0] * np.abs(Sm)) & \
             (np.abs(S[w] + S[w + 1] - 2 * Sm) < nsig * 2 * sdS + rel[1] * np.abs(Sm))
        p, q = b[w][:, ok], b[w + 1][:, ok]                                                 # [6, n]
        r = 0.5 * (b[w - 1][:, ok] + b[w + 2][:, ok])
        Ls.append((0.5 * (p + q)).ravel()); vs.append((0.5 * (q - p) ** 2).ravel()); Rs.append(r.ravel())
        ws.append(np.full(p.size, w, np.int32))
    return np.concatenate(Ls), np.concatenate(vs), np.concatenate(Rs), np.concatenate(ws)


def binned_ref(R, L, v, edges):
    """bins on the reference level R; per bin mean L (abscissa), mean v, its standard error, count"""
    i = np.digitize(R, edges) - 1; good = (i >= 0) & (i < len(edges) - 1); i = i[good]; L, v = L[good], v[good]
    nb = len(edges) - 1
    n = np.bincount(i, minlength=nb)
    with np.errstate(invalid='ignore', divide='ignore'):
        mL = np.bincount(i, L, nb) / n; mv = np.bincount(i, v, nb) / n
        se = np.sqrt(np.maximum(np.bincount(i, v * v, nb) / n - mv ** 2, 0) / n)
    return mL, mv, se, n


def binned(L, v, edges):
    i = np.digitize(L, edges) - 1
    n = np.bincount(i[(i >= 0) & (i < len(edges) - 1)], minlength=len(edges) - 1)
    good = (i >= 0) & (i < len(edges) - 1)
    sL = np.bincount(i[good], L[good], len(edges) - 1); sv = np.bincount(i[good], v[good], len(edges) - 1)
    sv2 = np.bincount(i[good], v[good] ** 2, len(edges) - 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        mL, mv = sL / n, sv / n
        se = np.sqrt(np.maximum(sv2 / n - mv ** 2, 0) / n)
    return mL, mv, se, n


def fit(mL, mv, se, n, Lmax, quad=False, nmin=200, Lmin=-np.inf):
    s = (n >= nmin) & np.isfinite(mv) & (mL <= Lmax) & (mL >= Lmin)
    A = np.c_[np.ones(s.sum()), mL[s]] if not quad else np.c_[np.ones(s.sum()), mL[s], mL[s] ** 2]
    wgt = 1 / se[s]
    coef, *_ = np.linalg.lstsq(A * wgt[:, None], mv[s] * wgt, rcond=None)
    chi2 = float(np.sum(((A @ coef - mv[s]) * wgt) ** 2) / max(s.sum() - A.shape[1], 1))
    return coef, chi2


def run(out_dir, work, data, K=1200):
    """work: the 4Pi work folder, data: the folder of the raw 4Pi movies (the movie file lists)"""
    os.makedirs(out_dir, exist_ok=True)
    res, curves = {}, {}
    cols = lit_columns()
    edges = np.r_[np.arange(-4, 20, 1.0), np.geomspace(20, 600, 30)]
    for cell, fi, f0 in SETS:
        fn = [f for f in movie_files(cell, work, data) if f.endswith(f'_{fi:03d}.dcimg')][0]
        cache = f'/tmp/camera_ptc_raw_{cell}_{fi:03d}_{f0}_{K}.npy'
        if os.path.exists(cache): a = np.load(cache).astype(np.float32)
        else:
            raw = Dcimg(fn).raw(f0, K); np.save(cache, raw); a = raw.astype(np.float32); del raw
        # offset: dark strip + column FPN (camera.py recipe, clipped temporal mean)
        med = np.median(a, 0); sd = 1.4826 * np.median(np.abs(a - med), 0)
        ok = a < med + 3 * sd
        Lc = np.where(ok, a, 0).sum(0) / ok.sum(0); Lc[0] = Lc[1]
        dark = (slice(1, 200), slice(20, 200))
        off0 = float(Lc[dark].mean())
        col = (Lc - median_filter(Lc, size=(1, 15), mode='nearest'))[1:].mean(0)
        off = off0 + np.broadcast_to(col[None], Lc.shape)
        # read noise in the dark strip: same-step lag-6 differences
        dd = a[6:, 1:, 20:200] - a[:-6, 1:, 20:200]
        rn2_pix = dd.var(0) / 2
        rn2 = float(np.median(rn2_pix)); rn2_mean = float(rn2_pix.mean())
        dark_pix_off_sd = float(Lc[dark].std())
        # bead-free background PTC across lit pixels (classical; small range)
        d6 = a[6:, 1:, :][..., cols] - a[:-6, 1:, :][..., cols]
        vb = (1.4826 * np.median(np.abs(d6), 0)) ** 2 / 2
        Lb = (Lc - off)[1:, cols]
        qs = np.quantile(Lb, np.linspace(0.02, 0.98, 16))
        bx = np.array([np.median(Lb[(Lb >= q0) & (Lb < q1)]) for q0, q1 in zip(qs[:-1], qs[1:])])
        by = np.array([np.median(vb[(Lb >= q0) & (Lb < q1)]) for q0, q1 in zip(qs[:-1], qs[1:])])
        sl_b, ic_b = np.polyfit(bx, by, 1)
        # pattern-matched pair PTC
        L, v, R, wi = ptc_pairs(a, off, rn2, cols)
        mL, mv, se, n = binned_ref(R, L, v, edges)
        r = dict(file=os.path.basename(fn), frames=[f0, f0 + K], offset_dark_strip_adu=off0, dark_strip_pixel_offset_sd_adu=dark_pix_off_sd,
                 column_fpn_sd_adu=float(col.std()), rn2_dark_median_adu2=rn2, rn2_dark_mean_adu2=rn2_mean,
                 n_pairs=int(L.size), frac_lit_pairs_kept=float(L.size / (6 * (K // 6 - 3) * 199 * cols.sum())),
                 bgptc=dict(levels_adu=[float(bx.min()), float(bx.max())], slope_adu_per_e=float(sl_b), gain_e_per_adu=float(1 / sl_b), intercept_adu2=float(ic_b)))
        for Lmax in (50, 150, 400):
            c, chi = fit(mL, mv, se, n, Lmax)
            r[f'lin_Lmax{Lmax}'] = dict(G_adu_per_e=float(c[1]), gain_e_per_adu=float(1 / c[1]), rn2_adu2=float(c[0]), chi2_dof=chi)
        cq, chiq = fit(mL, mv, se, n, 400, quad=True)
        r['quad_Lmax400'] = dict(G_adu_per_e=float(cq[1]), gain_e_per_adu=float(1 / cq[1]), rn2_adu2=float(cq[0]), c_per_adu=float(cq[2]),
                                 excess_frac_at_200adu=float(cq[2] * 200 / cq[1]), chi2_dof=chiq)
        # time-block jackknife (5 disjoint blocks of windows)
        blk = np.array_split(np.arange(1, K // 6 - 2), 5)
        Gs = []
        for bb in blk:
            s = np.isin(wi, bb)
            c, _ = fit(*binned_ref(R[s], L[s], v[s], edges), 150, nmin=50)
            Gs.append(c[1])
        Gs = np.array(Gs)
        r['blocks_G_adu_per_e'] = Gs.tolist()
        r['G_block_mean'] = float(Gs.mean()); r['G_block_sem'] = float(Gs.std(ddof=1) / np.sqrt(len(Gs)))
        # per channel (raw column ranges)
        r['per_channel_G_Lmax150'] = {}; r['per_channel_G_L8_80'] = {}; r['per_channel_G_L8_80_block_sem'] = {}
        for ch, x0 in enumerate(X0):
            cm = np.zeros(2304, bool); cm[x0:x0 + CW] = True
            Lch, vch, Rch, wch = ptc_pairs(a, off, rn2, cols & cm)
            c, _ = fit(*binned_ref(Rch, Lch, vch, edges), 150)
            r['per_channel_G_Lmax150'][f'ch{ch}'] = float(c[1])
            c, _ = fit(*binned_ref(Rch, Lch, vch, edges), 80, Lmin=8)
            r['per_channel_G_L8_80'][f'ch{ch}'] = float(c[1])
            gb = [fit(*binned_ref(Rch[np.isin(wch, bb)], Lch[np.isin(wch, bb)], vch[np.isin(wch, bb)], edges), 80, nmin=50, Lmin=8)[0][1] for bb in blk]
            r['per_channel_G_L8_80_block_sem'][f'ch{ch}'] = float(np.std(gb, ddof=1) / np.sqrt(len(gb)))
        pc = np.array(list(r['per_channel_G_L8_80'].values()))
        r['G_channel_mean_L8_80'] = float(pc.mean()); r['G_channel_sd_L8_80'] = float(pc.std(ddof=1))
        s_ = n >= 200
        r['curve'] = dict(L=mL[s_].tolist(), v=mv[s_].tolist(), se=se[s_].tolist(), n=n[s_].tolist())
        res[cell] = r; curves[cell] = (mL, mv, se, n)
        print(cell, json.dumps(r, indent=1), flush=True)
        del a, d6
    json.dump(res, open(f'{out_dir}/camera_ptc.json', 'w'), indent=1)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, cell in zip(axs, res):
        mL, mv, se, n = curves[cell]; s = n >= 200
        ax.errorbar(mL[s], mv[s], se[s], fmt='o', ms=3)
        c = res[cell]['lin_Lmax150']
        xx = np.linspace(0, np.nanmax(mL[s]), 50); ax.plot(xx, c['rn2_adu2'] + c['G_adu_per_e'] * xx, '-', label=f"fit L<=150: {c['G_adu_per_e']:.2f} ADU/e = {c['gain_e_per_adu']:.3f} e/ADU")
        ax.plot(xx, res[cell]['rn2_dark_median_adu2'] + G0 * xx, ':', color='0.4', label='4.35 ADU/e (their code)')
        ax.set_xlabel('pair mean - offset (ADU)'); ax.set_ylabel('var(d)/2 (ADU^2)'); ax.set_title(f"{cell} {res[cell]['file']}"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f'{out_dir}/camera_ptc.png', dpi=120)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir', nargs='?')
    ap.add_argument('--work', required=True, help='4Pi work folder')
    ap.add_argument('--data', required=True, help='folder of the raw 4Pi movies')
    A = ap.parse_args()
    run(A.out_dir or f'{A.work}/camera_ptc', A.work, A.data)
