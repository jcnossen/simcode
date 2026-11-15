"""SI figures of the v2 pattern estimation on one run (default: 1809_2, simcode_v2_psfv2_20261115):

  si_phase_decouple  per-step pattern phases vs time = shared per-angle drift eps_a(t) + fixed per-step offsets psi_s
                     (the shared-drift model of simcode/processing/pattern_estimation.py), and the per-block test of it.
  si_pattern_plots   the preprint's layout (2 x 2: rows SIMFLUX / SIMCODE, columns 1809_2 / 2.5nM_2; 1 column while the
                     2.5 nM run is not done): normalized intensity I_k / sum_6 I vs pattern phase for pattern 0 (angle 0,
                     step 0), included (blue, the method's own filter) vs excluded (red), with the illumination pattern
                     (fig_pattern_plots, simflux_pattern_rows, simcode_pattern_rows; see their docstrings).
  si_pattern_plots_extended  SIMCODE only, both angles, binned means (fig_pattern_plots_extended).

Pattern (pattern_est.pt of the run; simcode/processing/pattern_estimation.py): frame t, step s = t % 6, angle a = s // 3,
    I_t = B rel_s (1 + m_a cos(k_a.r - psi_s - eps_a(t))),
k [2, 2] rad/px, phi = psi [6] rad, m [6] (depth per angle, repeated), logrel [6], dphi = eps [2, K] rad on knots every
knot_frames (150) frames (linear interpolation), mean removed (gauge in psi).

si_phase_decouple (CPU only):
  The pass-1 rows are loaded as the pipeline does (simcode.processing.common.load_pass1_rows: each window once, the cal's
  prior_off removed, 'out' = central half & inside the run's crop) and the clean emitter-windows are re-selected with
  pattern_estimation.select (same nn_min / min_phot as the run). Per time block of B frames (rows by window start) and per
  step s, the weighted regression of pattern_estimation.fit_steps (y = I_t / mean_6 I = r + U cos th + V sin th,
  th = k_a.r, weights min(photons, 4000), two passes with 4 x MAD outlier rejection) is solved with eps = 0, so it returns
  the TOTAL per-step phase phi_s(b) = psi_s + eps_a(b) of that block, estimated independently for every step and block.
  Shared-drift test: within each block and angle, the deviation of the 3 steps from a common shift,
      d_s(b) = (phi_s(b) - psi_s) - mean_{s' in a}(phi_s'(b) - psi_s'),
  must be pure estimation noise if the 3 steps share one drift. Its noise level is measured on the data: the rows of each
  block are split by site (3-px checkerboard of the position, so all windows of one emitter fall in the same half) into
  two halves, both halves are fitted, and the same statistic is computed on (phi_s^h1(b) - phi_s^h2(b)) / 2, which has the
  noise variance of the full-block estimate and no signal. Also reported: r_s(b) = phi_s(b) - psi_s - <eps_a>_b (residual
  w.r.t. the full model incl. the estimated drift; <eps_a>_b = eps_a at the rows' frames of step s in block b, averaged).
  Persistence: autocovariance of d_s(b) between blocks b and b + L (estimation noise is independent between blocks, a
  step-specific phase wander is not); the mean autocovariance over lags 2-10 estimates the persistent (non-noise) variance.
  Panel A uses 3000-frame blocks (readability), panel B and the numbers 150-frame blocks (= the eps knot spacing).

si_pattern_plots_extended:
  locs_p1.hdf5 of the run (pass-1 rows: per-frame network intensities I [n, 6] of pass 1 and the position of the explicit
  pattern solve with the estimated pattern; simcode.dataset.Dataset). Phase of frame j of a row: k_a.r - psi_s - eps_a(t),
  t = window start + j, wrapped to [0, 2 pi). Selection = Dataset.mask() (the paper's: score p >= 0.63, calibrated
  max(sigma_x, sigma_y) <= 12 nm, inside the crop); the comparison set = the other rows inside the crop. A random subsample of
  rows is plotted (all 3 frames of the angle for each row); binned means and the effective-depth fit use all rows of the set.
  Model curve: rel_s (1 + m_a cos(phase)) / sum_6 rel (the sum over the 3 steps of the other angle of the cosine terms is ~0
  for ~120 deg step spacing), with rel averaged over the 3 steps of the angle.

Outputs (PDF, SVG, PNG, <name>_caption.txt, <name>_values.json) in --outdir. The block fits are cached in
<outdir>/si_phase_decouple_blockfits.npz and the clean rows in <outdir>/si_phase_decouple_cleanrows.npz (delete to refit).

  PYTHONPATH=. CUDA_VISIBLE_DEVICES= python analysis/paper/si_pattern.py [--run <run dir>] [--outdir d]
      [--which all|decouple|pattern|pattern_ext] [--datasets auto|1809_2,2.5nM_2]
"""
import os, sys, json, math, argparse
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_common  # noqa: F401  (paper matplotlib settings)
import matplotlib.pyplot as plt

RUN = '/mnt/data/dmdsf/bn_dnapaint/1809/cover1/2/runs/simcode_v2_psfv2_20261115'
OUT = '/home/jc/simcode2/out/paper_figs_v2'
T6 = 6
STEP_COLORS = ['tab:blue', 'tab:orange', 'tab:green']
SEL_COLOR, REJ_COLOR = 'red', 'grey'     # SIMCODE red = selected rows


def load_pattern(run):
    import torch
    P = torch.load(f'{run}/pattern_est.pt', weights_only=False)
    K = P['k'].double().numpy(); psi = P['phi'].double().numpy(); m = P['m'].double().numpy()[[0, 3]]
    lr = P['logrel'].double().numpy(); rel = np.exp(lr - lr.mean())
    eps = P['dphi'].double().numpy(); kf = int(P['knot_frames'])
    return dict(K=K, psi=psi, m=m, rel=rel, eps=eps, kf=kf)


def eps_at(pat, f):
    from simcode.processing.pattern_estimation import _interp_eps
    return _interp_eps(pat['eps'], pat['kf'], np.asarray(f, np.float64))


def save(fig, outdir, name):
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(f'{outdir}/{name}.{ext}', dpi=200 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------------------------------- phase decoupling
def load_clean(run):
    """the clean emitter-windows of the run's pattern estimation (as simcode.run does it, CPU)"""
    import glob
    from simcode.processing import pattern_estimation as PE
    from simcode.processing.common import load_pass1_rows
    PE.NB_DEVICE = 'cpu'
    cfg = json.load(open(f'{run}/run_config.json'))
    chunks = sorted(int(os.path.basename(f)[6:-4]) for f in glob.glob(f'{run}/pass1_*.npz'))
    crop = tuple(float(v) for v in np.load(f'{run}/pass1_{chunks[0]}.npz')['crop'])
    po = json.load(open(cfg['settings']['cal'])).get('prior_off')
    R = load_pass1_rows(run, chunks, crop, po)
    nn_min, min_phot = cfg['fixed']['nn_min'], cfg['fixed']['min_phot']
    D = PE.select(R, nn_min=nn_min, min_phot=min_phot)
    sel = D['clean']
    C = {k: D[k][sel] for k in ('w', 'x', 'y', 'I', 'phot')}
    return C, dict(n_rows=int(len(D['w'])), n_on=int(D['on'].sum()), n_clean=int(sel.sum()), crop=crop, prior_off=po,
                   nn_min=nn_min, min_phot=min_phot, chunks=len(chunks))


def _wls(Y, X, wt):
    """pattern_estimation.fit_steps' per-step regression (2 passes, 4 x MAD rejection) -> (U, V) / r -> phase"""
    ww = wt
    for _ in range(2):
        sw = np.sqrt(ww)
        c, *_ = np.linalg.lstsq(X * sw[:, None], Y * sw, rcond=None)
        res = Y - X @ c
        rs = 1.4826 * np.median(np.abs(res))
        ww = np.where(np.abs(res) < 4 * rs, wt, 0.0)
    return math.atan2(c[2], c[1])


def block_phases(C, pat, block, seed=0):
    """per block and step: total phase phi (eps = 0) of all rows and of two site halves; <eps> per block/step; n rows"""
    K = pat['K']
    w = C['w']; f = w[:, None] + np.arange(T6)[None]; st = f % 6; ang = st // 3
    th = K[ang, 0] * C['x'][:, None] + K[ang, 1] * C['y'][:, None]
    yn = C['I'] / C['I'].mean(1, keepdims=True)
    wt = np.clip(C['phot'], 0, 4000)
    E = eps_at(pat, f); eps = np.where(ang == 0, E[0], E[1])
    b = (w // block).astype(int); nb = int(b.max()) + 1
    # halves by site (3-px checkerboard): all windows of one emitter (overlapping windows, long binding events, shared
    # position error) fall in the same half, so the two halves are statistically independent
    half = ((np.floor(C['x'] / 3) + np.floor(C['y'] / 3)) % 2).astype(int)
    phi = np.full((3, nb, 6), np.nan); emean = np.full((nb, 6), np.nan); n = np.zeros(nb, int)
    order = np.argsort(b, kind='stable'); bounds = np.searchsorted(b[order], np.arange(nb + 1))
    for i in range(nb):
        idx = order[bounds[i]:bounds[i + 1]]; n[i] = len(idx)
        if len(idx) < 60: continue
        for s in range(6):
            col = np.argmax(st[idx] == s, 1)
            Y = yn[idx, col]; t_ = th[idx, col]
            X = np.stack([np.ones_like(t_), np.cos(t_), np.sin(t_)], 1)
            emean[i, s] = eps[idx, col].mean()
            for h, m_ in enumerate((np.ones(len(idx), bool), half[idx] == 0, half[idx] == 1)):
                phi[h, i, s] = _wls(Y[m_], X[m_], wt[idx][m_])
    return dict(phi=phi, emean=emean, n=n, centres=(np.arange(nb) + 0.5) * block, block=block)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def decouple_stats(B, psi):
    """d_s (deviation from the common shift), its half-split noise reference, r_s (residual incl. eps) -> dict"""
    phi, ph1, ph2 = B['phi']
    ok = np.isfinite(phi).all(1)
    dev = wrap(phi - psi[None])                                   # phi_s(b) - psi_s
    dn = wrap(ph1 - ph2) / 2                                      # noise reference (no signal)
    out = dict(ok=ok)
    for a in (0, 1):
        sl = slice(3 * a, 3 * a + 3)
        d = dev[:, sl] - np.angle(np.exp(1j * dev[:, sl]).mean(1, keepdims=True))
        d = wrap(d)
        z = dn[:, sl] - dn[:, sl].mean(1, keepdims=True)
        r = wrap(dev[:, sl] - B['emean'][:, sl])
        dd = d[ok]
        acov = {L_: float(np.mean(dd[:-L_] * dd[L_:])) for L_ in (1, 2, 5, 10)}
        pers = float(np.mean([np.mean(dd[:-L_] * dd[L_:]) for L_ in range(2, 11)]))
        out[a] = dict(d=d, dnoise=z, r=r, acorr={L_: v / float(np.mean(dd ** 2)) for L_, v in acov.items()},
                      rms_persistent=float(np.sqrt(max(pers, 0))),
                      prec_step=float(np.sqrt(np.mean(z[ok] ** 2)) * math.sqrt(1.5)),
                      std3=d.std(1, ddof=1), std3_noise=z.std(1, ddof=1),
                      rms_d=float(np.sqrt(np.mean(d[ok] ** 2))), rms_d_noise=float(np.sqrt(np.mean(z[ok] ** 2))),
                      rms_r=float(np.sqrt(np.mean(r[ok] ** 2))), rms_noise_step=float(np.sqrt(np.mean(dn[ok][:, sl] ** 2))),
                      mean_std3=float(np.mean(d.std(1, ddof=1)[ok])), mean_std3_noise=float(np.mean(z.std(1, ddof=1)[ok])))
    return out


def fig_phase_decouple(run, outdir, nframes=120000):
    pat = load_pattern(run)
    pitch_nm = [2 * np.pi / np.hypot(*pat['K'][a]) * 108.33 for a in (0, 1)]
    cache = f'{outdir}/si_phase_decouple_blockfits.npz'
    if os.path.exists(cache):
        Z = np.load(cache, allow_pickle=True)
        BA = Z['BA'].item(); BB = Z['BB'].item(); cinfo = Z['cinfo'].item()
    else:
        cc = f'{outdir}/si_phase_decouple_cleanrows.npz'
        if os.path.exists(cc):
            Z = np.load(cc, allow_pickle=True); C = {k: Z[k] for k in ('w', 'x', 'y', 'I', 'phot')}; cinfo = Z['cinfo'].item()
        else:
            C, cinfo = load_clean(run); np.savez(cc, cinfo=cinfo, **C)
        print('clean rows', cinfo, flush=True)
        BA = block_phases(C, pat, 3000); BB = block_phases(C, pat, 150)
        np.savez(cache, BA=BA, BB=BB, cinfo=cinfo)
    psi = pat['psi']; deg = np.degrees
    SA = decouple_stats(BA, psi); SB = decouple_stats(BB, psi)
    fk = np.arange(pat['eps'].shape[1]) * pat['kf']; keep = fk <= nframes

    # display in the preprint's convention: sin(q_i.(x, y) - phi_ij - eps_i(t)), phi_ij = psi_s - pi/2 (code: cos(... - psi_s ...))
    phi = wrap(psi - np.pi / 2)
    fig = plt.figure(figsize=(15, 9.6))
    gs = fig.add_gridspec(3, 5, width_ratios=[1, 0.2, 1, 0.3, 1], height_ratios=[1, 1, 1.05], hspace=0.55, wspace=0.08,
                          left=0.07, right=0.98, top=0.94, bottom=0.07)
    for a in (0, 1):
        sl = range(3 * a, 3 * a + 3)
        ax1 = fig.add_subplot(gs[a, 0]); ax2 = fig.add_subplot(gs[a, 2]); ax3 = fig.add_subplot(gs[a, 4])
        okA = SA['ok']
        for j, s in enumerate(sl):
            y = deg(phi[s] + wrap(BA['phi'][0, :, s] - psi[s]))
            ax1.plot(BA['centres'][okA] / 1e3, y[okA], '.', ms=5, color=STEP_COLORS[j], label=f'step j = {j + 1}')
            ax3.axhline(deg(phi[s]), color=STEP_COLORS[j], lw=2.5)
        ax2.plot(fk[keep] / 1e3, deg(pat['eps'][a, keep]), color='k', lw=1.2)
        ax2.plot(BA['centres'][okA] / 1e3, deg(np.angle(np.exp(1j * wrap(BA['phi'][0][:, list(sl)] - psi[list(sl)][None])).mean(1)))[okA],
                 'o', ms=3.5, mfc='none', mec='0.4', mew=0.8)
        lo = min(deg(phi[list(sl)]).min(), np.nanmin(deg(phi[list(sl)][None] + wrap(BA['phi'][0][:, list(sl)] - psi[list(sl)][None])))) - 25
        hi = max(deg(phi[list(sl)]).max(), np.nanmax(deg(phi[list(sl)][None] + wrap(BA['phi'][0][:, list(sl)] - psi[list(sl)][None])))) + 25
        for ax in (ax1, ax3): ax.set_ylim(lo, hi + 70)
        ax2.set_ylim(-25, 25)
        ax1.set_ylabel(f'angle i = {a + 1}\nphase (deg)')
        sec = ax2.secondary_yaxis('right', functions=(lambda v, p=pitch_nm[a]: v / 360 * p, lambda v, p=pitch_nm[a]: v * 360 / p))
        sec.set_ylabel('nm', labelpad=2)
        ax3.set_yticklabels([])
        ax1.legend(loc='upper center', fontsize=10, ncol=3, handletextpad=0.1, columnspacing=0.8, frameon=False, markerscale=1.5)
        for ax in (ax1, ax2, ax3):
            ax.set_xlim(0, nframes / 1e3)
            if a == 1: ax.set_xlabel('frame (x1000)')
        if a == 0:
            ax1.set_title('Per-step phases\n(independent fits, 3000-frame blocks)', fontsize=13)
            ax2.set_title('Shared phase drift $\\epsilon_i(t)$ (deg)\n(pattern estimation)', fontsize=13)
            ax3.set_title('Fixed per-step phases $\\phi_{i,j}$', fontsize=13)
            pass
        for x_, t_, xx in ((gs[a, 1], '=', 0.3), (gs[a, 3], '+', 0.72)):
            axt = fig.add_subplot(x_); axt.axis('off'); axt.text(xx, 0.5, t_, ha='center', va='center', fontsize=24)
    fig.text(0.01, 0.955, 'A', fontsize=20, weight='bold')
    # B: per-block (150-frame) deviation of the 3 steps from a common shift
    axb = fig.add_subplot(gs[2, :])
    okB = SB['ok']
    for a, c in ((0, 'k'), (1, 'tab:purple')):
        axb.plot(BB['centres'][okB] / 1e3, deg(SB[a]['std3'][okB]), lw=0.7, color=c, alpha=0.8,
                 label=f'angle {a + 1}: mean {deg(SB[a]["mean_std3"]):.2f} deg (noise ref. {deg(SB[a]["mean_std3_noise"]):.2f})')
        axb.axhline(deg(SB[a]['mean_std3_noise']), color='w', ls='-', lw=4, zorder=5)
        axb.axhline(deg(SB[a]['mean_std3_noise']), color=c, ls='--', lw=2, zorder=6)
    axb.set_xlim(0, nframes / 1e3); axb.set_ylim(0, 11)
    axb.set_xlabel('frame (x1000)'); axb.set_ylabel('std over 3 steps (deg)')
    axb.set_title('Per-block deviation of the 3 phase steps from a shared shift (150-frame blocks; dashed: noise reference)', fontsize=13)
    axb.legend(loc='upper right', fontsize=10, frameon=False, ncol=2)
    fig.text(0.01, 0.335, 'B', fontsize=20, weight='bold')
    save(fig, outdir, 'si_phase_decouple')

    # numbers
    eps_rng = {a: [float(deg(pat['eps'][a, keep].min())), float(deg(pat['eps'][a, keep].max()))] for a in (0, 1)}
    eps_rms = {a: float(deg(np.std(pat['eps'][a, keep]))) for a in (0, 1)}
    V = dict(run=run, pattern=f'{run}/pattern_est.pt', clean=cinfo, psi_deg=deg(psi).round(2).tolist(),
             phi_deg=deg(phi).round(2).tolist(),
             step_spacing_deg=[float(deg((psi[3 * a + j + 1] - psi[3 * a + j]) % (2 * np.pi))) for a in (0, 1) for j in (0, 1)],
             pitch_nm=pitch_nm, eps_range_deg=eps_rng, eps_rms_deg=eps_rms,
             eps_range_nm={a: [v / 360 * pitch_nm[a] for v in eps_rng[a]] for a in (0, 1)},
             eps_rms_nm={a: eps_rms[a] / 360 * pitch_nm[a] for a in (0, 1)},
             blocks150=dict(n_blocks_used=int(okB.sum()), median_rows=float(np.median(BB['n'][okB])),
                            **{f'angle{a}': dict({k: float(deg(SB[a][k])) for k in ('rms_d', 'rms_d_noise', 'rms_r', 'rms_noise_step', 'mean_std3', 'mean_std3_noise', 'rms_persistent', 'prec_step')}, acorr_d=SB[a]['acorr']) for a in (0, 1)}),
             blocks3000=dict(n_blocks_used=int(SA['ok'].sum()), median_rows=float(np.median(BA['n'][SA['ok']])),
                             **{f'angle{a}': dict({k: float(deg(SA[a][k])) for k in ('rms_d', 'rms_d_noise', 'rms_r', 'rms_noise_step', 'mean_std3', 'mean_std3_noise', 'rms_persistent', 'prec_step')}, acorr_d=SA[a]['acorr']) for a in (0, 1)}))
    json.dump(V, open(f'{outdir}/si_phase_decouple_values.json', 'w'), indent=1, default=float)
    V = json.loads(json.dumps(V, default=float))                   # str keys as in the json
    open(f'{outdir}/si_phase_decouple_caption.txt', 'w').write(caption_decouple(V, pat))
    return V


# ------------------------------------------------------------------------------------------- pattern plots
def fig_pattern_plots_extended(run, outdir, n_show=700, seed=1, name='si_pattern_plots_extended'):
    import torch
    from simcode.dataset import Dataset
    pat = load_pattern(run)
    L = Dataset.load(f'{run}/locs_p1.hdf5')
    sel = L.mask().numpy()
    inreg = L.mask(max_precision_nm=None, min_score=None).numpy()
    F = L.numpy()
    x, y = F['pos'][:, 0].astype(np.float64), F['pos'][:, 1].astype(np.float64)
    xp, yp = F['x_prior'].astype(np.float64), F['y_prior'].astype(np.float64)    # pass-1 head (pattern-free) positions
    I = F['intensities'].astype(np.float64); w = F['frame'].astype(np.int64)
    good = inreg & np.isfinite(I).all(1) & (I.sum(1) > 0) & np.isfinite(x) & np.isfinite(y)
    sets = dict(sel=good & sel, rest=good & ~sel)
    K, psi, m, rel = pat['K'], pat['psi'], pat['m'], pat['rel']
    rng = np.random.default_rng(seed)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True, gridspec_kw=dict(wspace=0.12))
    V = dict(run=run, table=f'{run}/locs_p1.hdf5', n_in_crop=int(good.sum()), n_selected=int(sets['sel'].sum()),
             n_rest=int(sets['rest'].sum()), n_show_rows=n_show, seed=seed, depth=m.tolist(),
             pitch_nm=[2 * np.pi / np.hypot(*K[a]) * 108.33 for a in (0, 1)],
             angle_deg=[float(np.degrees(np.arctan2(K[a, 1], K[a, 0]))) for a in (0, 1)], rel=rel.tolist(), fits={})
    bins = np.linspace(0, 2 * np.pi, 25); bc = 0.5 * (bins[1:] + bins[:-1])
    for a in (0, 1):
        ax = axs[a]
        for key, col, lab in (('rest', REJ_COLOR, 'not selected'), ('sel', SEL_COLOR, 'selected')):
            idx = np.nonzero(sets[key])[0]
            f = w[idx, None] + np.arange(T6)[None]; st = f % 6
            cols = np.stack([np.argmax(st == s, 1) for s in range(3 * a, 3 * a + 3)], 1)
            ft = np.take_along_axis(f, cols, 1)
            e = eps_at(pat, ft)[a]
            ph = (K[a, 0] * x[idx, None] + K[a, 1] * y[idx, None] - psi[np.arange(3 * a, 3 * a + 3)][None] - e) % (2 * np.pi)
            yn = np.take_along_axis(I[idx], cols, 1) / I[idx].sum(1, keepdims=True)
            # effective modulation of this set: 6 y / rel_a = A + B cos ph + C sin ph (all rows, least squares)
            ra = rel[3 * a:3 * a + 3].mean() / rel.sum() * 6
            yy = (yn * 6 / ra).ravel(); pp = ph.ravel()
            okf = np.isfinite(yy) & (np.abs(yy) < 10)
            X = np.stack([np.ones(okf.sum()), np.cos(pp[okf]), np.sin(pp[okf])], 1)
            c, *_ = np.linalg.lstsq(X, yy[okf], rcond=None)
            # the same fit with the phase from the pattern-free pass-1 prior position (independent of the pattern solve)
            php = (K[a, 0] * xp[idx, None] + K[a, 1] * yp[idx, None] - psi[np.arange(3 * a, 3 * a + 3)][None] - e) % (2 * np.pi)
            pq = php.ravel(); okq = okf & np.isfinite(pq)
            cq, *_ = np.linalg.lstsq(np.stack([np.ones(okq.sum()), np.cos(pq[okq]), np.sin(pq[okq])], 1), yy[okq], rcond=None)
            mb = np.array([np.mean(yn.ravel()[(pp >= lo) & (pp < hi)]) for lo, hi in zip(bins[:-1], bins[1:])])
            V['fits'][f'angle{a}_{key}'] = dict(n_rows=int(len(idx)), m_eff=float(np.hypot(c[1], c[2]) / c[0]),
                                               phase_offset_deg=float(np.degrees(np.arctan2(c[2], c[1]))), mean_level=float(c[0]),
                                               m_eff_prior_pos=float(np.hypot(cq[1], cq[2]) / cq[0]),
                                               phase_offset_prior_pos_deg=float(np.degrees(np.arctan2(cq[2], cq[1]))))
            sh = rng.choice(len(idx), min(n_show, len(idx)), replace=False)
            ax.scatter(ph[sh].ravel(), yn[sh].ravel(), s=2.5, color=col, alpha=0.45, lw=0, rasterized=True,
                       label=f'{lab} ({len(idx) / 1e6:.2f} M rows; $m_{{eff}}$ {np.hypot(c[1], c[2]) / c[0]:.2f})')
            ax.plot(bc, mb, 'o', ms=5, color=col, mec='k', mew=0.6, zorder=4)
        pp = np.linspace(0, 2 * np.pi, 300)
        ax.plot(pp, rel[3 * a:3 * a + 3].mean() / rel.sum() * (1 + m[a] * np.cos(pp)), 'k', lw=2,
                label=f'estimated pattern (m = {m[a]:.3f})', zorder=5)
        ax.set_xlim(0, 2 * np.pi); ax.set_ylim(-0.05, 0.6)
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]); ax.set_xticklabels(['0', '$\\pi$/2', '$\\pi$', '3$\\pi$/2', '2$\\pi$'])
        ax.set_xlabel('pattern phase (rad)')
        ax.set_title(f'angle {a} (pitch {V["pitch_nm"][a]:.1f} nm, {V["angle_deg"][a]:.1f} deg)', fontsize=14)
        ax.legend(loc='upper right', fontsize=9.5, frameon=True, framealpha=0.85, markerscale=4)
        ax.text(-0.02, 1.04, 'AB'[a], transform=ax.transAxes, fontsize=20, weight='bold', ha='right')
    axs[0].set_ylabel('normalized intensity $I_t / \\sum_6 I$')
    save(fig, outdir, name)
    json.dump(V, open(f'{outdir}/{name}_values.json', 'w'), indent=1, default=float)
    dv = f'{outdir}/si_phase_decouple_values.json'
    open(f'{outdir}/{name}_caption.txt', 'w').write(
        caption_pattern(V, json.load(open(dv)) if os.path.exists(dv) else None))
    return V


def phase_rms_line(D):
    """the one-line phase-estimation rms summary (degrees) from the si_phase_decouple values"""
    b = D['blocks150']; q = lambda k: [b[f'angle{a}'][k] for a in (0, 1)]
    o = lambda v: float(np.sqrt(np.mean(np.square(v))))
    r, n, dd, dn, pp = q('rms_r'), q('prec_step'), q('rms_d'), q('rms_d_noise'), q('rms_persistent')
    return (f'PHASE-ESTIMATION RMS (150-frame blocks, {b["n_blocks_used"]} blocks, median {b["median_rows"]:.0f} clean emitter-windows '
            f'per block): per-step phase scatter around the shared-drift model (phi_ij + eps_i(t)) rms {r[0]:.2f} / {r[1]:.2f} deg '
            f'(angle 1 / 2; overall {o(r):.2f} deg = {o(r) / 360 * np.mean(D["pitch_nm"]):.2f} nm); statistical precision of one per-step '
            f'block phase (site-split halves, sqrt(3/2) x noise rms of d) {n[0]:.2f} / {n[1]:.2f} deg (overall {o(n):.2f} deg). Step-specific part (deviation of the 3 '
            f'steps from a common shift) rms {dd[0]:.2f} / {dd[1]:.2f} deg vs noise {dn[0]:.2f} / {dn[1]:.2f} deg; persistent '
            f'(lag 2-10 autocovariance) {pp[0]:.2f} / {pp[1]:.2f} deg. 3000-frame blocks: residual rms '
            f'{D["blocks3000"]["angle0"]["rms_r"]:.2f} / {D["blocks3000"]["angle1"]["rms_r"]:.2f} deg, precision '
            f'{D["blocks3000"]["angle0"]["prec_step"]:.2f} / {D["blocks3000"]["angle1"]["prec_step"]:.2f} deg.')


def interp(V):
    b = V['blocks150']
    x = [b[f'angle{a}']['rms_d'] / b[f'angle{a}']['rms_d_noise'] for a in (0, 1)]
    t = (f'INTERPRETATION: the step-specific deviation d is {x[0]:.2f} / {x[1]:.2f} x its noise reference (150-frame blocks); '
         f'persistent part {b["angle0"]["rms_persistent"]:.2f} / {b["angle1"]["rms_persistent"]:.2f} deg. ')
    t += ('So the 3 steps of an angle follow one shared drift within the estimation noise (shared-drift model supported). '
          if max(x) < 1.15 else 'The 3 steps deviate from a shared drift beyond the noise (step-specific wander). ')
    t += ('The residual to phi + eps (res) exceeds the per-step precision because it also contains the error of the estimated eps '
          '(3-block running mean, other weighting) and the block-to-block change of the spatial phase field over the emitter '
          'population; both are common to the 3 steps and cancel in d. The lag-1 autocorrelation of d is expected from emitters whose '
          'windows span adjacent blocks.')
    return t


def caption_decouple(V, pat):
    b = V['blocks150']; c = V['clean']; deg = np.degrees
    L = ['DRAFT CAPTION',
         'Shared phase drift of the illumination pattern (low-density DNA-PAINT movie of main-text Fig. 3, 1809/cover1/2, '
         'SIMCODE pattern estimation). SIMCODE models the phase of each of the 6 illumination patterns as a fixed phase phi_ij '
         '(angle i, phase step j) plus a phase drift eps_i(t) at time t, shared by the 3 phase steps of angle i (the DMD shifts the '
         'pattern in discrete pixel steps, so a drift of the optics moves all 3 steps together). (A) Per angle: the per-step phases '
         'fitted independently in 3000-frame blocks (left) decompose into the shared drift estimated by the pattern estimation (middle; '
         'black: eps_i(t) on 150-frame knots; grey circles: the block mean of the left panel minus phi_ij) and the fixed phases '
         'phi_ij (right). '
         '(B) Test of the shared-drift assumption in 150-frame blocks: standard deviation over the 3 steps of the per-step phase '
         "deviation from the block's common shift (solid) and the same statistic for pure estimation noise (dashed; from two "
         f'independent halves of the emitters, split by position). The 3 steps deviate from a common shift by '
         f'{b["angle0"]["rms_d"]:.1f} / {b["angle1"]["rms_d"]:.1f} deg rms (angle 1 / 2), as expected from the estimation noise '
         f'alone ({b["angle0"]["rms_d_noise"]:.1f} / {b["angle1"]["rms_d_noise"]:.1f} deg), against a drift range of {V["eps_range_deg"]["0"][0]:.0f} to '
         f'{V["eps_range_deg"]["0"][1]:.0f} deg and {V["eps_range_deg"]["1"][0]:.0f} to {V["eps_range_deg"]["1"][1]:.0f} deg.',
         '',
         phase_rms_line(V),
         '',
         'NUMBERS (si_phase_decouple_values.json)',
         f'Pattern: pitch {V["pitch_nm"][0]:.1f} / {V["pitch_nm"][1]:.1f} nm, depth m {pat["m"][0]:.3f} / {pat["m"][1]:.3f}, '
         f'phi_ij (deg; i = 1, 2, j = 1, 2, 3) {V.get("phi_deg", V["psi_deg"])} (= code psi_s - 90 deg, sin convention), step spacing (deg) '
         f'{[round(v, 2) for v in V["step_spacing_deg"]]}.',
         f'Shared drift eps_i(t): range {V["eps_range_deg"]["0"][0]:.1f} to {V["eps_range_deg"]["0"][1]:.1f} deg (angle 1; '
         f'{V["eps_range_nm"]["0"][0]:.1f} to {V["eps_range_nm"]["0"][1]:.1f} nm), {V["eps_range_deg"]["1"][0]:.1f} to '
         f'{V["eps_range_deg"]["1"][1]:.1f} deg (angle 2; {V["eps_range_nm"]["1"][0]:.1f} to {V["eps_range_nm"]["1"][1]:.1f} nm); rms '
         f'{V["eps_rms_deg"]["0"]:.2f} / {V["eps_rms_deg"]["1"]:.2f} deg = {V["eps_rms_nm"]["0"]:.2f} / {V["eps_rms_nm"]["1"]:.2f} nm '
         '(knots every 150 frames, 3-block running mean in the estimation, mean removed; nm = deg / 360 x pitch).',
         f'Rows: {c["n_rows"]} pass-1 localizations, {c["n_on"]} on through the window, {c["n_clean"]} clean (nn >= {c["nn_min"]} px, '
         f'>= {c["min_phot"]:.0f} photons) = the rows of the run\'s pattern estimation (pattern_estimation.select).']
    for key, lab in (('blocks150', '150-frame blocks'), ('blocks3000', '3000-frame blocks')):
        B = V[key]
        L.append(f'{lab}: {B["n_blocks_used"]} blocks, median {B["median_rows"]:.0f} clean rows per block.')
        for a in (0, 1):
            A = B[f'angle{a}']
            L.append(f'  angle {a + 1}: step-specific deviation d rms {A["rms_d"]:.2f} deg (noise ref. {A["rms_d_noise"]:.2f}); mean std over 3 '
                     f'steps {A["mean_std3"]:.2f} (noise {A["mean_std3_noise"]:.2f}); residual to phi + eps rms {A["rms_r"]:.2f}; '
                     f'per-step precision {A["prec_step"]:.2f} (site-split difference incl. the phase-field difference of the halves: '
                     f'{A["rms_noise_step"]:.2f}); persistent part {A["rms_persistent"]:.2f} deg; autocorrelation of d at '
                     f'lag 1/2/5/10 blocks {", ".join(f"{v:.2f}" for v in A["acorr_d"].values())}.')
    L += ['Definitions: blockphase_ij(b) = phase of angle i, step j in block b fitted with eps = 0 (pattern_estimation.fit_steps '
          'regression); d_ij(b) = (blockphase_ij(b) - phi_ij) - circular mean over the angle\'s 3 steps; noise reference = same '
          'statistic on (blockphase_ij^h1 - blockphase_ij^h2) / 2 of two halves split by a 3-px checkerboard of the position (all '
          'windows of an emitter in one half); residual res_ij(b) = blockphase_ij(b) - phi_ij - mean eps_i at the rows\' frames; '
          'persistent part = sqrt(mean autocovariance of d over lags 2-10 blocks) (noise is independent between blocks). Code '
          'names: phi_ij = psi_s - pi/2 (s = 3 (i - 1) + j - 1), eps_i = eps_a (a = i - 1).',
          interp(V),
          '',
          f'FILES: {V["pattern"]} (k, phi, m, logrel, dphi, knot_frames); pass-1 chunks {V["run"]}/pass1_*.npz.',
          'CODE: analysis/paper/si_pattern.py (fig_phase_decouple, block_phases, decouple_stats); simcode/processing/pattern_estimation.py '
          '(select, fit_steps, fit_eps, _interp_eps); simcode/processing/common.py (load_pass1_rows).',
          'Rerun: PYTHONPATH=. CUDA_VISIBLE_DEVICES= python analysis/paper/si_pattern.py --which decouple (delete '
          'si_phase_decouple_blockfits.npz to refit)']
    return '\n'.join(L) + '\n'


def caption_pattern(V, D=None):
    f = V['fits']
    L = ['DRAFT CAPTION',
         'Pattern modulation of the SIMCODE localizations (low-density DNA-PAINT movie of main-text Fig. 3, 1809/cover1/2). '
         'Each point is one frame of a localization: its pass-1 network intensity normalized by the sum of the 6 intensities of '
         'the window, I_t / sum_6 I, against the illumination phase k_a.r - psi_s - eps_a(t) at the position from the explicit '
         'pattern solve, for orientation 0 (A) and 1 (B). Red: localizations in the paper\'s selection (detection score p >= 0.63, '
         'calibrated precision max(sigma_x, sigma_y) <= 12 nm); grey: all other localizations in the same region (mostly low-score '
         f'detections). {V["n_show_rows"]} randomly chosen localizations per set and orientation are shown (3 frames each); circles = '
         'mean per phase bin (24 bins) over all localizations of the set. Black: the estimated pattern rel_s (1 + m_a cos phase) / '
         f'sum rel with depth m = {V["depth"][0]:.3f} / {V["depth"][1]:.3f} and pitch {V["pitch_nm"][0]:.1f} / {V["pitch_nm"][1]:.1f} nm. '
         f'The selected localizations follow the pattern with an effective modulation of {f["angle0_sel"]["m_eff"]:.2f} / '
         f'{f["angle1_sel"]["m_eff"]:.2f}, the others only {f["angle0_rest"]["m_eff"]:.2f} / {f["angle1_rest"]["m_eff"]:.2f}. '
         f'(The solved position itself uses these intensities and the pattern; with the pattern-free pass-1 position the '
         f'selected localizations give {f["angle0_sel"]["m_eff_prior_pos"]:.2f} / {f["angle1_sel"]["m_eff_prior_pos"]:.2f}, lowered by '
         f'its larger position error.)',
         '']
    if D is not None: L += [phase_rms_line(D), '']
    L += ['NUMBERS (si_pattern_plots_extended_values.json)',
          f'Pattern (pattern_est.pt): depth m {V["depth"][0]:.4f} / {V["depth"][1]:.4f} (cross-cycle estimator, clean emitter-windows), '
          f'pitch {V["pitch_nm"][0]:.2f} / {V["pitch_nm"][1]:.2f} nm, orientation {V["angle_deg"][0]:.2f} / {V["angle_deg"][1]:.2f} deg '
          f'(sensor x axis), rel intensities {[round(v, 3) for v in V["rel"]]}.',
          f'Rows in the crop: {V["n_in_crop"]}; selected {V["n_selected"]}; not selected {V["n_rest"]}. Shown: {V["n_show_rows"]} rows per '
          f'set and angle (seed {V["seed"]}).']
    for k, v in f.items():
        L.append(f'  {k}: n {v["n_rows"]}, m_eff {v["m_eff"]:.3f}, phase offset {v["phase_offset_deg"]:.2f} deg, mean level '
                 f'{v["mean_level"]:.3f}; with the pattern-free pass-1 prior position: m_eff {v["m_eff_prior_pos"]:.3f}, offset '
                 f'{v["phase_offset_prior_pos_deg"]:.2f} deg')
    L += ['m_eff: least-squares fit of 6 (I_t / sum I) / rel_a = A + B cos phase + C sin phase over all frames of the set, m_eff = '
          'hypot(B, C) / A, offset = atan2(C, B). NB the solved position uses these intensities and the pattern, which pulls the '
          'phase toward agreement; the prior-position m_eff (pattern-free head position) is the independent check.',
          '',
          f'FILES: {V["table"]} (pos = pattern-solve x, y; intensities = pass-1 network I_t; score; crlb = calibrated sigma); '
          f'{V["run"]}/pattern_est.pt.',
          'CODE: analysis/paper/si_pattern.py (fig_pattern_plots); simcode/dataset.py (Dataset.load, mask = simcode.table.select).',
          'Rerun: PYTHONPATH=. CUDA_VISIBLE_DEVICES= python analysis/paper/si_pattern.py --which pattern_ext']
    return '\n'.join(L) + '\n'


# ------------------------------------------------------------------------------- pattern plots, preprint layout
# SIMFLUX rows: the PSF-v2 + camera-v2 refit baselines (out/paper2/<ds>/baselines_psfv2, run_base_psfv2.sh; decision 2026-11-15,
# consistent with SIMCODE and Figs 3 / 4); the pre-refit tables were out/paper2/<ds>
DS = {'1809_2': dict(title='Low density DNA-PAINT (1809/2)', run=RUN, base='/home/jc/simcode2/out/paper2/1809_2/baselines_psfv2'),
      '2.5nM_2': dict(title='Medium density DNA-PAINT (2.5 nM/2)',
                      run='/mnt/data/dmdsf/bn_dnapaint/dnapaint/2.5nM/2/runs/simcode_v2_psfv2_final',
                      base='/home/jc/simcode2/out/paper2/2.5nM_2/baselines_psfv2')}
CROP = (5.0, 5.0, 110.0)
INC_COLOR, EXC_COLOR = 'b', 'r'          # the preprint's colours (included blue, excluded red)


def run_ready(run):
    try:
        return (json.load(open(f'{run}/run_config.json')).get('status') == 'done' and os.path.exists(f'{run}/locs_p1.hdf5')
                and os.path.exists(f'{run}/pattern_est.pt'))
    except Exception:
        return False


def _fit_meff(x, y, sin_conv):
    """y = A + B cos x + C sin x (least squares) -> m_eff = hypot(B, C) / A, offset of the max from the model's max (deg)"""
    ok = np.isfinite(x) & np.isfinite(y) & (np.abs(y) < 10)
    X = np.stack([np.ones(ok.sum()), np.cos(x[ok]), np.sin(x[ok])], 1)
    c, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    off = np.degrees(np.arctan2(c[2], c[1])) - (90.0 if sin_conv else 0.0)
    return float(np.hypot(c[1], c[2]) / c[0]), float((off + 180) % 360 - 180)


def simflux_pattern_rows(base, ep=0, me_thr=0.1, chunk=500000):
    """SIMFLUX baseline (analysis/baselines/paper_pipeline.py): the SMLM rows smlm/smlm.npy (per-frame intensities ibg of the
    6-frame window, SMLM 2D Gaussian position) and the SIMFLUX pattern sf/mp_sf.pickle (smlmtorch ModulationPattern); included =
    the SIMFLUX input filter mp_sf.mod_error(smlm_ds) < 0.1 (PAPER['mod_error_threshold']). Pattern ep: the frame of the window
    with that pattern (column (ep - frame) % 6), phase = k.pos - phase(frame) (smlmtorch draw_pattern), x = phase mod 2 pi;
    model relint (1 + depth sin x). Rows inside CROP."""
    from smlmtorch.simflux.pattern_estimator import ModulationPattern
    from simcode.dataset import Dataset
    mp = ModulationPattern.load(f'{base}/sf/mp_sf.pickle')
    F = Dataset.load(f'{base}/smlm/smlm.npy', window=6).numpy()
    pos = F['pos'][:, :2].astype(np.float64); fr = F['frame'].astype(np.int64); I = F['intensities'].astype(np.float64)
    if I.ndim == 3: I = I[..., 0]
    keep = (pos[:, 0] >= CROP[0]) & (pos[:, 0] < CROP[0] + CROP[2]) & (pos[:, 1] >= CROP[1]) & (pos[:, 1] < CROP[1] + CROP[2])
    pos, fr, I = pos[keep], fr[keep], I[keep]
    npat = mp.pattern_frames.size
    me = np.zeros(len(fr)); x = np.zeros(len(fr))
    k = mp.mod['k'][:, :2].astype(np.float64)
    for i in range(0, len(fr), chunk):
        sl = slice(i, i + chunk)
        mod = mp.mod_at_frame(fr[sl], npat)
        ph = (mod['k'][..., :2] * pos[sl, None]).sum(-1) - mod['phase']
        exc = mod['relint'] * (1 + mod['depth'] * np.sin(ph))
        me[sl] = np.abs(I[sl] / I[sl].sum(1, keepdims=True) - exc).max(1)
        e = (ep - fr[sl]) % npat
        x[sl] = ph[np.arange(len(e)), e] % (2 * np.pi)
    e = (ep - fr) % npat
    y = I[np.arange(len(fr)), e] / I.sum(1)
    m = mp.mod[ep]
    return dict(x=x, y=y, inc=me < me_thr, curve=lambda t: m['relint'] * (1 + m['depth'] * np.sin(t)), sin_conv=True,
                relint=float(m['relint']), depth=float(m['depth']), pitch_nm=float(2 * np.pi / np.hypot(*k[ep]) * 108.33),
                n=int(len(fr)), n_inc=int((me < me_thr).sum()), rule=f'modulation error < {me_thr} (own pattern, SMLM per-frame intensities at the SMLM position; the SIMFLUX input filter)',
                base='SMLM fits accepted by the baseline (chi^2 <= 4 x ROI^2 on the 8 x 8 px ROI, smlm.npy) in the crop',
                files=[f'{base}/smlm/smlm.npy', f'{base}/sf/mp_sf.pickle'])


def simcode_pattern_rows(run, ep=0, me_thr=0.1, chunk=4000000):
    """SIMCODE: locs_p1.hdf5 (pass-1 network intensities and the pattern-free pass-1 head position (x_prior, y_prior)),
    pattern_est.pt (its own estimated pattern). Base population = the detections p >= 0.63 (the detection threshold of the
    paper's selection, no precision cut) inside the crop with finite, positive-sum intensities and finite positions. Included =
    modulation error gamma < me_thr, the SIMFLUX input rule applied to SIMCODE's own data: gamma = max over the 6 frames t of
    |I_t / sum_6 I - e_t|, e_t = rel_s / sum rel (1 + m_a cos(k_a.r - psi_s - eps_a(w + t))), s = (w + t) % 6, a = s // 3, r =
    the head position (as SIMFLUX: its relint (1 + depth sin(k.r - phase)) at the SMLM position). Pattern ep: the frame of the
    window with t % 6 == ep, phase x at the head position (plotted); x_solved = at the pattern-solve position (reference);
    paper_sel = the paper's selection (p >= 0.63, max(sigma) <= 12 nm), reference only."""
    from simcode.dataset import Dataset
    pat = load_pattern(run); K, psi, m, rel = pat['K'], pat['psi'], pat['m'], pat['rel']
    L = Dataset.load(f'{run}/locs_p1.hdf5')
    sel = L.mask().numpy(); base = L.mask(max_precision_nm=None).numpy()          # p >= 0.63, crop
    F = L.numpy(); del L
    I = F['intensities']
    good = base & np.isfinite(I).all(1) & (I.sum(1) > 0) & np.isfinite(F['pos'][:, :2]).all(1) & \
        np.isfinite(F['x_prior']) & np.isfinite(F['y_prior'])
    gi = np.nonzero(good)[0]; n = len(gi)
    a = ep // 3; rn = rel / rel.sum()
    x = np.empty(n); xs = np.empty(n); y = np.empty(n); gam = np.empty(n)
    for c0 in range(0, n, chunk):
        ii = gi[c0:c0 + chunk]
        w = F['frame'][ii].astype(np.int64); Ic = I[ii].astype(np.float64); Ic = Ic / Ic.sum(1, keepdims=True)
        xp = F['x_prior'][ii].astype(np.float64); yp = F['y_prior'][ii].astype(np.float64)
        ft = w[:, None] + np.arange(T6)[None]; st = ft % 6; at = st // 3
        E = eps_at(pat, ft)                                                       # [2, n, 6]
        ea = np.where(at == 0, E[0], E[1])
        ph6 = K[at, 0] * xp[:, None] + K[at, 1] * yp[:, None] - psi[st] - ea
        exc = rn[st] * (1 + m[at] * np.cos(ph6))
        gam[c0:c0 + chunk] = np.abs(Ic - exc).max(1)
        j = (ep - w) % 6; t = w + j; e = eps_at(pat, t)[a]
        x[c0:c0 + chunk] = (K[a, 0] * xp + K[a, 1] * yp - psi[ep] - e) % (2 * np.pi)
        xs[c0:c0 + chunk] = (K[a, 0] * F['pos'][ii, 0] + K[a, 1] * F['pos'][ii, 1] - psi[ep] - e) % (2 * np.pi)
        y[c0:c0 + chunk] = Ic[np.arange(len(ii)), j]
    inc = gam < me_thr
    r0 = rn[ep]
    return dict(x=x, x_solved=xs, y=y, inc=inc, gamma=gam, paper_sel=sel[gi], curve=lambda tt: r0 * (1 + m[a] * np.cos(tt)),
                sin_conv=False, relint=float(r0), depth=float(m[a]), pitch_nm=float(2 * np.pi / np.hypot(*K[a]) * 108.33), n=int(n),
                n_inc=int(inc.sum()), rule=f'modulation error < {me_thr} (own pattern, pass-1 intensities at the pass-1 head position)',
                base='detections p >= 0.63 in the crop (no precision cut), finite positive intensities',
                files=[f'{run}/locs_p1.hdf5', f'{run}/pattern_est.pt'])


def cached_rows(meth, ds, ep, outdir):
    """simflux_pattern_rows / simcode_pattern_rows with an npz cache in <outdir>/cache_si_pattern/ (keyed on the input files'
    mtimes, so a rerun for the layout does not re-read the tables)"""
    cfg = DS[ds]
    fn = (lambda: simflux_pattern_rows(cfg['base'], ep)) if meth == 'SIMFLUX' else (lambda: simcode_pattern_rows(cfg['run'], ep))
    cd = os.path.join(outdir, 'cache_si_pattern'); os.makedirs(cd, exist_ok=True)
    import hashlib
    src = cfg['base'] if meth == 'SIMFLUX' else cfg['run']
    ver = '_gamma'                                                                # the common modulation-error rule (2026-11-15)
    cf = os.path.join(cd, f'{ds}_{meth}_p{ep}_{hashlib.md5(src.encode()).hexdigest()[:8]}{ver}.npz')     # keyed on the source path too
    if os.path.exists(cf):
        z = np.load(cf, allow_pickle=False); M = json.loads(str(z['meta']))
        if all(os.path.exists(f) and abs(os.path.getmtime(f) - t) < 1e-3 for f, t in M['mtimes'].items()):
            R = {k: z[k] for k in z.files if k != 'meta'}; R.update(M['scalars'])
            r0, dep, sc = R['relint'], R['depth'], R['sin_conv']
            R['curve'] = (lambda t: r0 * (1 + dep * np.sin(t))) if sc else (lambda t: r0 * (1 + dep * np.cos(t)))
            return R
    R = fn()
    arr = {k: v for k, v in R.items() if isinstance(v, np.ndarray)}
    sca = {k: v for k, v in R.items() if not isinstance(v, np.ndarray) and not callable(v)}
    np.savez(cf, meta=json.dumps(dict(mtimes={f: os.path.getmtime(f) for f in R['files'] if os.path.exists(f)}, scalars=sca)), **arr)
    return R


def fig_pattern_plots(datasets, outdir, n_show=2000, seed=1, ep=0, name='si_pattern_plots'):
    """A (SIMFLUX) | B (SIMCODE) side by side, one row per dataset (2026-11-15 layout request: page aspect h / w ~0.8 with
    two datasets); pattern ep (default 0 = angle 0, step 0)"""
    nd = len(datasets)
    W = 10.0; H = 0.8 * W if nd > 1 else 0.5 * W
    fig, axs = plt.subplots(nd, 2, figsize=(W, H), squeeze=False,
                            gridspec_kw=dict(hspace=0.36, wspace=0.22, left=0.12 if nd > 1 else 0.09, right=0.98,
                                             top=0.93 if nd > 1 else 0.88, bottom=0.15 if nd > 1 else 0.22))
    rng = np.random.default_rng(seed)
    V = dict(pattern=ep, n_show_per_panel=n_show, sampling='one random sample of the base population per panel (true proportions); red drawn first, blue on top', seed=seed, panels={}, layout='A SIMFLUX | B SIMCODE; rows = datasets')
    for ri, ds in enumerate(datasets):
        cfg = DS[ds]
        for ci, meth in enumerate(('SIMFLUX', 'SIMCODE')):
            R = cached_rows(meth, ds, ep, outdir); ax = axs[ri, ci]
            if not R['sin_conv']:
                # SIMCODE rows are in the code's cos convention (phase k_a.r - psi_s - eps_a(t)); plot them in the preprint's sin
                # convention, phase q_i.(x, y) - phi_ij - eps_i(t) with phi_ij = psi_s - pi/2, as the SIMFLUX panels
                R = dict(R); r0_, dep_ = R['relint'], R['depth']
                for k_ in ('x', 'x_solved'):
                    if k_ in R: R[k_] = (R[k_] + np.pi / 2) % (2 * np.pi)
                R['curve'] = lambda t, r0_=r0_, dep_=dep_: r0_ * (1 + dep_ * np.sin(t)); R['sin_conv'] = True
            P = dict(files=R['files'], rule=R['rule'], base=R.get('base'), n=R['n'], n_included=R['n_inc'],
                     frac_included=R['n_inc'] / R['n'], depth=R['depth'], relint=R['relint'], pitch_nm=R['pitch_nm'])
            if 'paper_sel' in R:
                ps = R['paper_sel'].astype(bool)
                P['paper_selection_frac_of_base'] = float(ps.mean())
                P['paper_selection_frac_within_0.1'] = float(R['inc'][ps].mean()) if ps.any() else float('nan')
            samp = rng.choice(R['n'], min(n_show, R['n']), replace=False)        # one sample per panel: true proportions
            P['n_shown'] = int(len(samp)); P['n_shown_included'] = int(R['inc'][samp].sum())
            for key, msk, col, lab in (('excluded', ~R['inc'], EXC_COLOR, 'modulation error $\\geq$ 0.1'),     # red first, blue on top
                                       ('included', R['inc'], INC_COLOR, 'modulation error < 0.1')):
                idx = np.nonzero(msk)[0]
                sh = samp[msk[samp]]
                ax.scatter(R['x'][sh], R['y'][sh], s=4, marker='o', lw=0, color=col, label=lab, alpha=0.8, rasterized=True)
                me, off = _fit_meff(R['x'][idx], R['y'][idx] / R['relint'], R['sin_conv'])
                P[f'm_eff_{key}'] = me; P[f'phase_offset_deg_{key}'] = off
                if 'x_solved' in R:
                    P[f'm_eff_{key}_solved_position'], P[f'phase_offset_deg_{key}_solved_position'] = _fit_meff(
                        R['x_solved'][idx], R['y'][idx] / R['relint'], R['sin_conv'])
            if 'x_solved' in R: P['phase_position'] = 'pass-1 head (x_prior, y_prior), pattern-free'
            tt = np.linspace(0, 2 * np.pi, 400)
            ax.plot(tt, R['curve'](tt), 'k', lw=2.5, label='Illumination pattern')
            ax.set_xlim(0, 2 * np.pi); ax.set_ylim(0, 0.5)
            ax.text(0.97, 0.96, f'{100 * P["frac_included"]:.0f} % within 0.1', transform=ax.transAxes, ha='right', va='top',
                    fontsize=12, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5))
            ax.set_xticks([0, np.pi, 2 * np.pi]); ax.set_xticklabels(['0', '$\\pi$', '2$\\pi$'])
            if ri == 0:
                ax.set_title(meth, fontsize=15)
                ax.text(-0.16, 1.04, 'AB'[ci], transform=ax.transAxes, fontsize=20, weight='bold')
            if ci == 0:
                ax.set_ylabel(('' if nd == 1 else cfg['title'].replace(' (', '\n(') + '\n') + 'Normalized intensity', fontsize=12)
            else:
                ax.set_ylabel('Normalized intensity', fontsize=12)
            if ri == nd - 1: ax.set_xlabel('Phase [radians]', fontsize=13)
            V['panels'][f'{ds}_{meth}'] = P
    if nd == 1: fig.suptitle(DS[datasets[0]]['title'], fontsize=15, y=0.995)
    h, l = axs[-1, 0].get_legend_handles_labels()
    h, l = [h[1], h[0]] + h[2:], [l[1], l[0]] + l[2:]                             # legend: blue, red, pattern
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=12, markerscale=3, frameon=True, bbox_to_anchor=(0.5, 0.0))
    save(fig, outdir, name)
    V['datasets'] = datasets; V['figsize_in'] = [W, H]
    json.dump(V, open(f'{outdir}/{name}_values.json', 'w'), indent=1, default=float)
    open(f'{outdir}/{name}_caption.txt', 'w').write(caption_pattern_preprint(V))
    return V


def caption_pattern_preprint(V):
    P = V['panels']; ds = V['datasets']
    fr = lambda d, m_: f'{100 * P[f"{d}_{m_}"]["frac_included"]:.0f} %'
    L = ['DRAFT CAPTION',
         'Localizations and their intensities mapped onto one period of the illumination pattern (angle i = 1, phase step '
         'j = 1), for SIMFLUX (A) and SIMCODE (B), on the ' + ' and '.join(DS[d]['title'].split(' (')[0].lower() for d in ds) + ' movie' + ('s (rows)' if len(ds) > 1 else '') + '. Intensities are normalized per '
         'localization by the sum of the 6 intensities of its window, mu_N_ij / sum_ij mu_N_ij, and plotted against the pattern '
         'phase q_i . (x, y) - phi_ij (- eps_i(t) for SIMCODE). Both methods are classified with the same rule, the SIMFLUX input filter: '
         'a localization is included (blue) if its modulation error gamma = max_ij |mu_N_ij / sum mu_N - r_ij (1 + m_i sin(phase_ij))| '
         'is below 0.1, and excluded (red) otherwise; each method uses its own intensities, its own pattern-free position and its own '
         'estimated pattern. SIMFLUX: per-frame intensities of the SMLM ROI fits and the SMLM position; population = the SMLM fits '
         'accepted by the baseline (chi^2 <= 4 x ROI^2, 8 x 8 px ROI). SIMCODE: pass-1 network intensities and the pass-1 network '
         'position (before the pattern is used); population = the detections with score p >= 0.63 (no precision cut). Both inside '
         'the field of view x, y in [5, 115) px. '
         f'{V["n_show_per_panel"]} random localizations per panel, in their true proportions (red drawn first, blue on top); the '
         'panel text gives the fraction within 0.1 of the whole population. Black: the estimated illumination pattern of each method (SIMFLUX: depth fixed to '
         '0.9, as in the SIMFLUX pipeline; SIMCODE: the estimated depth), r_ij (1 + m_i sin(phase)).',
         '', 'NUMBERS (si_pattern_plots_values.json; m_eff = effective modulation depth, least-squares sine fit of y / relint over '
         'ALL rows of the class)']
    for k, p in P.items():
        L.append(f'  {k}: base {p["n"]} ({p.get("base")}); within 0.1: {p["n_included"]} ({100 * p["frac_included"]:.1f} %); '
                 f'pattern depth {p["depth"]:.3f}, pitch {p["pitch_nm"]:.1f} nm; m_eff < 0.1 {p["m_eff_included"]:.3f} '
                 f'(peak offset {p["phase_offset_deg_included"]:.1f} deg), >= 0.1 {p["m_eff_excluded"]:.3f}' +
                 (f'; reference: paper selection = {100 * p["paper_selection_frac_of_base"]:.1f} % of the base, of which '
                  f'{100 * p["paper_selection_frac_within_0.1"]:.1f} % within 0.1; m_eff at the pattern-solve position '
                  f'{p["m_eff_included_solved_position"]:.3f} / {p["m_eff_excluded_solved_position"]:.3f}'
                  if 'paper_selection_frac_of_base' in p else ''))
    L += ['  Files: ' + '; '.join(f'{k}: {", ".join(p["files"])}' for k, p in P.items()),
          '', 'NOTES: the rule (gamma < 0.1) is the preprint\'s, now for both methods (decision 2026-11-15; before, the SIMCODE '
          'panels used the paper selection p >= 0.63 & max(sigma) <= 12 nm, which is blind to the pattern). Neither the classification '
          'nor the plotted phase uses a pattern-solved position, but the included points lie near the curve by construction for '
          'both methods (they are selected on their distance to it); the fraction within 0.1 and m_eff of the excluded rows are '
          'the informative numbers.',
          'Conventions: both panels in the preprint\'s sin convention, model r_ij (1 + m_i sin(phase)): SIMFLUX (smlmtorch) relint '
          '(1 + depth sin phase) as is; SIMCODE\'s code model rel_s (1 + m_a cos(k_a.r - psi_s - eps_a(t))) is plotted with phase + '
          'pi/2 (phi_ij = psi_s - pi/2).',
          'Extended version (both angles, SIMCODE only, binned means): si_pattern_plots_extended.*',
          '', 'CODE: analysis/paper/si_pattern.py (fig_pattern_plots, simflux_pattern_rows, simcode_pattern_rows); plot style: '
          'smlmtorch/simflux/localizer_report.py draw_pattern; SIMFLUX baseline: analysis/baselines/paper_pipeline.py.',
          'Rerun: PYTHONPATH=. CUDA_VISIBLE_DEVICES= python analysis/paper/si_pattern.py --which pattern --datasets 1809_2,2.5nM_2']
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', default=RUN, help='simcode.run output directory (pattern_est.pt, pass1_*.npz, locs_p1.hdf5)')
    ap.add_argument('--outdir', default=OUT)
    ap.add_argument('--which', default='all', choices=['all', 'decouple', 'pattern', 'pattern_ext'])
    ap.add_argument('--datasets', default='auto', help="comma list of DS keys for si_pattern_plots; auto = 1809_2 (+ 2.5nM_2 once done)")
    ap.add_argument('--nframes', type=int, default=120000)
    ap.add_argument('--n_show', type=int, default=2000, help='rows plotted per panel (one random sample, true proportions) in si_pattern_plots')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    if a.which in ('all', 'decouple'): print(json.dumps(fig_phase_decouple(a.run, a.outdir, a.nframes), indent=1, default=float))
    if a.which in ('all', 'pattern'):
        dss = ['1809_2'] + (['2.5nM_2'] if run_ready(DS['2.5nM_2']['run']) else []) if a.datasets == 'auto' else a.datasets.split(',')
        print('si_pattern_plots datasets:', dss, flush=True)
        print(json.dumps(fig_pattern_plots(dss, a.outdir, a.n_show), indent=1, default=float))
    if a.which in ('all', 'pattern_ext'): print(json.dumps(fig_pattern_plots_extended(a.run, a.outdir, 700), indent=1, default=float))
